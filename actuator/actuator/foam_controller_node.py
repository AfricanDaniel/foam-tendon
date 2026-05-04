#!/usr/bin/env python3

import csv
import math
import os
import signal
import threading
import time

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from actuator_interfaces.srv import MoveFoam, MoveFoamCircle, MoveFoamSquare

from dynamixel_sdk import (
    PortHandler,
    PacketHandler,
    COMM_SUCCESS,
)

try:
    from actuator.natnet import NatNetClient as _NatNetClient
    _NATNET_AVAILABLE = True
except ImportError:
    _NatNetClient = None
    _NATNET_AVAILABLE = False

# ── Hardware ──────────────────────────────────────────────────────────────────
DEVICENAME = '/dev/ttyUSB0'
BAUDRATE = 1_000_000
PROTOCOL_VERSION = 2.0

# ── Motor IDs ─────────────────────────────────────────────────────────────────
MOTOR_NORTH = 1  # CW = pull north,  CCW = release
MOTOR_EAST  = 2  # CW = pull east,   CCW = release
MOTOR_SOUTH = 3  # CCW = pull south, CW  = release
MOTOR_WEST  = 4  # CCW = pull west,  CW  = release
ALL_MOTORS  = [MOTOR_NORTH, MOTOR_EAST, MOTOR_SOUTH, MOTOR_WEST]

# ── Dynamixel register addresses ──────────────────────────────────────────────
ADDR_OPERATING_MODE   = 11
ADDR_TORQUE_ENABLE    = 64
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION    = 116
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132

EXTENDED_POSITION_CONTROL_MODE = 4
TORQUE_ENABLE    = 1
TORQUE_DISABLE   = 0
DEFAULT_VELOCITY = 20             # profile velocity (lower = slower / safer)
MOVING_THRESHOLD = 10             # pulses; within this = "arrived"
PULSES_PER_DEGREE = 4096 / 360.0

# ── Direction table ───────────────────────────────────────────────────────────
_S2 = 1.0 / math.sqrt(2.0)
DIRECTION_VECTORS: dict[str, tuple[float, float]] = {
    'N':         (0.0,  1.0),
    'NORTH':     (0.0,  1.0),
    'S':         (0.0, -1.0),
    'SOUTH':     (0.0, -1.0),
    'E':         (1.0,  0.0),
    'EAST':      (1.0,  0.0),
    'W':         (-1.0, 0.0),
    'WEST':      (-1.0, 0.0),
    'NE':        (_S2,  _S2),
    'NORTHEAST': (_S2,  _S2),
    'NW':        (-_S2, _S2),
    'NORTHWEST': (-_S2, _S2),
    'SE':        (_S2, -_S2),
    'SOUTHEAST': (_S2, -_S2),
    'SW':        (-_S2, -_S2),
    'SOUTHWEST': (-_S2, -_S2),
}

# ── OptiTrack ─────────────────────────────────────────────────────────────────
NATNET_SERVER_IP  = '129.105.73.172'
OPTITRACK_TIMEOUT = 2.0   # seconds of silence → treated as unavailable
COLLECT_HZ        = 50    # data collection sample rate

# ── State / data directories ──────────────────────────────────────────────────
def _find_workspace_root() -> str:
    path = os.path.abspath(__file__)
    for _ in range(10):
        parent = os.path.dirname(path)
        if parent == path:
            break
        if os.path.isdir(os.path.join(parent, 'src')):
            return parent
        path = parent
    return os.path.expanduser('~')

STATE_FILE_DEFAULT  = os.path.join(_find_workspace_root(), 'foam_motor_state.csv')
DATA_COLLECTION_DIR = os.path.join(
    _find_workspace_root(), 'src', 'actuator', 'data_collection'
)
CSV_FIELDS = ['motor_id', 'current_position', 'home_position']


class FoamControllerNode(Node):
    """
    Controls a foam cylinder suspended by four Dynamixel motors via string spools.

    Pull / release conventions
    --------------------------
    ID 1 (North): CW  → pull,    CCW → release
    ID 2 (East):  CW  → pull,    CCW → release
    ID 3 (South): CCW → pull,    CW  → release
    ID 4 (West):  CCW → pull,    CW  → release

    All four motors share one serial bus.  Commands are sent sequentially
    via write4ByteTxRx (~1 ms per motor at 1 Mbps), so all four are
    dispatched within ~4 ms — negligible relative to motor travel time.

    Motor-command formula
    ---------------------
    Given a displacement vector (dx = East, dy = North):
        motor_1 (North) degrees = -dy
        motor_3 (South) degrees = -dy   ← same value; opposite pull convention cancels
        motor_2 (East)  degrees = -dx
        motor_4 (West)  degrees = -dx   ← same value; opposite pull convention cancels

    Positive degrees → CCW rotation (increase position register).
    Negative degrees → CW  rotation (decrease position register).

    State persistence
    -----------------
    Motor positions are written to a CSV after every move and immediately
    on Ctrl-C / SIGTERM / motor disconnection.

    On first run (no CSV), the current hardware positions become the home
    positions. On all subsequent runs, home positions are loaded from the CSV
    so that /go_home can always return the foam to its origin.

    Data collection
    ---------------
    When /move_foam, /move_foam_circle, or /move_foam_square is called and
    OptiTrack is live, a background thread records motor positions, motor
    velocities (register 128, units of 0.229 RPM), and OptiTrack rigid-body
    pose at ~50 Hz into a timestamped CSV under data_collection/.

    If OptiTrack is not available the move executes normally with no CSV.
    """

    def __init__(self) -> None:
        super().__init__('foam_controller_node')

        self.declare_parameter('state_file', STATE_FILE_DEFAULT)
        self.state_file: str = (
            self.get_parameter('state_file').get_parameter_value().string_value
        )

        self._stop_requested = False
        self._port_open = False
        self._destroyed = False
        self.current_positions: dict[int, int] = {mid: 0 for mid in ALL_MOTORS}
        self.home_positions:    dict[int, int] = {mid: 0 for mid in ALL_MOTORS}

        # Serialises all Dynamixel port access so the data-collection thread
        # and the main executor thread never race on the serial bus.
        self._port_lock = threading.Lock()

        # OptiTrack live state (written by NatNet callbacks, read by collector)
        self._optitrack_client = None
        self._optitrack_available = False
        self._optitrack_pos = None        # list [x, y, z]
        self._optitrack_rot = None        # list [qx, qy, qz, qw]
        self._optitrack_home_natnet = None  # NatNet [x,y,z] captured when at home
        self._optitrack_lock = threading.Lock()
        self._last_optitrack_time = 0.0
        self._optitrack_frame_count = 0   # total NatNet frames received
        self._optitrack_rb_count = 0      # total rigid-body callbacks received

        # Ensure the data collection directory exists
        os.makedirs(DATA_COLLECTION_DIR, exist_ok=True)

        # Load last known state BEFORE touching hardware so that if a motor is
        # offline on startup we still have its last commanded position on hand.
        csv_existed = self._load_state_from_csv()

        # Open serial port
        self.port_handler   = PortHandler(DEVICENAME)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)

        if not self.port_handler.openPort():
            raise RuntimeError(f'Failed to open serial port {DEVICENAME}')
        if not self.port_handler.setBaudRate(BAUDRATE):
            raise RuntimeError('Failed to set baud rate')

        self._port_open = True

        # Init motors; live hardware position overwrites CSV current when reachable
        for mid in ALL_MOTORS:
            self._init_motor(mid)

        # First ever run: promote current hardware positions to home
        if not csv_existed:
            for mid in ALL_MOTORS:
                self.home_positions[mid] = self.current_positions[mid]
            self.get_logger().info(
                'No state file found – current positions saved as home.'
            )

        self._save_state()

        # Signal handlers guarantee save + stop even on hard Ctrl-C
        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        self.create_service(MoveFoam,       '/move_foam',        self._cb_move_foam)
        self.create_service(MoveFoamCircle, '/move_foam_circle', self._cb_move_circle)
        self.create_service(MoveFoamSquare, '/move_foam_square', self._cb_move_square)
        self.create_service(Trigger,        '/go_home',          self._cb_go_home)
        self.create_service(Trigger,        '/set_home',         self._cb_set_home)
        self.create_service(Trigger,        '/optitrack_status', self._cb_optitrack_status)

        self._start_optitrack()

        # One-shot timer: report optitrack data flow 5 s after startup
        self._startup_check_timer = self.create_timer(5.0, self._cb_optitrack_startup_check)

        home_str = ', '.join(f'M{m}={self.home_positions[m]}' for m in ALL_MOTORS)
        self.get_logger().info(
            f'FoamControllerNode ready.\n'
            f'  State file     : {self.state_file}\n'
            f'  Data directory : {DATA_COLLECTION_DIR}\n'
            f'  Home poses     : {home_str}\n'
            f'  OptiTrack      : {"connected" if self._optitrack_available else "not available"}'
        )

    # ── State file ────────────────────────────────────────────────────────────

    def _load_state_from_csv(self) -> bool:
        if not os.path.exists(self.state_file):
            return False
        try:
            with open(self.state_file, newline='') as f:
                rows = {int(r['motor_id']): r for r in csv.DictReader(f)}

            if not all(mid in rows for mid in ALL_MOTORS):
                self.get_logger().warn('State file incomplete – reinitialising.')
                return False

            for mid in ALL_MOTORS:
                self.current_positions[mid] = int(rows[mid]['current_position'])
                self.home_positions[mid]    = int(rows[mid]['home_position'])

            self.get_logger().info(f'State loaded from {self.state_file}')
            return True

        except Exception as exc:
            self.get_logger().warn(f'Could not read state file ({exc}) – reinitialising.')
            return False

    def _save_state(self) -> None:
        try:
            with open(self.state_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()
                for mid in ALL_MOTORS:
                    writer.writerow({
                        'motor_id':         mid,
                        'current_position': self.current_positions[mid],
                        'home_position':    self.home_positions[mid],
                    })
        except Exception as exc:
            self.get_logger().error(f'Failed to save state: {exc}')

    # ── Motor initialisation ──────────────────────────────────────────────────

    def _init_motor(self, motor_id: int) -> None:
        ph, port = self.packet_handler, self.port_handler
        ph.write1ByteTxRx(port, motor_id, ADDR_TORQUE_ENABLE,    TORQUE_DISABLE)
        ph.write1ByteTxRx(port, motor_id, ADDR_OPERATING_MODE,   EXTENDED_POSITION_CONTROL_MODE)
        ph.write1ByteTxRx(port, motor_id, ADDR_TORQUE_ENABLE,    TORQUE_ENABLE)
        ph.write4ByteTxRx(port, motor_id, ADDR_PROFILE_VELOCITY, DEFAULT_VELOCITY)

        try:
            pos = self._read_position(motor_id)
            self.current_positions[motor_id] = pos
            self.get_logger().info(f'Motor {motor_id} ready  pos={pos}')
        except RuntimeError:
            self.get_logger().warn(
                f'Motor {motor_id} did not respond – '
                f'using last known position {self.current_positions[motor_id]}'
            )

    # ── Low-level helpers ─────────────────────────────────────────────────────

    def _read_position(self, motor_id: int) -> int:
        """Read present position; acquires port lock on each attempt."""
        for _ in range(5):
            with self._port_lock:
                raw, comm, err = self.packet_handler.read4ByteTxRx(
                    self.port_handler, motor_id, ADDR_PRESENT_POSITION
                )
            if comm == COMM_SUCCESS and err == 0:
                return raw if raw <= 0x7FFFFFFF else raw - 0x100000000
            time.sleep(0.01)
        raise RuntimeError(f'Motor {motor_id}: position read failed after 5 attempts')

    def _compute_motor_commands(self, dx: float, dy: float) -> dict[int, float]:
        return {
            MOTOR_NORTH: -dy,
            MOTOR_SOUTH: -dy,
            MOTOR_EAST:  -dx,
            MOTOR_WEST:  -dx,
        }

    def _write_goal(self, motor_id: int, target: int) -> bool:
        with self._port_lock:
            comm, err = self.packet_handler.write4ByteTxRx(
                self.port_handler, motor_id, ADDR_GOAL_POSITION, target & 0xFFFFFFFF
            )
        if comm != COMM_SUCCESS or err != 0:
            self.get_logger().error(
                f'Motor {motor_id}: goal write FAILED  comm={comm}  err={err}'
            )
            return False
        return True

    def _execute(self, commands: dict[int, float]) -> None:
        for motor_id, degrees in commands.items():
            if abs(degrees) < 1e-4:
                continue
            target = int(self.current_positions[motor_id] + degrees * PULSES_PER_DEGREE)
            if self._write_goal(motor_id, target):
                self.current_positions[motor_id] = target
        self._save_state()

    def _send_absolute(self, positions: dict[int, int]) -> None:
        for motor_id, target in positions.items():
            if self._write_goal(motor_id, target):
                self.current_positions[motor_id] = target
        self._save_state()

    def _wait_for_all(self, timeout: float = 10.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._stop_requested:
                return False
            try:
                if all(
                    abs(self._read_position(mid) - self.current_positions[mid]) <= MOVING_THRESHOLD
                    for mid in ALL_MOTORS
                ):
                    return True
            except RuntimeError as exc:
                self.get_logger().error(f'Motor disconnected during move: {exc}')
                self._save_state()
                return False
            time.sleep(0.02)
        return False

    def _emergency_stop(self) -> None:
        if not self._port_open:
            self._save_state()
            return
        for mid in ALL_MOTORS:
            try:
                raw, comm, err = self.packet_handler.read4ByteTxRx(
                    self.port_handler, mid, ADDR_PRESENT_POSITION
                )
                if comm == COMM_SUCCESS and err == 0:
                    signed = raw if raw <= 0x7FFFFFFF else raw - 0x100000000
                    self.current_positions[mid] = signed
                    self.packet_handler.write4ByteTxRx(
                        self.port_handler, mid, ADDR_GOAL_POSITION, raw
                    )
            except Exception:
                pass
        self._save_state()

    # ── Signal handling ───────────────────────────────────────────────────────

    def _handle_signal(self, sig, frame) -> None:
        self._stop_requested = True
        if rclpy.ok():
            rclpy.shutdown()

    # ── OptiTrack ─────────────────────────────────────────────────────────────

    def _start_optitrack(self) -> None:
        if not _NATNET_AVAILABLE:
            self.get_logger().warn(
                'NatNet SDK not found – OptiTrack disabled.'
            )
            return
        try:
            client = _NatNetClient()
            client.set_client_address('0.0.0.0')
            client.set_server_address(NATNET_SERVER_IP)
            client.set_use_multicast(False)
            client.set_print_level(0)
            client.new_frame_listener    = self._optitrack_frame_cb
            client.rigid_body_listener   = self._optitrack_rigid_body_cb
            # 'd' = datastream mode (rigid body + frame data)
            if client.run('d'):
                self._optitrack_client    = client
                self._optitrack_available = True
                self.get_logger().info(f'OptiTrack connected to {NATNET_SERVER_IP}')
            else:
                self.get_logger().warn(
                    'OptiTrack connection failed – moves will execute without data collection.'
                )
        except Exception as exc:
            self.get_logger().warn(f'OptiTrack init error ({exc}) – continuing without it.')

    def _optitrack_frame_cb(self, data_frame) -> None:
        self._optitrack_frame_count += 1
        if self._optitrack_frame_count == 1:
            self.get_logger().info('OptiTrack: first NatNet frame received.')

    def _optitrack_rigid_body_cb(self, rigid_body_id, pos, rot) -> None:
        self._optitrack_rb_count += 1
        if self._optitrack_rb_count == 1:
            self.get_logger().info(
                f'OptiTrack: first rigid body received  id={rigid_body_id}'
                f'  pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})'
            )
        with self._optitrack_lock:
            self._optitrack_pos = list(pos)
            self._optitrack_rot = list(rot)
            self._last_optitrack_time = time.time()

    def _is_optitrack_live(self) -> bool:
        """True if the NatNet client successfully connected."""
        return self._optitrack_available

    def _cb_optitrack_startup_check(self) -> None:
        """Fires once 5 s after startup to report whether data is flowing."""
        self.destroy_timer(self._startup_check_timer)   # one-shot
        if not self._optitrack_available:
            return
        if self._optitrack_frame_count == 0:
            self.get_logger().warn(
                'OptiTrack: connected to Motive but ZERO frames received after 5 s.\n'
                '  Check Motive → Edit → Settings → Streaming:\n'
                '    • "Broadcast Frame Data" must be ON\n'
                '    • Transmission type must be "Unicast"\n'
                '    • Local Interface should match this machine\'s IP on the OptiTrack network'
            )
        elif self._optitrack_rb_count == 0:
            self.get_logger().warn(
                f'OptiTrack: {self._optitrack_frame_count} frames received but ZERO rigid bodies.\n'
                '  Make sure at least one Rigid Body asset exists and is active in Motive.'
            )
        else:
            self.get_logger().info(
                f'OptiTrack OK: {self._optitrack_frame_count} frames, '
                f'{self._optitrack_rb_count} rigid-body callbacks in first 5 s.'
            )

    def _cb_optitrack_status(
        self, request: Trigger.Request, response: Trigger.Response
    ):
        """Return a human-readable OptiTrack status string."""
        if not self._optitrack_available:
            response.success = False
            response.message = 'OptiTrack not connected (NatNet client failed to start).'
            return response

        with self._optitrack_lock:
            pos = list(self._optitrack_pos) if self._optitrack_pos is not None else None
            rot = list(self._optitrack_rot) if self._optitrack_rot is not None else None
            last_t = self._last_optitrack_time

        age = time.time() - last_t if last_t > 0 else float('inf')
        lines = [
            f'NatNet frames received : {self._optitrack_frame_count}',
            f'Rigid-body callbacks   : {self._optitrack_rb_count}',
            f'Last rigid-body data   : {age:.2f} s ago' if age < 1e9 else 'Last rigid-body data   : never',
        ]
        if pos is not None:
            lines.append(f'Last position (x,y,z)  : ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})')
            lines.append(f'Last rotation (qx,y,z,w): ({rot[0]:.4f}, {rot[1]:.4f}, {rot[2]:.4f}, {rot[3]:.4f})')
        response.success = self._optitrack_rb_count > 0
        response.message = '\n'.join(lines)
        return response

    # ── Data collection ───────────────────────────────────────────────────────

    def _make_csv_path(self, label: str) -> str:
        """Return a unique, systematically named path for a new collection CSV."""
        existing = sum(1 for f in os.listdir(DATA_COLLECTION_DIR) if f.endswith('.csv'))
        run_num  = existing + 1
        ts       = time.strftime('%Y%m%d_%H%M%S')
        filename = f'run_{run_num:04d}_{ts}_{label}.csv'
        return os.path.join(DATA_COLLECTION_DIR, filename)

    def _collect_data(self, csv_path: str, stop_event: threading.Event) -> None:
        """
        Background thread: samples motor positions+velocities and OptiTrack pose
        at ~COLLECT_HZ Hz, writing one row per sample to csv_path.

        Motor velocity units: raw register value (1 unit = 0.229 RPM).
        Position units: Dynamixel pulses (4096 per revolution).
        OptiTrack position: NatNet coordinate units (meters if Motive default).
        OptiTrack rotation: quaternion (qx, qy, qz, qw).
        """
        fields = [
            'timestamp_s',
            'motor_1_pos', 'motor_2_pos', 'motor_3_pos', 'motor_4_pos',
            'motor_1_vel', 'motor_2_vel', 'motor_3_vel', 'motor_4_vel',
            'optitrack_x', 'optitrack_y', 'optitrack_z',
            'optitrack_qx', 'optitrack_qy', 'optitrack_qz', 'optitrack_qw',
        ]
        interval = 1.0 / COLLECT_HZ
        t0 = time.time()

        with self._optitrack_lock:
            home_nat = list(self._optitrack_home_natnet) if self._optitrack_home_natnet else None

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            # Row with timestamp < 0 is a home-reference marker for the replayer.
            # It records the NatNet position at home so the replayer can correctly
            # zero any CSV that starts at a shifted position (e.g. centred circles).
            if home_nat is not None:
                writer.writerow({
                    'timestamp_s':  '-1.0',
                    'motor_1_pos':  '0', 'motor_2_pos': '0',
                    'motor_3_pos':  '0', 'motor_4_pos': '0',
                    'motor_1_vel':  '',  'motor_2_vel': '',
                    'motor_3_vel':  '',  'motor_4_vel': '',
                    'optitrack_x':  f'{home_nat[0]:.6f}',
                    'optitrack_y':  f'{home_nat[1]:.6f}',
                    'optitrack_z':  f'{home_nat[2]:.6f}',
                    'optitrack_qx': '', 'optitrack_qy': '',
                    'optitrack_qz': '', 'optitrack_qw': '',
                })

            while not stop_event.is_set():
                row: dict = {'timestamp_s': f'{time.time() - t0:.4f}'}

                for mid in ALL_MOTORS:
                    pos_val = vel_val = ''
                    try:
                        with self._port_lock:
                            raw_p, cp, ep = self.packet_handler.read4ByteTxRx(
                                self.port_handler, mid, ADDR_PRESENT_POSITION)
                            raw_v, cv, ev = self.packet_handler.read4ByteTxRx(
                                self.port_handler, mid, ADDR_PRESENT_VELOCITY)
                        if cp == COMM_SUCCESS and ep == 0:
                            abs_pos = raw_p if raw_p <= 0x7FFFFFFF else raw_p - 0x100000000
                            pos_val = abs_pos - self.home_positions[mid]
                        if cv == COMM_SUCCESS and ev == 0:
                            vel_val = raw_v if raw_v <= 0x7FFFFFFF else raw_v - 0x100000000
                    except Exception:
                        pass
                    row[f'motor_{mid}_pos'] = pos_val
                    row[f'motor_{mid}_vel'] = vel_val

                with self._optitrack_lock:
                    p = list(self._optitrack_pos) if self._optitrack_pos is not None else None
                    r = list(self._optitrack_rot) if self._optitrack_rot is not None else None
                    last_t = self._last_optitrack_time

                if last_t > 0 and (time.time() - last_t) > OPTITRACK_TIMEOUT:
                    p = r = None

                if p is not None:
                    row['optitrack_x'] = f'{p[0]:.6f}'
                    row['optitrack_y'] = f'{p[1]:.6f}'
                    row['optitrack_z'] = f'{p[2]:.6f}'
                else:
                    row['optitrack_x'] = row['optitrack_y'] = row['optitrack_z'] = ''

                if r is not None:
                    row['optitrack_qx'] = f'{r[0]:.6f}'
                    row['optitrack_qy'] = f'{r[1]:.6f}'
                    row['optitrack_qz'] = f'{r[2]:.6f}'
                    row['optitrack_qw'] = f'{r[3]:.6f}'
                else:
                    row['optitrack_qx'] = row['optitrack_qy'] = \
                        row['optitrack_qz'] = row['optitrack_qw'] = ''

                writer.writerow(row)
                f.flush()
                stop_event.wait(interval)

        self.get_logger().info(f'Collection saved → {csv_path}')

    def _start_collection(self, label: str):
        """Create CSV path, spawn collection thread. Returns (stop_event, thread)."""
        csv_path   = self._make_csv_path(label)
        self.get_logger().info(f'Data collection started → {os.path.basename(csv_path)}')
        stop_event = threading.Event()
        thread     = threading.Thread(
            target=self._collect_data, args=(csv_path, stop_event), daemon=True
        )
        thread.start()
        return stop_event, thread

    def _stop_collection(self, stop_event: threading.Event, thread: threading.Thread) -> None:
        stop_event.set()
        thread.join(timeout=2.0)

    # ── Service callbacks ─────────────────────────────────────────────────────

    def _cb_go_home(self, request: Trigger.Request, response: Trigger.Response):
        if self._stop_requested:
            response.success = False
            response.message = 'Node is shutting down'
            return response
        self.get_logger().info('GoHome: returning to home positions.')

        collecting = self._is_optitrack_live()
        if collecting:
            stop_event, collector = self._start_collection('go_home')
        else:
            self.get_logger().warn('OptiTrack not connected – skipping data collection.')

        max_delta_pulses = max(
            abs(self.current_positions[mid] - self.home_positions[mid])
            for mid in ALL_MOTORS
        )
        max_delta_deg = max_delta_pulses / PULSES_PER_DEGREE
        home_timeout = max(10.0, max_delta_deg / (DEFAULT_VELOCITY * 0.229 * 6.0) * 1.5 + 5.0)
        self._send_absolute(dict(self.home_positions))
        moved_ok = self._wait_for_all(timeout=home_timeout)

        if collecting:
            self._stop_collection(stop_event, collector)

        if not moved_ok:
            response.success = False
            response.message = 'Motors did not reach home (timeout or interrupt)'
            return response

        for mid in ALL_MOTORS:
            self.current_positions[mid] = self.home_positions[mid]
        self._save_state()

        with self._optitrack_lock:
            if self._optitrack_pos is not None:
                self._optitrack_home_natnet = list(self._optitrack_pos)

        response.success = True
        response.message = (
            'Home reached. '
            + ', '.join(f'M{m}={self.home_positions[m]}' for m in ALL_MOTORS)
        )
        return response

    def _cb_set_home(self, request: Trigger.Request, response: Trigger.Response):
        if self._stop_requested:
            response.success = False
            response.message = 'Node is shutting down'
            return response
        self.get_logger().info('SetHome: saving current positions as home.')
        for mid in ALL_MOTORS:
            try:
                pos = self._read_position(mid)
            except RuntimeError:
                pos = self.current_positions[mid]
            self.home_positions[mid]    = pos
            self.current_positions[mid] = pos
        with self._optitrack_lock:
            if self._optitrack_pos is not None:
                self._optitrack_home_natnet = list(self._optitrack_pos)
        self._save_state()
        response.success = True
        response.message = (
            'Home updated. '
            + ', '.join(f'M{m}={self.home_positions[m]}' for m in ALL_MOTORS)
        )
        return response

    def _cb_move_foam(self, request: MoveFoam.Request, response: MoveFoam.Response):
        if self._stop_requested:
            response.success = False
            response.message = 'Node is shutting down'
            return response

        direction = request.direction.upper().strip()

        if direction not in DIRECTION_VECTORS:
            valid = ', '.join(sorted({k for k in DIRECTION_VECTORS if len(k) <= 2}))
            response.success = False
            response.message = f'Unknown direction "{request.direction}". Valid: {valid}'
            return response

        if request.degrees <= 0.0:
            response.success = False
            response.message = 'degrees must be positive'
            return response

        dx_n, dy_n = DIRECTION_VECTORS[direction]
        commands = self._compute_motor_commands(dx_n * request.degrees, dy_n * request.degrees)

        self.get_logger().info(f'MoveFoam  dir={direction}  deg={request.degrees:.2f}')

        collecting = self._is_optitrack_live()
        if collecting:
            label = f'move_{direction}_{request.degrees:.1f}deg'
            stop_event, collector = self._start_collection(label)
        else:
            self.get_logger().warn('OptiTrack not connected – skipping data collection.')

        self._execute(commands)
        move_timeout = max(10.0, request.degrees / (DEFAULT_VELOCITY * 0.229 * 6.0) * 1.5 + 5.0)
        moved_ok = self._wait_for_all(timeout=move_timeout)

        if collecting:
            self._stop_collection(stop_event, collector)

        if not moved_ok:
            response.success = False
            response.message = 'Motors did not reach target (timeout or interrupt)'
            return response

        response.success = True
        response.message = f'Moved {direction} by {request.degrees:.2f} deg'
        return response

    def _cb_move_circle(
        self, request: MoveFoamCircle.Request, response: MoveFoamCircle.Response
    ):
        if self._stop_requested:
            response.success = False
            response.message = 'Node is shutting down'
            return response

        radius     = request.radius
        steps      = request.steps if request.steps > 0 else 36
        step_delay = max(request.step_delay, 0.0)
        clockwise  = request.clockwise

        if radius <= 0.0:
            response.success = False
            response.message = 'radius must be positive'
            return response

        self.get_logger().info(
            f'MoveFoamCircle  r={radius:.2f}  steps={steps}  '
            f'delay={step_delay:.2f}s  cw={clockwise}'
        )

        collecting = self._is_optitrack_live()
        if collecting:
            label = request.label.strip() or f'circle_r{radius:.1f}_s{steps}_{"cw" if clockwise else "ccw"}'
            stop_event, collector = self._start_collection(label)
        else:
            self.get_logger().warn('OptiTrack not connected – skipping data collection.')

        sign      = 1.0 if clockwise else -1.0
        result_ok = True
        fail_step = -1

        for i in range(steps):
            if self._stop_requested:
                result_ok = False
                fail_step = i
                break

            theta_a = sign * 2.0 * math.pi * i       / steps
            theta_b = sign * 2.0 * math.pi * (i + 1) / steps

            dx = radius * (math.sin(theta_b) - math.sin(theta_a))
            dy = radius * (math.cos(theta_b) - math.cos(theta_a))

            step_deg = math.sqrt(dx * dx + dy * dy)
            step_timeout = max(2.0, step_deg / (DEFAULT_VELOCITY * 0.229 * 6.0) * 1.5 + step_delay + 0.5)
            self._execute(self._compute_motor_commands(dx, dy))
            if not self._wait_for_all(timeout=step_timeout):
                result_ok = False
                fail_step = i + 1
                break
            if step_delay > 0.0:
                time.sleep(step_delay)

        if collecting:
            self._stop_collection(stop_event, collector)

        if not result_ok:
            response.success = False
            response.message = f'Circle interrupted at step {fail_step}/{steps}'
            return response

        response.success = True
        response.message = (
            f'Circle complete: r={radius:.2f} deg, {steps} steps, '
            f'{"CW" if clockwise else "CCW"}'
        )
        return response

    def _cb_move_square(
        self, request: MoveFoamSquare.Request, response: MoveFoamSquare.Response
    ):
        if self._stop_requested:
            response.success = False
            response.message = 'Node is shutting down'
            return response

        side       = request.side_length
        step_delay = max(request.step_delay, 0.0)

        if side <= 0.0:
            response.success = False
            response.message = 'side_length must be positive'
            return response

        self.get_logger().info(
            f'MoveFoamSquare  side={side:.2f}  delay={step_delay:.2f}s'
        )

        collecting = self._is_optitrack_live()
        if collecting:
            csv_label = request.label.strip() or f'square_side{side:.1f}'
            stop_event, collector = self._start_collection(csv_label)
        else:
            self.get_logger().warn('OptiTrack not connected – skipping data collection.')

        result_ok = True
        fail_side = ''

        for side_dir, dx, dy in [
            ('N', 0.0,  side),
            ('E', side, 0.0),
            ('S', 0.0, -side),
            ('W', -side, 0.0),
        ]:
            if self._stop_requested:
                result_ok = False
                fail_side = side_dir
                break

            side_timeout = max(5.0, side / (DEFAULT_VELOCITY * 0.229 * 6.0) * 1.5 + step_delay + 0.5)
            self._execute(self._compute_motor_commands(dx, dy))
            if not self._wait_for_all(timeout=side_timeout):
                result_ok = False
                fail_side = side_dir
                break
            if step_delay > 0.0:
                time.sleep(step_delay)

        if collecting:
            self._stop_collection(stop_event, collector)

        if not result_ok:
            response.success = False
            response.message = f'Square interrupted on {fail_side} side (timeout or interrupt)'
            return response

        response.success = True
        response.message = f'Square complete: side={side:.2f} deg'
        return response

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        if self._optitrack_client is not None:
            try:
                self._optitrack_client.shutdown()
            except Exception:
                pass
            self._optitrack_client = None
        self._emergency_stop()
        if self._port_open:
            for mid in ALL_MOTORS:
                try:
                    self.packet_handler.write1ByteTxRx(
                        self.port_handler, mid, ADDR_TORQUE_ENABLE, TORQUE_DISABLE
                    )
                except Exception:
                    pass
            self.port_handler.closePort()
            self._port_open = False
        self.get_logger().info('Torque disabled. Port closed. State saved.')
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = FoamControllerNode()
        while rclpy.ok() and not node._stop_requested:
            rclpy.spin_once(node, timeout_sec=0.5)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    except RuntimeError as exc:
        print(f'[foam_controller_node] Fatal: {exc}')
    finally:
        if node is not None:
            node._stop_requested = True
            try:
                node.destroy_node()
            except Exception:
                pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
