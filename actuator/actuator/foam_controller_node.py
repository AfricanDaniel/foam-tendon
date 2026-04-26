#!/usr/bin/env python3

import csv
import math
import os
import signal
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
ADDR_PRESENT_POSITION = 132

EXTENDED_POSITION_CONTROL_MODE = 4
TORQUE_ENABLE    = 1
TORQUE_DISABLE   = 0
DEFAULT_VELOCITY = 20             # profile velocity (lower = slower / safer)
MOVING_THRESHOLD = 10             # pulses; within this = "arrived"
PULSES_PER_DEGREE = 4096 / 360.0

# ── Direction table ───────────────────────────────────────────────────────────
# Each entry is a normalised (dx, dy) unit vector.
# dx > 0 → East,  dx < 0 → West
# dy > 0 → North, dy < 0 → South
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

# ── State file ────────────────────────────────────────────────────────────────
# Walk up from this file until we find the directory that contains `src/`.
# That is the workspace root regardless of whether we're running from source
# (src/actuator/actuator/) or from the install tree (install/actuator/lib/actuator/).
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

STATE_FILE_DEFAULT = os.path.join(_find_workspace_root(), 'foam_motor_state.csv')
CSV_FIELDS = ['motor_id', 'current_position', 'home_position']


class FoamControllerNode(Node):
    """
    Controls a foam cube suspended by four Dynamixel motors via string spools.

    Pull / release conventions
    --------------------------
    ID 1 (North): CW  → pull,    CCW → release
    ID 2 (East):  CW  → pull,    CCW → release
    ID 3 (South): CCW → pull,    CW  → release
    ID 4 (West):  CCW → pull,    CW  → release

    All four motors share one serial bus and are commanded simultaneously
    via GroupSyncWrite so pull and release motors move in the same packet.

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

        home_str = ', '.join(f'M{m}={self.home_positions[m]}' for m in ALL_MOTORS)
        self.get_logger().info(
            f'FoamControllerNode ready.\n'
            f'  State file : {self.state_file}\n'
            f'  Home poses : {home_str}'
        )

    # ── State file ────────────────────────────────────────────────────────────

    def _load_state_from_csv(self) -> bool:
        """
        Populate current_positions and home_positions from CSV.
        Returns True if the file existed and was fully valid.
        """
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
        """Overwrite the CSV with the latest current and home positions."""
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
        """Read present position with unsigned→signed conversion and retry."""
        for _ in range(5):
            raw, comm, err = self.packet_handler.read4ByteTxRx(
                self.port_handler, motor_id, ADDR_PRESENT_POSITION
            )
            if comm == COMM_SUCCESS and err == 0:
                # Extended position control uses signed 32-bit values
                return raw if raw <= 0x7FFFFFFF else raw - 0x100000000
            time.sleep(0.01)
        raise RuntimeError(f'Motor {motor_id}: position read failed after 5 attempts')

    def _compute_motor_commands(self, dx: float, dy: float) -> dict[int, float]:
        """
        Convert (dx=East, dy=North) displacement to {motor_id: degrees}.
        See class docstring for the derivation of the -dy / -dx formula.
        """
        return {
            MOTOR_NORTH: -dy,
            MOTOR_SOUTH: -dy,
            MOTOR_EAST:  -dx,
            MOTOR_WEST:  -dx,
        }

    def _write_goal(self, motor_id: int, target: int) -> bool:
        """
        Write a goal position to one motor using write4ByteTxRx (the same API
        used by motor_service_node, proven reliable).
        target is a signed Python int; it is masked to unsigned 32-bit for the SDK.
        Returns True on success.
        """
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
        """
        Send relative degree commands to each motor individually (write4ByteTxRx),
        then persist state.  Individual writes are used instead of GroupSyncWrite
        because GroupSyncWrite silently drops commands on some motor IDs.
        """
        for motor_id, degrees in commands.items():
            if abs(degrees) < 1e-4:
                continue
            target = int(self.current_positions[motor_id] + degrees * PULSES_PER_DEGREE)
            if self._write_goal(motor_id, target):
                self.current_positions[motor_id] = target
        self._save_state()

    def _send_absolute(self, positions: dict[int, int]) -> None:
        """Send absolute goal positions to each motor individually."""
        for motor_id, target in positions.items():
            if self._write_goal(motor_id, target):
                self.current_positions[motor_id] = target
        self._save_state()

    def _wait_for_all(self, timeout: float = 10.0) -> bool:
        """
        Block until every motor is within MOVING_THRESHOLD of its target.
        Returns False on timeout, stop request, or motor disconnection.
        Saves state immediately if a disconnection is detected.
        """
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
        """
        Read actual motor positions, command goal = present (halt in place),
        then save state to CSV.  Safe to call from a signal handler or on
        power-loss detection; all hardware errors are silently swallowed so
        the CSV write always happens.
        """
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
                    # raw is already unsigned 32-bit; write goal = present to halt
                    self.packet_handler.write4ByteTxRx(
                        self.port_handler, mid, ADDR_GOAL_POSITION, raw
                    )
            except Exception:
                pass  # motor unreachable; keep last commanded position
        self._save_state()

    # ── Signal handling ───────────────────────────────────────────────────────

    def _handle_signal(self, sig, frame) -> None:
        # Keep this handler minimal — calling get_logger() or any DDS API from
        # a signal handler deadlocks because the rclpy executor holds a lock.
        # All cleanup (emergency stop, torque disable, CSV save) runs in
        # destroy_node(), which is called from the finally block in main().
        self._stop_requested = True
        if rclpy.ok():
            rclpy.shutdown()

    # ── Service callbacks ─────────────────────────────────────────────────────

    def _cb_go_home(self, request: Trigger.Request, response: Trigger.Response):
        """
        Move all four motors to their saved home positions simultaneously,
        then reset the CSV so current_position == home_position for each motor.
        """
        if self._stop_requested:
            response.success = False
            response.message = 'Node is shutting down'
            return response
        self.get_logger().info('GoHome: returning to home positions.')

        self._send_absolute(dict(self.home_positions))
        if not self._wait_for_all():
            response.success = False
            response.message = 'Motors did not reach home (timeout or interrupt)'
            return response

        # Synchronise tracked positions with home and write a clean CSV
        for mid in ALL_MOTORS:
            self.current_positions[mid] = self.home_positions[mid]
        self._save_state()

        response.success = True
        response.message = (
            'Home reached. '
            + ', '.join(f'M{m}={self.home_positions[m]}' for m in ALL_MOTORS)
        )
        return response

    def _cb_set_home(self, request: Trigger.Request, response: Trigger.Response):
        """
        Declare the foam's current position as the new home.
        Reads live encoder values, overwrites home_position in the CSV,
        and resets current_position to match so the file is clean.
        """
        if self._stop_requested:
            response.success = False
            response.message = 'Node is shutting down'
            return response
        self.get_logger().info('SetHome: saving current positions as home.')
        for mid in ALL_MOTORS:
            try:
                pos = self._read_position(mid)
            except RuntimeError:
                pos = self.current_positions[mid]  # fallback to last known
            self.home_positions[mid]    = pos
            self.current_positions[mid] = pos
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
        self._execute(commands)
        if not self._wait_for_all():
            response.success = False
            response.message = f'Motors did not reach target (timeout or interrupt)'
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

        # CW increments θ from North; exact finite differences guarantee the
        # foam returns to its starting position after one full revolution.
        sign = 1.0 if clockwise else -1.0

        for i in range(steps):
            if self._stop_requested:
                response.success = False
                response.message = 'Interrupted by shutdown signal'
                return response

            theta_a = sign * 2.0 * math.pi * i       / steps
            theta_b = sign * 2.0 * math.pi * (i + 1) / steps

            dx = radius * (math.sin(theta_b) - math.sin(theta_a))
            dy = radius * (math.cos(theta_b) - math.cos(theta_a))

            self._execute(self._compute_motor_commands(dx, dy))
            if not self._wait_for_all(timeout=max(step_delay * 3, 2.0)):
                response.success = False
                response.message = f'Circle interrupted at step {i + 1}/{steps}'
                return response
            if step_delay > 0.0:
                time.sleep(step_delay)

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

        # North → East → South → West traces a closed square
        for label, dx, dy in [('N', 0.0, side), ('E', side, 0.0),
                               ('S', 0.0, -side), ('W', -side, 0.0)]:
            if self._stop_requested:
                response.success = False
                response.message = 'Interrupted by shutdown signal'
                return response

            self._execute(self._compute_motor_commands(dx, dy))
            if not self._wait_for_all(timeout=max(step_delay * 3, 5.0)):
                response.success = False
                response.message = f'Square interrupted on {label} side (timeout or interrupt)'
                return response
            if step_delay > 0.0:
                time.sleep(step_delay)

        response.success = True
        response.message = f'Square complete: side={side:.2f} deg'
        return response

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self) -> None:
        """
        Called on any clean exit path.  Stops motors in place, saves state,
        disables torque, and closes the serial port.
        Guard against rclpy calling this a second time during context shutdown.
        """
        if self._destroyed:
            return
        self._destroyed = True
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
        # spin_once with a short timeout so the loop wakes and checks
        # _stop_requested even if the C-level rcl_wait doesn't exit immediately
        # after rclpy.shutdown() is called from the signal handler.
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
