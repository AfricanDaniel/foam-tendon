#!/usr/bin/env python3
"""
Automated training-data collector for the single-column foam robot.

Calls /move_foam, /move_foam_circle, /move_foam_square, and /go_home
in a systematic sequence to generate diverse trajectory data for neural
network training.  The foam_controller_node writes motor + OptiTrack CSV
files to data_collection/ automatically whenever a move service is called,
so no extra logging is needed here.

Edit the Parameters section below to change amplitudes, safety limits, etc.
Run a dry run first to preview the full sequence before any hardware moves.
"""

import math
import sys
import time

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from actuator_interfaces.srv import MoveFoam, MoveFoamCircle, MoveFoamSquare


# ══════════════════════════════════════════════════════════════════════════════
# Parameters — edit these before running
# ══════════════════════════════════════════════════════════════════════════════

# ── Safety ────────────────────────────────────────────────────────────────────
# Hard cap (degrees) applied to every single /move_foam call.
# Reduce this first if the foam bends too far during single-step moves.
MAX_DEGREES = 400.0 #150.0

# Maximum amplitude (degrees) used inside multi-step combo sequences.
# With two orthogonal steps at COMBO_AMPLITUDE, max displacement from home
# is ≈ √2 × COMBO_AMPLITUDE ≈ 99° at default — comparable to a single 100°
# move.  Lower this before MAX_DEGREES when combos over-bend the foam.
COMBO_AMPLITUDE = 100.0

# ── Single-direction sweeps ───────────────────────────────────────────────────
# The script moves each of the 8 cardinal/diagonal directions at every
# amplitude listed here, returning home after each move.
# Values above MAX_DEGREES are automatically clamped.
AMPLITUDES = [60.0, 100.0, 140.0, 180.0, 220.0, 260.0, 300.0, 340.0, 380.0]

# ── Circles ───────────────────────────────────────────────────────────────────
CIRCLE_RADII = [50.0, 80.0, 110.0, 140.0, 170.0,
                200.0, 250.0, 300.0, 350.0]  # degrees of string travel; clamped to MAX_DEGREES
CIRCLE_STEPS = 36                     # discrete steps per full revolution
CIRCLE_STEP_DELAY = 0.2              # seconds to pause between steps

# ── Squares ───────────────────────────────────────────────────────────────────
SQUARE_SIDES = [60.0, 90.0, 120.0, 150.0, 180.0,
                210.0, 270.0, 330.0]  # degrees; clamped to MAX_DEGREES
SQUARE_STEP_DELAY = 0.5              # seconds to pause between sides

# ── Large-amplitude combos (phases 9–10) ─────────────────────────────────────
# Extra amplitudes for two-step combos to sample the mid/outer dome.
LARGE_COMBO_AMPLITUDES = [150.0, 200.0, 250.0, 300.0]

# ── Timing ────────────────────────────────────────────────────────────────────
HOME_WAIT = 1.5      # seconds to pause after each /go_home before the next move
MOVE_TIMEOUT = 30.0  # ROS service call timeout for simple moves (seconds)

# Motor speed in deg/s: DEFAULT_VELOCITY=20 × 0.229 RPM/unit × 6 deg/RPM-unit
_MOTOR_DEGS_PER_SEC = 20 * 0.229 * 6.0

# ── Run control ───────────────────────────────────────────────────────────────
# Set to True to print the full command sequence without moving any hardware.
# Always do a dry run first to verify the sequence looks correct.
DRY_RUN = False

# Repeat the entire sequence this many times (set >1 to multiply dataset size).
REPETITIONS = 1

# ══════════════════════════════════════════════════════════════════════════════

DIRECTIONS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

# Two-step combos: (dir_A, dir_B).  Foam moves dir_A, then dir_B, then home.
# All steps use COMBO_AMPLITUDE.  Covers orthogonal, diagonal, and reverse paths.
_COMBO_PAIRS = [
    ('N',  'E'),
    ('N',  'W'),
    ('S',  'E'),
    ('S',  'W'),
    ('NE', 'NW'),
    ('SE', 'SW'),
    ('N',  'S'),   # cross-axis: foam passes through home region
    ('E',  'W'),
    ('NE', 'SW'),
    ('NW', 'SE'),
]

# Three-step combos: (dir_A, dir_B, dir_C) at COMBO_AMPLITUDE each.
# Represents richer "approach from non-home" trajectories.
_TRIPLE_COMBOS = [
    ('N',  'E',  'S'),
    ('N',  'W',  'S'),
    ('E',  'N',  'W'),
    ('E',  'S',  'W'),
    ('NE', 'SE', 'SW'),
    ('NW', 'SW', 'SE'),
]

# Asymmetric two-step combos: (dir_A, amp_A, dir_B, amp_B).
# Using unequal amplitudes reaches intermediate angles (e.g. NNE, ENE) that
# the uniform-amplitude combos and single-direction sweeps never hit.
_ASYM_COMBOS = [
    # NNE region — far N, small E
    ('N', 200.0, 'E', 100.0), ('N', 300.0, 'E', 100.0),
    # ENE region — far E, small N
    ('E', 200.0, 'N', 100.0), ('E', 300.0, 'N', 100.0),
    # ESE region — far E, small S
    ('E', 200.0, 'S', 100.0), ('E', 300.0, 'S', 100.0),
    # SSE region — far S, small E
    ('S', 200.0, 'E', 100.0), ('S', 300.0, 'E', 100.0),
    # SSW region — far S, small W
    ('S', 200.0, 'W', 100.0), ('S', 300.0, 'W', 100.0),
    # WSW region — far W, small S
    ('W', 200.0, 'S', 100.0), ('W', 300.0, 'S', 100.0),
    # WNW region — far W, small N
    ('W', 200.0, 'N', 100.0), ('W', 300.0, 'N', 100.0),
    # NNW region — far N, small W
    ('N', 200.0, 'W', 100.0), ('N', 300.0, 'W', 100.0),
    # Same pairs reversed (small primary, large secondary)
    ('N', 100.0, 'E', 200.0), ('N', 100.0, 'E', 300.0),
    ('E', 100.0, 'N', 200.0), ('E', 100.0, 'N', 300.0),
    ('E', 100.0, 'S', 200.0), ('E', 100.0, 'S', 300.0),
    ('S', 100.0, 'E', 200.0), ('S', 100.0, 'E', 300.0),
    ('S', 100.0, 'W', 200.0), ('S', 100.0, 'W', 300.0),
    ('W', 100.0, 'S', 200.0), ('W', 100.0, 'S', 300.0),
    ('W', 100.0, 'N', 200.0), ('W', 100.0, 'N', 300.0),
    ('N', 100.0, 'W', 200.0), ('N', 100.0, 'W', 300.0),
]


class TrainingDataCollector(Node):

    def __init__(self) -> None:
        super().__init__('training_data_collector')

        self._cli_move   = self.create_client(MoveFoam,       '/move_foam')
        self._cli_circle = self.create_client(MoveFoamCircle, '/move_foam_circle')
        self._cli_square = self.create_client(MoveFoamSquare, '/move_foam_square')
        self._cli_home   = self.create_client(Trigger,        '/go_home')

        self._n_calls   = 0
        self._n_failed  = 0
        self._move_num  = 0   # increments for progress display

    # ── Service helpers ────────────────────────────────────────────────────────

    def _wait_for_servers(self, timeout: float = 10.0) -> bool:
        pairs = [
            (self._cli_move,   '/move_foam'),
            (self._cli_circle, '/move_foam_circle'),
            (self._cli_square, '/move_foam_square'),
            (self._cli_home,   '/go_home'),
        ]
        for cli, name in pairs:
            self.get_logger().info(f'Waiting for {name} ...')
            if not cli.wait_for_service(timeout_sec=timeout):
                self.get_logger().error(
                    f'Service {name} not available after {timeout:.0f} s. '
                    'Is foam_controller_node running?'
                )
                return False
        return True

    def _call(self, client, request, label: str, timeout: float = MOVE_TIMEOUT) -> bool:
        self._move_num += 1
        prefix = f'[{self._move_num:4d}]'

        if DRY_RUN:
            print(f'{prefix} DRY-RUN  {label}')
            return True

        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        self._n_calls += 1

        if future.result() is None:
            self.get_logger().error(f'{prefix} TIMEOUT  {label}')
            self._n_failed += 1
            return False

        result = future.result()
        if not result.success:
            self.get_logger().error(f'{prefix} FAILED   {label}  →  {result.message}')
            self._n_failed += 1
            return False

        self.get_logger().info(f'{prefix} OK       {label}  →  {result.message}')
        return True

    # ── Move primitives ────────────────────────────────────────────────────────

    def go_home(self) -> bool:
        ok = self._call(self._cli_home, Trigger.Request(), 'go_home')
        if ok and not DRY_RUN:
            time.sleep(HOME_WAIT)
        return ok

    def move(self, direction: str, degrees: float) -> bool:
        deg = min(abs(degrees), MAX_DEGREES)
        req = MoveFoam.Request()
        req.direction = direction
        req.degrees   = float(deg)
        return self._call(
            self._cli_move, req,
            f'move_foam  dir={direction:3s}  deg={deg:.1f}'
        )

    def circle(self, radius: float, clockwise: bool, label: str = '') -> bool:
        r = min(radius, MAX_DEGREES)
        req = MoveFoamCircle.Request()
        req.radius     = float(r)
        req.steps      = CIRCLE_STEPS
        req.step_delay = float(CIRCLE_STEP_DELAY)
        req.clockwise  = clockwise
        req.label      = label
        step_deg = r * 2.0 * math.sin(math.pi / CIRCLE_STEPS)
        circle_timeout = max(MOVE_TIMEOUT,
            CIRCLE_STEPS * (step_deg / _MOTOR_DEGS_PER_SEC + CIRCLE_STEP_DELAY) * 1.5 + 10.0)
        return self._call(
            self._cli_circle, req,
            f'circle     r={r:.1f}  {"CW " if clockwise else "CCW"}  steps={CIRCLE_STEPS}',
            circle_timeout,
        )

    def square(self, side: float, label: str = '') -> bool:
        s = min(side, MAX_DEGREES)
        req = MoveFoamSquare.Request()
        req.side_length = float(s)
        req.step_delay  = float(SQUARE_STEP_DELAY)
        req.label       = label
        square_timeout = max(MOVE_TIMEOUT,
            4.0 * (s / _MOTOR_DEGS_PER_SEC + SQUARE_STEP_DELAY) * 1.5 + 10.0)
        return self._call(
            self._cli_square, req,
            f'square     side={s:.1f}',
            square_timeout,
        )

    # ── Sequence ───────────────────────────────────────────────────────────────

    def run(self) -> None:
        if not DRY_RUN:
            if not self._wait_for_servers():
                return

        total_groups = self._count_groups()
        self.get_logger().info(
            f'\nTraining data collection  |  '
            f'{total_groups} move-groups × {REPETITIONS} rep(s)'
            + ('  [DRY RUN — no hardware moves]' if DRY_RUN else '')
        )

        for rep in range(REPETITIONS):
            if REPETITIONS > 1:
                self.get_logger().info(f'\n═══ Repetition {rep + 1}/{REPETITIONS} ═══')
            self._run_sequence()

        if not DRY_RUN:
            rate = 100 * (self._n_calls - self._n_failed) / max(self._n_calls, 1)
            self.get_logger().info(
                f'\nDone.  '
                f'Service calls: {self._n_calls}  |  '
                f'Failed: {self._n_failed}  |  '
                f'Success rate: {rate:.0f}%'
            )

    def _run_sequence(self) -> None:
        # ── Phase 1: Single-direction sweeps ──────────────────────────────────
        # 8 directions × len(AMPLITUDES) moves, return home after each.
        # Covers all cardinal + diagonal displacements at multiple magnitudes.
        self.get_logger().info('\n--- Phase 1: Single-direction sweeps ---')
        for direction in DIRECTIONS:
            for amp in AMPLITUDES:
                self.move(direction, amp)
                self.go_home()

        # ── Phase 2: Two-step combos ──────────────────────────────────────────
        # Move dir_A then dir_B (without going home in between) then home.
        # Gives the network data about non-axis-aligned starting positions.
        self.get_logger().info('\n--- Phase 2: Two-step direction combos ---')
        for dir_a, dir_b in _COMBO_PAIRS:
            self.move(dir_a, COMBO_AMPLITUDE)
            self.move(dir_b, COMBO_AMPLITUDE)
            self.go_home()

        # ── Phase 3: Three-step combos ────────────────────────────────────────
        # Three sequential moves before going home — richer non-home trajectories.
        self.get_logger().info('\n--- Phase 3: Three-step direction combos ---')
        for dir_a, dir_b, dir_c in _TRIPLE_COMBOS:
            self.move(dir_a, COMBO_AMPLITUDE)
            self.move(dir_b, COMBO_AMPLITUDE)
            self.move(dir_c, COMBO_AMPLITUDE)
            self.go_home()

        # ── Phase 4: Axis oscillations ────────────────────────────────────────
        # Move out on one axis, come partially back, then go home.
        # Teaches the dynamics of deceleration and approach from both sides.
        self.get_logger().info('\n--- Phase 4: Axis oscillations ---')
        half = COMBO_AMPLITUDE * 0.5
        for fwd, bwd in [('N', 'S'), ('E', 'W'), ('NE', 'SW'), ('NW', 'SE')]:
            for amp in AMPLITUDES[:2]:   # only two amplitudes; triple combos cover the rest
                self.move(fwd, amp)
                self.move(bwd, half)          # partial return — foam still off-center
                self.go_home()
                self.move(bwd, amp)
                self.move(fwd, half)
                self.go_home()

        # ── Phase 5: Circles (edge-start) ────────────────────────────────────
        # Foam starts at home; the circle path has home on its edge.
        self.get_logger().info('\n--- Phase 5: Circles (edge-start) ---')
        for radius in CIRCLE_RADII:
            for clockwise in [False, True]:
                d = 'cw' if clockwise else 'ccw'
                self.circle(radius, clockwise,
                    label=f'circle_edge_r{min(radius, MAX_DEGREES):.1f}_s{CIRCLE_STEPS}_{d}')
                self.go_home()

        # ── Phase 6: Squares (corner-start) ──────────────────────────────────
        # Foam starts at home; home is the SW corner of the square.
        self.get_logger().info('\n--- Phase 6: Squares (corner-start) ---')
        for side in SQUARE_SIDES:
            self.square(side,
                label=f'square_corner_side{min(side, MAX_DEGREES):.1f}')
            self.go_home()

        # ── Phase 7: Centered circles ─────────────────────────────────────────
        # Shift North by radius so home (0,0) is the center of the circle.
        # The circle parameterisation traces x=r·sin(θ), y=r·(cos(θ)−1) from
        # the start point, so the centre sits at (0, −radius) relative to
        # start.  Starting at (0, +radius) puts that centre exactly at home.
        self.get_logger().info('\n--- Phase 7: Centered circles ---')
        for radius in CIRCLE_RADII:
            for clockwise in [False, True]:
                d = 'cw' if clockwise else 'ccw'
                self.move('N', radius)
                self.circle(radius, clockwise,
                    label=f'circle_centered_r{min(radius, MAX_DEGREES):.1f}_s{CIRCLE_STEPS}_{d}')
                self.go_home()

        # ── Phase 8: Centered squares ─────────────────────────────────────────
        # Shift to the SW corner (West side/2, South side/2) so the square
        # N→E→S→W path is centred on home (0,0).
        self.get_logger().info('\n--- Phase 8: Centered squares ---')
        for side in SQUARE_SIDES:
            half = side / 2.0
            self.move('W', half)
            self.move('S', half)
            self.square(side,
                label=f'square_centered_side{min(side, MAX_DEGREES):.1f}')
            self.go_home()

        # ── Phase 9: Two-step combos at larger amplitudes ─────────────────────
        # Same direction pairs as Phase 2 but at LARGE_COMBO_AMPLITUDES so
        # the mid/outer dome regions are sampled.
        self.get_logger().info('\n--- Phase 9: Two-step combos at large amplitudes ---')
        for amp in LARGE_COMBO_AMPLITUDES:
            for dir_a, dir_b in _COMBO_PAIRS:
                self.move(dir_a, amp)
                self.move(dir_b, amp)
                self.go_home()

        # ── Phase 10: Asymmetric two-step combos ──────────────────────────────
        # Unequal amplitudes per step reach intermediate angles (NNE, ENE, …)
        # that uniform-amplitude combos and single-direction sweeps never hit.
        self.get_logger().info('\n--- Phase 10: Asymmetric two-step combos ---')
        for dir_a, amp_a, dir_b, amp_b in _ASYM_COMBOS:
            self.move(dir_a, amp_a)
            self.move(dir_b, amp_b)
            self.go_home()

    def _count_groups(self) -> int:
        phase1  = len(DIRECTIONS) * len(AMPLITUDES)
        phase2  = len(_COMBO_PAIRS)
        phase3  = len(_TRIPLE_COMBOS)
        phase4  = len([('N','S'), ('E','W'), ('NE','SW'), ('NW','SE')]) * len(AMPLITUDES[:2]) * 2
        phase5  = len(CIRCLE_RADII) * 2
        phase6  = len(SQUARE_SIDES)
        phase7  = len(CIRCLE_RADII) * 2
        phase8  = len(SQUARE_SIDES)
        phase9  = len(LARGE_COMBO_AMPLITUDES) * len(_COMBO_PAIRS)
        phase10 = len(_ASYM_COMBOS)
        return phase1 + phase2 + phase3 + phase4 + phase5 + phase6 + phase7 + phase8 + phase9 + phase10


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrainingDataCollector()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted — hardware state preserved by foam_controller_node.')
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
