# `actuator` package

ROS 2 package for motor control, OptiTrack integration, and data collection
on the single-column foam robot.

---

## Nodes

| Entry point | Class | Purpose |
|---|---|---|
| `foam_controller_node` | `FoamControllerNode` | Main 4-motor omnidirectional control node |
| `motor_service_node` | `DynamixelServiceNode` | Single-motor low-level dev/debug node |
| `collect_training_data` | `TrainingDataCollector` | Automated multi-move dataset collection script |

---

## Node: `foam_controller_node`

Controls all four motors simultaneously. Uses Dynamixel extended-position
(multi-turn) mode and a per-call `GroupSyncWrite` packet so opposing pull and
release motors move in the same serial transaction.

### Motor layout and pull convention

```
              Motor 1 — North
                    │
Motor 4 (West) ─[FOAM]─ Motor 2 (East)
                    │
              Motor 3 — South
```

| ID | Direction | CW action      | CCW action     |
|----|-----------|----------------|----------------|
| 1  | North     | Pull → North   | Release string |
| 2  | East      | Pull → East    | Release string |
| 3  | South     | Release string | Pull → South   |
| 4  | West      | Release string | Pull → West    |

### Motor command formula

Given a unit displacement vector `(dx = East, dy = North)` scaled by the
requested degrees:

```
Motor 1 (North) command = −dy   [CW pulls N; CCW releases N]
Motor 3 (South) command = −dy   [same sign; opposite pull convention cancels]
Motor 2 (East)  command = −dx
Motor 4 (West)  command = −dx
```

Positive command → CCW (position register increases).
Negative command → CW (position register decreases).

### Launch

```bash
ros2 run actuator foam_controller_node
```

Custom state file path (optional):

```bash
ros2 run actuator foam_controller_node --ros-args -p state_file:=/path/to/state.csv
```

### State persistence

Motor positions are persisted in `<ws_root>/foam_motor_state.csv` after
every move and on Ctrl-C / SIGTERM / motor disconnection.

```csv
motor_id,current_position,home_position
1,2048,2048
2,3914,3914
3,2048,2048
4,2048,2048
```

**First run:** if no CSV exists the current encoder positions become home.
Move the foam to the intended center before starting the node for the first
time, or call `/set_home` afterwards to redefine the origin.

---

### Services

#### `/move_foam` — `actuator_interfaces/srv/MoveFoam`

Move the foam in a cardinal or diagonal direction.

| Field | Type | Description |
|---|---|---|
| `direction` | string | `N S E W NE NW SE SW` (case-insensitive) |
| `degrees` | float32 | String-travel distance in degrees (must be > 0) |

```bash
# Bend north 90°
ros2 service call /move_foam actuator_interfaces/srv/MoveFoam \
  "{direction: 'N', degrees: 90.0}"

# Bend northeast 60°
ros2 service call /move_foam actuator_interfaces/srv/MoveFoam \
  "{direction: 'NE', degrees: 60.0}"

# Bend southwest 45°
ros2 service call /move_foam actuator_interfaces/srv/MoveFoam \
  "{direction: 'SW', degrees: 45.0}"
```

Diagonal moves use equal East and North components (`degrees / √2` each) so
the actual string travel equals `degrees` regardless of direction.

---

#### `/move_foam_circle` — `actuator_interfaces/srv/MoveFoamCircle`

Move the foam in a circle by decomposing the path into discrete tangential
increments. Uses exact finite differences so the foam returns precisely to
its starting position after one full revolution.

| Field | Type | Default | Description |
|---|---|---|---|
| `radius` | float32 | — | Circle radius in degrees of string travel |
| `steps` | int32 | 36 | Discrete steps per revolution (0 = 36) |
| `step_delay` | float32 | 0.1 | Pause between steps (seconds) |
| `clockwise` | bool | false | `true` = CW when viewed from above |

At step `i` the incremental motor command is:
```
dx = radius · (sin θ_{i+1} − sin θ_i)
dy = radius · (cos θ_{i+1} − cos θ_i)
θ  = ±2π · i / steps   (sign set by clockwise flag)
```

```bash
# Circle: radius 50°, 36 steps, 0.2 s delay, counter-clockwise
ros2 service call /move_foam_circle actuator_interfaces/srv/MoveFoamCircle \
  "{radius: 50.0, steps: 36, step_delay: 0.2, clockwise: false}"

# Clockwise, faster (18 steps, no delay)
ros2 service call /move_foam_circle actuator_interfaces/srv/MoveFoamCircle \
  "{radius: 50.0, steps: 18, step_delay: 0.0, clockwise: true}"
```

---

#### `/move_foam_square` — `actuator_interfaces/srv/MoveFoamSquare`

Move the foam in a square (North → East → South → West), returning to the
starting position.

| Field | Type | Default | Description |
|---|---|---|---|
| `side_length` | float32 | — | Side length in degrees of string travel |
| `step_delay` | float32 | 0.5 | Pause between sides (seconds) |

```bash
# Square with 60° sides, 0.5 s pauses
ros2 service call /move_foam_square actuator_interfaces/srv/MoveFoamSquare \
  "{side_length: 60.0, step_delay: 0.5}"

# Tighter square, no pauses
ros2 service call /move_foam_square actuator_interfaces/srv/MoveFoamSquare \
  "{side_length: 30.0, step_delay: 0.0}"
```

---

#### `/go_home` — `std_srvs/srv/Trigger`

Return all four motors simultaneously to the saved home positions.

```bash
ros2 service call /go_home std_srvs/srv/Trigger {}
# success: True
# message: "Home reached. M1=2048, M2=3914, M3=2048, M4=2048"
```

---

#### `/set_home` — `std_srvs/srv/Trigger`

Read the current live encoder positions and save them as the new home.
Use this to redefine the origin at any time.

```bash
ros2 service call /set_home std_srvs/srv/Trigger {}
# success: True
# message: "Home updated. M1=2048, M2=3914, M3=2048, M4=2048"
```

---

#### `/optitrack_status` — `std_srvs/srv/Trigger`

Report OptiTrack connection status and the most recently received rigid-body
pose. Useful for quickly verifying the NatNet stream is flowing.

```bash
ros2 service call /optitrack_status std_srvs/srv/Trigger {}
# message: "NatNet frames received : 1523
#           Rigid-body callbacks   : 1521
#           Last rigid-body data   : 0.02 s ago
#           Last position (x,y,z)  : (0.1319, -0.1457, 0.2324)
#           Last rotation (qx,y,z,w): (0.0591, -0.0455, 0.0062, 0.9972)"
```

---

## Node: `motor_service_node`

Low-level single-motor control node for **Motor ID 2** (East). Intended for
hardware bring-up, cable checks, and Dynamixel calibration — not for foam
motion control.

```bash
ros2 run actuator motor_service_node
```

### Services

| Service | Type | Description |
|---|---|---|
| `/home_motor` | Trigger | Move motor 2 to home position (pulse 3914) |
| `/max_motor` | Trigger | Move motor 2 to max position (pulse 6060) |
| `/sequence_motor` | Trigger | Test routine: 360° CCW, pause 5 s, return |
| `/move_degrees` | MoveByDegrees | Relative move in degrees (+ = CCW, − = CW) |
| `/get_pose` | Trigger | Read and print current position |

```bash
# Move 90° CCW from current position
ros2 service call /move_degrees actuator_interfaces/srv/MoveByDegrees \
  "{degrees: 90.0}"

# Move 180° CW from current position
ros2 service call /move_degrees actuator_interfaces/srv/MoveByDegrees \
  "{degrees: -180.0}"

# Read current position
ros2 service call /get_pose std_srvs/srv/Trigger {}
# message: "pulses=3914, degrees=343.59"
```

---

## Script: `collect_training_data`

Drives the foam through a systematic sequence of moves to generate a diverse
dataset. `foam_controller_node` handles all CSV logging automatically — this
script only sequences the service calls.

### Prerequisites

`foam_controller_node` must be running and OptiTrack must be live.

```bash
# Terminal 1
ros2 run actuator foam_controller_node

# Terminal 2 — wait for "FoamControllerNode ready" before starting
ros2 run actuator collect_training_data
```

### Always do a dry run first

In `actuator/actuator/collect_training_data.py` set:
```python
DRY_RUN = True
```

Rebuild and run to print the full command sequence without moving any hardware.

```bash
colcon build --packages-select actuator && source install/setup.bash
ros2 run actuator collect_training_data
```

### Collection phases

| Phase | Content | Groups |
|---|---|---|
| 1 | 8 directions × 3 amplitudes, single move → home | 24 |
| 2 | 10 two-step combos (dir A → dir B → home) | 10 |
| 3 | 6 three-step combos (dir A → B → C → home) | 6 |
| 4 | Axis oscillations (out → partial return → home) | 16 |
| 5 | Full circles CW + CCW at 3 radii | 6 |
| 6 | Squares at 3 side lengths | 3 |

Each group produces one or more CSV files in `data_collection/`.

### Key parameters (top of `collect_training_data.py`)

#### Safety — adjust these before the first real run

| Parameter | Default | Description |
|---|---|---|
| `MAX_DEGREES` | `150.0` | Hard cap on every single `/move_foam` call. **Lower this first** if individual sweeps over-bend the foam. |
| `COMBO_AMPLITUDE` | `70.0` | Amplitude for all multi-step combos (Phases 2–4). Two orthogonal steps at this value → ~99° peak displacement. **Lower this** if combos look dangerous. |

#### Coverage

| Parameter | Default | Description |
|---|---|---|
| `AMPLITUDES` | `[60, 100, 140]` | Degrees for Phase 1 sweeps |
| `CIRCLE_RADII` | `[50, 80, 110]` | Circle radii (degrees) |
| `CIRCLE_STEPS` | `36` | Steps per revolution |
| `CIRCLE_STEP_DELAY` | `0.2` | Seconds between circle steps |
| `SQUARE_SIDES` | `[60, 90, 120]` | Square side lengths (degrees) |
| `SQUARE_STEP_DELAY` | `0.5` | Seconds between square sides |

#### Run control

| Parameter | Default | Description |
|---|---|---|
| `DRY_RUN` | `False` | Print sequence without moving. Set `True` to preview. |
| `REPETITIONS` | `1` | How many times to run the full sequence |
| `HOME_WAIT` | `1.5` | Seconds to wait after each `/go_home` |
| `MOVE_TIMEOUT` | `30.0` | ROS service call timeout (seconds) |

---

## Data collection CSV format

Files are written to `src/actuator/data_collection/` by `foam_controller_node`
every time a move service is called while OptiTrack is streaming.

**Filename pattern:** `run_NNNN_YYYYMMDD_HHMMSS_<label>.csv`

**Example:** `run_0001_20260430_002346_move_N_90.0deg.csv`

**Columns:**

| Column | Unit | Description |
|---|---|---|
| `timestamp_s` | s | Elapsed time since collection start |
| `motor_1_pos` … `motor_4_pos` | pulses | Position relative to home (signed 32-bit) |
| `motor_1_vel` … `motor_4_vel` | 0.229 RPM | Present velocity register value |
| `optitrack_x` | m | NatNet X position (= robot East) |
| `optitrack_y` | m | NatNet Y position (= robot North) |
| `optitrack_z` | m | NatNet Z position (= robot Up) |
| `optitrack_qx/qy/qz/qw` | — | Rigid-body orientation quaternion |

Sample rate: 50 Hz (`COLLECT_HZ` constant in `foam_controller_node.py`).

**Example rows:**
```
timestamp_s,motor_1_pos,motor_2_pos,motor_3_pos,motor_4_pos,...,optitrack_x,optitrack_y,optitrack_z,...
0.0002,-14,-17,-14,-19,...,0.131931,-0.145707,0.232382,...
0.0452,-18,-17,-17,-19,...,0.131877,-0.145746,0.232308,...
```

---

## OptiTrack / Motive setup

1. Open Motive and create or verify a **Rigid Body** asset on the foam tip markers.
2. Go to **Edit → Settings → Streaming**:
   - **Broadcast Frame Data**: ON
   - **Transmission Type**: Unicast
   - **Local Interface**: set to the IP of the network interface facing the
     ROS machine
3. Set `NATNET_SERVER_IP` in `foam_controller_node.py` to match Motive's IP.
4. Start `foam_controller_node` — it will connect automatically and log a
   confirmation after 5 s.

Verify the stream is flowing:
```bash
ros2 service call /optitrack_status std_srvs/srv/Trigger {}
```
