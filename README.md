# Single-Column Foam Robot — ROS 2 Workspace

A ROS 2 (Kilted) workspace for controlling and studying a tendon-driven
single-column foam robot. Four Dynamixel motors pull strings attached to the
top of a flexible foam column to bend it in any horizontal direction. An
OptiTrack motion-capture system tracks the rigid body at the foam tip, and
all motor + pose data is logged to CSV for downstream ML training.

---

## Hardware overview

```
          [Rigid-body markers]
                  |
         ┌────────┴────────┐
         │   Foam column   │  ← strings attach at the top
         └────────┬────────┘
                  │ (vertical column on base)
    ──────────────┴──────────────  ground plane
    Motor 1 (N)  Motor 2 (E)  Motor 3 (S)  Motor 4 (W)
    [spools; CW/CCW pull convention per motor — see actuator/README.md]
```

OptiTrack streams rigid-body pose over NatNet (unicast) to the machine
running `foam_controller_node`.

---

## Workspace layout

```
<ws_root>/
├── src/
│   ├── README.md                        ← you are here
│   │
│   ├── actuator/                        ← motor control, OptiTrack, data collection
│   │   ├── README.md
│   │   ├── actuator/
│   │   │   ├── foam_controller_node.py  # main 4-motor control node
│   │   │   ├── motor_service_node.py    # single-motor dev/debug node
│   │   │   ├── collect_training_data.py # automated dataset collection
│   │   │   ├── single_dynamixel.py      # standalone motor smoke-test script
│   │   │   └── natnet/                  # NatNet SDK client (OptiTrack)
│   │   └── data_collection/             # timestamped CSVs written here at runtime
│   │
│   ├── actuator_interfaces/             ← custom ROS 2 service definitions
│   │   └── srv/
│   │       ├── MoveFoam.srv
│   │       ├── MoveFoamCircle.srv
│   │       ├── MoveFoamSquare.srv
│   │       ├── MoveByDegrees.srv
│   │       └── ExecuteMotorTrajectory.srv
│   │
│   ├── foam_ml/                         ← ML models and interactive interfaces
│   │   ├── README.md
│   │   ├── models/                      # saved model files (generated at train time)
│   │   └── foam_ml/
│   │       ├── train_model.py           # retrain forward + inverse models from CSVs
│   │       ├── model_utils.py           # inference, trajectory generation, ROS2 executor
│   │       ├── option1_dome.py          # hemisphere workspace + click-to-place waypoints
│   │       ├── option2_coordinate.py    # enter (East, North) mm target
│   │       └── option3_path_draw.py     # click-drag to draw a free-hand path
│   │
│   └── foam_viz/                        ← trajectory replay and visualisation
│       ├── README.md
│       └── foam_viz/
│           ├── trajectory_replayer.py
│           └── compare_trajectories.py  # overlay predicted vs real trajectory
│
├── foam_motor_state.csv                 ← persisted motor + home positions
└── install/ build/ log/                 ← colcon output (not committed)
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| ROS 2 Kilted | `source /opt/ros/kilted/setup.bash` |
| Dynamixel SDK | `pip install dynamixel-sdk` |
| matplotlib, numpy, scipy | included with ROS |
| scikit-learn | `sudo apt install python3-sklearn` (required for `foam_ml`) |
| OptiTrack / Motive | Required for live data collection; optional for replay |

---

## Build

```bash
cd <ws_root>
source /opt/ros/kilted/setup.bash
colcon build
source install/setup.bash
```

Build a single package (faster during development):

```bash
colcon build --packages-select actuator_interfaces   # rebuild first if services changed
colcon build --packages-select actuator
colcon build --packages-select foam_ml
colcon build --packages-select foam_viz
```

---

## Quick start

### 1 — Start the foam controller

```bash
ros2 run actuator foam_controller_node
```

On first launch the current motor positions become the home positions.
The node connects OptiTrack automatically if Motive is reachable.

### 2 — Move the foam

```bash
# Bend north 90° of string travel
ros2 service call /move_foam actuator_interfaces/srv/MoveFoam \
  "{direction: 'N', degrees: 90.0}"

# Return to home
ros2 service call /go_home std_srvs/srv/Trigger {}
```

### 3 — Collect a full training dataset (requires OptiTrack live)

```bash
# Terminal 1 — controller must already be running
ros2 run actuator collect_training_data
```

CSVs land in `src/actuator/data_collection/` automatically.

### 4 — Replay and visualise a recorded run

```bash
ros2 run foam_viz trajectory_replayer --file run_0001_20260430_002346_move_N_90.0deg.csv

# Static (non-animated) overview
ros2 run foam_viz trajectory_replayer --file move_N_90 --no-anim

# 3× speed replay
ros2 run foam_viz trajectory_replayer --file move_N_90 --speed 3.0
```

### 5 — Train the ML model and plan trajectories

```bash
# Train (or retrain after collecting new data)
ros2 run foam_ml train_model

# Option 1: hemisphere dome — click waypoints on the workspace
ros2 run foam_ml option1_dome

# Option 2: enter a target coordinate (East, North in mm)
ros2 run foam_ml option2_coordinate

# Option 3: draw a free-hand path on a top-down grid
ros2 run foam_ml option3_path_draw

# Compare a predicted trajectory against the recorded real one
ros2 run foam_viz compare_trajectories --predicted <predicted_csv> --real <real_csv>
```

All three option scripts accept `--dry-run` to preview predictions without
moving the hardware.

---

## Coordinate frame

All visualisation and logged data uses the **robot frame**:

| Display axis | Positive direction | NatNet source column |
|---|---|---|
| East  | physical East  | `optitrack_x` |
| North | physical North | `optitrack_y` |
| Up    | vertically up  | `optitrack_z` |

Home position (foam at rest) is `(0, 0, 0)`. Verified empirically:
commanding `/move_foam N` produces a positive NatNet-Y displacement;
commanding `/move_foam E` produces a positive NatNet-X displacement.

---

## Packages

| Package | Type | Purpose | README |
|---|---|---|---|
| `actuator` | ament_python | Motor control, OptiTrack, data logging | [actuator/README.md](actuator/README.md) |
| `actuator_interfaces` | ament_cmake | Custom service message definitions | — |
| `foam_ml` | ament_python | ML models, trajectory planning, interactive GUIs | [foam_ml/README.md](foam_ml/README.md) |
| `foam_viz` | ament_python | Trajectory replay and comparison visualisation | [foam_viz/README.md](foam_viz/README.md) |
