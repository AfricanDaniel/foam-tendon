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
│   │       └── MoveByDegrees.srv
│   │
│   └── foam_viz/                        ← trajectory replay and visualisation
│       ├── README.md
│       └── foam_viz/
│           └── trajectory_replayer.py
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
| matplotlib, numpy | `pip install matplotlib numpy` |
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
colcon build --packages-select actuator
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
| `foam_viz` | ament_python | Trajectory replay and 2D/3D visualisation | [foam_viz/README.md](foam_viz/README.md) |
