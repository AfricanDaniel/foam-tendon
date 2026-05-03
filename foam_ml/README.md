# `foam_ml` package

Machine-learning models and interactive control interfaces for the
single-column foam robot.  Trains a forward model (motors → tip position)
and an inverse model (tip position → motors) from logged OptiTrack data,
then provides three GUI options for planning, predicting, and executing
trajectories on the real hardware.

---

## Quick start

```bash
# 1 – install scikit-learn if not already present
sudo apt install python3-sklearn

# 2 – build all packages
cd <ws_root>
source /opt/ros/kilted/setup.bash
colcon build
source install/setup.bash

# 3 – train (or retrain) the model from all collected CSVs
ros2 run foam_ml train_model

# 4 – launch any option interface
ros2 run foam_ml option1_dome        # hemisphere workspace + click waypoints
ros2 run foam_ml option2_coordinate  # enter (East, North) in mm
ros2 run foam_ml option3_path_draw   # click-drag to draw a free-hand path
```

All three option scripts accept `--dry-run` to show predictions without
moving the hardware.

---

## Prerequisites

| Requirement | Install |
|---|---|
| ROS 2 Kilted | `source /opt/ros/kilted/setup.bash` |
| numpy | included with ROS |
| scipy | included with ROS |
| matplotlib | included with ROS |
| scikit-learn | `sudo apt install python3-sklearn` |
| `actuator_interfaces` | built in this workspace |
| `foam_controller_node` | must be **running** for hardware execution |

---

## Training the model

```bash
ros2 run foam_ml train_model
# or, standalone (no ROS environment needed):
python3 src/foam_ml/foam_ml/train_model.py
```

The script:

1. Scans every CSV in `src/actuator/data_collection/`.
2. For each file, zeroes the OptiTrack position at the first valid row
   (home = first frame) to produce `(east, north, up)` displacements in metres.
3. Trains a **forward model** — `[m1, m2, m3, m4] → [east, north, up]`.
4. Trains an **inverse model** — `[east, north, up] → [m1, m2, m3, m4]`.
5. Computes workspace statistics (max reach, axis ranges).
6. Saves everything to `src/foam_ml/models/`.

**Re-run this command any time you collect new data.** It always retrains
from scratch so the model reflects the latest dataset.

### Example output

```
============================================================
Foam ML — Model Training
============================================================
Loaded 171 files (13 skipped), 17,715 data points.

Training forward model (motors → xyz)...
  Train RMSE: 4.51 mm  |  Test RMSE: 4.63 mm
  Converged in 212 iterations

Training inverse model (xyz → motors)...
  Train RMSE: 195.6 pulses  |  Test RMSE: 210.2 pulses
  Converged in 124 iterations

Models saved to: .../src/foam_ml/models/
  Max horizontal reach : 77.0 mm
  Max total reach      : 80.2 mm
  Data points used     : 17,715
```

### Model architecture

Both models are multi-layer perceptrons trained with scikit-learn's
`MLPRegressor`:

| Property | Value |
|---|---|
| Hidden layers | `(256, 256, 128)` |
| Activation | ReLU |
| Solver | Adam |
| Max iterations | 1000 |
| Early stopping | yes (10% validation split, 20-iteration patience) |
| Input normalisation | `StandardScaler` per model |

### Saved files

```
src/foam_ml/models/
├── forward_model.pkl    # sklearn MLPRegressor: motors → xyz
├── fwd_m_scaler.pkl     # StandardScaler for motor inputs
├── fwd_xyz_scaler.pkl   # StandardScaler for xyz outputs (shared with inverse)
├── inverse_model.pkl    # sklearn MLPRegressor: xyz → motors
├── inv_m_scaler.pkl     # StandardScaler for motor outputs
└── workspace.json       # workspace statistics (max reach, axis ranges)
```

---

## Option 1 — Dome interface

```bash
ros2 run foam_ml option1_dome [--dry-run]
```

Shows the full 3D reachable workspace as a hemisphere, with training data
points as a background scatter and two dotted reference circles:

- **Orange dashed** — 95th-percentile horizontal reach (typical safe limit)
- **Red dotted** — absolute maximum horizontal reach observed in training data

### Controls

| Action | Effect |
|---|---|
| Left-click in top-down panel | Add a waypoint (clipped to max reach if outside) |
| Right-click in top-down panel | Remove the last waypoint |
| **Preview** button | Compute and display the predicted trajectory |
| **Execute** button | Send trajectory to hardware; record real OptiTrack data |
| **Compare** button | Open `compare_trajectories` with predicted vs real |
| **Clear** button | Reset all waypoints and predictions |

### Workflow

1. Click one or more positions on the top-down map.
2. Click **Preview** — the predicted motor path and foam tip trajectory appear in
   both the top-down and 3D panels.
3. Optionally click **Execute** (requires `foam_controller_node` running).
4. Click **Compare** to see predicted vs real side-by-side.

---

## Option 2 — Coordinate input

```bash
ros2 run foam_ml option2_coordinate [--dry-run]
```

Type a target position in millimetres (East, North) directly into text boxes.

### Controls

| Widget | Description |
|---|---|
| East (mm) text box | Desired East displacement from home |
| North (mm) text box | Desired North displacement from home |
| **Preview** button | Predict and display the trajectory to the target |
| **Execute** button | Move hardware; record real trajectory |
| **Compare** button | Open comparison viewer |
| **Clear** button | Reset prediction |

Targets outside the maximum reach are automatically clipped to the boundary
and the adjusted values are shown in the status bar.

---

## Option 3 — Draw a path

```bash
ros2 run foam_ml option3_path_draw [--dry-run] [--waypoints N]
```

Free-hand path drawing on a top-down grid.  Click and drag to trace a path;
release to finalise.  The path is smoothed with a parametric spline and
resampled to `N` waypoints (default 20) before being fed to the model.

### Options

| Flag | Default | Description |
|---|---|---|
| `--dry-run` | off | Skip hardware execution |
| `--waypoints` | `20` | Number of waypoints to resample the drawn path to |

### Controls

| Action | Effect |
|---|---|
| Click and drag in drawing panel | Trace a path |
| **Preview** button | Predict and show the foam tip trajectory |
| **Execute** button | Run on hardware; record real trajectory |
| **Compare** button | Open comparison viewer |
| **Clear** button | Erase current drawing and predictions |

Points drawn outside the max-reach circle are clipped to the boundary
as you draw.

---

## How prediction works

For all three options the same pipeline runs:

```
User input (East/North waypoints)
        ↓
Inverse model: [E, N, 0] → [m1, m2, m3, m4]   (pulses from home)
        ↓
Linear interpolation in motor space
(home → waypoint 1 → waypoint 2 → … in n_interp_steps per segment)
        ↓
Forward model applied at each interpolated step
        ↓
Predicted foam tip trajectory [East, North, Up] (metres from home)
        ↓
Saved as predicted_NNNN_TIMESTAMP_<label>.csv in data_collection/
```

The predicted CSV is in the standard 16-column format so it can be loaded
by `trajectory_replayer` or `compare_trajectories`.

---

## Hardware execution

When **Execute** is clicked, the option script calls the
`/execute_motor_trajectory` ROS 2 service on `foam_controller_node`:

```
Service   : /execute_motor_trajectory
Type      : actuator_interfaces/srv/ExecuteMotorTrajectory
```

| Request field | Type | Description |
|---|---|---|
| `motor_1_waypoints` … `motor_4_waypoints` | `int32[]` | Motor positions in pulses relative to home |
| `step_delay` | float32 | Seconds to wait after each waypoint (default 0.3) |
| `label` | string | Label used in the output CSV filename |

| Response field | Type | Description |
|---|---|---|
| `success` | bool | Whether all waypoints were reached |
| `message` | string | Human-readable summary |
| `data_file` | string | Absolute path to the recorded OptiTrack CSV |

`foam_controller_node` records a continuous OptiTrack stream across the entire
multi-waypoint move and returns the path to the collected CSV.

---

## Comparing predicted and real trajectories

```bash
ros2 run foam_viz compare_trajectories \
  --predicted predicted_0001_20260502_120000_dome_predicted.csv \
  --real      run_0185_20260502_120100_dome_real.csv
```

Both CSVs are searched in `src/actuator/data_collection/`.  Partial name
matching is supported.  See [`foam_viz/README.md`](../foam_viz/README.md)
for full details.

---

## File layout

```
src/foam_ml/
├── README.md                   ← you are here
├── package.xml
├── setup.py
├── resource/
│   └── foam_ml
├── models/                     ← created by train_model; not committed
│   ├── forward_model.pkl
│   ├── fwd_m_scaler.pkl
│   ├── fwd_xyz_scaler.pkl
│   ├── inverse_model.pkl
│   ├── inv_m_scaler.pkl
│   └── workspace.json
└── foam_ml/
    ├── __init__.py
    ├── train_model.py          # training entry point
    ├── model_utils.py          # inference, trajectory generation, ROS2 executor
    ├── option1_dome.py         # hemisphere GUI
    ├── option2_coordinate.py   # coordinate input GUI
    └── option3_path_draw.py    # free-hand drawing GUI
```

---

## Troubleshooting

**`Models not found` error when running an option script**

Run `ros2 run foam_ml train_model` first.  The `models/` directory is not
included in the repository and must be generated from your collected data.

**`/execute_motor_trajectory service not available`**

`foam_controller_node` must be running before clicking **Execute**:

```bash
ros2 run actuator foam_controller_node
```

**Low prediction accuracy after collecting new data**

Re-run `ros2 run foam_ml train_model` to incorporate the new CSVs.

**`sklearn` not found**

```bash
sudo apt install python3-sklearn
```

**Hardware executes but no real trajectory CSV appears**

OptiTrack must be live (Motive streaming, rigid body visible).  Check:

```bash
ros2 service call /optitrack_status std_srvs/srv/Trigger {}
```
