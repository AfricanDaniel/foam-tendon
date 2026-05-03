# `foam_viz` package

ROS 2 package for replaying and visualising rigid-body trajectories recorded
by the `actuator` package during foam robot operation.

---

## Tools

| Entry point | Description |
|---|---|
| `trajectory_replayer` | Animate or plot a single data-collection CSV in a 4-panel matplotlib window |
| `compare_trajectories` | Overlay a predicted trajectory (from `foam_ml`) against a real recorded one |

---

## Node: `trajectory_replayer`

Reads a CSV from `src/actuator/data_collection/`, subtracts the home position
so `(0, 0, 0)` = foam at rest, and shows a 4-panel figure:

| Panel | Content |
|---|---|
| 3D view | Animated trajectory coloured by time; grey ghost of full path |
| Top-down (E–N) | Bird's-eye view — confirms direction relative to North/East |
| Side (E–Up) | East vs height — shows how far east the top bends and drops |
| Side (N–Up) | North vs height — shows how far north the top bends and drops |
| Motor strip | All four motor positions vs time with animated time cursor |

The file can be specified as a full path, a bare filename, or a **partial
string** — if only one file in `data_collection/` matches, it is used
automatically.

### Launch

```bash
ros2 run foam_viz trajectory_replayer --file <csv>
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--file` / `-f` | required | CSV filename, partial name, or absolute path |
| `--speed` / `-s` | `1.0` | Playback speed multiplier (e.g. `3.0` = 3× real time) |
| `--no-anim` | off | Show a static full-trajectory plot instead of animation |

---

## Examples

### Animated replay (default)

```bash
# Full filename
ros2 run foam_viz trajectory_replayer \
  --file run_0001_20260430_002346_move_N_90.0deg.csv

# Partial match — finds the file automatically if unambiguous
ros2 run foam_viz trajectory_replayer --file move_N_90

# 3× speed
ros2 run foam_viz trajectory_replayer --file move_N_90 --speed 3.0

# Half speed (useful for inspecting fast transients)
ros2 run foam_viz trajectory_replayer --file move_N_90 --speed 0.5
```

### Static overview

```bash
# Non-animated — shows full trajectory at once with colour-coded time
ros2 run foam_viz trajectory_replayer --file move_N_90 --no-anim

# Static plot of a circle trajectory
ros2 run foam_viz trajectory_replayer \
  --file circle_r50 --no-anim

# Static plot of a go_home run
ros2 run foam_viz trajectory_replayer --file go_home --no-anim
```

### Comparing a move to its return home

Open two terminals and run a move file and the following go_home side by side:

```bash
# Terminal 1
ros2 run foam_viz trajectory_replayer --file run_0004_20260430_002504_move_N_60.0deg.csv

# Terminal 2
ros2 run foam_viz trajectory_replayer --file run_0005_20260430_002507_go_home.csv
```

---

## Coordinate frame

OptiTrack NatNet columns map to the robot frame as follows — verified
empirically by checking which NatNet axis changes when each direction is
commanded:

| Robot axis | Positive direction | NatNet column | Verified by |
|---|---|---|---|
| East | physical East | `optitrack_x` | Commanding `move E` → X increases |
| North | physical North | `optitrack_y` | Commanding `move N` → Y increases |
| Up | vertically up | `optitrack_z` | Z constant during horizontal moves; slight drop when foam bends |

At startup the node prints the home position and the exact offset applied:

```
Home (NatNet): x=0.1319  y=-0.1457  z=0.2324 m

Coordinate frame used for display
  East  = NatNet X − 0.1319
  North = NatNet Y − (−0.1457)
  Up    = NatNet Z − 0.2324
```

No axis flips or sign inversions are needed. If you add new markers or
recalibrate Motive, re-verify by running a short `move_N` and `move_E` and
checking the sign of the resulting displacement in the top-down panel.

---

## Reading the plots

### Animated mode

- **Grey path** — full recorded trajectory (visible immediately for context).
- **Blue trail** — portion of trajectory up to the current frame.
- **Red dot** — current position of the rigid body.
- **Green dot** — home position `(0, 0, 0)`.
- **Red star** (static plot only) — final position.
- **Time colourbar** — plasma colour map from start (dark) to end (bright).

### Top-down panel (E–N)

The most useful panel for checking direction. If you commanded North, the
trail should run upward along the North axis. A diagonal move should appear
at 45°. Any unexpected lateral drift here indicates the foam is not centred
or the strings have unequal tension.

### Side panels (E–Up, N–Up)

The Up displacement should be small (a few mm) and negative — the foam top
drops slightly as the column bends. Large positive Up values would suggest
the rigid body was disturbed or the markers moved.

### Motor strip

Positive pulses = CCW rotation from home. Motor 1 (North, red) and Motor 2
(East, green) decrease for pull; Motor 3 (South, blue) and Motor 4 (West,
orange) increase for pull. During a pure North move you should see M1 and M3
change while M2 and M4 stay flat.

---

---

## Tool: `compare_trajectories`

Loads a **predicted** trajectory (generated by `foam_ml`) and optionally a
**real** recorded trajectory and overlays them on the same axes for direct
comparison.

```bash
# Static overlay (full paths visible immediately)
ros2 run foam_viz compare_trajectories \
  --predicted predicted_0001_20260502_120000_dome_predicted.csv \
  --real      run_0185_20260502_120100_dome_real.csv

# Animated replay — both dots advance in sync
ros2 run foam_viz compare_trajectories \
  --predicted predicted_0001_20260502_120000_dome_predicted.csv \
  --real      run_0185_20260502_120100_dome_real.csv \
  --animate

# Animated at 2× speed
ros2 run foam_viz compare_trajectories \
  --predicted dome_predicted --real dome_real --animate --speed 2.0
```

If `--real` is omitted only the predicted trajectory is shown.

### Options

| Flag | Required | Description |
|---|---|---|
| `--predicted` / `-p` | yes | Predicted CSV filename, partial name, or absolute path |
| `--real` / `-r` | no | Real CSV filename, partial name, or absolute path |
| `--animate` / `-a` | no | Animated replay — both dots advance in sync from start to end |
| `--speed` / `-s` | no | Playback speed multiplier for `--animate` (default `1.0`) |

### Static layout

| Panel | Content |
|---|---|
| 3D view | Both full trajectories overlaid — **orange** = predicted, **blue** = real |
| Top-down (E–N) | Bird's-eye overlay |
| Side (E–Up) | East vs height overlay |
| Side (N–Up) | North vs height overlay |
| Error plot | 3D Euclidean error in mm over normalised time (mean shown as dashed line) |

### Animated layout

Same 5-panel layout but with live-updating trails and dots.  Ghost paths
(faint lines) show the full trajectory at all times so you always know where
each is headed.

| Extra feature | Description |
|---|---|
| **Orange dot** | Predicted tip position at current normalised time |
| **Blue dot** | Real tip position at the same normalised time |
| **Error badge** | Live 3D Euclidean error in mm shown on the top-down panel |
| **Motor strip** | Dashed = predicted motor positions, solid = real motor positions |
| **Time label** | Shows predicted and real elapsed seconds for each dot position |

Both trajectories are resampled to the same number of frames so they start
and finish together regardless of their original lengths or sample rates.

### Typical workflow (called from a foam_ml option script)

The option scripts have a **Compare** button that assembles and runs this
command automatically after a hardware execution completes.  You can also
run it manually:

```bash
# After option2_coordinate predicted + executed:
ros2 run foam_viz compare_trajectories \
  --predicted predicted_0001_20260502_coord_predicted.csv \
  --real      run_0186_20260502_coord_real.csv
```

---

## Data collection CSV location

CSVs are searched in:

```
<ws_root>/src/actuator/data_collection/
```

Predicted trajectory CSVs are also written to this same directory
(with a `predicted_` prefix) so both `trajectory_replayer` and
`compare_trajectories` can find them with partial-name matching.

The path is resolved automatically from the package install location. No
environment variable or symlink is needed after `colcon build`.
