# `foam_viz` package

ROS 2 package for replaying and visualising rigid-body trajectories recorded
by the `actuator` package during foam robot operation.

---

## Nodes

| Entry point | Description |
|---|---|
| `trajectory_replayer` | Animate or plot a data-collection CSV in a 4-panel matplotlib window |

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

## Data collection CSV location

CSVs are searched in:

```
<ws_root>/src/actuator/data_collection/
```

The path is resolved automatically from the package install location. No
environment variable or symlink is needed after `colcon build`.
