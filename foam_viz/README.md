# `foam_viz` package

ROS 2 package for visualising rigid-body trajectories recorded by the `actuator` package during foam robot operation.

---

## Tools

| Entry point | Description |
|---|---|
| `trajectory_replayer` | Animate or plot a single data-collection CSV in a 4-panel matplotlib window |

---

## Node: `trajectory_replayer`

Reads a CSV from `src/actuator/data_collection/`, subtracts the home position
so `(0, 0, 0)` = foam at rest, and shows a 4-panel figure:

| Panel | Content |
|---|---|
| 3D view | Animated trajectory coloured by time; grey ghost of full path |
| Top-down (E–N) | Bird's-eye view — confirms direction relative to North/East |
| Side (E–Up) | East vs height |
| Side (N–Up) | North vs height |
| Motor strip | All four motor positions vs time with animated time cursor |

### Launch

```bash
ros2 run foam_viz trajectory_replayer --file <csv>
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--file` / `-f` | required | CSV filename, partial name, or absolute path |
| `--speed` / `-s` | `1.0` | Playback speed multiplier |
| `--no-anim` | off | Show a static full-trajectory plot instead of animation |

### Examples

```bash
ros2 run foam_viz trajectory_replayer --file run_0001_20260430_002346_move_N_90.0deg.csv
ros2 run foam_viz trajectory_replayer --file move_N_90 --speed 3.0
ros2 run foam_viz trajectory_replayer --file move_N_90 --no-anim
```

---

## Coordinate frame

| Robot axis | Positive direction | NatNet column |
|---|---|---|
| East | physical East | `optitrack_x` |
| North | physical North | `optitrack_y` |
| Up | vertically up | `optitrack_z` |

---

## TODO

- `compare_trajectories.py` — overlay a predicted trajectory (from `foam_ml`) against a real recorded one, with static and animated modes
