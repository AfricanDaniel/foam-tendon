# Single Column Foam – ROS 2 Workspace

Controls a foam cube suspended by four Dynamixel motors via string spools.
Each motor retracts or releases its string to move the foam omnidirectionally.

## Hardware Layout

```
              ID 1 (North)
                  │
ID 4 (West) ──[FOAM]── ID 2 (East)
                  │
              ID 3 (South)
```

| Motor ID | Direction | CW action      | CCW action     |
|----------|-----------|----------------|----------------|
| 1        | North     | Pull → N       | Release string |
| 2        | East      | Pull → E       | Release string |
| 3        | South     | Release string | Pull → S       |
| 4        | West      | Release string | Pull → W       |

All four motors share one serial bus (`/dev/ttyUSB1`, 1 Mbps).

---

## Packages

| Package               | Type          | Purpose                            |
|-----------------------|---------------|------------------------------------|
| `actuator`            | ament_python  | ROS 2 nodes                        |
| `actuator_interfaces` | ament_cmake   | Custom service definitions         |

### Build

```bash
cd <ws_root>
colcon build
source install/setup.bash
```

---

## Node: `motor_service_node`

Single-motor low-level control node for **Motor ID 2** (East).
Useful for direct testing and calibration of an individual Dynamixel.

**Launch:**
```bash
ros2 run actuator motor_service_node
```

---

### Services – `motor_service_node`

#### `/home_motor`
Moves motor 2 to the predefined home position (pulse 3914 ≈ 343.6°).

**Type:** `std_srvs/srv/Trigger`

```bash
ros2 service call /home_motor std_srvs/srv/Trigger
```

---

#### `/max_motor`
Moves motor 2 to the predefined maximum position (pulse 6060 ≈ 532.1°).

**Type:** `std_srvs/srv/Trigger`

```bash
ros2 service call /max_motor std_srvs/srv/Trigger
```

---

#### `/sequence_motor`
Executes a fixed test routine: rotate 360° CCW, pause 5 s, return to start.

**Type:** `std_srvs/srv/Trigger`

```bash
ros2 service call /sequence_motor std_srvs/srv/Trigger
```

---

#### `/move_degrees`
Moves motor 2 by a relative number of degrees from its current position.
Positive → CCW. Negative → CW.

**Type:** `actuator_interfaces/srv/MoveByDegrees`

| Field     | Type    | Description                    |
|-----------|---------|--------------------------------|
| `degrees` | float32 | Degrees to rotate (±)          |

```bash
# Rotate 90° CCW
ros2 service call /move_degrees actuator_interfaces/srv/MoveByDegrees "{degrees: 90.0}"

# Rotate 180° CW
ros2 service call /move_degrees actuator_interfaces/srv/MoveByDegrees "{degrees: -180.0}"
```

---

#### `/get_pose`
Returns motor 2's current position in pulses and degrees.

**Type:** `std_srvs/srv/Trigger`

```bash
ros2 service call /get_pose std_srvs/srv/Trigger {}
# Response: success: True, message: "pulses=3914, degrees=343.59"
```

---

## Node: `foam_controller_node`

High-level omnidirectional foam movement node.
Controls all four motors simultaneously using Dynamixel GroupSyncWrite so
every motor moves in the same packet — no lag between opposing motors.

**Motor command formula** – given displacement vector (dx = East, dy = North):

```
Motor 1 (North) degrees = -dy
Motor 3 (South) degrees = -dy   ← same value; opposite pull convention cancels
Motor 2 (East)  degrees = -dx
Motor 4 (West)  degrees = -dx   ← same value; opposite pull convention cancels
```

**Launch:**
```bash
ros2 run actuator foam_controller_node
```

Custom state file path (optional):
```bash
ros2 run actuator foam_controller_node --ros-args -p state_file:=/path/to/state.csv
```

---

### State persistence (`<ws_root>/foam_motor_state.csv`)

The node tracks motor positions in a CSV file at the workspace root
(`single_column_foam_ros2_ws/foam_motor_state.csv`) so it stays with the
project and can be inspected or edited manually at any time.
The path is resolved automatically from the node's file location, so it works
whether the node runs from source or from the `colcon build` install tree.
Override with the `state_file` ROS parameter if needed.

```
motor_id,current_position,home_position
1,2048,2048
2,3914,3914
3,2048,2048
4,2048,2048
```

| Column             | Meaning                                                 |
|--------------------|---------------------------------------------------------|
| `motor_id`         | Dynamixel ID (1–4)                                      |
| `current_position` | Last commanded position in encoder pulses (signed 32-bit) |
| `home_position`    | Reference origin; set once on first run, never changes  |

**When is the file written?**

| Event                          | What happens                                          |
|-------------------------------|-------------------------------------------------------|
| After every movement command   | `current_position` updated immediately                |
| Ctrl-C / SIGTERM               | Motors halt in place; actual encoder values saved     |
| Motor disconnection detected   | Last known positions saved before the node errors out |
| `/go_home` completes           | `current_position` reset to equal `home_position`     |

**First run behaviour:** if no CSV exists, the node reads the live encoder
positions and saves them as the home positions. Move the foam to its
intended center point before starting the node for the first time,
or call `/set_home` at any time to redefine the origin.

---

### Services – `foam_controller_node`

#### `/go_home`
Moves all four motors simultaneously back to the saved home positions (the
positions recorded in `home_position` column of the CSV). After arrival the
CSV is updated so `current_position == home_position` for all motors, giving
a clean reference state.

**Type:** `std_srvs/srv/Trigger`

```bash
ros2 service call /go_home std_srvs/srv/Trigger
# Response example:
# success: True
# message: "Home reached. M1=2048, M2=3914, M3=2048, M4=2048"
```

---

#### `/set_home`
Reads the current live encoder positions and saves them as the new home positions
in the CSV. Use this any time you want to redefine "home" — for example after
manually repositioning the foam or after the first run to confirm the origin.

**Type:** `std_srvs/srv/Trigger`

```bash
ros2 service call /set_home std_srvs/srv/Trigger
# Response example:
# success: True
# message: "Home updated. M1=2100, M2=3914, M3=1950, M4=2048"
```

After this call, `/go_home` will return to the positions reported in this response.

---

#### `/move_foam`
Moves the foam in a cardinal or diagonal direction by a given string-travel distance.

**Type:** `actuator_interfaces/srv/MoveFoam`

| Field       | Type    | Description                                             |
|-------------|---------|---------------------------------------------------------|
| `direction` | string  | `N`, `S`, `E`, `W`, `NE`, `NW`, `SE`, `SW` (case-insensitive) |
| `degrees`   | float32 | String travel distance in degrees (must be positive)    |

**What each motor does per direction:**

| Direction | Motor 1 (N)   | Motor 2 (E)   | Motor 3 (S)   | Motor 4 (W)   |
|-----------|---------------|---------------|---------------|---------------|
| N         | CW – pull     | –             | CW – release  | –             |
| S         | CCW – release | –             | CCW – pull    | –             |
| E         | –             | CW – pull     | –             | CW – release  |
| W         | –             | CCW – release | –             | CCW – pull    |
| NE        | CW – pull     | CW – pull     | CW – release  | CW – release  |
| NW        | CW – pull     | CCW – release | CW – release  | CCW – pull    |
| SE        | CCW – release | CW – pull     | CCW – pull    | CW – release  |
| SW        | CCW – release | CCW – release | CCW – pull    | CCW – pull    |

Diagonal movements use equal components (`degrees / √2`) on each axis.

```bash
# Move north 90°
ros2 service call /move_foam actuator_interfaces/srv/MoveFoam "{direction: 'N', degrees: 90.0}"

# Move northeast 45°
ros2 service call /move_foam actuator_interfaces/srv/MoveFoam "{direction: 'NE', degrees: 45.0}"

# Move southwest 30°
ros2 service call /move_foam actuator_interfaces/srv/MoveFoam "{direction: 'SW', degrees: 30.0}"
```

---

#### `/move_foam_circle`
Moves the foam in a circle by decomposing the path into discrete tangential steps.
Uses exact finite differences (`sin θ`, `cos θ`) so the foam returns precisely to
its start after one full revolution.

**Type:** `actuator_interfaces/srv/MoveFoamCircle`

| Field        | Type    | Default | Description                                        |
|--------------|---------|---------|----------------------------------------------------|
| `radius`     | float32 | –       | Circle radius in degrees of string travel          |
| `steps`      | int32   | 36      | Discrete steps per revolution (0 = use default)   |
| `step_delay` | float32 | 0.1     | Pause between steps in seconds                     |
| `clockwise`  | bool    | false   | `true` = CW when viewed from above                 |

At each step `i` (angle `θ = 2π·i / steps`, clockwise from North):
```
dx = r · (sin θ_{i+1} − sin θ_i)
dy = r · (cos θ_{i+1} − cos θ_i)
```
Both motors on each axis move simultaneously.

```bash
# Circle: radius 45°, 36 steps, 0.1 s delay, counter-clockwise
ros2 service call /move_foam_circle actuator_interfaces/srv/MoveFoamCircle \
  "{radius: 45.0, steps: 36, step_delay: 0.1, clockwise: false}"

# Faster circle: 18 steps, no delay, clockwise
ros2 service call /move_foam_circle actuator_interfaces/srv/MoveFoamCircle \
  "{radius: 30.0, steps: 18, step_delay: 0.0, clockwise: true}"
```

---

#### `/move_foam_square`
Moves the foam in a square (North → East → South → West), returning to start.

**Type:** `actuator_interfaces/srv/MoveFoamSquare`

| Field         | Type    | Default | Description                              |
|---------------|---------|---------|------------------------------------------|
| `side_length` | float32 | –       | Side length in degrees of string travel  |
| `step_delay`  | float32 | 0.5     | Pause between sides in seconds           |

Side sequence and motor actions:

| Side  | dx    | dy    | Motors active                          |
|-------|-------|-------|----------------------------------------|
| North | 0     | +side | M1 CW pull, M3 CW release              |
| East  | +side | 0     | M2 CW pull, M4 CW release              |
| South | 0     | -side | M3 CCW pull, M1 CCW release            |
| West  | -side | 0     | M4 CCW pull, M2 CCW release            |

```bash
# Square: 60° sides, 0.5 s pause between sides
ros2 service call /move_foam_square actuator_interfaces/srv/MoveFoamSquare \
  "{side_length: 60.0, step_delay: 0.5}"

# Square: 90° sides, no pause
ros2 service call /move_foam_square actuator_interfaces/srv/MoveFoamSquare \
  "{side_length: 90.0, step_delay: 0.0}"
```

---

## Custom Service Definitions (`actuator_interfaces`)

### `MoveByDegrees.srv`
```
float32 degrees
---
bool success
string message
```

### `MoveFoam.srv`
```
string direction
float32 degrees
---
bool success
string message
```

### `MoveFoamCircle.srv`
```
float32 radius
int32 steps
float32 step_delay
bool clockwise
---
bool success
string message
```

### `MoveFoamSquare.srv`
```
float32 side_length
float32 step_delay
---
bool success
string message
```
