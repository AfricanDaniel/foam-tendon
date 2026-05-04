#!/usr/bin/env python3
"""
Shared inference utilities for all foam_ml option scripts.

Provides:
  load_models()                        – load saved models + workspace
  predict_xyz(motors_4d)               – forward model
  predict_motors(xyz_3d)               – inverse model
  generate_predicted_trajectory(...)   – interpolated motor path + forward predictions
  save_predicted_trajectory(...)       – write predicted CSV in data_collection format
  execute_on_hardware(...)             – call ROS2 /execute_motor_trajectory service
"""

import csv
import json
import os
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).parent


def _find_models_dir() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory('foam_ml')) / 'models'
    except Exception:
        return _HERE.parent / 'models'


MODELS_DIR = _find_models_dir()

_REQUIRED_FILES = [
    "forward_model.pkl",
    "fwd_m_scaler.pkl",
    "fwd_xyz_scaler.pkl",
    "inverse_model.pkl",
    "inv_m_scaler.pkl",
    "workspace.json",
]


# ── Model loading ─────────────────────────────────────────────────────────────

_cache: Optional[dict] = None


def load_models(force_reload: bool = False) -> dict:
    """
    Load saved models from MODELS_DIR.  Returns a dict with keys:
        fwd_model, fwd_m_scaler, fwd_xyz_scaler,
        inv_model, inv_m_scaler, workspace
    Caches the result after first load.
    """
    global _cache
    if _cache is not None and not force_reload:
        return _cache

    missing = [f for f in _required_files() if not (MODELS_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Models not found in {MODELS_DIR}.\n"
            f"  Missing: {missing}\n"
            f"  Run first:  python3 train_model.py  (or  ros2 run foam_ml train_model)"
        )

    def _load(name):
        with open(MODELS_DIR / name, "rb") as f:
            return pickle.load(f)

    with open(MODELS_DIR / "workspace.json") as f:
        workspace = json.load(f)

    _cache = {
        "fwd_model":      _load("forward_model.pkl"),
        "fwd_m_scaler":   _load("fwd_m_scaler.pkl"),
        "fwd_xyz_scaler": _load("fwd_xyz_scaler.pkl"),
        "inv_model":      _load("inverse_model.pkl"),
        "inv_m_scaler":   _load("inv_m_scaler.pkl"),
        "workspace":      workspace,
    }
    return _cache


def _required_files():
    return _REQUIRED_FILES


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_xyz(motors_4d: np.ndarray) -> np.ndarray:
    """
    Forward model: motors (pulses from home) → tip position (m from home).
    motors_4d : (N, 4) or (4,)
    Returns   : (N, 3) or (3,) [east, north, up]
    """
    m = load_models()
    single = motors_4d.ndim == 1
    x = np.atleast_2d(motors_4d).astype(np.float32)
    xs = m["fwd_m_scaler"].transform(x)
    ys = m["fwd_model"].predict(xs)
    out = m["fwd_xyz_scaler"].inverse_transform(ys)
    return out[0] if single else out


def predict_motors(xyz_3d: np.ndarray) -> np.ndarray:
    """
    Inverse model: tip position (m from home) → motors (pulses from home).
    xyz_3d  : (N, 3) or (3,) [east, north, up]
    Returns : (N, 4) or (4,)
    """
    m = load_models()
    single = xyz_3d.ndim == 1
    x = np.atleast_2d(xyz_3d).astype(np.float32)
    xs = m["fwd_xyz_scaler"].transform(x)
    ys = m["inv_model"].predict(xs)
    out = m["inv_m_scaler"].inverse_transform(ys)
    return out[0] if single else out


# ── Trajectory generation ─────────────────────────────────────────────────────

def generate_predicted_trajectory(
    xyz_waypoints: np.ndarray,
    n_interp_steps: int = 40,
) -> tuple:
    """
    Given a list of (east, north, up) waypoints in metres:
    1. Use inverse model to get motor positions at each waypoint.
    2. Linearly interpolate motor positions between consecutive waypoints.
    3. Apply forward model at each interpolated step.

    Returns (motor_seq, xyz_pred_seq):
        motor_seq   : (M, 4) int   – interpolated motor positions (pulses from home)
        xyz_pred_seq: (M, 3) float – predicted tip position at each step (m from home)
    """
    waypoints = np.atleast_2d(xyz_waypoints).astype(np.float32)
    # Prepend home (0,0,0)
    home = np.zeros((1, 3), dtype=np.float32)
    all_waypoints = np.vstack([home, waypoints])

    # Compute motor positions at each waypoint
    motor_waypoints = predict_motors(all_waypoints)   # (K, 4)

    motor_seq = []
    for i in range(len(motor_waypoints) - 1):
        start = motor_waypoints[i]
        end   = motor_waypoints[i + 1]
        for t in np.linspace(0.0, 1.0, n_interp_steps, endpoint=(i == len(motor_waypoints) - 2)):
            motor_seq.append(start + t * (end - start))

    motor_seq = np.array(motor_seq, dtype=np.float32)

    # Apply forward model
    xyz_pred_seq = predict_xyz(motor_seq)

    return motor_seq.astype(int), xyz_pred_seq


# ── Save predicted trajectory as CSV ─────────────────────────────────────────

def _find_data_dir() -> Path:
    path = _HERE
    for _ in range(10):
        for candidate in [
            path / "src" / "actuator" / "data_collection" / "training_data",
            path / "actuator" / "data_collection" / "training_data",
        ]:
            if candidate.is_dir():
                return candidate
        parent = path.parent
        if parent == path:
            break
        path = parent
    fallback = _HERE.parent.parent / "actuator" / "data_collection" / "training_data"
    if fallback.is_dir():
        return fallback
    raise FileNotFoundError("Cannot find training_data directory.")


def _find_ml_data_dir() -> Path:
    d = _find_data_dir() / "ml_runs"
    d.mkdir(exist_ok=True)
    return d


_CSV_FIELDS = [
    "timestamp_s",
    "motor_1_pos", "motor_2_pos", "motor_3_pos", "motor_4_pos",
    "motor_1_vel", "motor_2_vel", "motor_3_vel", "motor_4_vel",
    "optitrack_x", "optitrack_y", "optitrack_z",
    "optitrack_qx", "optitrack_qy", "optitrack_qz", "optitrack_qw",
]


def save_predicted_trajectory(
    motor_seq: np.ndarray,
    xyz_pred_seq: np.ndarray,
    label: str = "predicted",
    duration_s: float = 5.0,
) -> tuple:
    """
    Save a predicted trajectory to data_collection/ml_runs/.
    Returns (csv_path, run_num) so callers can pass run_num to execute_on_hardware.
    """
    data_dir = _find_ml_data_dir()
    existing = sum(1 for f in os.listdir(data_dir) if f.endswith(".csv"))
    run_num  = existing + 1
    ts       = time.strftime("%Y%m%d_%H%M%S")
    filename = f"run_{run_num:04d}_{ts}_{label}.csv"
    path     = str(data_dir / filename)

    N       = len(motor_seq)
    t_array = np.linspace(0.0, duration_s, N)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for i in range(N):
            m  = motor_seq[i]
            vel = [0, 0, 0, 0]
            if i > 0:
                dt = t_array[i] - t_array[i - 1]
                if dt > 0:
                    vel = ((motor_seq[i] - motor_seq[i - 1]) / dt).tolist()
            xyz = xyz_pred_seq[i]
            writer.writerow({
                "timestamp_s":  f"{t_array[i]:.4f}",
                "motor_1_pos":  int(m[0]),
                "motor_2_pos":  int(m[1]),
                "motor_3_pos":  int(m[2]),
                "motor_4_pos":  int(m[3]),
                "motor_1_vel":  f"{vel[0]:.1f}",
                "motor_2_vel":  f"{vel[1]:.1f}",
                "motor_3_vel":  f"{vel[2]:.1f}",
                "motor_4_vel":  f"{vel[3]:.1f}",
                # Store as absolute positions (first row = 0,0,0 so replayer zeros correctly)
                "optitrack_x":  f"{xyz[0]:.6f}",
                "optitrack_y":  f"{xyz[1]:.6f}",
                "optitrack_z":  f"{xyz[2]:.6f}",
                "optitrack_qx": "0.000000",
                "optitrack_qy": "0.000000",
                "optitrack_qz": "0.000000",
                "optitrack_qw": "1.000000",
            })

    print(f"Predicted trajectory saved → {filename}")
    return path, run_num


# ── ROS2 hardware execution ───────────────────────────────────────────────────

def execute_on_hardware(
    motor_waypoints_rel: np.ndarray,
    step_delay: float = 0.5,
    label: str = "ml_trajectory",
    run_num: int = 0,
) -> str:
    """
    Send motor waypoints to the running foam_controller_node via the
    /execute_motor_trajectory service.

    motor_waypoints_rel : (N, 4) int – pulses relative to home
    step_delay          : seconds between waypoints
    label               : used for the output CSV filename
    run_num             : if > 0, forces this run number so the real CSV
                          is paired with the predicted CSV from save_predicted_trajectory

    Returns the path to the collected real trajectory CSV, or '' if ROS2
    is unavailable or the service call fails.
    """
    try:
        import rclpy
        from rclpy.node import Node
        from actuator_interfaces.srv import ExecuteMotorTrajectory
    except ImportError as e:
        print(f"ROS2 not available ({e}) — skipping hardware execution.")
        return ""

    waypoints = np.atleast_2d(motor_waypoints_rel).astype(int)
    N = len(waypoints)

    rclpy.init()
    node = Node("foam_ml_executor")

    client = node.create_client(ExecuteMotorTrajectory, "/execute_motor_trajectory")
    if not client.wait_for_service(timeout_sec=5.0):
        print("ERROR: /execute_motor_trajectory service not available.")
        print("       Make sure foam_controller_node is running.")
        node.destroy_node()
        rclpy.shutdown()
        return ""

    req = ExecuteMotorTrajectory.Request()
    req.motor_1_waypoints = [int(waypoints[i, 0]) for i in range(N)]
    req.motor_2_waypoints = [int(waypoints[i, 1]) for i in range(N)]
    req.motor_3_waypoints = [int(waypoints[i, 2]) for i in range(N)]
    req.motor_4_waypoints = [int(waypoints[i, 3]) for i in range(N)]
    req.step_delay        = float(step_delay)
    req.label             = label
    req.output_dir        = str(_find_ml_data_dir())
    req.run_num           = int(run_num)

    print(f"Sending {N} waypoints to hardware (step_delay={step_delay:.2f}s)...")
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=300.0)

    node.destroy_node()
    rclpy.shutdown()

    if future.result() is None:
        print("ERROR: Service call timed out.")
        return ""

    result = future.result()
    if not result.success:
        print(f"ERROR: {result.message}")
    else:
        print(f"Hardware execution complete: {result.message}")
        if result.data_file:
            print(f"Real trajectory saved → {os.path.basename(result.data_file)}")
    return result.data_file if result.success else ""
