#!/usr/bin/env python3
"""
Train forward and inverse neural-network models for the foam robot.

Run directly at any time to retrain from scratch:
    python3 train_model.py
or:
    ros2 run foam_ml train_model

Requirements:
    sudo apt install python3-sklearn

Forward model : [motor_1..4 pos (pulses from home)] -> [east, north, up (m)]
Inverse model : [east, north, up (m)]               -> [motor_1..4 pos (pulses from home)]

Saved to: <this package>/models/
"""

import csv
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
except ImportError:
    print("ERROR: scikit-learn not found.")
    print("Install it:  sudo apt install python3-sklearn")
    sys.exit(1)


# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent


def _find_models_dir() -> Path:
    # Always save to the source tree so colcon build picks them up.
    # Walk up from __file__ until we find the workspace root (has both src/ and install/).
    path = Path(__file__).resolve().parent
    for _ in range(15):
        if (path / 'src').is_dir() and (path / 'install').is_dir():
            candidate = path / 'src' / 'foam_ml' / 'models'
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        parent = path.parent
        if parent == path:
            break
        path = parent
    # Fallback: running directly from source checkout
    return (_HERE.parent / 'models').resolve()


MODELS_DIR = _find_models_dir()


def _find_data_dir() -> Path:
    path = _HERE
    for _ in range(10):
        candidate = path / "src" / "actuator" / "data_collection" / "training_data"
        if candidate.is_dir():
            return candidate
        candidate2 = path / "actuator" / "data_collection" / "training_data"
        if candidate2.is_dir():
            return candidate2
        parent = path.parent
        if parent == path:
            break
        path = parent
    fallback = _HERE.parent.parent / "actuator" / "data_collection" / "training_data"
    if fallback.is_dir():
        return fallback
    raise FileNotFoundError(
        f"Cannot find training_data directory. Searched from {_HERE}"
    )


# ── CSV loading ────────────────────────────────────────────────────────────────

_MOTOR_COLS  = ["motor_1_pos", "motor_2_pos", "motor_3_pos", "motor_4_pos"]
_OPT_COLS    = ["optitrack_x", "optitrack_y", "optitrack_z"]


def _load_csv(path: Path):
    """
    Return (motors, xyz) arrays for one CSV.
    motors : (N, 4) float – pulses relative to home
    xyz    : (N, 3) float – east/north/up in metres, zeroed at first valid row
    Returns None if the file has too few valid rows.
    """
    motors_rows, xyz_rows = [], []
    home = None

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row["optitrack_x"])
                y = float(row["optitrack_y"])
                z = float(row["optitrack_z"])
            except (ValueError, KeyError):
                continue

            m = []
            ok = True
            for col in _MOTOR_COLS:
                try:
                    m.append(float(row[col]))
                except (ValueError, KeyError):
                    ok = False
                    break
            if not ok:
                continue

            if home is None:
                home = (x, y, z)

            # NatNet X = East, Y = North, Z = Up (empirically verified)
            xyz_rows.append([x - home[0], y - home[1], z - home[2]])
            motors_rows.append(m)

    if len(motors_rows) < 5:
        return None, None

    return np.array(motors_rows, dtype=np.float32), np.array(xyz_rows, dtype=np.float32)


def load_dataset(data_dir: Path = None):
    """Load all CSVs and return (motors_4d, xyz_3d) as stacked arrays."""
    if data_dir is None:
        data_dir = _find_data_dir()

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    all_m, all_xyz = [], []
    skipped = 0
    for f in csv_files:
        m, xyz = _load_csv(f)
        if m is None:
            skipped += 1
            continue
        all_m.append(m)
        all_xyz.append(xyz)

    if not all_m:
        raise ValueError("All CSV files were empty or invalid.")

    motors = np.vstack(all_m)
    xyz    = np.vstack(all_xyz)
    print(f"Loaded {len(all_m)} files ({skipped} skipped), {len(motors):,} data points.")
    return motors, xyz


# ── Workspace statistics ───────────────────────────────────────────────────────

def compute_workspace_stats(xyz: np.ndarray) -> dict:
    """Compute workspace boundaries from tip-position data."""
    east  = xyz[:, 0]
    north = xyz[:, 1]
    up    = xyz[:, 2]

    horiz_dist = np.sqrt(east**2 + north**2)
    total_dist = np.sqrt(east**2 + north**2 + up**2)

    # 95th-percentile radii to avoid outlier artefacts
    max_horiz_95  = float(np.percentile(horiz_dist, 95))
    max_total_95  = float(np.percentile(total_dist, 95))
    max_horiz_abs = float(horiz_dist.max())
    max_total_abs = float(total_dist.max())

    return {
        "max_horizontal_reach_m": max_horiz_abs,
        "max_horizontal_reach_p95_m": max_horiz_95,
        "max_total_reach_m": max_total_abs,
        "max_total_reach_p95_m": max_total_95,
        "east_range_m":  [float(east.min()),  float(east.max())],
        "north_range_m": [float(north.min()), float(north.max())],
        "up_range_m":    [float(up.min()),    float(up.max())],
        "n_datapoints": int(len(xyz)),
    }


# ── Training ──────────────────────────────────────────────────────────────────

_HIDDEN = (256, 256, 128)
_MAX_ITER = 1000


def train_forward(motors: np.ndarray, xyz: np.ndarray):
    """Train forward model: [m1..4] -> [east, north, up]."""
    print("\nTraining forward model (motors → xyz)...")

    m_scaler   = StandardScaler().fit(motors)
    xyz_scaler = StandardScaler().fit(xyz)
    Xm = m_scaler.transform(motors)
    Xy = xyz_scaler.transform(xyz)

    Xm_tr, Xm_te, Xy_tr, Xy_te = train_test_split(Xm, Xy, test_size=0.15, random_state=42)

    model = MLPRegressor(
        hidden_layer_sizes=_HIDDEN,
        activation="relu",
        solver="adam",
        max_iter=_MAX_ITER,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=False,
    )
    model.fit(Xm_tr, Xy_tr)

    pred_tr = model.predict(Xm_tr)
    pred_te = model.predict(Xm_te)
    rmse_tr = float(np.sqrt(mean_squared_error(Xy_tr, pred_tr)))
    rmse_te = float(np.sqrt(mean_squared_error(Xy_te, pred_te)))

    # Convert RMSE back to metres (xyz_scaler scale)
    scale_m = xyz_scaler.scale_.mean()
    print(f"  Train RMSE: {rmse_tr * scale_m * 1000:.2f} mm  |  Test RMSE: {rmse_te * scale_m * 1000:.2f} mm")
    print(f"  Converged in {model.n_iter_} iterations")

    return model, m_scaler, xyz_scaler


def train_inverse(motors: np.ndarray, xyz: np.ndarray, xyz_scaler: StandardScaler):
    """Train inverse model: [east, north, up] -> [m1..4]."""
    print("\nTraining inverse model (xyz → motors)...")

    m_inv_scaler = StandardScaler().fit(motors)
    Xm = m_inv_scaler.transform(motors)
    Xy = xyz_scaler.transform(xyz)   # reuse the xyz scaler from forward model

    Xy_tr, Xy_te, Xm_tr, Xm_te = train_test_split(Xy, Xm, test_size=0.15, random_state=42)

    model = MLPRegressor(
        hidden_layer_sizes=_HIDDEN,
        activation="relu",
        solver="adam",
        max_iter=_MAX_ITER,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=False,
    )
    model.fit(Xy_tr, Xm_tr)

    pred_tr = model.predict(Xy_tr)
    pred_te = model.predict(Xy_te)
    rmse_tr = float(np.sqrt(mean_squared_error(Xm_tr, pred_tr)))
    rmse_te = float(np.sqrt(mean_squared_error(Xm_te, pred_te)))

    scale_p = m_inv_scaler.scale_.mean()
    print(f"  Train RMSE: {rmse_tr * scale_p:.1f} pulses  |  Test RMSE: {rmse_te * scale_p:.1f} pulses")
    print(f"  Converged in {model.n_iter_} iterations")

    return model, m_inv_scaler


# ── Save / load ────────────────────────────────────────────────────────────────

def save_models(
    fwd_model,
    fwd_m_scaler,
    fwd_xyz_scaler,
    inv_model,
    inv_m_scaler,
    workspace_stats: dict,
):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODELS_DIR / "forward_model.pkl",   "wb") as f: pickle.dump(fwd_model, f)
    with open(MODELS_DIR / "fwd_m_scaler.pkl",    "wb") as f: pickle.dump(fwd_m_scaler, f)
    with open(MODELS_DIR / "fwd_xyz_scaler.pkl",  "wb") as f: pickle.dump(fwd_xyz_scaler, f)
    with open(MODELS_DIR / "inverse_model.pkl",   "wb") as f: pickle.dump(inv_model, f)
    with open(MODELS_DIR / "inv_m_scaler.pkl",    "wb") as f: pickle.dump(inv_m_scaler, f)

    with open(MODELS_DIR / "workspace.json", "w") as f:
        json.dump(workspace_stats, f, indent=2)

    print(f"\nModels saved to: {MODELS_DIR}/")
    print(f"  Max horizontal reach : {workspace_stats['max_horizontal_reach_m']*1000:.1f} mm")
    print(f"  Max total reach      : {workspace_stats['max_total_reach_m']*1000:.1f} mm")
    print(f"  Data points used     : {workspace_stats['n_datapoints']:,}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Foam ML — Model Training")
    print("=" * 60)

    motors, xyz = load_dataset()

    ws = compute_workspace_stats(xyz)

    fwd_model, fwd_m_scaler, fwd_xyz_scaler = train_forward(motors, xyz)
    inv_model, inv_m_scaler = train_inverse(motors, xyz, fwd_xyz_scaler)

    save_models(fwd_model, fwd_m_scaler, fwd_xyz_scaler, inv_model, inv_m_scaler, ws)

    print("\nDone.  You can now run any of:")
    print("  ros2 run foam_ml option1_dome")
    print("  ros2 run foam_ml option2_coordinate")
    print("  ros2 run foam_ml option3_path_draw")


if __name__ == "__main__":
    main()
