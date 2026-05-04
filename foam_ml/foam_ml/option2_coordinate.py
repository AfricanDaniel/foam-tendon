#!/usr/bin/env python3
"""
Option 2 – Coordinate Input

Enter a desired tip position (East, North) in millimetres.  The model
predicts motor commands and the foam tip trajectory.

Controls
--------
  Type coordinates in the text boxes, then click [Preview].
  [Execute]  sends the trajectory to the hardware.
  [Compare]  opens the comparison viewer.

Usage:
    python3 option2_coordinate.py  [--dry-run]
    ros2 run foam_ml option2_coordinate [--dry-run]
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from matplotlib.widgets import Button, TextBox
import numpy as np

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from model_utils import (
    load_models,
    predict_motors,
    predict_xyz,
    generate_predicted_trajectory,
    save_predicted_trajectory,
    execute_on_hardware,
    _find_data_dir,
)


def _load_all_training_xyz():
    import csv as _csv
    data_dir = _find_data_dir()
    east, north, up = [], [], []
    for f in sorted(data_dir.glob("*.csv")):
        if os.path.basename(str(f)).startswith("predicted_"):
            continue
        home = None
        with open(f, newline="") as fh:
            reader = _csv.DictReader(fh)
            for row in reader:
                try:
                    x = float(row["optitrack_x"])
                    y = float(row["optitrack_y"])
                    z = float(row["optitrack_z"])
                except (ValueError, KeyError):
                    continue
                if home is None:
                    home = (x, y, z)
                east.append(x - home[0])
                north.append(y - home[1])
                up.append(z - home[2])
    return np.array(east), np.array(north), np.array(up)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    raw = [a for a in sys.argv[1:] if not a.startswith("__")]
    sep = raw.index("--ros-args") if "--ros-args" in raw else len(raw)
    opts = parser.parse_args(raw[:sep])
    dry_run = opts.dry_run

    try:
        mdl = load_models()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    ws        = mdl["workspace"]
    max_r     = ws["max_horizontal_reach_m"]
    max_r_p95 = ws["max_horizontal_reach_p95_m"]

    east_d, north_d, up_d = _load_all_training_xyz()

    pred_csv_path = [None]
    real_csv_path = [None]
    predicted_path = [None]
    pred_run_num   = [0]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("Foam ML – Option 2: Coordinate Input", fontsize=12, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           left=0.07, right=0.97, top=0.88, bottom=0.22,
                           hspace=0.35, wspace=0.3)

    ax_top = fig.add_subplot(gs[0, 0])
    ax_eu  = fig.add_subplot(gs[1, 0])
    ax_3d  = fig.add_subplot(gs[:, 1:], projection="3d")

    pad = max_r * 1.3 * 1000

    def _setup_2d(ax, xlabel, ylabel, title):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.axhline(0, c="k", lw=0.4, alpha=0.3)
        ax.axvline(0, c="k", lw=0.4, alpha=0.3)
        ax.set_aspect("equal")

    _setup_2d(ax_top, "East (mm)", "North (mm)", "Top-down (E–N)")
    ax_top.scatter(east_d * 1000, north_d * 1000, s=1, c="lightgray", alpha=0.4, zorder=1)
    for ax, xs, ys in [(ax_eu, east_d * 1000, up_d * 1000)]:
        _setup_2d(ax, "East (mm)", "Up (mm)", "Side view (E–Up)")
        ax.scatter(xs, ys, s=1, c="lightgray", alpha=0.4, zorder=1)
        ax.set_xlim(-pad, pad)
        ax.set_ylim(-pad * 0.1, pad)

    # Max range circles on top-down
    ax_top.add_patch(Circle((0, 0), max_r_p95 * 1000, fill=False,
                            linestyle="--", color="orange", lw=1.5,
                            label=f"95% reach ({max_r_p95*1000:.0f} mm)"))
    ax_top.add_patch(Circle((0, 0), max_r * 1000, fill=False,
                            linestyle=":", color="red", lw=1.0,
                            label=f"Max ({max_r*1000:.0f} mm)"))
    ax_top.scatter([0], [0], c="lime", s=80, zorder=5, label="Home")
    ax_top.set_xlim(-pad, pad)
    ax_top.set_ylim(-pad, pad)
    ax_top.legend(fontsize=7, loc="upper right")
    ax_top.text(pad * 0.95, 0, "E→", fontsize=7, color="gray", va="center")
    ax_top.text(0, pad * 0.95, "↑N", fontsize=7, color="gray", ha="center")

    pred_top,  = ax_top.plot([], [], "-", color="darkorange", lw=2, zorder=5, label="Predicted")
    target_dot, = ax_top.plot([], [], "*", color="red", ms=12, zorder=6)
    pred_eu,   = ax_eu.plot([], [], "-", color="darkorange", lw=2, zorder=5)

    # 3D
    ax_3d.set_title("3D view", fontsize=9)
    ax_3d.set_xlabel("East (mm)")
    ax_3d.set_ylabel("North (mm)")
    ax_3d.set_zlabel("Up (mm)")
    ax_3d.scatter(east_d * 1000, north_d * 1000, up_d * 1000,
                  s=1, c="lightgray", alpha=0.3, zorder=1)
    ang = np.linspace(0, 2 * np.pi, 200)
    ax_3d.plot(max_r_p95 * 1000 * np.cos(ang),
               max_r_p95 * 1000 * np.sin(ang),
               np.zeros(200), "--", color="orange", lw=1.5, alpha=0.8)
    ax_3d.scatter([0], [0], [0], c="lime", s=80, zorder=5)
    lim = max_r * 1000 * 1.3
    ax_3d.set_xlim(-lim, lim)
    ax_3d.set_ylim(-lim, lim)
    ax_3d.set_zlim(-lim * 0.1, lim)
    pred_3d, = ax_3d.plot([], [], [], "-", color="darkorange", lw=2, zorder=5)

    # ── Widgets ───────────────────────────────────────────────────────────────
    ax_te = fig.add_axes([0.10, 0.12, 0.12, 0.05])
    ax_tn = fig.add_axes([0.26, 0.12, 0.12, 0.05])
    tb_east  = TextBox(ax_te, "East (mm) ", initial="50")
    tb_north = TextBox(ax_tn, "North (mm) ", initial="50")

    ax_info = fig.add_axes([0.05, 0.01, 0.55, 0.08])
    ax_info.axis("off")
    info_text = ax_info.text(0, 0.5,
        "Enter East/North coordinates and click Preview.",
        va="center", fontsize=9, color="navy")

    btns = {}
    for name, x in [("preview", 0.58), ("execute", 0.68), ("compare", 0.78), ("clear", 0.88)]:
        ax_b = fig.add_axes([x, 0.08, 0.08, 0.06])
        btns[name] = Button(ax_b, name.capitalize())
    btns["execute"].color = "lightyellow"
    btns["compare"].color = "lightcyan"
    if dry_run:
        btns["execute"].label.set_text("Execute\n(dry)")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_preview(event):
        try:
            e_mm = float(tb_east.text)
            n_mm = float(tb_north.text)
        except ValueError:
            info_text.set_text("Invalid coordinates.")
            fig.canvas.draw_idle()
            return

        e_m = e_mm / 1000.0
        n_m = n_mm / 1000.0
        dist = np.sqrt(e_m**2 + n_m**2)
        if dist > max_r * 1.05:
            scale = max_r / dist
            e_m *= scale
            n_m *= scale
            info_text.set_text(
                f"Target clipped to max reach: E={e_m*1000:.1f} mm, N={n_m*1000:.1f} mm"
            )

        target_xyz = np.array([[e_m, n_m, 0.0]], dtype=np.float32)
        motor_seq, xyz_seq = generate_predicted_trajectory(target_xyz, n_interp_steps=60)
        predicted_path[0] = (motor_seq, xyz_seq)

        # Update plots
        pred_top.set_data(xyz_seq[:, 0] * 1000, xyz_seq[:, 1] * 1000)
        pred_eu.set_data(xyz_seq[:, 0] * 1000, xyz_seq[:, 2] * 1000)
        pred_3d.set_data(xyz_seq[:, 0] * 1000, xyz_seq[:, 1] * 1000)
        pred_3d.set_3d_properties(xyz_seq[:, 2] * 1000)
        target_dot.set_data([xyz_seq[-1, 0] * 1000], [xyz_seq[-1, 1] * 1000])

        path, rn = save_predicted_trajectory(motor_seq, xyz_seq, label="coord_predicted")
        pred_csv_path[0] = path
        pred_run_num[0]  = rn

        # Show predicted endpoint
        ep = xyz_seq[-1]
        info_text.set_text(
            f"Predicted endpoint: E={ep[0]*1000:.1f} mm  N={ep[1]*1000:.1f} mm  Up={ep[2]*1000:.1f} mm\n"
            f"CSV: {os.path.basename(path)}"
        )
        fig.canvas.draw_idle()

    def _on_execute(event):
        if predicted_path[0] is None:
            info_text.set_text("Click Preview first.")
            fig.canvas.draw_idle()
            return
        motor_seq, xyz_seq = predicted_path[0]
        if dry_run:
            info_text.set_text("Dry-run: hardware execution skipped.")
            fig.canvas.draw_idle()
            return
        info_text.set_text("Executing on hardware...")
        fig.canvas.draw_idle()
        plt.pause(0.1)
        real_path = execute_on_hardware(motor_seq, step_delay=0.3, label="coord_real", run_num=pred_run_num[0])
        real_csv_path[0] = real_path
        if real_path:
            info_text.set_text(
                f"Done!  Real: {os.path.basename(real_path)}\nClick Compare to view side-by-side."
            )
        else:
            info_text.set_text("Hardware execution failed or ROS2 not available.")
        fig.canvas.draw_idle()

    def _on_compare(event):
        p = pred_csv_path[0]
        r = real_csv_path[0]
        if p is None:
            info_text.set_text("No predicted trajectory yet — click Preview first.")
            fig.canvas.draw_idle()
            return
        cmd = f"ros2 run foam_viz compare_trajectories --animate --predicted '{p}'"
        if r:
            cmd += f" --real '{r}'"
        os.system(cmd + " &")

    def _on_clear(event):
        predicted_path[0] = None
        pred_top.set_data([], [])
        pred_eu.set_data([], [])
        pred_3d.set_data([], [])
        pred_3d.set_3d_properties([])
        target_dot.set_data([], [])
        info_text.set_text("Cleared.")
        fig.canvas.draw_idle()

    btns["preview"].on_clicked(_on_preview)
    btns["execute"].on_clicked(_on_execute)
    btns["compare"].on_clicked(_on_compare)
    btns["clear"].on_clicked(_on_clear)

    plt.show()


if __name__ == "__main__":
    main()
