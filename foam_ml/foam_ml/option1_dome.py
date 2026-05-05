#!/usr/bin/env python3
"""
Option 1 – Dome Interface

Shows the reachable workspace as a 3D hemisphere.  Click in the top-down
panel to place waypoints on the dome; the system predicts the motor commands
and simulates the foam tip trajectory.

Controls
--------
  Left-click  (top-down panel)  : add a waypoint
  Right-click (top-down panel)  : remove last waypoint
  [Preview]  button             : show predicted trajectory
  [Execute]  button             : send to hardware and record real trajectory
  [Compare]  button             : open compare_trajectories for predicted vs real
  [Clear]    button             : clear all waypoints

Usage:
    python3 option1_dome.py  [--dry-run]
    ros2 run foam_ml option1_dome [--dry-run]

--dry-run : show predictions only, no hardware execution.
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
from matplotlib.widgets import Button
import numpy as np

# Local imports
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from model_utils import (
    load_models,
    generate_predicted_trajectory,
    save_predicted_trajectory,
    execute_on_hardware,
    _find_data_dir,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_all_training_xyz():
    """Load all collected East/North/Up points for workspace display."""
    import csv as _csv
    data_dir = _find_data_dir()
    east, north, up = [], [], []
    for f in sorted(data_dir.glob("*.csv")):
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


def _make_hemisphere_mesh(radius, n=30):
    """Return (X, Y, Z) arrays for a hemisphere wireframe."""
    theta = np.linspace(0, np.pi / 2, n)   # 0=top, pi/2=equator
    phi   = np.linspace(0, 2 * np.pi, n)
    T, P  = np.meshgrid(theta, phi)
    X = radius * np.sin(T) * np.cos(P)
    Y = radius * np.sin(T) * np.sin(P)
    Z = radius * np.cos(T)
    return X, Y, Z


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Show predictions only, skip hardware execution")
    raw = [a for a in sys.argv[1:] if not a.startswith("__")]
    sep = raw.index("--ros-args") if "--ros-args" in raw else len(raw)
    opts = parser.parse_args(raw[:sep])
    dry_run = opts.dry_run

    # ── Load models ────────────────────────────────────────────────────────────
    try:
        mdl = load_models()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    ws        = mdl["workspace"]
    max_r     = ws["max_horizontal_reach_m"]
    max_r_p95 = ws["max_horizontal_reach_p95_m"]

    # Training data for background scatter
    east_d, north_d, up_d = _load_all_training_xyz()

    # State
    waypoints_en   = []        # [(east, north), ...]   user-placed
    predicted_path = [None]    # [xyz_seq or None]
    real_csv_path  = [None]    # [str or None]
    pred_csv_path  = [None]
    pred_run_num   = [0]       # shared run number for predicted+real pair

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle("Foam ML – Option 1: Dome Interface", fontsize=12, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, left=0.05, right=0.97, top=0.88, bottom=0.15,
                           hspace=0.35, wspace=0.3)

    ax_top  = fig.add_subplot(gs[0, 0])           # top-down click canvas
    ax_3d   = fig.add_subplot(gs[:, 1:], projection="3d")  # 3D dome

    # ── Top-down panel ─────────────────────────────────────────────────────────
    ax_top.set_title("Top-down (click to add waypoints)", fontsize=9)
    ax_top.set_xlabel("East (mm)")
    ax_top.set_ylabel("North (mm)")
    pad_m = max_r * 1.25

    # Training data background
    ax_top.scatter(east_d * 1000, north_d * 1000, s=1, c="lightgray", alpha=0.4, zorder=1)

    # Max-range dotted circle (p95 = guide, abs = hard limit)
    circ_p95 = Circle((0, 0), max_r_p95 * 1000, fill=False,
                      linestyle="--", color="orange", lw=1.5, label=f"95% reach ({max_r_p95*1000:.0f} mm)")
    circ_abs = Circle((0, 0), max_r * 1000, fill=False,
                      linestyle=":", color="red", lw=1.0, label=f"Max reach ({max_r*1000:.0f} mm)")
    ax_top.add_patch(circ_p95)
    ax_top.add_patch(circ_abs)
    ax_top.scatter([0], [0], c="lime", s=80, zorder=5, label="Home")
    ax_top.set_xlim(-pad_m * 1000, pad_m * 1000)
    ax_top.set_ylim(-pad_m * 1000, pad_m * 1000)
    ax_top.set_aspect("equal")
    ax_top.axhline(0, c="k", lw=0.4, alpha=0.3)
    ax_top.axvline(0, c="k", lw=0.4, alpha=0.3)
    ax_top.text(pad_m * 990, 0, "E→", fontsize=7, color="gray", va="center")
    ax_top.text(0, pad_m * 990, "↑N", fontsize=7, color="gray", ha="center")
    ax_top.legend(fontsize=7, loc="upper right")

    wp_scatter,  = ax_top.plot([], [], "o", color="royalblue", ms=8, zorder=6, label="Waypoints")
    pred_line_2d, = ax_top.plot([], [], "-", color="darkorange", lw=2, zorder=5, label="Predicted")

    # ── 3D panel ──────────────────────────────────────────────────────────────
    ax_3d.set_title("3D Workspace Dome", fontsize=9)
    ax_3d.set_xlabel("East (mm)")
    ax_3d.set_ylabel("North (mm)")
    ax_3d.set_zlabel("Up (mm)")
    # Training scatter — projected onto dome surface
    dome_z_train = np.sqrt(np.maximum(0.0, max_r**2 - east_d**2 - north_d**2)) * 1000
    ax_3d.scatter(east_d * 1000, north_d * 1000, dome_z_train,
                  s=1, c="lightgray", alpha=0.3, zorder=1)
    # Hemisphere wireframe at max reach
    HX, HY, HZ = _make_hemisphere_mesh(max_r * 1000)
    ax_3d.plot_wireframe(HX, HY, HZ, color="orange", alpha=0.15, lw=0.5, zorder=2)
    # Dotted equatorial circle at p95
    ang = np.linspace(0, 2 * np.pi, 200)
    ax_3d.plot(max_r_p95 * 1000 * np.cos(ang),
               max_r_p95 * 1000 * np.sin(ang),
               np.zeros(200), "--", color="orange", lw=1.5, alpha=0.8)
    def _dome_z(e_m, n_m):
        """Project (east, north) in metres onto the dome surface, returning display Z in mm."""
        return float(np.sqrt(max(0.0, max_r**2 - e_m**2 - n_m**2))) * 1000.0

    lim = max_r * 1000 * 1.3
    ax_3d.set_xlim(-lim, lim)
    ax_3d.set_ylim(-lim, lim)
    ax_3d.set_zlim(-lim * 0.1, lim)

    wp_3d,   = ax_3d.plot([], [], [], "o", color="royalblue", ms=8, zorder=6)
    pred_3d, = ax_3d.plot([], [], [], "-", color="darkorange", lw=2, zorder=5)

    # ── Bottom info text ───────────────────────────────────────────────────────
    ax_info = fig.add_axes([0.05, 0.03, 0.5, 0.08])
    ax_info.axis("off")
    info_text = ax_info.text(0, 0.5, "Left-click to add waypoints.  Right-click to undo.",
                             va="center", fontsize=9, color="navy")

    # ── Buttons ────────────────────────────────────────────────────────────────
    btn_ax = {
        "preview": fig.add_axes([0.58, 0.03, 0.08, 0.06]),
        "clear":   fig.add_axes([0.67, 0.03, 0.08, 0.06]),
        "execute": fig.add_axes([0.76, 0.03, 0.08, 0.06]),
        "compare": fig.add_axes([0.85, 0.03, 0.08, 0.06]),
    }
    btns = {k: Button(v, k.capitalize()) for k, v in btn_ax.items()}
    btns["execute"].color = "lightyellow"
    btns["compare"].color = "lightcyan"
    if dry_run:
        btns["execute"].label.set_text("Execute\n(dry)")

    # ── Interaction callbacks ─────────────────────────────────────────────────

    def _refresh_wp_display():
        if waypoints_en:
            xs = [w[0] * 1000 for w in waypoints_en]
            ys = [w[1] * 1000 for w in waypoints_en]
            wp_scatter.set_data(xs, ys)
            # Project waypoints onto dome surface for 3D display
            zs = [_dome_z(w[0], w[1]) for w in waypoints_en]
            wp_3d.set_data(xs, ys)
            wp_3d.set_3d_properties(zs)
        else:
            wp_scatter.set_data([], [])
            wp_3d.set_data([], [])
            wp_3d.set_3d_properties([])
        fig.canvas.draw_idle()

    def _on_click(event):
        if event.inaxes is not ax_top:
            return
        e_m = event.xdata / 1000.0
        n_m = event.ydata / 1000.0
        dist = np.sqrt(e_m**2 + n_m**2)
        if event.button == 1:
            if dist > max_r * 1.05:
                # Clip to max range
                scale = max_r / dist
                e_m  *= scale
                n_m  *= scale
            waypoints_en.append((e_m, n_m))
            info_text.set_text(f"{len(waypoints_en)} waypoint(s) placed.  "
                               "Left-click: add | Right-click: undo")
            _refresh_wp_display()
        elif event.button == 3 and waypoints_en:
            waypoints_en.pop()
            info_text.set_text(f"{len(waypoints_en)} waypoint(s).  Removed last.")
            _refresh_wp_display()

    def _on_preview(event):
        if not waypoints_en:
            info_text.set_text("No waypoints yet — click the top-down map to add some.")
            fig.canvas.draw_idle()
            return

        xyz_wps = np.array([[e, n, 0.0] for e, n in waypoints_en], dtype=np.float32)
        motor_seq, xyz_seq = generate_predicted_trajectory(xyz_wps, n_interp_steps=50)
        predicted_path[0] = xyz_seq

        # Display — project path onto dome surface for 3D, flat east/north for top-down
        pred_line_2d.set_data(xyz_seq[:, 0] * 1000, xyz_seq[:, 1] * 1000)
        dome_zs = [_dome_z(float(xyz_seq[i, 0]), float(xyz_seq[i, 1])) for i in range(len(xyz_seq))]
        pred_3d.set_data(xyz_seq[:, 0] * 1000, xyz_seq[:, 1] * 1000)
        pred_3d.set_3d_properties(dome_zs)

        # Save predicted CSV
        path, rn = save_predicted_trajectory(motor_seq, xyz_seq, label="dome_predicted")
        pred_csv_path[0] = path
        pred_run_num[0]  = rn
        info_text.set_text(f"Predicted trajectory shown.  CSV: {os.path.basename(path)}\n"
                           "Click Execute to run on hardware, or Compare (after executing) to compare.")
        fig.canvas.draw_idle()

    def _on_clear(event):
        waypoints_en.clear()
        predicted_path[0] = None
        pred_line_2d.set_data([], [])
        pred_3d.set_data([], [])
        pred_3d.set_3d_properties([])
        wp_scatter.set_data([], [])
        wp_3d.set_data([], [])
        wp_3d.set_3d_properties([])
        info_text.set_text("Cleared.  Left-click to add waypoints.")
        fig.canvas.draw_idle()

    def _on_execute(event):
        if not waypoints_en:
            info_text.set_text("No waypoints — click Preview first.")
            fig.canvas.draw_idle()
            return

        xyz_wps = np.array([[e, n, 0.0] for e, n in waypoints_en], dtype=np.float32)
        motor_seq, xyz_seq = generate_predicted_trajectory(xyz_wps, n_interp_steps=50)

        if pred_csv_path[0] is None:
            path, rn = save_predicted_trajectory(motor_seq, xyz_seq, "dome_predicted")
            pred_csv_path[0] = path
            pred_run_num[0]  = rn

        if dry_run:
            info_text.set_text("Dry-run: hardware execution skipped.")
            fig.canvas.draw_idle()
            return

        info_text.set_text("Executing on hardware... (this may take a while)")
        fig.canvas.draw_idle()
        plt.pause(0.1)

        real_path = execute_on_hardware(motor_seq, step_delay=0.3, label="dome_real", run_num=pred_run_num[0])
        real_csv_path[0] = real_path

        if real_path:
            info_text.set_text(
                f"Done!  Real: {os.path.basename(real_path)}\n"
                f"Click Compare to view side-by-side."
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
        info_text.set_text(f"Opening comparison...\n{cmd}")
        fig.canvas.draw_idle()
        os.system(cmd + " &")

    fig.canvas.mpl_connect("button_press_event", _on_click)
    btns["preview"].on_clicked(_on_preview)
    btns["clear"].on_clicked(_on_clear)
    btns["execute"].on_clicked(_on_execute)
    btns["compare"].on_clicked(_on_compare)

    plt.show()


if __name__ == "__main__":
    main()
