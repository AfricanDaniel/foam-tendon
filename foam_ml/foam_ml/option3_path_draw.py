#!/usr/bin/env python3
"""
Option 3 – 2D Path Drawing

Draw a free-hand path on the top-down (East–North) grid.  The model predicts
motor commands and simulates the foam tip trajectory along the drawn path.

Controls
--------
  Click and drag in the top-down panel to draw a path.
  [Preview]  : show predicted trajectory for the drawn path
  [Execute]  : send to hardware and record real trajectory
  [Compare]  : open comparison viewer
  [Clear]    : erase the drawn path

Usage:
    python3 option3_path_draw.py  [--dry-run]
    ros2 run foam_ml option3_path_draw [--dry-run]
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
from scipy.interpolate import splprep, splev

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from model_utils import (
    load_models,
    generate_predicted_trajectory,
    save_predicted_trajectory,
    execute_on_hardware,
    _find_data_dir,
)


# ── Resample a drawn path to N evenly-spaced waypoints ────────────────────────

def _resample_path(xs, ys, n_out=20):
    """Smooth and resample a hand-drawn path to n_out equally-spaced waypoints."""
    pts = np.array([xs, ys])
    if pts.shape[1] < 4:
        # Too few points: just interpolate linearly
        t_in  = np.linspace(0, 1, pts.shape[1])
        t_out = np.linspace(0, 1, n_out)
        xs_r  = np.interp(t_out, t_in, xs)
        ys_r  = np.interp(t_out, t_in, ys)
        return xs_r, ys_r

    # Parametric spline smoothing
    try:
        tck, u = splprep(pts, s=len(xs) * 0.5, k=min(3, pts.shape[1] - 1))
        u_new = np.linspace(0, 1, n_out)
        xs_r, ys_r = splev(u_new, tck)
        return xs_r, ys_r
    except Exception:
        t_in  = np.linspace(0, 1, pts.shape[1])
        t_out = np.linspace(0, 1, n_out)
        return np.interp(t_out, t_in, xs), np.interp(t_out, t_in, ys)


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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--waypoints", type=int, default=20,
                        help="Number of waypoints to resample the path to (default 20)")
    raw = [a for a in sys.argv[1:] if not a.startswith("__")]
    sep = raw.index("--ros-args") if "--ros-args" in raw else len(raw)
    opts = parser.parse_args(raw[:sep])
    dry_run   = opts.dry_run
    n_wpoints = opts.waypoints

    try:
        mdl = load_models()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    ws        = mdl["workspace"]
    max_r     = ws["max_horizontal_reach_m"]
    max_r_p95 = ws["max_horizontal_reach_p95_m"]

    east_d, north_d, up_d = _load_all_training_xyz()

    # Drawing state
    drawing        = [False]
    drawn_e        = []
    drawn_n        = []
    pred_csv_path  = [None]
    real_csv_path  = [None]
    predicted      = [None]   # (motor_seq, xyz_seq)
    pred_run_num   = [0]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle("Foam ML – Option 3: Draw a Path", fontsize=12, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           left=0.05, right=0.97, top=0.88, bottom=0.22,
                           hspace=0.35, wspace=0.3)

    ax_draw = fig.add_subplot(gs[:, 0])   # drawing canvas
    ax_3d   = fig.add_subplot(gs[0, 1:], projection="3d")
    ax_eu   = fig.add_subplot(gs[1, 1])
    ax_nu   = fig.add_subplot(gs[1, 2])

    pad = max_r * 1.3 * 1000

    # Drawing canvas setup
    ax_draw.set_title("Draw path here (click & drag)", fontsize=9)
    ax_draw.set_xlabel("East (mm)")
    ax_draw.set_ylabel("North (mm)")
    ax_draw.scatter(east_d * 1000, north_d * 1000,
                    s=1, c="lightgray", alpha=0.4, zorder=1)
    ax_draw.add_patch(Circle((0, 0), max_r_p95 * 1000, fill=False,
                             linestyle="--", color="orange", lw=1.5,
                             label=f"95% reach ({max_r_p95*1000:.0f} mm)"))
    ax_draw.add_patch(Circle((0, 0), max_r * 1000, fill=False,
                             linestyle=":", color="red", lw=1.0,
                             label=f"Max ({max_r*1000:.0f} mm)"))
    ax_draw.scatter([0], [0], c="lime", s=80, zorder=5, label="Home")
    ax_draw.set_xlim(-pad, pad)
    ax_draw.set_ylim(-pad, pad)
    ax_draw.set_aspect("equal")
    ax_draw.axhline(0, c="k", lw=0.4, alpha=0.3)
    ax_draw.axvline(0, c="k", lw=0.4, alpha=0.3)
    ax_draw.text(pad * 0.92, 0, "E→", fontsize=7, color="gray", va="center")
    ax_draw.text(0, pad * 0.92, "↑N", fontsize=7, color="gray", ha="center")
    ax_draw.legend(fontsize=7, loc="upper right")

    drawn_line,  = ax_draw.plot([], [], "-", color="steelblue", lw=2, zorder=4, label="Drawn")
    wp_dots,     = ax_draw.plot([], [], "o", color="steelblue", ms=5, zorder=5)
    pred_draw,   = ax_draw.plot([], [], "-", color="darkorange", lw=2.5, zorder=6, label="Predicted")

    # 3D panel
    ax_3d.set_title("3D predicted trajectory", fontsize=9)
    ax_3d.set_xlabel("East (mm)")
    ax_3d.set_ylabel("North (mm)")
    ax_3d.set_zlabel("Up (mm)")
    ax_3d.scatter(east_d * 1000, north_d * 1000, up_d * 1000,
                  s=1, c="lightgray", alpha=0.3, zorder=1)
    ang = np.linspace(0, 2 * np.pi, 200)
    ax_3d.plot(max_r_p95 * 1000 * np.cos(ang),
               max_r_p95 * 1000 * np.sin(ang),
               np.zeros(200), "--", color="orange", lw=1.5, alpha=0.7)
    ax_3d.scatter([0], [0], [0], c="lime", s=80, zorder=5)
    lim = max_r * 1000 * 1.3
    ax_3d.set_xlim(-lim, lim)
    ax_3d.set_ylim(-lim, lim)
    ax_3d.set_zlim(-lim * 0.1, lim)
    pred_3d, = ax_3d.plot([], [], [], "-", color="darkorange", lw=2, zorder=5)

    # Side panels
    for ax, xl, yl, tl in [
        (ax_eu, "East (mm)",  "Up (mm)",   "Side (E–Up)"),
        (ax_nu, "North (mm)", "Up (mm)",   "Side (N–Up)"),
    ]:
        ax.set_title(tl, fontsize=9)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.axhline(0, c="k", lw=0.4, alpha=0.3)
        ax.axvline(0, c="k", lw=0.4, alpha=0.3)
        ax.set_xlim(-pad, pad)
        ax.set_ylim(-pad * 0.1, pad)
    pred_eu, = ax_eu.plot([], [], "-", color="darkorange", lw=2)
    pred_nu, = ax_nu.plot([], [], "-", color="darkorange", lw=2)

    # ── Info + buttons ────────────────────────────────────────────────────────
    ax_info = fig.add_axes([0.05, 0.01, 0.55, 0.18])
    ax_info.axis("off")
    info_text = ax_info.text(0, 0.5,
        "Click and drag in the drawing panel to trace a path.\n"
        "Points outside the max range are clipped to the boundary.",
        va="center", fontsize=9, color="navy")

    for name, x in [("preview", 0.58), ("execute", 0.68), ("compare", 0.78), ("clear", 0.88)]:
        ax_b = fig.add_axes([x, 0.08, 0.08, 0.06])
        if name == "preview":
            btn_preview = Button(ax_b, "Preview")
        elif name == "execute":
            btn_execute = Button(ax_b, "Execute" if not dry_run else "Execute\n(dry)")
            btn_execute.color = "lightyellow"
        elif name == "compare":
            btn_compare = Button(ax_b, "Compare")
            btn_compare.color = "lightcyan"
        else:
            btn_clear = Button(ax_b, "Clear")

    # ── Mouse interaction ─────────────────────────────────────────────────────

    def _clip(e_m, n_m):
        d = np.sqrt(e_m**2 + n_m**2)
        if d > max_r:
            s = max_r / d
            return e_m * s, n_m * s
        return e_m, n_m

    def _on_press(event):
        if event.inaxes is not ax_draw:
            return
        if event.button != 1:
            return
        drawing[0] = True
        drawn_e.clear()
        drawn_n.clear()
        e, n = _clip(event.xdata / 1000.0, event.ydata / 1000.0)
        drawn_e.append(e)
        drawn_n.append(n)

    def _on_motion(event):
        if not drawing[0] or event.inaxes is not ax_draw:
            return
        e, n = _clip(event.xdata / 1000.0, event.ydata / 1000.0)
        # Downsample: only add if moved > 2 mm
        if drawn_e and abs(e - drawn_e[-1]) < 0.002 and abs(n - drawn_n[-1]) < 0.002:
            return
        drawn_e.append(e)
        drawn_n.append(n)
        drawn_line.set_data([v * 1000 for v in drawn_e],
                            [v * 1000 for v in drawn_n])
        fig.canvas.draw_idle()

    def _on_release(event):
        if event.button != 1:
            return
        drawing[0] = False
        if len(drawn_e) < 2:
            return
        # Resample to waypoints and show dots
        xe, xn = _resample_path(drawn_e, drawn_n, n_out=n_wpoints)
        wp_dots.set_data([v * 1000 for v in xe], [v * 1000 for v in xn])
        info_text.set_text(
            f"Path drawn: {len(drawn_e)} raw points → {n_wpoints} waypoints.  "
            "Click Preview to see predicted trajectory."
        )
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", _on_press)
    fig.canvas.mpl_connect("motion_notify_event", _on_motion)
    fig.canvas.mpl_connect("button_release_event", _on_release)

    # ── Button callbacks ──────────────────────────────────────────────────────

    def _on_preview(event):
        if len(drawn_e) < 2:
            info_text.set_text("Draw a path first.")
            fig.canvas.draw_idle()
            return

        xe, xn = _resample_path(drawn_e, drawn_n, n_out=n_wpoints)
        xyz_wps = np.column_stack([xe, xn, np.zeros(n_wpoints)]).astype(np.float32)
        motor_seq, xyz_seq = generate_predicted_trajectory(xyz_wps, n_interp_steps=30)
        predicted[0] = (motor_seq, xyz_seq)

        pred_draw.set_data(xyz_seq[:, 0] * 1000, xyz_seq[:, 1] * 1000)
        pred_3d.set_data(xyz_seq[:, 0] * 1000, xyz_seq[:, 1] * 1000)
        pred_3d.set_3d_properties(xyz_seq[:, 2] * 1000)
        pred_eu.set_data(xyz_seq[:, 0] * 1000, xyz_seq[:, 2] * 1000)
        pred_nu.set_data(xyz_seq[:, 1] * 1000, xyz_seq[:, 2] * 1000)

        path, rn = save_predicted_trajectory(motor_seq, xyz_seq, label="path_predicted")
        pred_csv_path[0] = path
        pred_run_num[0]  = rn
        info_text.set_text(
            f"Predicted: {len(xyz_seq)} steps across {n_wpoints} waypoints.\n"
            f"CSV: {os.path.basename(path)}"
        )
        fig.canvas.draw_idle()

    def _on_execute(event):
        if predicted[0] is None:
            info_text.set_text("Click Preview first.")
            fig.canvas.draw_idle()
            return
        motor_seq, xyz_seq = predicted[0]
        if dry_run:
            info_text.set_text("Dry-run: hardware execution skipped.")
            fig.canvas.draw_idle()
            return
        info_text.set_text("Executing on hardware...")
        fig.canvas.draw_idle()
        plt.pause(0.1)
        real_path = execute_on_hardware(motor_seq, step_delay=0.3, label="path_real", run_num=pred_run_num[0])
        real_csv_path[0] = real_path
        if real_path:
            info_text.set_text(
                f"Done!  Real: {os.path.basename(real_path)}\nClick Compare."
            )
        else:
            info_text.set_text("Hardware execution failed or ROS2 not available.")
        fig.canvas.draw_idle()

    def _on_compare(event):
        p = pred_csv_path[0]
        r = real_csv_path[0]
        if p is None:
            info_text.set_text("No predicted trajectory yet.")
            fig.canvas.draw_idle()
            return
        cmd = f"ros2 run foam_viz compare_trajectories --animate --predicted '{p}'"
        if r:
            cmd += f" --real '{r}'"
        os.system(cmd + " &")

    def _on_clear(event):
        drawn_e.clear()
        drawn_n.clear()
        predicted[0] = None
        for artist in [drawn_line, wp_dots, pred_draw, pred_eu, pred_nu]:
            artist.set_data([], [])
        pred_3d.set_data([], [])
        pred_3d.set_3d_properties([])
        info_text.set_text("Cleared.  Click and drag to draw a new path.")
        fig.canvas.draw_idle()

    btn_preview.on_clicked(_on_preview)
    btn_execute.on_clicked(_on_execute)
    btn_compare.on_clicked(_on_compare)
    btn_clear.on_clicked(_on_clear)

    plt.show()


if __name__ == "__main__":
    main()
