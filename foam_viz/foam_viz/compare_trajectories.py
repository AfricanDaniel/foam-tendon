#!/usr/bin/env python3
"""
Compare a predicted trajectory (from foam_ml) with a real recorded trajectory.

Modes
-----
  Static (default)  : full trajectories overlaid on all panels at once
  Animated          : both dots advance in sync like trajectory_replayer

Usage:
    ros2 run foam_viz compare_trajectories --predicted <csv> [--real <csv>]
    ros2 run foam_viz compare_trajectories --predicted <csv> --real <csv> --animate
    ros2 run foam_viz compare_trajectories --predicted <csv> --real <csv> --animate --speed 2.0

<csv> can be a filename searched in data_collection/, a partial name, or an
absolute path.  If --real is omitted only the predicted trajectory is shown.
"""

import argparse
import os
import sys

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from trajectory_replayer import _resolve_path, load_csv, _find_data_dir

DATA_DIR = _find_data_dir()

_P_COLOR = "darkorange"   # predicted
_R_COLOR = "royalblue"    # real


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _resample(arr: np.ndarray, n_out: int) -> np.ndarray:
    idx_in  = np.linspace(0, 1, len(arr))
    idx_out = np.linspace(0, 1, n_out)
    return np.column_stack([
        np.interp(idx_out, idx_in, arr[:, i]) for i in range(arr.shape[1])
    ])


def _align(pred: dict, real: dict):
    """Resample both to the same length. Returns (p_xyz, r_xyz) in metres."""
    p = np.column_stack([pred["east"], pred["north"], pred["up"]])
    r = np.column_stack([real["east"], real["north"], real["up"]])
    n = max(len(p), len(r))
    return _resample(p, n), _resample(r, n)


def _axis_limits(arrays, margin_frac=0.15, min_half=2.0):
    """Return (lo, hi) that fits all arrays with a margin, ≥ min_half mm half-span."""
    all_vals = np.concatenate([a.ravel() for a in arrays])
    lo, hi   = all_vals.min(), all_vals.max()
    half     = max((hi - lo) / 2 * (1 + margin_frac), min_half)
    mid      = (lo + hi) / 2
    return mid - half, mid + half


def _set_3d_equal(ax, *arrays):
    """Force equal aspect on a 3D axes given several value arrays."""
    lo, hi = _axis_limits(arrays)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)


# ── Static comparison ──────────────────────────────────────────────────────────

def plot_comparison(pred: dict, real: dict | None = None) -> None:
    has_real = real is not None
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        f"{'Predicted vs Real' if has_real else 'Predicted Trajectory'}  "
        f"({'static — pass --animate for animated replay'})",
        fontsize=11, fontweight="bold",
    )

    pe = pred["east"]  * 1000
    pn = pred["north"] * 1000
    pu = pred["up"]    * 1000
    if has_real:
        re = real["east"]  * 1000
        rn = real["north"] * 1000
        ru = real["up"]    * 1000

    # ── 3D ────────────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(2, 3, (1, 4), projection="3d")
    ax3.plot(pe, pn, pu, color=_P_COLOR, lw=2, label="Predicted")
    ax3.scatter(pe[-1], pn[-1], pu[-1], color=_P_COLOR, marker="*", s=120, zorder=5)
    if has_real:
        ax3.plot(re, rn, ru, color=_R_COLOR, lw=2, label="Real")
        ax3.scatter(re[-1], rn[-1], ru[-1], color=_R_COLOR, marker="*", s=120, zorder=5)
    ax3.scatter([0], [0], [0], color="lime", s=80, zorder=6, label="Home")
    ax3.set_xlabel("East (mm)"); ax3.set_ylabel("North (mm)"); ax3.set_zlabel("Up (mm)")
    ax3.set_title("3D trajectories"); ax3.legend(fontsize=9)
    _xs = [pe, pn, pu] + ([re, rn, ru] if has_real else [])
    _set_3d_equal(ax3, *_xs)

    def _2d_panel(pos, xlabel, ylabel, title, xd, yd, xr=None, yr=None):
        ax = fig.add_subplot(2, 3, pos)
        ax.plot(xd, yd, color=_P_COLOR, lw=2, label="Predicted")
        ax.scatter(xd[0], yd[0], c="lime", s=60, zorder=5)
        if xr is not None:
            ax.plot(xr, yr, color=_R_COLOR, lw=2, label="Real")
        ax.scatter([0], [0], c="lime", s=80, zorder=6)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.set_aspect("equal")
        ax.axhline(0, c="k", lw=0.4, alpha=0.3); ax.axvline(0, c="k", lw=0.4, alpha=0.3)
        ax.legend(fontsize=8)
        lo, hi = _axis_limits([xd, yd] + ([xr, yr] if xr is not None else []))
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        return ax

    _2d_panel(2, "East (mm)",  "North (mm)", "Top-down (E–N)",
              pe, pn, re if has_real else None, rn if has_real else None)
    _2d_panel(3, "East (mm)",  "Up (mm)",    "Side (E–Up)",
              pe, pu, re if has_real else None, ru if has_real else None)
    _2d_panel(6, "North (mm)", "Up (mm)",    "Side (N–Up)",
              pn, pu, rn if has_real else None, ru if has_real else None)

    # ── Error / info panel ────────────────────────────────────────────────────
    ax_err = fig.add_subplot(2, 3, 5)
    if has_real:
        p_arr, r_arr = _align(pred, real)
        err = np.sqrt(np.sum((p_arr - r_arr) ** 2, axis=1)) * 1000
        t   = np.linspace(0, 1, len(err))
        ax_err.plot(t, err, color="crimson", lw=1.5)
        ax_err.fill_between(t, err, alpha=0.2, color="crimson")
        ax_err.axhline(err.mean(), color="crimson", lw=1.0, linestyle="--",
                       label=f"Mean {err.mean():.1f} mm")
        ax_err.set_xlabel("Normalised time")
        ax_err.set_ylabel("3D error (mm)")
        ax_err.set_title("Prediction error (3D Euclidean)")
        ax_err.legend(fontsize=8)
    else:
        ax_err.text(0.5, 0.5, "No real trajectory\nto compare against",
                    ha="center", va="center", transform=ax_err.transAxes,
                    fontsize=10, color="gray")
        ax_err.set_title("Prediction error"); ax_err.axis("off")

    fig.tight_layout()
    plt.show()


# ── Animated comparison ────────────────────────────────────────────────────────

def plot_animated(pred: dict, real: dict | None = None, speed: float = 1.0) -> None:
    """
    Animate both trajectories advancing in sync.

    Both trajectories are resampled to the same number of frames (the longer
    one's frame count) so they finish together regardless of original length.
    The dots advance at normalised-time pace so you always see the full motion
    from start to end.
    """
    has_real = real is not None

    pe = pred["east"]  * 1000
    pn = pred["north"] * 1000
    pu = pred["up"]    * 1000
    pt = pred["t"]
    pm = pred["motors"]

    if has_real:
        re = real["east"]  * 1000
        rn = real["north"] * 1000
        ru = real["up"]    * 1000
        rm = real["motors"]
        # Resample both to the same N frames for sync'd playback
        N  = max(len(pe), len(re))
        p_xyz = _resample(np.column_stack([pe, pn, pu]), N)
        r_xyz = _resample(np.column_stack([re, rn, ru]), N)
        p_mot = _resample(pm, N)
        r_mot = _resample(rm, N)
        t_norm = np.linspace(0, 1, N)
        # Recover approximate real durations for time label
        p_dur = pt[-1]
        r_dur = real["t"][-1]
    else:
        N     = len(pe)
        p_xyz = np.column_stack([pe, pn, pu])
        r_xyz = None
        p_mot = pm
        r_mot = None
        t_norm = np.linspace(0, 1, N)
        p_dur = pt[-1]
        r_dur = None

    pe_a, pn_a, pu_a = p_xyz[:, 0], p_xyz[:, 1], p_xyz[:, 2]
    if has_real:
        re_a, rn_a, ru_a = r_xyz[:, 0], r_xyz[:, 1], r_xyz[:, 2]

    # Pre-compute error at each frame for the live error strip
    if has_real:
        err_arr = np.sqrt(np.sum((p_xyz - r_xyz) ** 2, axis=1))  # mm

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9))
    title_str = (
        f"Predicted (orange) vs Real (blue)  –  speed ×{speed:.1f}"
        if has_real else
        f"Predicted trajectory  –  speed ×{speed:.1f}"
    )
    fig.suptitle(title_str, fontsize=11, fontweight="bold")

    # Axis limits — computed once from the full data
    all_e = [pe_a] + ([re_a] if has_real else [])
    all_n = [pn_a] + ([rn_a] if has_real else [])
    all_u = [pu_a] + ([ru_a] if has_real else [])
    all_h = all_e + all_n   # for top-down equal-aspect

    e_lo, e_hi = _axis_limits(all_e)
    n_lo, n_hi = _axis_limits(all_n)
    u_lo, u_hi = _axis_limits(all_u)
    h_lo = min(e_lo, n_lo);  h_hi = max(e_hi, n_hi)

    # ── 3D panel ──────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(2, 3, (1, 4), projection="3d")
    ax3.plot(pe_a, pn_a, pu_a, color=_P_COLOR, lw=0.8, alpha=0.35)
    if has_real:
        ax3.plot(re_a, rn_a, ru_a, color=_R_COLOR, lw=0.8, alpha=0.35)
    ax3.scatter([0], [0], [0], color="lime", s=60, zorder=4, label="Home")
    _set_3d_equal(ax3, pe_a, pn_a, pu_a,
                  *(([re_a, rn_a, ru_a]) if has_real else []))
    ax3.set_xlabel("East (mm)"); ax3.set_ylabel("North (mm)"); ax3.set_zlabel("Up (mm)")
    ax3.set_title("3D view"); ax3.legend(fontsize=8)

    trail_3d_p, = ax3.plot([], [], [], color=_P_COLOR, lw=2)
    dot_3d_p,   = ax3.plot([], [], [], "o", color=_P_COLOR, ms=8, zorder=5)
    if has_real:
        trail_3d_r, = ax3.plot([], [], [], color=_R_COLOR, lw=2)
        dot_3d_r,   = ax3.plot([], [], [], "o", color=_R_COLOR, ms=8, zorder=5)

    # ── Top-down ──────────────────────────────────────────────────────────────
    ax_top = fig.add_subplot(2, 3, 2)
    ax_top.plot(pe_a, pn_a, color=_P_COLOR, lw=0.8, alpha=0.35, label="Predicted")
    if has_real:
        ax_top.plot(re_a, rn_a, color=_R_COLOR, lw=0.8, alpha=0.35, label="Real")
    ax_top.scatter([0], [0], c="lime", s=60, zorder=4)
    ax_top.set_xlabel("East (mm)"); ax_top.set_ylabel("North (mm)")
    ax_top.set_title("Top-down (E–N)"); ax_top.set_aspect("equal")
    ax_top.set_xlim(h_lo, h_hi); ax_top.set_ylim(h_lo, h_hi)
    ax_top.axhline(0, c="k", lw=0.4, alpha=0.3); ax_top.axvline(0, c="k", lw=0.4, alpha=0.3)
    ax_top.legend(fontsize=8, loc="upper right")

    trail_top_p, = ax_top.plot([], [], color=_P_COLOR, lw=2)
    dot_top_p,   = ax_top.plot([], [], "o", color=_P_COLOR, ms=8, zorder=5)
    if has_real:
        trail_top_r, = ax_top.plot([], [], color=_R_COLOR, lw=2)
        dot_top_r,   = ax_top.plot([], [], "o", color=_R_COLOR, ms=8, zorder=5)

    # ── Side E–Up ─────────────────────────────────────────────────────────────
    ax_e = fig.add_subplot(2, 3, 3)
    ax_e.plot(pe_a, pu_a, color=_P_COLOR, lw=0.8, alpha=0.35)
    if has_real:
        ax_e.plot(re_a, ru_a, color=_R_COLOR, lw=0.8, alpha=0.35)
    ax_e.scatter([0], [0], c="lime", s=60, zorder=4)
    ax_e.set_xlabel("East (mm)"); ax_e.set_ylabel("Up (mm)")
    ax_e.set_title("Side (E–Up)"); ax_e.set_aspect("equal")
    ax_e.set_xlim(e_lo, e_hi); ax_e.set_ylim(u_lo, u_hi)
    ax_e.axhline(0, c="k", lw=0.4, alpha=0.3); ax_e.axvline(0, c="k", lw=0.4, alpha=0.3)

    trail_e_p, = ax_e.plot([], [], color=_P_COLOR, lw=2)
    dot_e_p,   = ax_e.plot([], [], "o", color=_P_COLOR, ms=8, zorder=5)
    if has_real:
        trail_e_r, = ax_e.plot([], [], color=_R_COLOR, lw=2)
        dot_e_r,   = ax_e.plot([], [], "o", color=_R_COLOR, ms=8, zorder=5)

    # ── Side N–Up ─────────────────────────────────────────────────────────────
    ax_n = fig.add_subplot(2, 3, 6)
    ax_n.plot(pn_a, pu_a, color=_P_COLOR, lw=0.8, alpha=0.35)
    if has_real:
        ax_n.plot(rn_a, ru_a, color=_R_COLOR, lw=0.8, alpha=0.35)
    ax_n.scatter([0], [0], c="lime", s=60, zorder=4)
    ax_n.set_xlabel("North (mm)"); ax_n.set_ylabel("Up (mm)")
    ax_n.set_title("Side (N–Up)"); ax_n.set_aspect("equal")
    ax_n.set_xlim(n_lo, n_hi); ax_n.set_ylim(u_lo, u_hi)
    ax_n.axhline(0, c="k", lw=0.4, alpha=0.3); ax_n.axvline(0, c="k", lw=0.4, alpha=0.3)

    trail_n_p, = ax_n.plot([], [], color=_P_COLOR, lw=2)
    dot_n_p,   = ax_n.plot([], [], "o", color=_P_COLOR, ms=8, zorder=5)
    if has_real:
        trail_n_r, = ax_n.plot([], [], color=_R_COLOR, lw=2)
        dot_n_r,   = ax_n.plot([], [], "o", color=_R_COLOR, ms=8, zorder=5)

    # ── Motor / error strip ────────────────────────────────────────────────────
    ax_m = fig.add_subplot(2, 3, 5)
    _motor_labels  = ["M1 North", "M2 East", "M3 South", "M4 West"]
    _motor_colors  = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]

    for i in range(4):
        ax_m.plot(t_norm, p_mot[:, i],
                  color=_motor_colors[i], lw=1.0, alpha=0.35,
                  linestyle="--", label=f"{_motor_labels[i]} pred")
        if has_real:
            ax_m.plot(t_norm, r_mot[:, i],
                      color=_motor_colors[i], lw=1.5, alpha=0.6,
                      label=f"{_motor_labels[i]} real")

    ax_m.axhline(0, c="k", lw=0.4, alpha=0.3)
    ax_m.set_xlabel("Normalised time")
    ax_m.set_ylabel("Position (pulses)")
    ax_m.legend(fontsize=6, ncol=2, loc="upper left")
    cursor_line = ax_m.axvline(0, color="red", lw=1.2)
    time_title  = ax_m.set_title("Motors  t=0.00")

    # Live error annotation on top-down panel (only when has_real)
    if has_real:
        err_text = ax_top.text(
            0.02, 0.97, "", transform=ax_top.transAxes,
            fontsize=8, va="top", color="crimson",
            bbox=dict(facecolor="white", alpha=0.7, pad=2, edgecolor="none"),
        )

    # ── Animation update ──────────────────────────────────────────────────────
    def update(k):
        # Predicted trails
        trail_3d_p.set_data(pe_a[:k+1], pn_a[:k+1])
        trail_3d_p.set_3d_properties(pu_a[:k+1])
        dot_3d_p.set_data([pe_a[k]], [pn_a[k]])
        dot_3d_p.set_3d_properties([pu_a[k]])

        trail_top_p.set_data(pe_a[:k+1], pn_a[:k+1])
        dot_top_p.set_data([pe_a[k]], [pn_a[k]])

        trail_e_p.set_data(pe_a[:k+1], pu_a[:k+1])
        dot_e_p.set_data([pe_a[k]], [pu_a[k]])

        trail_n_p.set_data(pn_a[:k+1], pu_a[:k+1])
        dot_n_p.set_data([pn_a[k]], [pu_a[k]])

        artists = [trail_3d_p, dot_3d_p,
                   trail_top_p, dot_top_p,
                   trail_e_p, dot_e_p,
                   trail_n_p, dot_n_p,
                   cursor_line, time_title]

        if has_real:
            trail_3d_r.set_data(re_a[:k+1], rn_a[:k+1])
            trail_3d_r.set_3d_properties(ru_a[:k+1])
            dot_3d_r.set_data([re_a[k]], [rn_a[k]])
            dot_3d_r.set_3d_properties([ru_a[k]])

            trail_top_r.set_data(re_a[:k+1], rn_a[:k+1])
            dot_top_r.set_data([re_a[k]], [rn_a[k]])

            trail_e_r.set_data(re_a[:k+1], ru_a[:k+1])
            dot_e_r.set_data([re_a[k]], [ru_a[k]])

            trail_n_r.set_data(rn_a[:k+1], ru_a[:k+1])
            dot_n_r.set_data([rn_a[k]], [ru_a[k]])

            err_mm = err_arr[k]
            err_text.set_text(f"Error: {err_mm:.1f} mm")
            artists += [trail_3d_r, dot_3d_r,
                        trail_top_r, dot_top_r,
                        trail_e_r, dot_e_r,
                        trail_n_r, dot_n_r,
                        err_text]

        cursor_line.set_xdata([t_norm[k], t_norm[k]])

        # Time label showing approximate real durations
        if has_real:
            p_t = t_norm[k] * p_dur
            r_t = t_norm[k] * r_dur
            time_title.set_text(f"Motors  pred {p_t:.2f}s / real {r_t:.2f}s")
        else:
            time_title.set_text(f"Motors  t={t_norm[k]*p_dur:.2f}s")

        return artists

    # Interval based on the slower (longer) trajectory's natural frame rate
    base_dt    = 1.0 / 50.0          # assume 50 Hz collection
    interval_ms = max(10.0, base_dt * 1000.0 / speed)

    ani = animation.FuncAnimation(
        fig, update, frames=N, interval=interval_ms, blit=False, repeat=False
    )

    fig.tight_layout()
    plt.show()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    raw = [a for a in sys.argv[1:] if not a.startswith("__")]
    sep = raw.index("--ros-args") if "--ros-args" in raw else len(raw)
    args_to_parse = raw[:sep]

    parser = argparse.ArgumentParser(
        description="Compare predicted vs real foam trajectories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"CSV files are searched in:\n  {DATA_DIR}",
    )
    parser.add_argument("--predicted", "-p", required=True,
                        help="Predicted trajectory CSV (from foam_ml)")
    parser.add_argument("--real", "-r", default=None,
                        help="Real trajectory CSV (from data_collection)")
    parser.add_argument("--animate", "-a", action="store_true",
                        help="Animated replay mode (both dots advance in sync)")
    parser.add_argument("--speed", "-s", type=float, default=1.0,
                        help="Playback speed multiplier for --animate (default 1.0)")
    opts = parser.parse_args(args_to_parse)

    pred_path = _resolve_path(opts.predicted)
    print(f"Predicted : {pred_path}")
    pred_data = load_csv(pred_path)
    p = pred_data
    print(f"  {len(p['t'])} rows  |  dur {p['t'][-1]:.2f}s  |  "
          f"max disp {np.sqrt(p['east']**2+p['north']**2+p['up']**2).max()*1000:.1f} mm")

    real_data = None
    if opts.real:
        real_path = _resolve_path(opts.real)
        print(f"Real      : {real_path}")
        real_data = load_csv(real_path)
        r = real_data
        print(f"  {len(r['t'])} rows  |  dur {r['t'][-1]:.2f}s  |  "
              f"max disp {np.sqrt(r['east']**2+r['north']**2+r['up']**2).max()*1000:.1f} mm")

    if opts.animate:
        plot_animated(pred_data, real_data, speed=opts.speed)
    else:
        plot_comparison(pred_data, real_data)


if __name__ == "__main__":
    main()
