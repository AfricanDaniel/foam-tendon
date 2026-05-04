#!/usr/bin/env python3
"""
Replay and visualize a rigid-body trajectory from a data_collection CSV.

Usage:
    ros2 run foam_viz trajectory_replayer --file <csv_filename>

    <csv_filename> can be a bare filename (searched inside data_collection/)
    or an absolute path.

    Optional:
        --speed   playback speed multiplier (default 1.0, >1 = faster)
        --no-anim show static full-trajectory plot instead of animation

Coordinate convention (empirically verified from motor-direction data)
----------------------------------------------------------------------
    NatNet X  →  Robot East   (positive = East)
    NatNet Y  →  Robot North  (positive = North)
    NatNet Z  →  Robot Up     (positive = up)

    Home position (first valid row) is subtracted so (0, 0, 0) = home.

Motor layout reminder
---------------------
    Motor 1 (North): CW pulls North
    Motor 2 (East):  CW pulls East
    Motor 3 (South): CCW pulls South
    Motor 4 (West):  CCW pulls West

    When the foam bends North the marker travels in the +Y NatNet direction.
    When the foam bends East  the marker travels in the +X NatNet direction.
"""

import argparse
import os
import sys

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# ── Locate the data_collection directory ──────────────────────────────────────

def _find_data_dir() -> str:
    """Walk up from this file to find src/actuator/data_collection."""
    path = os.path.abspath(__file__)
    for _ in range(12):
        parent = os.path.dirname(path)
        if parent == path:
            break
        candidate = os.path.join(parent, 'src', 'actuator', 'data_collection')
        if os.path.isdir(candidate):
            return candidate
        path = parent
    # Fallback: look next to the foam_viz package
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        '..', 'actuator', 'data_collection'
    )


DATA_DIR = _find_data_dir()


# ── CSV loading ────────────────────────────────────────────────────────────────

def _resolve_path(name: str) -> str:
    if os.path.isabs(name) and os.path.isfile(name):
        return name
    # Search data_collection/ and its known subdirectories
    search_dirs = [
        DATA_DIR,
        os.path.join(DATA_DIR, 'training_data'),
        os.path.join(DATA_DIR, 'ml_runs'),
    ]
    for d in search_dirs:
        direct = os.path.join(d, name)
        if os.path.isfile(direct):
            return direct
    # Try substring match across all search dirs
    all_matches = []
    for d in search_dirs:
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if name in f and f.endswith('.csv'):
                    all_matches.append(os.path.join(d, f))
    if len(all_matches) == 1:
        return all_matches[0]
    if len(all_matches) > 1:
        print(f'Ambiguous match for "{name}":')
        for m in all_matches:
            print(f'  {m}')
        sys.exit(1)
    print(f'File not found: "{name}"\nSearched in: {", ".join(search_dirs)}')
    sys.exit(1)


def load_csv(path: str) -> dict:
    """
    Load a data_collection CSV.  Returns a dict with keys:
        t        – time array (seconds, relative to start)
        east     – East displacement from home (m)
        north    – North displacement from home (m)
        up       – Up displacement from home (m)
        motors   – (N, 4) array of motor positions (pulses relative to home)
        raw_x/y/z – raw NatNet positions (m)
        label    – filename stem
    """
    data: dict[str, list] = {k: [] for k in
        ['t', 'raw_x', 'raw_y', 'raw_z',
         'm1', 'm2', 'm3', 'm4']}

    with open(path, newline='') as f:
        import csv
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row['optitrack_x'])
                y = float(row['optitrack_y'])
                z = float(row['optitrack_z'])
            except (ValueError, KeyError):
                continue  # skip rows with missing optitrack data
            data['t'].append(float(row['timestamp_s']))
            data['raw_x'].append(x)
            data['raw_y'].append(y)
            data['raw_z'].append(z)
            for i, key in enumerate(['motor_1_pos', 'motor_2_pos',
                                      'motor_3_pos', 'motor_4_pos'], 1):
                try:
                    data[f'm{i}'].append(float(row[key]))
                except (ValueError, KeyError):
                    data[f'm{i}'].append(float('nan'))

    if not data['t']:
        print('No valid optitrack rows found in the CSV.')
        sys.exit(1)

    t   = np.array(data['t'])
    rx  = np.array(data['raw_x'])
    ry  = np.array(data['raw_y'])
    rz  = np.array(data['raw_z'])

    # A row with timestamp_s < 0 is the home-reference marker written by
    # foam_controller_node.  Use its NatNet coords as the zero reference and
    # drop it from the data so it doesn't appear in the plots.
    if len(t) > 1 and t[0] < 0:
        home_x, home_y, home_z = rx[0], ry[0], rz[0]
        t  = t[1:];  rx = rx[1:]; ry = ry[1:]; rz = rz[1:]
        for k in ['m1', 'm2', 'm3', 'm4']:
            data[k] = data[k][1:]
    else:
        # Legacy CSVs: first valid frame is home
        home_x, home_y, home_z = rx[0], ry[0], rz[0]

    # NatNet → robot frame (empirically verified):
    #   NatNet X = East,  NatNet Y = North,  NatNet Z = Up
    east  = rx - home_x
    north = ry - home_y
    up    = rz - home_z

    motors = np.column_stack([data['m1'], data['m2'], data['m3'], data['m4']])

    return {
        't':      t - t[0],
        'east':   east,
        'north':  north,
        'up':     up,
        'motors': motors,
        'raw_x':  rx,
        'raw_y':  ry,
        'raw_z':  rz,
        'home':   (home_x, home_y, home_z),
        'label':  os.path.splitext(os.path.basename(path))[0],
    }


# ── Plotting helpers ───────────────────────────────────────────────────────────

_MOTOR_LABELS = ['M1 North', 'M2 East', 'M3 South', 'M4 West']
_MOTOR_COLORS = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']


def _colormap_line(ax, x, y, z, t, cmap='plasma', lw=1.5):
    """Draw a 3D line coloured by time using a LineCollection-style approach."""
    from matplotlib.collections import LineCollection
    # Build segments in 3D by projecting to 2D isn't straightforward;
    # use scatter for colour and a plain line for the path.
    ax.plot(x, y, z, color='gray', lw=0.6, alpha=0.4)
    sc = ax.scatter(x, y, z, c=t, cmap=cmap, s=4, zorder=3, depthshade=False)
    return sc


def _axis_equal_3d(ax, x, y, z, margin=0.01):
    """Force equal aspect ratio on a 3D axis."""
    xr = [x.min() - margin, x.max() + margin]
    yr = [y.min() - margin, y.max() + margin]
    zr = [z.min() - margin, z.max() + margin]
    half = max(xr[1]-xr[0], yr[1]-yr[0], zr[1]-zr[0]) / 2
    cx = np.mean(xr); cy = np.mean(yr); cz = np.mean(zr)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)


# ── Static overview plot ───────────────────────────────────────────────────────

def plot_static(data: dict) -> None:
    """Show a 4-panel static overview of the full trajectory."""
    east  = data['east']  * 1000   # → mm for readability
    north = data['north'] * 1000
    up    = data['up']    * 1000
    t     = data['t']
    motors = data['motors']
    label  = data['label']

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(label, fontsize=11, fontweight='bold')

    # ── 3D trajectory ──────────────────────────────────────────────────────────
    ax3d = fig.add_subplot(2, 3, (1, 4), projection='3d')
    _colormap_line(ax3d, east, north, up, t)
    ax3d.scatter([0], [0], [0], color='lime', s=80, zorder=5, label='Home')
    ax3d.scatter([east[0]], [north[0]], [up[0]],
                 color='deepskyblue', marker='^', s=100, zorder=5, label='Start')
    ax3d.scatter(east[-1], north[-1], up[-1],
                 color='red', marker='*', s=120, zorder=5, label='End')
    ax3d.set_xlabel('East (mm)')
    ax3d.set_ylabel('North (mm)')
    ax3d.set_zlabel('Up (mm)')
    ax3d.set_title('3D trajectory')
    ax3d.legend(fontsize=8)
    _axis_equal_3d(ax3d, east, north, up)
    # Add cardinal direction text
    lim = ax3d.get_ylim()[1]
    ax3d.text(0, lim, 0, 'N', color='gray', fontsize=8, ha='center')
    ax3d.text(ax3d.get_xlim()[1], 0, 0, 'E', color='gray', fontsize=8, ha='center')

    # ── Top-down (East–North) ──────────────────────────────────────────────────
    ax_top = fig.add_subplot(2, 3, 2)
    sc = ax_top.scatter(east, north, c=t, cmap='plasma', s=6)
    ax_top.plot(east, north, color='gray', lw=0.5, alpha=0.4)
    ax_top.scatter(0, 0, color='lime', s=80, zorder=5, label='Home')
    ax_top.scatter(east[0], north[0], color='deepskyblue', marker='^', s=80, zorder=5, label='Start')
    ax_top.scatter(east[-1], north[-1], color='red', marker='*', s=100, zorder=5, label='End')
    ax_top.set_xlabel('East (mm)')
    ax_top.set_ylabel('North (mm)')
    ax_top.set_title('Top-down view (E–N)')
    ax_top.set_aspect('equal')
    ax_top.axhline(0, color='k', lw=0.4, alpha=0.3)
    ax_top.axvline(0, color='k', lw=0.4, alpha=0.3)
    ax_top.legend(fontsize=8)
    # Cardinal arrows
    _add_cardinal_arrows(ax_top)

    # ── Side view East (East–Up) ───────────────────────────────────────────────
    ax_e = fig.add_subplot(2, 3, 3)
    ax_e.scatter(east, up, c=t, cmap='plasma', s=6)
    ax_e.plot(east, up, color='gray', lw=0.5, alpha=0.4)
    ax_e.scatter(0, 0, color='lime', s=80, zorder=5)
    ax_e.scatter(east[0], up[0], color='deepskyblue', marker='^', s=80, zorder=5)
    ax_e.scatter(east[-1], up[-1], color='red', marker='*', s=100, zorder=5)
    ax_e.set_xlabel('East (mm)')
    ax_e.set_ylabel('Up (mm)')
    ax_e.set_title('Side view (E–Up)')
    ax_e.set_aspect('equal')
    ax_e.axhline(0, color='k', lw=0.4, alpha=0.3)
    ax_e.axvline(0, color='k', lw=0.4, alpha=0.3)

    # ── Side view North (North–Up) ─────────────────────────────────────────────
    ax_n = fig.add_subplot(2, 3, 6)
    ax_n.scatter(north, up, c=t, cmap='plasma', s=6)
    ax_n.plot(north, up, color='gray', lw=0.5, alpha=0.4)
    ax_n.scatter(0, 0, color='lime', s=80, zorder=5)
    ax_n.scatter(north[0], up[0], color='deepskyblue', marker='^', s=80, zorder=5)
    ax_n.scatter(north[-1], up[-1], color='red', marker='*', s=100, zorder=5)
    ax_n.set_xlabel('North (mm)')
    ax_n.set_ylabel('Up (mm)')
    ax_n.set_title('Side view (N–Up)')
    ax_n.set_aspect('equal')
    ax_n.axhline(0, color='k', lw=0.4, alpha=0.3)
    ax_n.axvline(0, color='k', lw=0.4, alpha=0.3)

    # Shared colourbar (time)
    cb = fig.colorbar(sc, ax=[ax_top, ax_e, ax_n], shrink=0.6, label='Time (s)')

    # ── Motor positions ────────────────────────────────────────────────────────
    ax_m = fig.add_subplot(2, 3, 5)
    for i in range(4):
        ax_m.plot(t, motors[:, i], color=_MOTOR_COLORS[i],
                  label=_MOTOR_LABELS[i], lw=1.2)
    ax_m.set_xlabel('Time (s)')
    ax_m.set_ylabel('Position (pulses from home)')
    ax_m.set_title('Motor positions')
    ax_m.legend(fontsize=7)
    ax_m.axhline(0, color='k', lw=0.4, alpha=0.3)

    fig.tight_layout()
    plt.show()


def _add_cardinal_arrows(ax):
    """Add N/E compass labels to a 2D East–North plot."""
    xl, xr = ax.get_xlim()
    yb, yt = ax.get_ylim()
    cx = (xl + xr) / 2
    cy = (yb + yt) / 2
    dx = (xr - xl) * 0.04
    dy = (yt - yb) * 0.04
    ax.text(xr - dx, cy, 'E→', fontsize=7, color='gray', va='center')
    ax.text(cx, yt - dy, '↑N', fontsize=7, color='gray', ha='center')


# ── Animation ──────────────────────────────────────────────────────────────────

def plot_animated(data: dict, speed: float = 1.0) -> None:
    """Animate the rigid-body replay with a 4-panel layout."""
    east  = data['east']  * 1000   # → mm
    north = data['north'] * 1000
    up    = data['up']    * 1000
    t     = data['t']
    motors = data['motors']
    label  = data['label']
    n      = len(t)

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f'{label}  –  replay (speed ×{speed:.1f})', fontsize=10, fontweight='bold')

    pad = max(abs(east).max(), abs(north).max(), abs(up).max()) * 0.15 + 2.0
    e_lim = (east.min()  - pad, east.max()  + pad)
    n_lim = (north.min() - pad, north.max() + pad)
    u_lim = (up.min()    - pad, up.max()    + pad)
    half = max(e_lim[1]-e_lim[0], n_lim[1]-n_lim[0], u_lim[1]-u_lim[0]) / 2

    # ── 3D panel ───────────────────────────────────────────────────────────────
    ax3d = fig.add_subplot(2, 3, (1, 4), projection='3d')
    ax3d.plot(east, north, up, color='lightgray', lw=0.8, alpha=0.5)
    ax3d.scatter([0], [0], [0], color='lime', s=60, zorder=4, label='Home')
    ax3d.scatter([east[0]], [north[0]], [up[0]],
                 color='deepskyblue', marker='^', s=80, zorder=4, label='Start')
    trail3d, = ax3d.plot([], [], [], color='royalblue', lw=1.5)
    dot3d,   = ax3d.plot([], [], [], 'o', color='red', ms=8, zorder=5)
    ax3d.set_xlabel('East (mm)')
    ax3d.set_ylabel('North (mm)')
    ax3d.set_zlabel('Up (mm)')
    ax3d.set_title('3D view')
    ax3d.legend(fontsize=8)
    cx = (e_lim[0]+e_lim[1])/2; cy = (n_lim[0]+n_lim[1])/2; cz = (u_lim[0]+u_lim[1])/2
    ax3d.set_xlim(cx-half, cx+half)
    ax3d.set_ylim(cy-half, cy+half)
    ax3d.set_zlim(cz-half, cz+half)

    # ── Top-down (E–N) ─────────────────────────────────────────────────────────
    ax_top = fig.add_subplot(2, 3, 2)
    ax_top.plot(east, north, color='lightgray', lw=0.8, alpha=0.5)
    ax_top.scatter([0], [0], color='lime', s=60, zorder=4)
    ax_top.scatter([east[0]], [north[0]], color='deepskyblue', marker='^', s=60, zorder=4)
    trail_top, = ax_top.plot([], [], color='royalblue', lw=1.5)
    dot_top,   = ax_top.plot([], [], 'o', color='red', ms=8, zorder=5)
    ax_top.set_xlabel('East (mm)')
    ax_top.set_ylabel('North (mm)')
    ax_top.set_title('Top-down (E–N)')
    ax_top.set_xlim(e_lim)
    ax_top.set_ylim(n_lim)
    ax_top.set_aspect('equal')
    ax_top.axhline(0, color='k', lw=0.4, alpha=0.3)
    ax_top.axvline(0, color='k', lw=0.4, alpha=0.3)

    # ── Side East–Up ───────────────────────────────────────────────────────────
    ax_e = fig.add_subplot(2, 3, 3)
    ax_e.plot(east, up, color='lightgray', lw=0.8, alpha=0.5)
    ax_e.scatter([0], [0], color='lime', s=60, zorder=4)
    ax_e.scatter([east[0]], [up[0]], color='deepskyblue', marker='^', s=60, zorder=4)
    trail_e, = ax_e.plot([], [], color='royalblue', lw=1.5)
    dot_e,   = ax_e.plot([], [], 'o', color='red', ms=8, zorder=5)
    ax_e.set_xlabel('East (mm)')
    ax_e.set_ylabel('Up (mm)')
    ax_e.set_title('Side (E–Up)')
    ax_e.set_xlim(e_lim)
    ax_e.set_ylim(u_lim)
    ax_e.set_aspect('equal')
    ax_e.axhline(0, color='k', lw=0.4, alpha=0.3)
    ax_e.axvline(0, color='k', lw=0.4, alpha=0.3)

    # ── Motor positions strip ──────────────────────────────────────────────────
    ax_m = fig.add_subplot(2, 3, 5)
    for i in range(4):
        ax_m.plot(t, motors[:, i], color=_MOTOR_COLORS[i],
                  lw=1.0, alpha=0.4, label=_MOTOR_LABELS[i])
    motor_lines = [ax_m.axvline(t[0], color='red', lw=1.0)]
    ax_m.set_xlabel('Time (s)')
    ax_m.set_ylabel('Position (pulses)')
    ax_m.set_title('Motor positions')
    ax_m.legend(fontsize=7)
    ax_m.axhline(0, color='k', lw=0.4, alpha=0.3)
    time_text = ax_m.set_title('Motor positions  t=0.00 s')

    # ── Side North–Up ──────────────────────────────────────────────────────────
    ax_n = fig.add_subplot(2, 3, 6)
    ax_n.plot(north, up, color='lightgray', lw=0.8, alpha=0.5)
    ax_n.scatter([0], [0], color='lime', s=60, zorder=4)
    ax_n.scatter([north[0]], [up[0]], color='deepskyblue', marker='^', s=60, zorder=4)
    trail_n, = ax_n.plot([], [], color='royalblue', lw=1.5)
    dot_n,   = ax_n.plot([], [], 'o', color='red', ms=8, zorder=5)
    ax_n.set_xlabel('North (mm)')
    ax_n.set_ylabel('Up (mm)')
    ax_n.set_title('Side (N–Up)')
    ax_n.set_xlim(n_lim)
    ax_n.set_ylim(u_lim)
    ax_n.set_aspect('equal')
    ax_n.axhline(0, color='k', lw=0.4, alpha=0.3)
    ax_n.axvline(0, color='k', lw=0.4, alpha=0.3)

    fig.tight_layout()

    # ── Animation callback ─────────────────────────────────────────────────────
    frame_dt = np.diff(t, prepend=t[0])
    frame_dt[0] = 0.0

    def update(frame):
        k = frame  # current row index

        # trails
        trail3d.set_data(east[:k+1], north[:k+1])
        trail3d.set_3d_properties(up[:k+1])
        dot3d.set_data([east[k]], [north[k]])
        dot3d.set_3d_properties([up[k]])

        trail_top.set_data(east[:k+1], north[:k+1])
        dot_top.set_data([east[k]], [north[k]])

        trail_e.set_data(east[:k+1], up[:k+1])
        dot_e.set_data([east[k]], [up[k]])

        trail_n.set_data(north[:k+1], up[:k+1])
        dot_n.set_data([north[k]], [up[k]])

        motor_lines[0].set_xdata([t[k], t[k]])
        time_text.set_text(f'Motor positions  t={t[k]:.2f} s')

        return (trail3d, dot3d, trail_top, dot_top,
                trail_e, dot_e, trail_n, dot_n, motor_lines[0])

    # Interval in ms between frames; use actual elapsed time scaled by speed
    base_interval = float(np.mean(np.diff(t))) * 1000.0 / speed
    interval_ms   = max(10.0, base_interval)

    ani = animation.FuncAnimation(
        fig, update, frames=n, interval=interval_ms,
        blit=False, repeat=False
    )

    plt.show()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    # Strip ROS2 args so argparse doesn't choke on --ros-args
    raw_args = [a for a in sys.argv[1:] if not a.startswith('__')]
    ros_sep  = raw_args.index('--ros-args') if '--ros-args' in raw_args else len(raw_args)
    args_to_parse = raw_args[:ros_sep]

    parser = argparse.ArgumentParser(
        description='Replay a rigid-body trajectory from a data_collection CSV.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'CSV files are searched in:\n  {DATA_DIR}',
    )
    parser.add_argument(
        '--file', '-f', required=True,
        help='CSV filename (or full path). Partial name matching is supported.',
    )
    parser.add_argument(
        '--speed', '-s', type=float, default=1.0,
        help='Playback speed multiplier (default 1.0).',
    )
    parser.add_argument(
        '--no-anim', action='store_true',
        help='Show a static overview plot instead of animation.',
    )

    opts = parser.parse_args(args_to_parse)

    csv_path = _resolve_path(opts.file)
    print(f'Loading: {csv_path}')

    data = load_csv(csv_path)
    n    = len(data['t'])
    dur  = data['t'][-1]
    disp = np.sqrt(data['east']**2 + data['north']**2 + data['up']**2).max() * 1000

    home = data['home']
    print(
        f'Rows         : {n}\n'
        f'Duration     : {dur:.2f} s\n'
        f'Max disp     : {disp:.1f} mm from home\n'
        f'Home (NatNet): x={home[0]:.4f}  y={home[1]:.4f}  z={home[2]:.4f} m\n'
        f'\n'
        f'Coordinate frame used for display\n'
        f'  East  = NatNet X − {home[0]:.4f}\n'
        f'  North = NatNet Y − {home[1]:.4f}\n'
        f'  Up    = NatNet Z − {home[2]:.4f}\n'
    )

    if opts.no_anim:
        plot_static(data)
    else:
        plot_animated(data, speed=opts.speed)


if __name__ == '__main__':
    main()
