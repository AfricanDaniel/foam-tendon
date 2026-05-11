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

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.renderers.default = 'browser'

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


# ── Shared layout helper ───────────────────────────────────────────────────────

_MOTOR_LABELS = ['M1 North', 'M2 East', 'M3 South', 'M4 West']
_MOTOR_COLORS = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']


def _base_fig(title: str) -> go.Figure:
    """2×3 subplot figure: 3D scene spans left column, 2D panels fill right."""
    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{'type': 'scene', 'rowspan': 2}, {'type': 'xy'}, {'type': 'xy'}],
            [None,                             {'type': 'xy'}, {'type': 'xy'}],
        ],
        subplot_titles=['3D View', 'Top-Down (E–N)', 'Side (E–Up)',
                        '',        'Motor Positions',  'Side (N–Up)'],
        column_widths=[0.40, 0.30, 0.30],
        row_heights=[0.50, 0.50],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )
    fig.update_layout(
        title_text=title,
        template='plotly_white',
        width=1350, height=720,
        scene=dict(
            xaxis_title='East (mm)',
            yaxis_title='North (mm)',
            zaxis_title='Up (mm)',
            aspectmode='data',
            camera=dict(eye=dict(x=2.0, y=2.0, z=1.5)),
        ),
        legend=dict(x=1.02, y=1.0, xanchor='left'),
    )
    # Top-down (E–N): equal aspect, dtick=10 on both axes
    fig.update_xaxes(title_text='East (mm)',  scaleanchor='y',  scaleratio=1, dtick=10, row=1, col=2)
    fig.update_yaxes(title_text='North (mm)',                                 dtick=10, row=1, col=2)
    # Side E–Up: equal aspect, dtick=10 on both axes
    fig.update_xaxes(title_text='East (mm)',  scaleanchor='y2', scaleratio=1, dtick=10, row=1, col=3)
    fig.update_yaxes(title_text='Up (mm)',                                    dtick=10, row=1, col=3)
    # Motor plot: no equal aspect needed
    fig.update_xaxes(title_text='Time (s)',                                             row=2, col=2)
    fig.update_yaxes(title_text='Position (pulses)',                                    row=2, col=2)
    # Side N–Up: equal aspect, dtick=10 on both axes
    fig.update_xaxes(title_text='North (mm)', scaleanchor='y4', scaleratio=1, dtick=10, row=2, col=3)
    fig.update_yaxes(title_text='Up (mm)',                                    dtick=10, row=2, col=3)
    return fig


# ── Static overview plot ───────────────────────────────────────────────────────

def plot_static(data: dict) -> None:
    """Open a 5-panel static overview of the full trajectory in the browser."""
    east   = data['east']  * 1000   # → mm
    north  = data['north'] * 1000
    up     = data['up']    * 1000
    t      = data['t']
    motors = data['motors']
    label  = data['label']

    fig = _base_fig(label)

    # ── 3D trajectory coloured by time ────────────────────────────────────────
    fig.add_trace(go.Scatter3d(
        x=east, y=north, z=up, mode='lines',
        line=dict(color='lightgray', width=2), opacity=0.5,
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=east, y=north, z=up, mode='markers',
        marker=dict(size=3, color=t, colorscale='Plasma', showscale=True,
                    colorbar=dict(title='Time (s)', len=0.45, x=0.37)),
        name='Trajectory', showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode='markers',
        marker=dict(size=9, color='lime'), name='Home',
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[east[0]], y=[north[0]], z=[up[0]], mode='markers',
        marker=dict(size=9, color='deepskyblue', symbol='diamond'), name='Start',
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[east[-1]], y=[north[-1]], z=[up[-1]], mode='markers',
        marker=dict(size=9, color='red', symbol='cross'), name='End',
    ), row=1, col=1)

    # ── Top-down (E–N) ────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=east, y=north, mode='lines+markers',
        marker=dict(size=3, color=t, colorscale='Plasma', showscale=False),
        line=dict(color='lightgray', width=1),
        showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers',
        marker=dict(size=9, color='lime'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[east[0]], y=[north[0]], mode='markers',
        marker=dict(size=9, color='deepskyblue', symbol='diamond'),
        showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[east[-1]], y=[north[-1]], mode='markers',
        marker=dict(size=9, color='red', symbol='cross'),
        showlegend=False), row=1, col=2)

    # ── Side E–Up ─────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=east, y=up, mode='lines+markers',
        marker=dict(size=3, color=t, colorscale='Plasma', showscale=False),
        line=dict(color='lightgray', width=1),
        showlegend=False,
    ), row=1, col=3)
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers',
        marker=dict(size=9, color='lime'), showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=[east[0]], y=[up[0]], mode='markers',
        marker=dict(size=9, color='deepskyblue', symbol='diamond'),
        showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=[east[-1]], y=[up[-1]], mode='markers',
        marker=dict(size=9, color='red', symbol='cross'),
        showlegend=False), row=1, col=3)

    # ── Motor positions ────────────────────────────────────────────────────────
    for i in range(4):
        fig.add_trace(go.Scatter(
            x=t, y=motors[:, i], mode='lines',
            line=dict(color=_MOTOR_COLORS[i], width=2),
            name=_MOTOR_LABELS[i],
        ), row=2, col=2)

    # ── Side N–Up ─────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=north, y=up, mode='lines+markers',
        marker=dict(size=3, color=t, colorscale='Plasma', showscale=False),
        line=dict(color='lightgray', width=1),
        showlegend=False,
    ), row=2, col=3)
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers',
        marker=dict(size=9, color='lime'), showlegend=False), row=2, col=3)
    fig.add_trace(go.Scatter(x=[north[0]], y=[up[0]], mode='markers',
        marker=dict(size=9, color='deepskyblue', symbol='diamond'),
        showlegend=False), row=2, col=3)
    fig.add_trace(go.Scatter(x=[north[-1]], y=[up[-1]], mode='markers',
        marker=dict(size=9, color='red', symbol='cross'),
        showlegend=False), row=2, col=3)

    fig.show()


# ── Animation ──────────────────────────────────────────────────────────────────

def plot_animated(data: dict, speed: float = 1.0) -> None:
    """Open an interactive animation in the browser with play/pause and a scrub bar."""
    east   = data['east']  * 1000   # → mm
    north  = data['north'] * 1000
    up     = data['up']    * 1000
    t      = data['t']
    motors = data['motors']
    label  = data['label']
    n      = len(t)

    # Subsample so the browser stays responsive (≤ 250 frames)
    MAX_FRAMES = 250
    step = max(1, n // MAX_FRAMES)
    frame_idx = list(range(0, n, step))
    if frame_idx[-1] != n - 1:
        frame_idx.append(n - 1)

    motor_min = float(motors.min()) - 10
    motor_max = float(motors.max()) + 10

    fig = _base_fig(f'{label}  –  replay  (speed ×{speed:.1f})')

    # ── Static base traces (trace index annotated) ───────────────────────────
    #
    # 3D panel
    fig.add_trace(go.Scatter3d(           # 0  ghost full path
        x=east, y=north, z=up, mode='lines',
        line=dict(color='lightgray', width=1.5), opacity=0.4,
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(           # 1  home
        x=[0], y=[0], z=[0], mode='markers',
        marker=dict(size=9, color='lime'), name='Home',
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(           # 2  start
        x=[east[0]], y=[north[0]], z=[up[0]], mode='markers',
        marker=dict(size=9, color='deepskyblue', symbol='diamond'), name='Start',
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(           # 3  ← animated trail 3D
        x=[east[0]], y=[north[0]], z=[up[0]], mode='lines',
        line=dict(color='royalblue', width=3),
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(           # 4  ← animated dot 3D
        x=[east[0]], y=[north[0]], z=[up[0]], mode='markers',
        marker=dict(size=10, color='red'), name='Current',
    ), row=1, col=1)

    # Top-down panel
    fig.add_trace(go.Scatter(             # 5  ghost top-down
        x=east, y=north, mode='lines',
        line=dict(color='lightgray', width=1), opacity=0.5,
        showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(             # 6  home top-down
        x=[0], y=[0], mode='markers',
        marker=dict(size=9, color='lime'), showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(             # 7  ← animated trail top-down
        x=[east[0]], y=[north[0]], mode='lines',
        line=dict(color='royalblue', width=2), showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(             # 8  ← animated dot top-down
        x=[east[0]], y=[north[0]], mode='markers',
        marker=dict(size=10, color='red'), showlegend=False,
    ), row=1, col=2)

    # Side E–Up panel
    fig.add_trace(go.Scatter(             # 9  ghost E-Up
        x=east, y=up, mode='lines',
        line=dict(color='lightgray', width=1), opacity=0.5,
        showlegend=False,
    ), row=1, col=3)
    fig.add_trace(go.Scatter(             # 10 home E-Up
        x=[0], y=[0], mode='markers',
        marker=dict(size=9, color='lime'), showlegend=False,
    ), row=1, col=3)
    fig.add_trace(go.Scatter(             # 11 ← animated trail E-Up
        x=[east[0]], y=[up[0]], mode='lines',
        line=dict(color='royalblue', width=2), showlegend=False,
    ), row=1, col=3)
    fig.add_trace(go.Scatter(             # 12 ← animated dot E-Up
        x=[east[0]], y=[up[0]], mode='markers',
        marker=dict(size=10, color='red'), showlegend=False,
    ), row=1, col=3)

    # Motor panel
    for i in range(4):
        fig.add_trace(go.Scatter(         # 13-16 motor lines
            x=t, y=motors[:, i], mode='lines',
            line=dict(color=_MOTOR_COLORS[i], width=1.5),
            opacity=0.5, name=_MOTOR_LABELS[i],
        ), row=2, col=2)
    fig.add_trace(go.Scatter(             # 17 ← animated time cursor
        x=[t[0], t[0]], y=[motor_min, motor_max],
        mode='lines', line=dict(color='red', width=2, dash='dash'),
        showlegend=False,
    ), row=2, col=2)

    # Side N–Up panel
    fig.add_trace(go.Scatter(             # 18 ghost N-Up
        x=north, y=up, mode='lines',
        line=dict(color='lightgray', width=1), opacity=0.5,
        showlegend=False,
    ), row=2, col=3)
    fig.add_trace(go.Scatter(             # 19 home N-Up
        x=[0], y=[0], mode='markers',
        marker=dict(size=9, color='lime'), showlegend=False,
    ), row=2, col=3)
    fig.add_trace(go.Scatter(             # 20 ← animated trail N-Up
        x=[north[0]], y=[up[0]], mode='lines',
        line=dict(color='royalblue', width=2), showlegend=False,
    ), row=2, col=3)
    fig.add_trace(go.Scatter(             # 21 ← animated dot N-Up
        x=[north[0]], y=[up[0]], mode='markers',
        marker=dict(size=10, color='red'), showlegend=False,
    ), row=2, col=3)

    # Indices of traces updated each frame (must match frame data order below)
    ANIM_TRACES = [3, 4, 7, 8, 11, 12, 17, 20, 21]

    # ── Build frames ──────────────────────────────────────────────────────────
    frames = []
    for k in frame_idx:
        frames.append(go.Frame(
            name=str(k),
            traces=ANIM_TRACES,
            data=[
                go.Scatter3d(x=east[:k+1].tolist(),
                             y=north[:k+1].tolist(),
                             z=up[:k+1].tolist()),           # 3  trail 3D
                go.Scatter3d(x=[east[k]], y=[north[k]],
                             z=[up[k]]),                     # 4  dot 3D
                go.Scatter(x=east[:k+1].tolist(),
                           y=north[:k+1].tolist()),          # 7  trail top-down
                go.Scatter(x=[east[k]], y=[north[k]]),       # 8  dot top-down
                go.Scatter(x=east[:k+1].tolist(),
                           y=up[:k+1].tolist()),             # 11 trail E-Up
                go.Scatter(x=[east[k]], y=[up[k]]),          # 12 dot E-Up
                go.Scatter(x=[t[k], t[k]],
                           y=[motor_min, motor_max]),        # 17 time cursor
                go.Scatter(x=north[:k+1].tolist(),
                           y=up[:k+1].tolist()),             # 20 trail N-Up
                go.Scatter(x=[north[k]], y=[up[k]]),         # 21 dot N-Up
            ],
        ))

    mean_dt    = float(np.mean(np.diff(t))) if len(t) > 1 else 0.05
    frame_dur  = max(30, int(mean_dt * 1000 * step / speed))

    slider_steps = [
        dict(
            args=[[str(k)], {'frame': {'duration': frame_dur, 'redraw': True},
                             'mode': 'immediate'}],
            label=f'{t[k]:.1f}s',
            method='animate',
        )
        for k in frame_idx
    ]

    fig.frames = frames
    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            y=-0.06, x=0.0, xanchor='left', yanchor='top',
            buttons=[
                dict(
                    label='▶ Play',
                    method='animate',
                    args=[None, {'frame': {'duration': frame_dur, 'redraw': True},
                                 'fromcurrent': True, 'mode': 'immediate'}],
                ),
                dict(
                    label='⏸ Pause',
                    method='animate',
                    args=[[None], {'frame': {'duration': 0}, 'mode': 'immediate'}],
                ),
            ],
        )],
        sliders=[dict(
            active=0,
            steps=slider_steps,
            x=0.07, len=0.90, xanchor='left',
            y=-0.04, yanchor='top',
            pad={'b': 10},
            currentvalue=dict(prefix='t = ', visible=True, xanchor='center'),
            transition=dict(duration=0),
        )],
    )

    fig.show()


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
