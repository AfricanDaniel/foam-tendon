# %%
"""
Visualise all move_E_ trajectories from training data.

Three views:
  1. Top-down   (North vs East)
  2. Side view  (East vs Up)
  3. 3D         (East, North, Up)

Plus a motor-position panel showing how Motors 2 & 4 change with East displacement.
All coordinates are zeroed at the first row of each file (home position).
All units: mm for position, pulses for motor counts.
"""

import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path

pio.renderers.default = "browser"

# ── Find data ─────────────────────────────────────────────────────────────────
def _find_training_data() -> Path:
    path = Path.cwd()
    for _ in range(10):
        c = path / "actuator" / "data_collection" / "training_data"
        if c.is_dir():
            return c
        path = path.parent
    raise FileNotFoundError(f"Cannot find training_data (started from {Path.cwd()})")

data_dir = _find_training_data()
move_e_files = sorted(data_dir.glob("*move_E_*.csv"))
print(f"Found {len(move_e_files)} move_E_ files")

# ── Load and zero each trajectory at home ────────────────────────────────────
trajectories = []

for f in move_e_files:
    m = re.search(r'move_E_(\d+\.?\d*)deg', f.name)
    amplitude = float(m.group(1)) if m else 0.0

    df = pd.read_csv(f)
    opt_cols = ['optitrack_x', 'optitrack_y', 'optitrack_z']
    mot_cols = ['motor_1_pos', 'motor_2_pos', 'motor_3_pos', 'motor_4_pos']

    valid_opt = df[opt_cols].dropna()
    if valid_opt.empty:
        continue

    home_xyz = valid_opt.iloc[0].values
    df = df.copy()
    df['optitrack_x'] -= home_xyz[0]
    df['optitrack_y'] -= home_xyz[1]
    df['optitrack_z'] -= home_xyz[2]

    valid_mot = df[mot_cols].dropna()
    if valid_mot.empty:
        continue
    home_mot = valid_mot.iloc[0].values
    for i, col in enumerate(mot_cols):
        df[col] -= int(home_mot[i])

    df['east_mm']  = df['optitrack_x'] * 1000
    df['north_mm'] = df['optitrack_y'] * 1000
    df['up_mm']    = df['optitrack_z'] * 1000

    trajectories.append({
        'name':      f.name,
        'amplitude': amplitude,
        'df':        df,
    })

trajectories.sort(key=lambda x: x['amplitude'])
print(f"Loaded {len(trajectories)} valid trajectories")

# ── Colour map ────────────────────────────────────────────────────────────────
amps      = sorted(set(t['amplitude'] for t in trajectories))
n         = max(len(amps) - 1, 1)
amp_color = {a: f"hsl({int(280 * i / n)},80%,50%)" for i, a in enumerate(amps)}

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Top-down view  (North vs East)
# ─────────────────────────────────────────────────────────────────────────────
# %%
fig1 = go.Figure()

for t in trajectories:
    df  = t['df']
    amp = t['amplitude']
    fig1.add_trace(go.Scatter(
        x=df['east_mm'],
        y=df['north_mm'],
        mode='lines',
        name=f"{amp:.0f}°",
        line=dict(color=amp_color[amp], width=2),
        legendgroup=f"{amp:.0f}",
        showlegend=True,
    ))

fig1.add_trace(go.Scatter(
    x=[0], y=[0], mode='markers',
    marker=dict(size=12, color='lime', symbol='circle',
                line=dict(color='black', width=1)),
    name='Home', showlegend=True
))

fig1.update_layout(
    title='move_E_ Trajectories — Top-Down View<br>'
          '<sup>North vs East displacement from home</sup>',
    xaxis_title='East Displacement (mm)',
    yaxis_title='North Displacement (mm)',
    xaxis=dict(scaleanchor='y', scaleratio=1),
    legend_title='Amplitude',
    template='plotly_white',
    width=750, height=650,
)
fig1.show()

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Side view  (East vs Up)
# ─────────────────────────────────────────────────────────────────────────────
# %%
fig2 = go.Figure()

for t in trajectories:
    df  = t['df']
    amp = t['amplitude']
    fig2.add_trace(go.Scatter(
        x=df['east_mm'],
        y=df['up_mm'],
        mode='lines',
        name=f"{amp:.0f}°",
        line=dict(color=amp_color[amp], width=2),
        legendgroup=f"{amp:.0f}",
        showlegend=True,
    ))

fig2.add_trace(go.Scatter(
    x=[0], y=[0], mode='markers',
    marker=dict(size=12, color='lime', symbol='circle',
                line=dict(color='black', width=1)),
    name='Home', showlegend=True
))

fig2.update_layout(
    title='move_E_ Trajectories — Side View<br>'
          '<sup>Up displacement vs East displacement from home</sup>',
    xaxis_title='East Displacement (mm)',
    yaxis_title='Up Displacement (mm)',
    legend_title='Amplitude',
    template='plotly_white',
    width=750, height=550,
)
fig2.show()

# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — 3D trajectory
# ─────────────────────────────────────────────────────────────────────────────
# %%
fig3 = go.Figure()

for t in trajectories:
    df  = t['df']
    amp = t['amplitude']
    fig3.add_trace(go.Scatter3d(
        x=df['east_mm'],
        y=df['north_mm'],
        z=df['up_mm'],
        mode='lines',
        name=f"{amp:.0f}°",
        line=dict(color=amp_color[amp], width=4),
        legendgroup=f"{amp:.0f}",
        showlegend=True,
    ))

fig3.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0], mode='markers',
    marker=dict(size=6, color='lime', symbol='circle',
                line=dict(color='black', width=1)),
    name='Home', showlegend=True
))

fig3.update_layout(
    title='move_E_ Trajectories — 3D View',
    scene=dict(
        xaxis_title='East (mm)',
        yaxis_title='North (mm)',
        zaxis_title='Up (mm)',
        aspectmode='data',
    ),
    legend_title='Amplitude',
    template='plotly_white',
    width=850, height=700,
)
fig3.show()

# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Motor 2 & 4 positions vs East displacement
# ─────────────────────────────────────────────────────────────────────────────
# %%
from plotly.subplots import make_subplots

fig4 = make_subplots(
    rows=1, cols=2,
    subplot_titles=['Motor 2 vs East', 'Motor 4 vs East'],
    shared_xaxes=False,
)

mot_cols = ['motor_2_pos', 'motor_4_pos']
positions = [(1, 1), (1, 2)]

for mi, (mot_col, pos) in enumerate(zip(mot_cols, positions)):
    row, col = pos
    for ti, t in enumerate(trajectories):
        df  = t['df']
        amp = t['amplitude']
        fig4.add_trace(go.Scatter(
            x=df['east_mm'],
            y=df[mot_col],
            mode='lines',
            name=f"{amp:.0f}°",
            line=dict(color=amp_color[amp], width=1.5),
            legendgroup=f"{amp:.0f}",
            showlegend=(mi == 0),
        ), row=row, col=col)

fig4.update_layout(
    title='Motor Positions vs East Displacement<br>'
          '<sup>Pulses from home — all move_E_ trajectories (Motor 2 = East pull, Motor 4 = West release)</sup>',
    template='plotly_white',
    legend_title='Amplitude',
    width=950, height=500,
)
for r, c in positions:
    fig4.update_xaxes(title_text='East Displacement (mm)', row=r, col=c)
    fig4.update_yaxes(title_text='Motor Position (pulses from home)', row=r, col=c)

fig4.show()

# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — Motor 2 & 4 positions vs Up displacement
# ─────────────────────────────────────────────────────────────────────────────
# %%
fig5 = make_subplots(
    rows=1, cols=2,
    subplot_titles=['Motor 2 vs Up', 'Motor 4 vs Up'],
    shared_xaxes=False,
)

mot_cols_up = ['motor_2_pos', 'motor_4_pos']
positions_up = [(1, 1), (1, 2)]

for mi, (mot_col, pos) in enumerate(zip(mot_cols_up, positions_up)):
    row, col = pos
    for ti, t in enumerate(trajectories):
        df  = t['df']
        amp = t['amplitude']
        fig5.add_trace(go.Scatter(
            x=df['up_mm'],
            y=df[mot_col],
            mode='lines',
            name=f"{amp:.0f}°",
            line=dict(color=amp_color[amp], width=1.5),
            legendgroup=f"{amp:.0f}",
            showlegend=(mi == 0),
        ), row=row, col=col)

fig5.update_layout(
    title='Motor Positions vs Up Displacement<br>'
          '<sup>Pulses from home — all move_E_ trajectories (Motor 2 = East pull, Motor 4 = West release)</sup>',
    template='plotly_white',
    legend_title='Amplitude',
    width=950, height=500,
)
for r, c in positions_up:
    fig5.update_xaxes(title_text='Up Displacement (mm)', row=r, col=c)
    fig5.update_yaxes(title_text='Motor Position (pulses from home)', row=r, col=c)

fig5.show()
