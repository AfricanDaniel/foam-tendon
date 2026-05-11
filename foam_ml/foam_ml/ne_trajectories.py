# %%
"""
Visualise all move_NE_ trajectories from training data.

Three views:
  1. Top-down   (North vs East)
  2. Side view  (NE diagonal magnitude vs Up)
  3. 3D         (East, North, Up)

Plus motor panels (all four active during diagonal moves):
  4. All motors vs NE diagonal magnitude
  5. All motors vs Up displacement

NE is the positive-X, positive-Y direction in the robot frame.
  diag_mm = sqrt(east_mm^2 + north_mm^2)  — horizontal distance from home

Motor behaviour during a NE move:
  Motor 1 (North pull):  CW → position decreases
  Motor 2 (East pull):   CW → position decreases
  Motor 3 (South release): CW → position decreases
  Motor 4 (West release):  CW → position decreases

All coordinates are zeroed at the first row of each file (home position).
All units: mm for position, pulses for motor counts.
"""

import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
from plotly.subplots import make_subplots

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
move_ne_files = sorted(data_dir.glob("*move_NE_*.csv"))
print(f"Found {len(move_ne_files)} move_NE_ files")

# ── Load and zero each trajectory at home ────────────────────────────────────
trajectories = []

for f in move_ne_files:
    m = re.search(r'move_NE_(\d+\.?\d*)deg', f.name)
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
    df['diag_mm']  = np.sqrt(df['east_mm']**2 + df['north_mm']**2)

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
    title='move_NE_ Trajectories — Top-Down View<br>'
          '<sup>North vs East displacement from home</sup>',
    xaxis_title='East Displacement (mm)',
    yaxis_title='North Displacement (mm)',
    xaxis=dict(scaleanchor='y', scaleratio=1, dtick=10),
    yaxis=dict(dtick=10),
    legend_title='Amplitude',
    template='plotly_white',
    width=750, height=650,
)
fig1.show()

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Side view  (NE diagonal magnitude vs Up)
# ─────────────────────────────────────────────────────────────────────────────
# %%
fig2 = go.Figure()

for t in trajectories:
    df  = t['df']
    amp = t['amplitude']
    fig2.add_trace(go.Scatter(
        x=df['diag_mm'],
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
    title='move_NE_ Trajectories — Side View<br>'
          '<sup>Up displacement vs NE diagonal magnitude from home</sup>',
    xaxis_title='NE Diagonal Magnitude (mm)',
    yaxis_title='Up Displacement (mm)',
    xaxis=dict(dtick=10),
    yaxis=dict(dtick=10),
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
    title='move_NE_ Trajectories — 3D View',
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
# Plot 4 — All motor positions vs NE diagonal magnitude
# ─────────────────────────────────────────────────────────────────────────────
# %%
fig4 = make_subplots(
    rows=2, cols=2,
    subplot_titles=['Motor 1 (North pull) vs NE', 'Motor 2 (East pull) vs NE',
                    'Motor 3 (South release) vs NE', 'Motor 4 (West release) vs NE'],
    shared_xaxes=False,
)

mot_cols = ['motor_1_pos', 'motor_2_pos', 'motor_3_pos', 'motor_4_pos']
positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

for mi, (mot_col, pos) in enumerate(zip(mot_cols, positions)):
    row, col = pos
    for ti, t in enumerate(trajectories):
        df  = t['df']
        amp = t['amplitude']
        fig4.add_trace(go.Scatter(
            x=df['diag_mm'],
            y=df[mot_col],
            mode='lines',
            name=f"{amp:.0f}°",
            line=dict(color=amp_color[amp], width=1.5),
            legendgroup=f"{amp:.0f}",
            showlegend=(mi == 0),
        ), row=row, col=col)

fig4.update_layout(
    title='Motor Positions vs NE Diagonal Magnitude<br>'
          '<sup>Pulses from home — all move_NE_ trajectories</sup>',
    template='plotly_white',
    legend_title='Amplitude',
    width=950, height=750,
)
for r, c in positions:
    fig4.update_xaxes(title_text='NE Diagonal Magnitude (mm)', row=r, col=c)
    fig4.update_yaxes(title_text='Motor Position (pulses from home)', row=r, col=c)

fig4.show()

# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — All motor positions vs Up displacement
# ─────────────────────────────────────────────────────────────────────────────
# %%
fig5 = make_subplots(
    rows=2, cols=2,
    subplot_titles=['Motor 1 (North pull) vs Up', 'Motor 2 (East pull) vs Up',
                    'Motor 3 (South release) vs Up', 'Motor 4 (West release) vs Up'],
    shared_xaxes=False,
)

for mi, (mot_col, pos) in enumerate(zip(mot_cols, positions)):
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
          '<sup>Pulses from home — all move_NE_ trajectories</sup>',
    template='plotly_white',
    legend_title='Amplitude',
    width=950, height=750,
)
for r, c in positions:
    fig5.update_xaxes(title_text='Up Displacement (mm)', row=r, col=c)
    fig5.update_yaxes(title_text='Motor Position (pulses from home)', row=r, col=c)

fig5.show()
