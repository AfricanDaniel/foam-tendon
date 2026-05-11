# %%
"""
Visualise ALL training trajectories (all 8 directions) to show the dome shape.

Two views:
  1. Top-Down (East vs North) — star/rose pattern; equal scale
  2. 3D       (East, North, Up) — dome surface emerges from data

Trajectories are coloured by direction.
All coordinates are zeroed at the first sample of each file (home position).
Units: mm.
"""

import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path

pio.renderers.default = "browser"

# ── Direction colour map ──────────────────────────────────────────────────────
DIR_COLORS = {
    'N':  '#e41a1c',
    'NE': '#ff7f00',
    'E':  '#daa520',
    'SE': '#4daf4a',
    'S':  '#377eb8',
    'SW': '#984ea3',
    'W':  '#a65628',
    'NW': '#f781bf',
}

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
all_move_files = sorted(f for f in data_dir.glob("*.csv") if "go_home" not in f.name)
print(f"Found {len(all_move_files)} move files")

# ── Load all trajectories ────────────────────────────────────────────────────
trajectories = []
OPT_COLS = ['optitrack_x', 'optitrack_y', 'optitrack_z']

for f in all_move_files:
    m = re.search(r'move_([A-Z]+)_(\d+\.?\d*)deg', f.name)
    if not m:
        continue
    direction = m.group(1)
    amplitude = float(m.group(2))

    df = pd.read_csv(f)
    valid = df[OPT_COLS].dropna()
    if valid.empty:
        continue

    home = valid.iloc[0].values
    df = df.copy()
    df['east_mm']  = (df['optitrack_x'] - home[0]) * 1000
    df['north_mm'] = (df['optitrack_y'] - home[1]) * 1000
    df['up_mm']    = (df['optitrack_z'] - home[2]) * 1000

    trajectories.append({
        'direction': direction,
        'amplitude': amplitude,
        'df':        df,
    })

print(f"Loaded {len(trajectories)} valid trajectories")

# Group by direction for legend deduplication
from collections import defaultdict
by_dir = defaultdict(list)
for t in trajectories:
    by_dir[t['direction']].append(t)

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Top-Down View
# ─────────────────────────────────────────────────────────────────────────────
# %%
fig1 = go.Figure()

for direction, group in sorted(by_dir.items()):
    color = DIR_COLORS.get(direction, '#888888')
    for i, t in enumerate(group):
        df = t['df']
        fig1.add_trace(go.Scatter(
            x=df['east_mm'],
            y=df['north_mm'],
            mode='lines',
            line=dict(color=color, width=1),
            opacity=0.6,
            name=direction,
            legendgroup=direction,
            showlegend=(i == 0),
        ))

fig1.add_trace(go.Scatter(
    x=[0], y=[0], mode='markers',
    marker=dict(size=14, color='lime', symbol='circle',
                line=dict(color='black', width=1.5)),
    name='Home', showlegend=True,
))

fig1.update_layout(
    title='All Trajectories — Top-Down View<br>'
          '<sup>East vs North displacement from home — all 8 directions</sup>',
    xaxis_title='East Displacement (mm)',
    yaxis_title='North Displacement (mm)',
    xaxis=dict(scaleanchor='y', scaleratio=1, dtick=10),
    yaxis=dict(dtick=10),
    legend_title='Direction',
    template='plotly_white',
    width=800, height=750,
)
fig1.show()

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — 3D Dome View
# ─────────────────────────────────────────────────────────────────────────────
# %%
fig2 = go.Figure()

for direction, group in sorted(by_dir.items()):
    color = DIR_COLORS.get(direction, '#888888')
    for i, t in enumerate(group):
        df = t['df']
        fig2.add_trace(go.Scatter3d(
            x=df['east_mm'],
            y=df['north_mm'],
            z=df['up_mm'],
            mode='lines',
            line=dict(color=color, width=3),
            opacity=0.7,
            name=direction,
            legendgroup=direction,
            showlegend=(i == 0),
        ))

fig2.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0], mode='markers',
    marker=dict(size=7, color='lime', symbol='circle',
                line=dict(color='black', width=1)),
    name='Home', showlegend=True,
))

fig2.update_layout(
    title='All Trajectories — 3D Dome View<br>'
          '<sup>East, North, Up — all 8 directions form the dome envelope</sup>',
    scene=dict(
        xaxis_title='East (mm)',
        yaxis_title='North (mm)',
        zaxis_title='Up (mm)',
        aspectmode='data',
    ),
    legend_title='Direction',
    template='plotly_white',
    width=900, height=750,
)
fig2.show()
