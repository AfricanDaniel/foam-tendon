# %%
"""
Dome height model — trained on ALL 8-direction training data.

The foam tip traces a dome in 3D space. This file fits MLP models that capture
the full dome geometry and the motor commands needed to reach any point.

Three models:
  A. (east, north) → up_mm
       "Given a 2D horizontal target, predict the foam height."
       Visualised as a predicted dome surface over the data.

  B. (east, north) → motors 1-4
       "Given a 2D horizontal target, predict all motor commands."

  C. (east, north, up) → motors 1-4
       "Full 3D position → motor commands (inverse kinematics)."
       Most useful for closed-loop control.
"""

import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error

pio.renderers.default = "browser"

# ── Find and load ALL move data ───────────────────────────────────────────────
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

NEEDED_COLS = ['optitrack_x', 'optitrack_y', 'optitrack_z',
               'motor_1_pos', 'motor_2_pos', 'motor_3_pos', 'motor_4_pos']

east_all, north_all, up_all = [], [], []
m1_all, m2_all, m3_all, m4_all = [], [], [], []

for f in all_move_files:
    df = pd.read_csv(f)
    valid = df[NEEDED_COLS].dropna()
    if valid.empty:
        continue
    home = valid.iloc[0]
    east_all.extend((valid['optitrack_x'] - home['optitrack_x']) * 1000)
    north_all.extend((valid['optitrack_y'] - home['optitrack_y']) * 1000)
    up_all.extend((valid['optitrack_z']   - home['optitrack_z']) * 1000)
    m1_all.extend(valid['motor_1_pos'] - int(home['motor_1_pos']))
    m2_all.extend(valid['motor_2_pos'] - int(home['motor_2_pos']))
    m3_all.extend(valid['motor_3_pos'] - int(home['motor_3_pos']))
    m4_all.extend(valid['motor_4_pos'] - int(home['motor_4_pos']))

east  = np.array(east_all,  dtype=float)
north = np.array(north_all, dtype=float)
up    = np.array(up_all,    dtype=float)
m1    = np.array(m1_all,    dtype=float)
m2    = np.array(m2_all,    dtype=float)
m3    = np.array(m3_all,    dtype=float)
m4    = np.array(m4_all,    dtype=float)

print(f"Loaded {len(east):,} data points from {len(all_move_files)} files")
print(f"East  range: {east.min():.1f} → {east.max():.1f} mm")
print(f"North range: {north.min():.1f} → {north.max():.1f} mm")
print(f"Up    range: {up.min():.1f} → {up.max():.1f} mm")

# ═════════════════════════════════════════════════════════════════════════════
# Section A: (east, north) → up
# ═════════════════════════════════════════════════════════════════════════════
# %%
X_en = np.column_stack([east, north])

Xen_tr, Xen_te, yu_tr, yu_te = train_test_split(X_en, up, test_size=0.2, random_state=42)

sc_en = StandardScaler().fit(Xen_tr)
sc_yu = StandardScaler().fit(yu_tr.reshape(-1, 1))
nn_a = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu',
                    max_iter=1000, random_state=42).fit(
    sc_en.transform(Xen_tr), sc_yu.transform(yu_tr.reshape(-1, 1)).ravel())
nn_a_pred = sc_yu.inverse_transform(
    nn_a.predict(sc_en.transform(Xen_te)).reshape(-1, 1)).ravel()
nn_a_r2   = r2_score(yu_te, nn_a_pred)
nn_a_rmse = np.sqrt(mean_squared_error(yu_te, nn_a_pred))

print(f"\nSection A: (east, north) → up")
print(f"  MLP  R²={nn_a_r2:.4f}  RMSE={nn_a_rmse:.2f} mm")

# ── Plot A1: Predicted dome surface + actual data ────────────────────────────
# %%
n_grid = 70
e_lin = np.linspace(east.min(),  east.max(),  n_grid)
n_lin = np.linspace(north.min(), north.max(), n_grid)
eg, ng = np.meshgrid(e_lin, n_lin)
grid_pts = np.column_stack([eg.ravel(), ng.ravel()])

up_surf = sc_yu.inverse_transform(
    nn_a.predict(sc_en.transform(grid_pts)).reshape(-1, 1)).ravel().reshape(eg.shape)

fig_a1 = go.Figure()
fig_a1.add_trace(go.Surface(
    x=e_lin, y=n_lin, z=up_surf,
    colorscale='Viridis', opacity=0.55,
    name='MLP predicted dome',
    showscale=True,
    colorbar=dict(title='Predicted Up (mm)', len=0.6),
))
fig_a1.add_trace(go.Scatter3d(
    x=east, y=north, z=up,
    mode='markers',
    marker=dict(size=1.5, color=up, colorscale='Reds', opacity=0.25),
    name='Actual data',
))
fig_a1.update_layout(
    title='Dome Height Model — Predicted Surface vs Actual Data<br>'
          f'<sup>MLP R²={nn_a_r2:.3f}  RMSE={nn_a_rmse:.1f} mm  |  trained on all 8 directions</sup>',
    scene=dict(
        xaxis_title='East (mm)',
        yaxis_title='North (mm)',
        zaxis_title='Up / Height (mm)',
        aspectmode='data',
    ),
    template='plotly_white',
    width=950, height=750,
)
fig_a1.show()

# ── Plot A2: Predicted vs Actual scatter ─────────────────────────────────────
# %%
mn, mx = yu_te.min(), yu_te.max()
fig_a2 = go.Figure()
fig_a2.add_trace(go.Scatter(
    x=yu_te, y=nn_a_pred, mode='markers',
    marker=dict(size=3, color='darkorange', opacity=0.4),
    name=f'MLP  R²={nn_a_r2:.3f}  RMSE={nn_a_rmse:.1f} mm'))
fig_a2.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode='lines',
    line=dict(color='black', dash='dot', width=1.5), name='Perfect'))
fig_a2.update_layout(
    title='(East, North) → Up: Predicted vs Actual — Test Set',
    xaxis_title='Actual Up (mm)', yaxis_title='Predicted Up (mm)',
    xaxis=dict(dtick=10), yaxis=dict(dtick=10),
    template='plotly_white', legend=dict(x=0.02, y=0.98),
    width=750, height=600,
)
fig_a2.show()

# ═════════════════════════════════════════════════════════════════════════════
# Section B: (east, north) → motors 1-4
# ═════════════════════════════════════════════════════════════════════════════
# %%
motors_all = np.column_stack([m1, m2, m3, m4])
Xen_tr_b, Xen_te_b, ym_tr_b, ym_te_b = train_test_split(
    X_en, motors_all, test_size=0.2, random_state=42)

sc_en_b = StandardScaler().fit(Xen_tr_b)
sc_mb   = StandardScaler().fit(ym_tr_b)
nn_b = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu',
                    max_iter=1000, random_state=42).fit(
    sc_en_b.transform(Xen_tr_b), sc_mb.transform(ym_tr_b))
nn_b_pred = sc_mb.inverse_transform(nn_b.predict(sc_en_b.transform(Xen_te_b)))
nn_b_r2   = [r2_score(ym_te_b[:, i], nn_b_pred[:, i]) for i in range(4)]
nn_b_rmse = [np.sqrt(mean_squared_error(ym_te_b[:, i], nn_b_pred[:, i])) for i in range(4)]

motor_labels = ['Motor 1 (North)', 'Motor 2 (East)', 'Motor 3 (South)', 'Motor 4 (West)']
print(f"\nSection B: (east, north) → motors 1-4")
for i, lbl in enumerate(motor_labels):
    print(f"  {lbl}  MLP R²={nn_b_r2[i]:.4f}  RMSE={nn_b_rmse[i]:.1f} p")

fig_b = make_subplots(rows=2, cols=2,
    subplot_titles=[f'{l}<br><sup>MLP R²={nn_b_r2[i]:.3f}  RMSE={nn_b_rmse[i]:.0f} p</sup>'
                    for i, l in enumerate(motor_labels)])
for idx in range(4):
    r, c = divmod(idx, 2)
    row, col = r + 1, c + 1
    mn_m, mx_m = ym_te_b[:, idx].min(), ym_te_b[:, idx].max()
    fig_b.add_trace(go.Scatter(
        x=ym_te_b[:, idx], y=nn_b_pred[:, idx], mode='markers',
        marker=dict(size=2, color='darkorange', opacity=0.3),
        name='MLP', showlegend=(idx == 0)), row=row, col=col)
    fig_b.add_trace(go.Scatter(
        x=[mn_m, mx_m], y=[mn_m, mx_m], mode='lines',
        line=dict(color='black', dash='dot', width=1),
        name='Perfect', showlegend=(idx == 0)), row=row, col=col)
    fig_b.update_xaxes(title_text='Actual (pulses)', row=row, col=col)
    fig_b.update_yaxes(title_text='Predicted (pulses)', row=row, col=col)

fig_b.update_layout(
    title='(East, North) → Motor Commands: Predicted vs Actual — Test Set',
    template='plotly_white', width=1000, height=800,
)
fig_b.show()

# ═════════════════════════════════════════════════════════════════════════════
# Section C: (east, north, up) → motors 1-4  [inverse kinematics]
# ═════════════════════════════════════════════════════════════════════════════
# %%
X_enu = np.column_stack([east, north, up])
Xenu_tr, Xenu_te, ym_tr_c, ym_te_c = train_test_split(
    X_enu, motors_all, test_size=0.2, random_state=42)

sc_enu = StandardScaler().fit(Xenu_tr)
sc_mc  = StandardScaler().fit(ym_tr_c)
nn_c = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu',
                    max_iter=1000, random_state=42).fit(
    sc_enu.transform(Xenu_tr), sc_mc.transform(ym_tr_c))
nn_c_pred = sc_mc.inverse_transform(nn_c.predict(sc_enu.transform(Xenu_te)))
nn_c_r2   = [r2_score(ym_te_c[:, i], nn_c_pred[:, i]) for i in range(4)]
nn_c_rmse = [np.sqrt(mean_squared_error(ym_te_c[:, i], nn_c_pred[:, i])) for i in range(4)]

print(f"\nSection C: (east, north, up) → motors 1-4  [inverse kinematics]")
for i, lbl in enumerate(motor_labels):
    print(f"  {lbl}  MLP R²={nn_c_r2[i]:.4f}  RMSE={nn_c_rmse[i]:.1f} p")

fig_c = make_subplots(rows=2, cols=2,
    subplot_titles=[f'{l}<br><sup>MLP R²={nn_c_r2[i]:.3f}  RMSE={nn_c_rmse[i]:.0f} p</sup>'
                    for i, l in enumerate(motor_labels)])
for idx in range(4):
    r, c = divmod(idx, 2)
    row, col = r + 1, c + 1
    mn_m, mx_m = ym_te_c[:, idx].min(), ym_te_c[:, idx].max()
    fig_c.add_trace(go.Scatter(
        x=ym_te_c[:, idx], y=nn_c_pred[:, idx], mode='markers',
        marker=dict(size=2, color='darkorange', opacity=0.3),
        name='MLP', showlegend=(idx == 0)), row=row, col=col)
    fig_c.add_trace(go.Scatter(
        x=[mn_m, mx_m], y=[mn_m, mx_m], mode='lines',
        line=dict(color='black', dash='dot', width=1),
        name='Perfect', showlegend=(idx == 0)), row=row, col=col)
    fig_c.update_xaxes(title_text='Actual (pulses)', row=row, col=col)
    fig_c.update_yaxes(title_text='Predicted (pulses)', row=row, col=col)

fig_c.update_layout(
    title='(East, North, Up) → Motor Commands: Predicted vs Actual — Test Set<br>'
          '<sup>Inverse kinematics: given 3D tip position, predict all motor commands</sup>',
    template='plotly_white', width=1000, height=800,
)
fig_c.show()
