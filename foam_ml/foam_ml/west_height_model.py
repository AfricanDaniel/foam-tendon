# %%
"""
Model: West displacement ↔ Up displacement (foam height) ↔ Motor positions

Trains two models on all move_W_ trajectory data:
  1. Linear Regression   (baseline)
  2. Neural Network MLP  (sklearn)

West displacement = −optitrack_x × 1000 mm  (positive = further west).
Active motors for West moves: Motor 2 (East, releases CCW) and Motor 4 (West, pulls CCW).
Both increase (go CCW / positive pulses) during a west move.

Sections:
  A. West (mm) → Up (mm)               "given how far west, predict height"
  B. Up (mm)   → West (mm)             "given height, predict west position"
  C. West (mm) → Motor 2 & 4 (pulses) "given west, predict motor commands"
  D. Up (mm)   → Motor 2 & 4 (pulses) "given height, predict motor commands"
"""

import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
from plotly.subplots import make_subplots

pio.renderers.default = "browser"

# ── Find and load data ───────────────────────────────────────────────────────
def _find_training_data() -> Path:
    path = Path.cwd()
    for _ in range(10):
        c = path / "actuator" / "data_collection" / "training_data"
        if c.is_dir():
            return c
        path = path.parent
    raise FileNotFoundError(f"Cannot find training_data (started from {Path.cwd()})")

data_dir = _find_training_data()
move_w_files = sorted(data_dir.glob("*move_W_*.csv"))

NEEDED_COLS = ['optitrack_x', 'optitrack_z', 'motor_2_pos', 'motor_4_pos']
west_all, up_all, m2_all, m4_all = [], [], [], []

for f in move_w_files:
    df = pd.read_csv(f)
    valid = df[NEEDED_COLS].dropna()
    if valid.empty:
        continue
    home = valid.iloc[0]
    # Negate optitrack_x so west displacement is positive
    west_all.extend(-(valid['optitrack_x'] - home['optitrack_x']) * 1000)  # mm
    up_all.extend(   (valid['optitrack_z'] - home['optitrack_z']) * 1000)  # mm
    m2_all.extend(   valid['motor_2_pos'] - int(home['motor_2_pos']))       # pulses
    m4_all.extend(   valid['motor_4_pos'] - int(home['motor_4_pos']))       # pulses

west = np.array(west_all)
up   = np.array(up_all)
m2   = np.array(m2_all, dtype=float)
m4   = np.array(m4_all, dtype=float)
print(f"Loaded {len(west):,} data points from {len(move_w_files)} files")
print(f"West    range: {west.min():.1f} mm  →  {west.max():.1f} mm")
print(f"Up      range: {up.min():.1f} mm  →  {up.max():.1f} mm")
print(f"Motor 2 range: {m2.min():.0f}  →  {m2.max():.0f} pulses")
print(f"Motor 4 range: {m4.min():.0f}  →  {m4.max():.0f} pulses")

# ── Train/test split ─────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error

X_w = west.reshape(-1, 1)
X_u = up.reshape(-1, 1)

# Direction A: West → Up
Xw_tr, Xw_te, yu_tr, yu_te = train_test_split(X_w, up, test_size=0.2, random_state=42)

# Direction B: Up → West
Xu_tr, Xu_te, yw_tr, yw_te = train_test_split(X_u, west, test_size=0.2, random_state=42)

# ─────────────────────────────────────────────────────────────────────────────
# Direction A: West → Up
# ─────────────────────────────────────────────────────────────────────────────
# %%

lr_a = LinearRegression().fit(Xw_tr, yu_tr)
lr_a_pred = lr_a.predict(Xw_te)
lr_a_r2   = r2_score(yu_te, lr_a_pred)
lr_a_rmse = np.sqrt(mean_squared_error(yu_te, lr_a_pred))

sc_w  = StandardScaler().fit(Xw_tr)
sc_yu = StandardScaler().fit(yu_tr.reshape(-1, 1))
nn_a = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                    max_iter=500, random_state=42).fit(
    sc_w.transform(Xw_tr),
    sc_yu.transform(yu_tr.reshape(-1, 1)).ravel()
)
nn_a_pred_sc = nn_a.predict(sc_w.transform(Xw_te))
nn_a_pred    = sc_yu.inverse_transform(nn_a_pred_sc.reshape(-1, 1)).ravel()
nn_a_r2      = r2_score(yu_te, nn_a_pred)
nn_a_rmse    = np.sqrt(mean_squared_error(yu_te, nn_a_pred))

print("\nDirection A: West → Up")
print(f"  Linear Regression  R²={lr_a_r2:.4f}  RMSE={lr_a_rmse:.2f} mm")
print(f"  Neural Network     R²={nn_a_r2:.4f}  RMSE={nn_a_rmse:.2f} mm")

w_grid = np.linspace(west.min(), west.max(), 300).reshape(-1, 1)
lr_curve_a = lr_a.predict(w_grid)
nn_curve_a = sc_yu.inverse_transform(
    nn_a.predict(sc_w.transform(w_grid)).reshape(-1, 1)
).ravel()

fig_a = go.Figure()
fig_a.add_trace(go.Scatter(
    x=west, y=up, mode='markers',
    marker=dict(size=2, color='lightsteelblue', opacity=0.3),
    name='Data'
))
fig_a.add_trace(go.Scatter(
    x=w_grid.ravel(), y=lr_curve_a, mode='lines',
    line=dict(color='crimson', width=2, dash='dash'),
    name=f'Linear  R²={lr_a_r2:.3f}  RMSE={lr_a_rmse:.1f} mm'
))
fig_a.add_trace(go.Scatter(
    x=w_grid.ravel(), y=nn_curve_a, mode='lines',
    line=dict(color='darkorange', width=2),
    name=f'MLP     R²={nn_a_r2:.3f}  RMSE={nn_a_rmse:.1f} mm'
))
fig_a.update_layout(
    title='West → Up: Model Fit<br>'
          '<sup>Given west displacement, predict foam height (Up)</sup>',
    xaxis_title='West Displacement (mm)',
    yaxis_title='Up Displacement / Height (mm)',
    template='plotly_white',
    legend=dict(x=0.02, y=0.98),
    width=800, height=550,
)
fig_a.show()

# ─────────────────────────────────────────────────────────────────────────────
# Direction B: Up → West
# ─────────────────────────────────────────────────────────────────────────────
# %%

lr_b = LinearRegression().fit(Xu_tr, yw_tr)
lr_b_pred = lr_b.predict(Xu_te)
lr_b_r2   = r2_score(yw_te, lr_b_pred)
lr_b_rmse = np.sqrt(mean_squared_error(yw_te, lr_b_pred))

sc_u  = StandardScaler().fit(Xu_tr)
sc_yw = StandardScaler().fit(yw_tr.reshape(-1, 1))
nn_b = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                    max_iter=500, random_state=42).fit(
    sc_u.transform(Xu_tr),
    sc_yw.transform(yw_tr.reshape(-1, 1)).ravel()
)
nn_b_pred_sc = nn_b.predict(sc_u.transform(Xu_te))
nn_b_pred    = sc_yw.inverse_transform(nn_b_pred_sc.reshape(-1, 1)).ravel()
nn_b_r2      = r2_score(yw_te, nn_b_pred)
nn_b_rmse    = np.sqrt(mean_squared_error(yw_te, nn_b_pred))

print("\nDirection B: Up → West")
print(f"  Linear Regression  R²={lr_b_r2:.4f}  RMSE={lr_b_rmse:.2f} mm")
print(f"  Neural Network     R²={nn_b_r2:.4f}  RMSE={nn_b_rmse:.2f} mm")

u_grid = np.linspace(up.min(), up.max(), 300).reshape(-1, 1)
lr_curve_b = lr_b.predict(u_grid)
nn_curve_b = sc_yw.inverse_transform(
    nn_b.predict(sc_u.transform(u_grid)).reshape(-1, 1)
).ravel()

fig_b = go.Figure()
fig_b.add_trace(go.Scatter(
    x=up, y=west, mode='markers',
    marker=dict(size=2, color='lightsteelblue', opacity=0.3),
    name='Data'
))
fig_b.add_trace(go.Scatter(
    x=u_grid.ravel(), y=lr_curve_b, mode='lines',
    line=dict(color='crimson', width=2, dash='dash'),
    name=f'Linear  R²={lr_b_r2:.3f}  RMSE={lr_b_rmse:.1f} mm'
))
fig_b.add_trace(go.Scatter(
    x=u_grid.ravel(), y=nn_curve_b, mode='lines',
    line=dict(color='darkorange', width=2),
    name=f'MLP     R²={nn_b_r2:.3f}  RMSE={nn_b_rmse:.1f} mm'
))
fig_b.update_layout(
    title='Up → West: Model Fit<br>'
          '<sup>Given foam height (Up), predict west displacement</sup>',
    xaxis_title='Up Displacement / Height (mm)',
    yaxis_title='West Displacement (mm)',
    template='plotly_white',
    legend=dict(x=0.02, y=0.98),
    width=800, height=550,
)
fig_b.show()

# ─────────────────────────────────────────────────────────────────────────────
# Residual comparison
# ─────────────────────────────────────────────────────────────────────────────
# %%
fig_r = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        'West → Up: Predicted vs Actual',
        'Up → West: Predicted vs Actual',
    ]
)

for pred, label, color in [
    (lr_a_pred, f'Linear R²={lr_a_r2:.3f}', 'crimson'),
    (nn_a_pred, f'MLP    R²={nn_a_r2:.3f}', 'darkorange'),
]:
    fig_r.add_trace(go.Scatter(
        x=yu_te, y=pred, mode='markers',
        marker=dict(size=3, color=color, opacity=0.5),
        name=label, legendgroup=label,
    ), row=1, col=1)

for pred, label, color in [
    (lr_b_pred, f'Linear R²={lr_b_r2:.3f}', 'crimson'),
    (nn_b_pred, f'MLP    R²={nn_b_r2:.3f}', 'darkorange'),
]:
    fig_r.add_trace(go.Scatter(
        x=yw_te, y=pred, mode='markers',
        marker=dict(size=3, color=color, opacity=0.5),
        name=label, legendgroup=label, showlegend=False,
    ), row=1, col=2)

for col, (mn, mx) in enumerate(
    [(yu_te.min(), yu_te.max()), (yw_te.min(), yw_te.max())], start=1
):
    fig_r.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx], mode='lines',
        line=dict(color='black', dash='dot', width=1),
        name='Perfect', showlegend=(col == 1),
    ), row=1, col=col)

fig_r.update_xaxes(title_text='Actual Up (mm)',     row=1, col=1)
fig_r.update_yaxes(title_text='Predicted Up (mm)',  row=1, col=1)
fig_r.update_xaxes(title_text='Actual West (mm)',    row=1, col=2)
fig_r.update_yaxes(title_text='Predicted West (mm)', row=1, col=2)

fig_r.update_layout(
    title='Predicted vs Actual — Test Set',
    template='plotly_white',
    width=1000, height=500,
)
fig_r.show()

# ─────────────────────────────────────────────────────────────────────────────
# Direction C: West → Motor 2 & 4
# ─────────────────────────────────────────────────────────────────────────────
# %%

motors_24 = np.column_stack([m2, m4])

Xw_tr_c, Xw_te_c, ym_tr_c, ym_te_c = train_test_split(
    X_w, motors_24, test_size=0.2, random_state=42
)

lr_c = LinearRegression().fit(Xw_tr_c, ym_tr_c)
lr_c_pred = lr_c.predict(Xw_te_c)
lr_c_r2   = [r2_score(ym_te_c[:, i], lr_c_pred[:, i]) for i in range(2)]
lr_c_rmse = [np.sqrt(mean_squared_error(ym_te_c[:, i], lr_c_pred[:, i])) for i in range(2)]

sc_w_c = StandardScaler().fit(Xw_tr_c)
sc_mc  = StandardScaler().fit(ym_tr_c)
nn_c = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                    max_iter=500, random_state=42).fit(
    sc_w_c.transform(Xw_tr_c),
    sc_mc.transform(ym_tr_c)
)
nn_c_pred = sc_mc.inverse_transform(nn_c.predict(sc_w_c.transform(Xw_te_c)))
nn_c_r2   = [r2_score(ym_te_c[:, i], nn_c_pred[:, i]) for i in range(2)]
nn_c_rmse = [np.sqrt(mean_squared_error(ym_te_c[:, i], nn_c_pred[:, i])) for i in range(2)]

print("\nDirection C: West → Motor 2 & 4")
for i, lbl in enumerate(['Motor 2', 'Motor 4']):
    print(f"  {lbl}  Linear R²={lr_c_r2[i]:.4f} RMSE={lr_c_rmse[i]:.1f} pulses"
          f"   MLP R²={nn_c_r2[i]:.4f} RMSE={nn_c_rmse[i]:.1f} pulses")

lr_curves_c = lr_c.predict(w_grid)
nn_curves_c = sc_mc.inverse_transform(nn_c.predict(sc_w_c.transform(w_grid)))

fig_c = make_subplots(rows=1, cols=2,
                      subplot_titles=['West → Motor 2', 'West → Motor 4'])

for col_idx, (motor_data, lbl) in enumerate([(m2, 'Motor 2'), (m4, 'Motor 4')], start=1):
    fig_c.add_trace(go.Scatter(
        x=west, y=motor_data, mode='markers',
        marker=dict(size=2, color='lightsteelblue', opacity=0.3),
        name='Data', showlegend=(col_idx == 1),
    ), row=1, col=col_idx)
    fig_c.add_trace(go.Scatter(
        x=w_grid.ravel(), y=lr_curves_c[:, col_idx - 1], mode='lines',
        line=dict(color='crimson', width=2, dash='dash'),
        name=f'Linear R²={lr_c_r2[col_idx-1]:.3f} RMSE={lr_c_rmse[col_idx-1]:.0f} p',
        showlegend=(col_idx == 1),
    ), row=1, col=col_idx)
    fig_c.add_trace(go.Scatter(
        x=w_grid.ravel(), y=nn_curves_c[:, col_idx - 1], mode='lines',
        line=dict(color='darkorange', width=2),
        name=f'MLP    R²={nn_c_r2[col_idx-1]:.3f} RMSE={nn_c_rmse[col_idx-1]:.0f} p',
        showlegend=(col_idx == 1),
    ), row=1, col=col_idx)
    fig_c.update_xaxes(title_text='West Displacement (mm)', row=1, col=col_idx)
    fig_c.update_yaxes(title_text=f'{lbl} Position (pulses from home)', row=1, col=col_idx)

fig_c.update_layout(
    title='West → Motor Positions: Model Fit<br>'
          '<sup>Given west displacement, predict Motor 2 & Motor 4 pulses from home</sup>',
    template='plotly_white',
    legend=dict(x=0.01, y=0.99),
    width=1000, height=500,
)
fig_c.show()

# ─────────────────────────────────────────────────────────────────────────────
# Direction D: Up (height) → Motor 2 & 4
# ─────────────────────────────────────────────────────────────────────────────
# %%

Xu_tr_d, Xu_te_d, ym_tr_d, ym_te_d = train_test_split(
    X_u, motors_24, test_size=0.2, random_state=42
)

lr_d = LinearRegression().fit(Xu_tr_d, ym_tr_d)
lr_d_pred = lr_d.predict(Xu_te_d)
lr_d_r2   = [r2_score(ym_te_d[:, i], lr_d_pred[:, i]) for i in range(2)]
lr_d_rmse = [np.sqrt(mean_squared_error(ym_te_d[:, i], lr_d_pred[:, i])) for i in range(2)]

sc_u_d = StandardScaler().fit(Xu_tr_d)
sc_md  = StandardScaler().fit(ym_tr_d)
nn_d = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                    max_iter=500, random_state=42).fit(
    sc_u_d.transform(Xu_tr_d),
    sc_md.transform(ym_tr_d)
)
nn_d_pred = sc_md.inverse_transform(nn_d.predict(sc_u_d.transform(Xu_te_d)))
nn_d_r2   = [r2_score(ym_te_d[:, i], nn_d_pred[:, i]) for i in range(2)]
nn_d_rmse = [np.sqrt(mean_squared_error(ym_te_d[:, i], nn_d_pred[:, i])) for i in range(2)]

print("\nDirection D: Up → Motor 2 & 4")
for i, lbl in enumerate(['Motor 2', 'Motor 4']):
    print(f"  {lbl}  Linear R²={lr_d_r2[i]:.4f} RMSE={lr_d_rmse[i]:.1f} pulses"
          f"   MLP R²={nn_d_r2[i]:.4f} RMSE={nn_d_rmse[i]:.1f} pulses")

lr_curves_d = lr_d.predict(u_grid)
nn_curves_d = sc_md.inverse_transform(nn_d.predict(sc_u_d.transform(u_grid)))

fig_d = make_subplots(rows=1, cols=2,
                      subplot_titles=['Up → Motor 2', 'Up → Motor 4'])

for col_idx, (motor_data, lbl) in enumerate([(m2, 'Motor 2'), (m4, 'Motor 4')], start=1):
    fig_d.add_trace(go.Scatter(
        x=up, y=motor_data, mode='markers',
        marker=dict(size=2, color='lightsteelblue', opacity=0.3),
        name='Data', showlegend=(col_idx == 1),
    ), row=1, col=col_idx)
    fig_d.add_trace(go.Scatter(
        x=u_grid.ravel(), y=lr_curves_d[:, col_idx - 1], mode='lines',
        line=dict(color='crimson', width=2, dash='dash'),
        name=f'Linear R²={lr_d_r2[col_idx-1]:.3f} RMSE={lr_d_rmse[col_idx-1]:.0f} p',
        showlegend=(col_idx == 1),
    ), row=1, col=col_idx)
    fig_d.add_trace(go.Scatter(
        x=u_grid.ravel(), y=nn_curves_d[:, col_idx - 1], mode='lines',
        line=dict(color='darkorange', width=2),
        name=f'MLP    R²={nn_d_r2[col_idx-1]:.3f} RMSE={nn_d_rmse[col_idx-1]:.0f} p',
        showlegend=(col_idx == 1),
    ), row=1, col=col_idx)
    fig_d.update_xaxes(title_text='Up Displacement / Height (mm)', row=1, col=col_idx)
    fig_d.update_yaxes(title_text=f'{lbl} Position (pulses from home)', row=1, col=col_idx)

fig_d.update_layout(
    title='Up (Height) → Motor Positions: Model Fit<br>'
          '<sup>Given foam height, predict Motor 2 & Motor 4 pulses from home</sup>',
    template='plotly_white',
    legend=dict(x=0.01, y=0.99),
    width=1000, height=500,
)
fig_d.show()
