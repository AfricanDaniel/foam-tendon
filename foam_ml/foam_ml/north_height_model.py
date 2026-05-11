# %%
"""
Model: North displacement ↔ Up displacement (foam height) ↔ Motor positions

Trains two models on all move_N_ trajectory data:
  1. Linear Regression   (baseline)
  2. Neural Network MLP  (sklearn)

Sections:
  A. North (mm)  → Up (mm)              "given how far north, predict height"
  B. Up (mm)     → North (mm)           "given height, predict north position"
  C. North (mm)  → Motor 1 & 3 (pulses) "given north, predict motor commands"
  D. Up (mm)     → Motor 1 & 3 (pulses) "given height, predict motor commands"
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
move_n_files = sorted(data_dir.glob("*move_N_*.csv"))

NEEDED_COLS = ['optitrack_y', 'optitrack_z', 'motor_1_pos', 'motor_3_pos']
north_all, up_all, m1_all, m3_all = [], [], [], []

for f in move_n_files:
    df = pd.read_csv(f)
    valid = df[NEEDED_COLS].dropna()
    if valid.empty:
        continue
    home = valid.iloc[0]
    north_all.extend((valid['optitrack_y'] - home['optitrack_y']) * 1000)   # mm
    up_all.extend(   (valid['optitrack_z'] - home['optitrack_z']) * 1000)   # mm
    m1_all.extend(    valid['motor_1_pos'] - int(home['motor_1_pos']))       # pulses
    m3_all.extend(    valid['motor_3_pos'] - int(home['motor_3_pos']))       # pulses

north = np.array(north_all)
up    = np.array(up_all)
m1    = np.array(m1_all, dtype=float)
m3    = np.array(m3_all, dtype=float)
print(f"Loaded {len(north):,} data points from {len(move_n_files)} files")
print(f"North   range: {north.min():.1f} mm  →  {north.max():.1f} mm")
print(f"Up      range: {up.min():.1f} mm  →  {up.max():.1f} mm")
print(f"Motor 1 range: {m1.min():.0f}  →  {m1.max():.0f} pulses")
print(f"Motor 3 range: {m3.min():.0f}  →  {m3.max():.0f} pulses")

# ── Train/test split ─────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error

X_n = north.reshape(-1, 1)   # north as input
X_u = up.reshape(-1, 1)      # up as input

# Direction A: North → Up
Xn_tr, Xn_te, yu_tr, yu_te = train_test_split(X_n, up, test_size=0.2, random_state=42)

# Direction B: Up → North
Xu_tr, Xu_te, yn_tr, yn_te = train_test_split(X_u, north, test_size=0.2, random_state=42)

# ─────────────────────────────────────────────────────────────────────────────
# Direction A: North → Up
# ─────────────────────────────────────────────────────────────────────────────
# %%

# Linear regression
lr_a = LinearRegression().fit(Xn_tr, yu_tr)
lr_a_pred = lr_a.predict(Xn_te)
lr_a_r2   = r2_score(yu_te, lr_a_pred)
lr_a_rmse = np.sqrt(mean_squared_error(yu_te, lr_a_pred))

# Neural network
sc_n  = StandardScaler().fit(Xn_tr)
sc_yu = StandardScaler().fit(yu_tr.reshape(-1, 1))
nn_a = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                    max_iter=500, random_state=42).fit(
    sc_n.transform(Xn_tr),
    sc_yu.transform(yu_tr.reshape(-1, 1)).ravel()
)
nn_a_pred_sc = nn_a.predict(sc_n.transform(Xn_te))
nn_a_pred    = sc_yu.inverse_transform(nn_a_pred_sc.reshape(-1, 1)).ravel()
nn_a_r2      = r2_score(yu_te, nn_a_pred)
nn_a_rmse    = np.sqrt(mean_squared_error(yu_te, nn_a_pred))

print("\nDirection A: North → Up")
print(f"  Linear Regression  R²={lr_a_r2:.4f}  RMSE={lr_a_rmse:.2f} mm")
print(f"  Neural Network     R²={nn_a_r2:.4f}  RMSE={nn_a_rmse:.2f} mm")

# Plot Direction A
n_grid = np.linspace(north.min(), north.max(), 300).reshape(-1, 1)
lr_curve_a = lr_a.predict(n_grid)
nn_curve_a = sc_yu.inverse_transform(
    nn_a.predict(sc_n.transform(n_grid)).reshape(-1, 1)
).ravel()

fig_a = go.Figure()

fig_a.add_trace(go.Scatter(
    x=north, y=up, mode='markers',
    marker=dict(size=2, color='lightsteelblue', opacity=0.3),
    name='Data'
))
fig_a.add_trace(go.Scatter(
    x=n_grid.ravel(), y=lr_curve_a, mode='lines',
    line=dict(color='crimson', width=2, dash='dash'),
    name=f'Linear  R²={lr_a_r2:.3f}  RMSE={lr_a_rmse:.1f} mm'
))
fig_a.add_trace(go.Scatter(
    x=n_grid.ravel(), y=nn_curve_a, mode='lines',
    line=dict(color='darkorange', width=2),
    name=f'MLP     R²={nn_a_r2:.3f}  RMSE={nn_a_rmse:.1f} mm'
))
fig_a.update_layout(
    title='North → Up: Model Fit<br>'
          '<sup>Given north displacement, predict foam height (Up)</sup>',
    xaxis_title='North Displacement (mm)',
    yaxis_title='Up Displacement / Height (mm)',
    template='plotly_white',
    legend=dict(x=0.02, y=0.98),
    width=800, height=550,
)
fig_a.show()

# ─────────────────────────────────────────────────────────────────────────────
# Direction B: Up → North
# ─────────────────────────────────────────────────────────────────────────────
# %%

# Linear regression
lr_b = LinearRegression().fit(Xu_tr, yn_tr)
lr_b_pred = lr_b.predict(Xu_te)
lr_b_r2   = r2_score(yn_te, lr_b_pred)
lr_b_rmse = np.sqrt(mean_squared_error(yn_te, lr_b_pred))

# Neural network
sc_u  = StandardScaler().fit(Xu_tr)
sc_yn = StandardScaler().fit(yn_tr.reshape(-1, 1))
nn_b = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                    max_iter=500, random_state=42).fit(
    sc_u.transform(Xu_tr),
    sc_yn.transform(yn_tr.reshape(-1, 1)).ravel()
)
nn_b_pred_sc = nn_b.predict(sc_u.transform(Xu_te))
nn_b_pred    = sc_yn.inverse_transform(nn_b_pred_sc.reshape(-1, 1)).ravel()
nn_b_r2      = r2_score(yn_te, nn_b_pred)
nn_b_rmse    = np.sqrt(mean_squared_error(yn_te, nn_b_pred))

print("\nDirection B: Up → North")
print(f"  Linear Regression  R²={lr_b_r2:.4f}  RMSE={lr_b_rmse:.2f} mm")
print(f"  Neural Network     R²={nn_b_r2:.4f}  RMSE={nn_b_rmse:.2f} mm")

# Plot Direction B
u_grid = np.linspace(up.min(), up.max(), 300).reshape(-1, 1)
lr_curve_b = lr_b.predict(u_grid)
nn_curve_b = sc_yn.inverse_transform(
    nn_b.predict(sc_u.transform(u_grid)).reshape(-1, 1)
).ravel()

fig_b = go.Figure()

fig_b.add_trace(go.Scatter(
    x=up, y=north, mode='markers',
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
    title='Up → North: Model Fit<br>'
          '<sup>Given foam height (Up), predict north displacement</sup>',
    xaxis_title='Up Displacement / Height (mm)',
    yaxis_title='North Displacement (mm)',
    template='plotly_white',
    legend=dict(x=0.02, y=0.98),
    width=800, height=550,
)
fig_b.show()

# ─────────────────────────────────────────────────────────────────────────────
# Residual comparison — actual vs predicted for both models, both directions
# ─────────────────────────────────────────────────────────────────────────────
# %%
fig_r = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        'North → Up: Predicted vs Actual',
        'Up → North: Predicted vs Actual',
    ]
)

# Direction A residuals
for pred, label, color in [
    (lr_a_pred, f'Linear R²={lr_a_r2:.3f}', 'crimson'),
    (nn_a_pred, f'MLP    R²={nn_a_r2:.3f}', 'darkorange'),
]:
    fig_r.add_trace(go.Scatter(
        x=yu_te, y=pred, mode='markers',
        marker=dict(size=3, color=color, opacity=0.5),
        name=label, legendgroup=label,
    ), row=1, col=1)

# Direction B residuals
for pred, label, color in [
    (lr_b_pred, f'Linear R²={lr_b_r2:.3f}', 'crimson'),
    (nn_b_pred, f'MLP    R²={nn_b_r2:.3f}', 'darkorange'),
]:
    fig_r.add_trace(go.Scatter(
        x=yn_te, y=pred, mode='markers',
        marker=dict(size=3, color=color, opacity=0.5),
        name=label, legendgroup=label, showlegend=False,
    ), row=1, col=2)

# Perfect-prediction diagonal
for col, (mn, mx) in enumerate(
    [(yu_te.min(), yu_te.max()), (yn_te.min(), yn_te.max())], start=1
):
    fig_r.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx], mode='lines',
        line=dict(color='black', dash='dot', width=1),
        name='Perfect', showlegend=(col == 1),
    ), row=1, col=col)

fig_r.update_xaxes(title_text='Actual Up (mm)',   row=1, col=1)
fig_r.update_yaxes(title_text='Predicted Up (mm)', row=1, col=1)
fig_r.update_xaxes(title_text='Actual North (mm)',   row=1, col=2)
fig_r.update_yaxes(title_text='Predicted North (mm)', row=1, col=2)

fig_r.update_layout(
    title='Predicted vs Actual — Test Set',
    template='plotly_white',
    width=1000, height=500,
)
fig_r.show()

# ─────────────────────────────────────────────────────────────────────────────
# Direction C: North → Motor 1 & Motor 3
# ─────────────────────────────────────────────────────────────────────────────
# %%

motors_13 = np.column_stack([m1, m3])   # (N, 2) multi-output target

Xn_tr_c, Xn_te_c, ym_tr_c, ym_te_c = train_test_split(
    X_n, motors_13, test_size=0.2, random_state=42
)

# Linear regression (multi-output natively)
lr_c = LinearRegression().fit(Xn_tr_c, ym_tr_c)
lr_c_pred = lr_c.predict(Xn_te_c)
lr_c_r2   = [r2_score(ym_te_c[:, i], lr_c_pred[:, i]) for i in range(2)]
lr_c_rmse = [np.sqrt(mean_squared_error(ym_te_c[:, i], lr_c_pred[:, i])) for i in range(2)]

# Neural network (multi-output)
sc_n_c  = StandardScaler().fit(Xn_tr_c)
sc_mc   = StandardScaler().fit(ym_tr_c)
nn_c = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                    max_iter=500, random_state=42).fit(
    sc_n_c.transform(Xn_tr_c),
    sc_mc.transform(ym_tr_c)
)
nn_c_pred = sc_mc.inverse_transform(nn_c.predict(sc_n_c.transform(Xn_te_c)))
nn_c_r2   = [r2_score(ym_te_c[:, i], nn_c_pred[:, i]) for i in range(2)]
nn_c_rmse = [np.sqrt(mean_squared_error(ym_te_c[:, i], nn_c_pred[:, i])) for i in range(2)]

print("\nDirection C: North → Motor 1 & 3")
for i, lbl in enumerate(['Motor 1', 'Motor 3']):
    print(f"  {lbl}  Linear R²={lr_c_r2[i]:.4f} RMSE={lr_c_rmse[i]:.1f} pulses"
          f"   MLP R²={nn_c_r2[i]:.4f} RMSE={nn_c_rmse[i]:.1f} pulses")

# Fit curves over the full North range
lr_curves_c = lr_c.predict(n_grid)
nn_curves_c = sc_mc.inverse_transform(nn_c.predict(sc_n_c.transform(n_grid)))

fig_c = make_subplots(rows=1, cols=2,
                      subplot_titles=['North → Motor 1', 'North → Motor 3'])

for col_idx, (motor_data, lbl) in enumerate([(m1, 'Motor 1'), (m3, 'Motor 3')], start=1):
    fig_c.add_trace(go.Scatter(
        x=north, y=motor_data, mode='markers',
        marker=dict(size=2, color='lightsteelblue', opacity=0.3),
        name='Data', showlegend=(col_idx == 1),
    ), row=1, col=col_idx)
    fig_c.add_trace(go.Scatter(
        x=n_grid.ravel(), y=lr_curves_c[:, col_idx - 1], mode='lines',
        line=dict(color='crimson', width=2, dash='dash'),
        name=f'Linear R²={lr_c_r2[col_idx-1]:.3f} RMSE={lr_c_rmse[col_idx-1]:.0f} p',
        showlegend=(col_idx == 1),
    ), row=1, col=col_idx)
    fig_c.add_trace(go.Scatter(
        x=n_grid.ravel(), y=nn_curves_c[:, col_idx - 1], mode='lines',
        line=dict(color='darkorange', width=2),
        name=f'MLP    R²={nn_c_r2[col_idx-1]:.3f} RMSE={nn_c_rmse[col_idx-1]:.0f} p',
        showlegend=(col_idx == 1),
    ), row=1, col=col_idx)
    fig_c.update_xaxes(title_text='North Displacement (mm)', row=1, col=col_idx)
    fig_c.update_yaxes(title_text=f'{lbl} Position (pulses from home)', row=1, col=col_idx)

fig_c.update_layout(
    title='North → Motor Positions: Model Fit<br>'
          '<sup>Given north displacement, predict Motor 1 & Motor 3 pulses from home</sup>',
    template='plotly_white',
    legend=dict(x=0.01, y=0.99),
    width=1000, height=500,
)
fig_c.show()

# ─────────────────────────────────────────────────────────────────────────────
# Direction D: Up (height) → Motor 1 & Motor 3
# ─────────────────────────────────────────────────────────────────────────────
# %%

Xu_tr_d, Xu_te_d, ym_tr_d, ym_te_d = train_test_split(
    X_u, motors_13, test_size=0.2, random_state=42
)

# Linear regression
lr_d = LinearRegression().fit(Xu_tr_d, ym_tr_d)
lr_d_pred = lr_d.predict(Xu_te_d)
lr_d_r2   = [r2_score(ym_te_d[:, i], lr_d_pred[:, i]) for i in range(2)]
lr_d_rmse = [np.sqrt(mean_squared_error(ym_te_d[:, i], lr_d_pred[:, i])) for i in range(2)]

# Neural network
sc_u_d  = StandardScaler().fit(Xu_tr_d)
sc_md   = StandardScaler().fit(ym_tr_d)
nn_d = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                    max_iter=500, random_state=42).fit(
    sc_u_d.transform(Xu_tr_d),
    sc_md.transform(ym_tr_d)
)
nn_d_pred = sc_md.inverse_transform(nn_d.predict(sc_u_d.transform(Xu_te_d)))
nn_d_r2   = [r2_score(ym_te_d[:, i], nn_d_pred[:, i]) for i in range(2)]
nn_d_rmse = [np.sqrt(mean_squared_error(ym_te_d[:, i], nn_d_pred[:, i])) for i in range(2)]

print("\nDirection D: Up → Motor 1 & 3")
for i, lbl in enumerate(['Motor 1', 'Motor 3']):
    print(f"  {lbl}  Linear R²={lr_d_r2[i]:.4f} RMSE={lr_d_rmse[i]:.1f} pulses"
          f"   MLP R²={nn_d_r2[i]:.4f} RMSE={nn_d_rmse[i]:.1f} pulses")

# Fit curves over the full Up range
lr_curves_d = lr_d.predict(u_grid)
nn_curves_d = sc_md.inverse_transform(nn_d.predict(sc_u_d.transform(u_grid)))

fig_d = make_subplots(rows=1, cols=2,
                      subplot_titles=['Up → Motor 1', 'Up → Motor 3'])

for col_idx, (motor_data, lbl) in enumerate([(m1, 'Motor 1'), (m3, 'Motor 3')], start=1):
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
          '<sup>Given foam height, predict Motor 1 & Motor 3 pulses from home</sup>',
    template='plotly_white',
    legend=dict(x=0.01, y=0.99),
    width=1000, height=500,
)
fig_d.show()
