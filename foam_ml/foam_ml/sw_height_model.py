# %%
"""
Model: SW diagonal displacement ↔ Up displacement (foam height) ↔ Motor positions

Trains two models on all move_SW_ trajectory data:
  1. Linear Regression   (baseline)
  2. Neural Network MLP  (sklearn)

SW displacement = sqrt(east_mm^2 + north_mm^2)  — horizontal magnitude from home.
All four motors are active during SW moves:
  Motor 1 (North release): CCW → position increases
  Motor 2 (East release):  CCW → position increases
  Motor 3 (South pull):    CCW → position increases
  Motor 4 (West pull):     CCW → position increases

Sections:
  A. SW (mm) → Up (mm)                 "given diagonal distance, predict height"
  B. Up (mm) → SW (mm)                 "given height, predict diagonal distance"
  C. SW (mm) → Motors 1, 2, 3, 4      "given SW distance, predict all motor commands"
  D. Up (mm) → Motors 1, 2, 3, 4      "given height, predict all motor commands"
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
move_sw_files = sorted(data_dir.glob("*move_SW_*.csv"))

NEEDED_COLS = ['optitrack_x', 'optitrack_y', 'optitrack_z',
               'motor_1_pos', 'motor_2_pos', 'motor_3_pos', 'motor_4_pos']
sw_all, up_all, m1_all, m2_all, m3_all, m4_all = [], [], [], [], [], []

for f in move_sw_files:
    df = pd.read_csv(f)
    valid = df[NEEDED_COLS].dropna()
    if valid.empty:
        continue
    home = valid.iloc[0]
    east = (valid['optitrack_x'] - home['optitrack_x']) * 1000
    north = (valid['optitrack_y'] - home['optitrack_y']) * 1000
    sw_all.extend(np.sqrt(east**2 + north**2))               # mm magnitude
    up_all.extend((valid['optitrack_z'] - home['optitrack_z']) * 1000)
    m1_all.extend(valid['motor_1_pos'] - int(home['motor_1_pos']))
    m2_all.extend(valid['motor_2_pos'] - int(home['motor_2_pos']))
    m3_all.extend(valid['motor_3_pos'] - int(home['motor_3_pos']))
    m4_all.extend(valid['motor_4_pos'] - int(home['motor_4_pos']))

sw = np.array(sw_all)
up = np.array(up_all)
m1 = np.array(m1_all, dtype=float)
m2 = np.array(m2_all, dtype=float)
m3 = np.array(m3_all, dtype=float)
m4 = np.array(m4_all, dtype=float)
print(f"Loaded {len(sw):,} data points from {len(move_sw_files)} files")
print(f"SW diag range: {sw.min():.1f} mm  →  {sw.max():.1f} mm")
print(f"Up      range: {up.min():.1f} mm  →  {up.max():.1f} mm")
for i, (m, lbl) in enumerate([(m1,'M1'),(m2,'M2'),(m3,'M3'),(m4,'M4')]):
    print(f"Motor {i+1}  range: {m.min():.0f}  →  {m.max():.0f} pulses")

# ── Train/test split ─────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error

X_sw = sw.reshape(-1, 1)
X_u  = up.reshape(-1, 1)

Xsw_tr, Xsw_te, yu_tr, yu_te = train_test_split(X_sw, up,  test_size=0.2, random_state=42)
Xu_tr,  Xu_te,  ysw_tr,ysw_te= train_test_split(X_u,  sw,  test_size=0.2, random_state=42)

# ─────────────────────────────────────────────────────────────────────────────
# Direction A: SW → Up
# ─────────────────────────────────────────────────────────────────────────────
# %%
lr_a = LinearRegression().fit(Xsw_tr, yu_tr)
lr_a_pred = lr_a.predict(Xsw_te)
lr_a_r2   = r2_score(yu_te, lr_a_pred)
lr_a_rmse = np.sqrt(mean_squared_error(yu_te, lr_a_pred))

sc_sw = StandardScaler().fit(Xsw_tr)
sc_yu = StandardScaler().fit(yu_tr.reshape(-1, 1))
nn_a = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                    max_iter=500, random_state=42).fit(
    sc_sw.transform(Xsw_tr), sc_yu.transform(yu_tr.reshape(-1,1)).ravel())
nn_a_pred = sc_yu.inverse_transform(
    nn_a.predict(sc_sw.transform(Xsw_te)).reshape(-1,1)).ravel()
nn_a_r2   = r2_score(yu_te, nn_a_pred)
nn_a_rmse = np.sqrt(mean_squared_error(yu_te, nn_a_pred))

print(f"\nDirection A: SW → Up")
print(f"  Linear  R²={lr_a_r2:.4f}  RMSE={lr_a_rmse:.2f} mm")
print(f"  MLP     R²={nn_a_r2:.4f}  RMSE={nn_a_rmse:.2f} mm")

sw_grid = np.linspace(sw.min(), sw.max(), 300).reshape(-1, 1)
fig_a = go.Figure()
fig_a.add_trace(go.Scatter(x=sw, y=up, mode='markers',
    marker=dict(size=2, color='lightsteelblue', opacity=0.3), name='Data'))
fig_a.add_trace(go.Scatter(x=sw_grid.ravel(), y=lr_a.predict(sw_grid), mode='lines',
    line=dict(color='crimson', width=2, dash='dash'),
    name=f'Linear  R²={lr_a_r2:.3f}  RMSE={lr_a_rmse:.1f} mm'))
fig_a.add_trace(go.Scatter(x=sw_grid.ravel(), y=sc_yu.inverse_transform(
    nn_a.predict(sc_sw.transform(sw_grid)).reshape(-1,1)).ravel(), mode='lines',
    line=dict(color='darkorange', width=2),
    name=f'MLP     R²={nn_a_r2:.3f}  RMSE={nn_a_rmse:.1f} mm'))
fig_a.update_layout(
    title='SW → Up: Model Fit<br><sup>Given SW diagonal distance, predict foam height</sup>',
    xaxis_title='SW Diagonal Magnitude (mm)', yaxis_title='Up Displacement / Height (mm)',
    template='plotly_white', legend=dict(x=0.02, y=0.98), width=800, height=550)
fig_a.show()

# ─────────────────────────────────────────────────────────────────────────────
# Direction B: Up → SW
# ─────────────────────────────────────────────────────────────────────────────
# %%
lr_b = LinearRegression().fit(Xu_tr, ysw_tr)
lr_b_pred = lr_b.predict(Xu_te)
lr_b_r2   = r2_score(ysw_te, lr_b_pred)
lr_b_rmse = np.sqrt(mean_squared_error(ysw_te, lr_b_pred))

sc_u  = StandardScaler().fit(Xu_tr)
sc_ysw = StandardScaler().fit(ysw_tr.reshape(-1, 1))
nn_b = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                    max_iter=500, random_state=42).fit(
    sc_u.transform(Xu_tr), sc_ysw.transform(ysw_tr.reshape(-1,1)).ravel())
nn_b_pred = sc_ysw.inverse_transform(
    nn_b.predict(sc_u.transform(Xu_te)).reshape(-1,1)).ravel()
nn_b_r2   = r2_score(ysw_te, nn_b_pred)
nn_b_rmse = np.sqrt(mean_squared_error(ysw_te, nn_b_pred))

print(f"\nDirection B: Up → SW")
print(f"  Linear  R²={lr_b_r2:.4f}  RMSE={lr_b_rmse:.2f} mm")
print(f"  MLP     R²={nn_b_r2:.4f}  RMSE={nn_b_rmse:.2f} mm")

u_grid = np.linspace(up.min(), up.max(), 300).reshape(-1, 1)
fig_b = go.Figure()
fig_b.add_trace(go.Scatter(x=up, y=sw, mode='markers',
    marker=dict(size=2, color='lightsteelblue', opacity=0.3), name='Data'))
fig_b.add_trace(go.Scatter(x=u_grid.ravel(), y=lr_b.predict(u_grid), mode='lines',
    line=dict(color='crimson', width=2, dash='dash'),
    name=f'Linear  R²={lr_b_r2:.3f}  RMSE={lr_b_rmse:.1f} mm'))
fig_b.add_trace(go.Scatter(x=u_grid.ravel(), y=sc_ysw.inverse_transform(
    nn_b.predict(sc_u.transform(u_grid)).reshape(-1,1)).ravel(), mode='lines',
    line=dict(color='darkorange', width=2),
    name=f'MLP     R²={nn_b_r2:.3f}  RMSE={nn_b_rmse:.1f} mm'))
fig_b.update_layout(
    title='Up → SW: Model Fit<br><sup>Given foam height, predict SW diagonal distance</sup>',
    xaxis_title='Up Displacement / Height (mm)', yaxis_title='SW Diagonal Magnitude (mm)',
    template='plotly_white', legend=dict(x=0.02, y=0.98), width=800, height=550)
fig_b.show()

# ─────────────────────────────────────────────────────────────────────────────
# Residual comparison
# ─────────────────────────────────────────────────────────────────────────────
# %%
fig_r = make_subplots(rows=1, cols=2,
    subplot_titles=['SW → Up: Predicted vs Actual', 'Up → SW: Predicted vs Actual'])
for pred, label, color in [(lr_a_pred, f'Linear R²={lr_a_r2:.3f}', 'crimson'),
                            (nn_a_pred, f'MLP    R²={nn_a_r2:.3f}', 'darkorange')]:
    fig_r.add_trace(go.Scatter(x=yu_te, y=pred, mode='markers',
        marker=dict(size=3, color=color, opacity=0.5),
        name=label, legendgroup=label), row=1, col=1)
for pred, label, color in [(lr_b_pred, f'Linear R²={lr_b_r2:.3f}', 'crimson'),
                            (nn_b_pred, f'MLP    R²={nn_b_r2:.3f}', 'darkorange')]:
    fig_r.add_trace(go.Scatter(x=ysw_te, y=pred, mode='markers',
        marker=dict(size=3, color=color, opacity=0.5),
        name=label, legendgroup=label, showlegend=False), row=1, col=2)
for col, (mn, mx) in enumerate([(yu_te.min(),yu_te.max()),(ysw_te.min(),ysw_te.max())], 1):
    fig_r.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode='lines',
        line=dict(color='black', dash='dot', width=1),
        name='Perfect', showlegend=(col==1)), row=1, col=col)
fig_r.update_xaxes(title_text='Actual Up (mm)', row=1, col=1)
fig_r.update_yaxes(title_text='Predicted Up (mm)', row=1, col=1)
fig_r.update_xaxes(title_text='Actual SW (mm)', row=1, col=2)
fig_r.update_yaxes(title_text='Predicted SW (mm)', row=1, col=2)
fig_r.update_layout(title='Predicted vs Actual — Test Set',
    template='plotly_white', width=1000, height=500)
fig_r.show()

# ─────────────────────────────────────────────────────────────────────────────
# Direction C: SW → All 4 motors
# ─────────────────────────────────────────────────────────────────────────────
# %%
motors_all = np.column_stack([m1, m2, m3, m4])
Xsw_tr_c, Xsw_te_c, ym_tr_c, ym_te_c = train_test_split(
    X_sw, motors_all, test_size=0.2, random_state=42)

lr_c = LinearRegression().fit(Xsw_tr_c, ym_tr_c)
lr_c_pred = lr_c.predict(Xsw_te_c)
lr_c_r2   = [r2_score(ym_te_c[:,i], lr_c_pred[:,i]) for i in range(4)]
lr_c_rmse = [np.sqrt(mean_squared_error(ym_te_c[:,i], lr_c_pred[:,i])) for i in range(4)]

sc_sw_c = StandardScaler().fit(Xsw_tr_c)
sc_mc   = StandardScaler().fit(ym_tr_c)
nn_c = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                    max_iter=500, random_state=42).fit(
    sc_sw_c.transform(Xsw_tr_c), sc_mc.transform(ym_tr_c))
nn_c_pred = sc_mc.inverse_transform(nn_c.predict(sc_sw_c.transform(Xsw_te_c)))
nn_c_r2   = [r2_score(ym_te_c[:,i], nn_c_pred[:,i]) for i in range(4)]
nn_c_rmse = [np.sqrt(mean_squared_error(ym_te_c[:,i], nn_c_pred[:,i])) for i in range(4)]

print("\nDirection C: SW → Motors 1-4")
for i, lbl in enumerate(['Motor 1','Motor 2','Motor 3','Motor 4']):
    print(f"  {lbl}  Linear R²={lr_c_r2[i]:.4f} RMSE={lr_c_rmse[i]:.1f} p"
          f"   MLP R²={nn_c_r2[i]:.4f} RMSE={nn_c_rmse[i]:.1f} p")

lr_curves_c = lr_c.predict(sw_grid)
nn_curves_c = sc_mc.inverse_transform(nn_c.predict(sc_sw_c.transform(sw_grid)))

motor_data_list = [m1, m2, m3, m4]
motor_labels = ['Motor 1 (N release)', 'Motor 2 (E release)',
                'Motor 3 (S pull)', 'Motor 4 (W pull)']
fig_c = make_subplots(rows=2, cols=2,
    subplot_titles=[f'SW → {l}' for l in motor_labels])
for idx, (motor_data, lbl) in enumerate(zip(motor_data_list, motor_labels)):
    r, c = divmod(idx, 2)
    row, col = r+1, c+1
    fig_c.add_trace(go.Scatter(x=sw, y=motor_data, mode='markers',
        marker=dict(size=2, color='lightsteelblue', opacity=0.3),
        name='Data', showlegend=(idx==0)), row=row, col=col)
    fig_c.add_trace(go.Scatter(x=sw_grid.ravel(), y=lr_curves_c[:,idx], mode='lines',
        line=dict(color='crimson', width=2, dash='dash'),
        name=f'Linear R²={lr_c_r2[idx]:.3f}', showlegend=(idx==0)), row=row, col=col)
    fig_c.add_trace(go.Scatter(x=sw_grid.ravel(), y=nn_curves_c[:,idx], mode='lines',
        line=dict(color='darkorange', width=2),
        name=f'MLP R²={nn_c_r2[idx]:.3f}', showlegend=(idx==0)), row=row, col=col)
    fig_c.update_xaxes(title_text='SW Diagonal Magnitude (mm)', row=row, col=col)
    fig_c.update_yaxes(title_text='Motor Position (pulses from home)', row=row, col=col)
fig_c.update_layout(
    title='SW → Motor Positions: Model Fit<br>'
          '<sup>Given SW diagonal distance, predict all motor pulses from home</sup>',
    template='plotly_white', legend=dict(x=0.01, y=0.99), width=1000, height=750)
fig_c.show()

# ─────────────────────────────────────────────────────────────────────────────
# Direction D: Up → All 4 motors
# ─────────────────────────────────────────────────────────────────────────────
# %%
Xu_tr_d, Xu_te_d, ym_tr_d, ym_te_d = train_test_split(
    X_u, motors_all, test_size=0.2, random_state=42)

lr_d = LinearRegression().fit(Xu_tr_d, ym_tr_d)
lr_d_pred = lr_d.predict(Xu_te_d)
lr_d_r2   = [r2_score(ym_te_d[:,i], lr_d_pred[:,i]) for i in range(4)]
lr_d_rmse = [np.sqrt(mean_squared_error(ym_te_d[:,i], lr_d_pred[:,i])) for i in range(4)]

sc_u_d = StandardScaler().fit(Xu_tr_d)
sc_md  = StandardScaler().fit(ym_tr_d)
nn_d = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                    max_iter=500, random_state=42).fit(
    sc_u_d.transform(Xu_tr_d), sc_md.transform(ym_tr_d))
nn_d_pred = sc_md.inverse_transform(nn_d.predict(sc_u_d.transform(Xu_te_d)))
nn_d_r2   = [r2_score(ym_te_d[:,i], nn_d_pred[:,i]) for i in range(4)]
nn_d_rmse = [np.sqrt(mean_squared_error(ym_te_d[:,i], nn_d_pred[:,i])) for i in range(4)]

print("\nDirection D: Up → Motors 1-4")
for i, lbl in enumerate(['Motor 1','Motor 2','Motor 3','Motor 4']):
    print(f"  {lbl}  Linear R²={lr_d_r2[i]:.4f} RMSE={lr_d_rmse[i]:.1f} p"
          f"   MLP R²={nn_d_r2[i]:.4f} RMSE={nn_d_rmse[i]:.1f} p")

lr_curves_d = lr_d.predict(u_grid)
nn_curves_d = sc_md.inverse_transform(nn_d.predict(sc_u_d.transform(u_grid)))

fig_d = make_subplots(rows=2, cols=2,
    subplot_titles=[f'Up → {l}' for l in motor_labels])
for idx, (motor_data, lbl) in enumerate(zip(motor_data_list, motor_labels)):
    r, c = divmod(idx, 2)
    row, col = r+1, c+1
    fig_d.add_trace(go.Scatter(x=up, y=motor_data, mode='markers',
        marker=dict(size=2, color='lightsteelblue', opacity=0.3),
        name='Data', showlegend=(idx==0)), row=row, col=col)
    fig_d.add_trace(go.Scatter(x=u_grid.ravel(), y=lr_curves_d[:,idx], mode='lines',
        line=dict(color='crimson', width=2, dash='dash'),
        name=f'Linear R²={lr_d_r2[idx]:.3f}', showlegend=(idx==0)), row=row, col=col)
    fig_d.add_trace(go.Scatter(x=u_grid.ravel(), y=nn_curves_d[:,idx], mode='lines',
        line=dict(color='darkorange', width=2),
        name=f'MLP R²={nn_d_r2[idx]:.3f}', showlegend=(idx==0)), row=row, col=col)
    fig_d.update_xaxes(title_text='Up Displacement / Height (mm)', row=row, col=col)
    fig_d.update_yaxes(title_text='Motor Position (pulses from home)', row=row, col=col)
fig_d.update_layout(
    title='Up (Height) → Motor Positions: Model Fit<br>'
          '<sup>Given foam height, predict all motor pulses from home (move_SW_ data)</sup>',
    template='plotly_white', legend=dict(x=0.01, y=0.99), width=1000, height=750)
fig_d.show()
