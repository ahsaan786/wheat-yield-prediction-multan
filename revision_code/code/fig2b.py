import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

plt.rcParams['font.family'] = 'DejaVu Sans'
W, H = 10.0, 13.6
fig = plt.figure(figsize=(W, H), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')

STAGES = [
    ('#1F5FA8', '#EAF2FB', 'Data and crop-season features'),
    ('#2E7D32', '#EBF6EC', 'Trend control and chronological partition'),
    ('#C26A00', '#FFF4E3', 'Standalone residual models and baselines'),
    ('#6A3D9A', '#F3ECFA', 'District-adaptive SERWI integration and model selection'),
    ('#B23A3A', '#FCEDED', 'Out-of-time evaluation, uncertainty and interpretation'),
    ('#37474F', '#EEF1F3', 'Conditional robustness check and outputs'),
]

def band(y0, h, k):
    dark, light, title = STAGES[k]
    ax.add_patch(FancyBboxPatch((0.25, y0), W - 0.5, h, boxstyle='round,pad=0,rounding_size=0.18',
                                fc=light, ec=dark, lw=1.2))
    ax.add_patch(Circle((0.72, y0 + h - 0.36), 0.25, fc=dark, ec='none'))
    ax.text(0.72, y0 + h - 0.36, str(k + 1), ha='center', va='center', color='white', fontsize=13, fontweight='bold')
    ax.text(1.12, y0 + h - 0.36, f'Stage {k + 1}  |  {title}', ha='left', va='center', color=dark,
            fontsize=12.5, fontweight='bold')
    return dark

def box(x, y, w, h, title, body, dark, fs=9.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.12',
                                fc='white', ec=dark, lw=1.1))
    ax.text(x + w / 2, y + h - 0.2, title, ha='center', va='top', fontsize=fs + 0.8, fontweight='bold', color='#1A1A1A')
    ax.text(x + w / 2, y + h - 0.52, body, ha='center', va='top', fontsize=fs, color='#333333', linespacing=1.35)

def harrow(x1, x2, y, c):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle='-|>', mutation_scale=14, color=c, lw=1.4))

def varrow(x, y1, y2, c='#555555'):
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle='-|>', mutation_scale=18, color=c, lw=1.8))

ax.text(W / 2, H - 0.35, 'Trend-adjusted, district-weighted wheat-yield nowcasting workflow',
        ha='center', va='center', fontsize=14.5, fontweight='bold', color='#111111')

# geometry
bh = [1.75, 1.75, 1.55, 2.05, 1.75, 1.75]
gap = 0.42
top = H - 0.75
ys = []
y = top
for h in bh:
    y -= h; ys.append(y); y -= gap

inner_x0, inner_x1 = 0.55, W - 0.55
iw = inner_x1 - inner_x0

# Stage 1
d = band(ys[0], bh[0], 0); by = ys[0] + 0.15; bhh = 1.0
w3 = (iw - 2 * 0.45) / 3
xs = [inner_x0 + i * (w3 + 0.45) for i in range(3)]
box(xs[0], by, w3, bhh, 'Data assembly', '4 districts × 25 years (2001–2025)\nofficial CRS/PBS wheat yield', d)
box(xs[1], by, w3, bhh, 'Crop-season features', 'MODIS wheat-mask NDVI and LST,\nCHIRPS rain, ERA5-Land VPD, stress days', d)
box(xs[2], by, w3, bhh, 'Feature screening', '|r| > 0.90 on 2001–2020; 9 predictors\nincl. yield lag and 2 interactions', d)
for i in range(2):
    harrow(xs[i] + w3 + 0.04, xs[i + 1] - 0.04, by + bhh / 2, d)

# Stage 2
d = band(ys[1], bh[1], 1); by = ys[1] + 0.15
w2 = (iw - 0.45) / 2
box(inner_x0, by, w2, bhh, 'Pre-test trend–residual decomposition', 'district Theil–Sen trend chosen on 2019–2020;\nmodels learn the detrended residual', d)
x2 = inner_x0 + w2 + 0.45
ax.add_patch(FancyBboxPatch((x2, by), w2, bhh, boxstyle='round,pad=0,rounding_size=0.12', fc='white', ec=d, lw=1.1))
ax.text(x2 + w2 / 2, by + bhh - 0.24, 'Chronological data partition', ha='center', va='top', fontsize=10.2, fontweight='bold')
tx0, tw, ty = x2 + 0.25, w2 - 0.5, by + 0.08
segs = [(0.44, 'Development', '2002–2018', '#8FBF8F'), (0.26, 'Validation', '2019–2020', '#F2C14E'), (0.30, 'Test (once)', '2021–2025', '#E57373')]
cx = tx0
for fr, lab, yrs, c in segs:
    sw = tw * fr
    ax.add_patch(FancyBboxPatch((cx, ty), sw, 0.46, boxstyle='square,pad=0', fc=c, ec='white', lw=1.5))
    ax.text(cx + sw / 2, ty + 0.32, lab, ha='center', va='center', fontsize=8.6, fontweight='bold', color='#1A1A1A')
    ax.text(cx + sw / 2, ty + 0.12, yrs, ha='center', va='center', fontsize=8.0, color='#1A1A1A')
    cx += sw
harrow(inner_x0 + w2 + 0.04, x2 - 0.04, by + bhh / 2, d)

# Stage 3
d = band(ys[2], bh[2], 2); by = ys[2] + 0.15; b3 = 0.85
box(xs[0], by, w3, b3, 'XGBoost', 'time-ordered CV; ten baselines', d)
box(xs[1], by, w3, b3, 'LSTM', '3-season sequences; 20 seeds', d)
box(xs[2], by, w3, b3, 'TCN', 'causal convolutions; 20 seeds', d)

# Stage 4
d = band(ys[3], bh[3], 3); by = ys[3] + 0.15
box(inner_x0, by, iw, 1.38, 'Inverse-error weighting in each district', '', d, fs=9.0)
ax.text(W/2, by + 0.80, 'one-year-ahead rolling forecasts 2014–2020  →  district RMSE of each model  →  district weights  →  trend added back', ha='center', va='center', fontsize=9.0, color='#333333')
ax.text(W/2, by + 0.50, r'$w_{m,d} = (1/\mathrm{RMSE}_{m,d})\ /\ \Sigma_{k}\,(1/\mathrm{RMSE}_{k,d})$', ha='center', va='center', fontsize=10.5, color='#4A2A70')
ax.text(W/2, by + 0.19, 'hybrids: XGB+LSTM, TCN+XGB, LSTM+TCN, XGB+LSTM+TCN   |   operational model = lowest pre-test RMSE', ha='center', va='center', fontsize=9.0, color='#333333')

# Stage 5
d = band(ys[4], bh[4], 4); by = ys[4] + 0.15
box(xs[0], by, w3, bhh, 'Out-of-time evaluation', 'test years used once; R², RMSE, MAE,\nbias, skill vs trend, DM tests', d)
box(xs[1], by, w3, bhh, 'Uncertainty', 'year-block bootstrap, 20-seed\nspread, rolling-origin forecasts', d)
box(xs[2], by, w3, bhh, 'Explainability', 'signed SHAP values and\ndependence plots', d)

# Stage 6
d = band(ys[5], bh[5], 5); by = ys[5] + 0.15
box(inner_x0, by, w2, bhh, 'Conditional 2026–2030 check', 'CMIP6 SSP2-4.5; NDVI Scenarios A and B;\ntrend frozen at 2025 or continued', d)
box(x2, by, w2, bhh, 'Outputs', 'district nowcasts, applicability-domain\ncheck, reproducible code', d)
harrow(inner_x0 + w2 + 0.04, x2 - 0.04, by + bhh / 2, d)

# stage-to-stage arrows
for k in range(5):
    varrow(W / 2, ys[k] - 0.02, ys[k + 1] + bh[k + 1] + 0.02)
# model fan-in/out arrows inside stage 3 -> 4 handled by stage arrow; add side arrows for clarity
for x in [xs[0] + w3 / 2, xs[2] + w3 / 2]:
    varrow(x, ys[2] - 0.02, ys[3] + bh[3] + 0.02, '#999999')

fig.savefig('figs/fig2_workflow.png', dpi=300, facecolor='white')
print('ok')
