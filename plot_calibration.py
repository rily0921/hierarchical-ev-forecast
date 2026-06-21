"""
方向3: 覆盖率-区间宽度曲线 (Reliability-Width Trade-off)
─────────────────────────────────────────────────
基于已有校准数据, 绘制"名义覆盖率 vs 实际覆盖率 + 区间宽度"双轴图
同时对比多个实验组, 直观展示校准度与锐度的权衡
"""
import numpy as np, json, os, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
                     'legend.fontsize': 9, 'figure.dpi': 150, 'savefig.dpi': 300})

OUT = r'E:/Desktop/毕业论文/code/output'
FIG = r'E:/Desktop/毕业论文/图'
os.makedirs(FIG, exist_ok=True)

# ── 加载各实验的校准数据 ──
def load_calib(exp_dir):
    """从实验结果加载校准指标"""
    p = f'{OUT}/{exp_dir}/mint_shrink/metrics.json'
    if not os.path.exists(p):
        # 尝试旧格式
        p = f'{OUT}/{exp_dir}/mint_shrink/metrics.json'
    with open(p) as f:
        m = json.load(f)
    return m['calibration']

# 手工构造所有实验的校准数据
experiments = {
    'E2 (Admin+Indep)':   {'reliability': {'nominal': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], 'actual': None}, 'interval_coverage': None},
}

# 加载可用实验的校准数据
available = {}
for d in sorted(os.listdir(OUT)):
    if d.startswith('E') and os.path.isdir(f'{OUT}/{d}'):
        try:
            cal = load_calib(d)
            name = d.split('_')[0]
            actual = cal['reliability']['actual']
            nominal = cal['reliability']['nominal']
            ic = cal['interval_coverage']
            available[name] = {
                'nominal': np.array(nominal),
                'actual': np.array(actual),
                'widths': {k: v['avg_interval_width'] for k, v in ic.items()},
                'coverages': {k: v['actual_coverage'] for k, v in ic.items()},
                'pit_dev': cal.get('pit_deviation', 0),
            }
            print(f'  {d}: PIT={cal.get("pit_deviation",0):.3f}')
        except Exception as e:
            print(f'  {d}: SKIP ({e})')

# ── 图1: 可靠性曲线 (Reliability Diagram) ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

colors = {'E5': '#2C7BB6', 'E7': '#D7191C', 'E8': '#FDAE61',
          'E2': '#888888', 'E3': '#888888', 'E4': '#888888'}
labels_map = {
    'E5': 'E5: Behav+Behav (best)', 'E7': 'E7: Behav+Admin (worst)',
    'E8': 'E8: Admin+Behav', 'E2': 'E2: Admin+Indep', 'E3': 'E3: Admin+Admin',
    'E4': 'E4: Behav+Indep',
}
styles = {'E5': '-', 'E7': '--', 'E8': '-.', 'E2': ':', 'E3': ':', 'E4': ':'}

for name, data in available.items():
    c = colors.get(name, '#999')
    ls = styles.get(name, '-')
    label = labels_map.get(name, name)
    ax1.plot(data['nominal'], data['actual'], color=c, linestyle=ls,
             marker='o', markersize=5, linewidth=2, label=label, alpha=0.9)

# 完美校准线
ax1.plot([0, 1], [0, 1], 'k-', linewidth=0.8, alpha=0.3)
ax1.fill_between([0, 1], [0, 0.95], [0, 1.05], alpha=0.1, color='green',
                  label='Acceptable range')
ax1.set_xlabel('Nominal Coverage'); ax1.set_ylabel('Actual Coverage')
ax1.set_title('Reliability Diagram'); ax1.legend(fontsize=7, loc='lower right')
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)

# ── 图2: 覆盖率-宽度权衡 ──
levels = [0.50, 0.80, 0.90]
markers = ['s', '^', 'D']
for name, data in available.items():
    c = colors.get(name, '#999')
    for j, level in enumerate(levels):
        k = f'{level:.0%}'
        if k in data['coverages']:
            ax2.scatter(data['coverages'][k], data['widths'][k],
                       color=c, marker=markers[j], s=80,
                       label=f'{labels_map.get(name,name)} ({level:.0%})' if j==0 else '',
                       alpha=0.85, edgecolors='white', linewidth=0.5)
# 连接同实验的点
for name, data in available.items():
    c = colors.get(name, '#999')
    pts_cov, pts_wid = [], []
    for level in levels:
        k = f'{level:.0%}'
        if k in data['coverages']:
            pts_cov.append(data['coverages'][k])
            pts_wid.append(data['widths'][k])
    if len(pts_cov) > 1:
        ax2.plot(pts_cov, pts_wid, color=c, linewidth=1, alpha=0.4, linestyle='--')

ax2.set_xlabel('Actual Coverage'); ax2.set_ylabel('Avg Interval Width (occ. %)')
ax2.set_title('Coverage-Width Trade-off')
ax2.legend(fontsize=6.5, loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIG}/Fig_reliability_coverage_tradeoff.png', dpi=300)
plt.close()
print(f'\nSaved: {FIG}/Fig_reliability_coverage_tradeoff.png')

# ── 图3: PIT 直方图对比 ──
# 加载实际预测数据做PIT
p_bot_e5 = np.load(f'{OUT}/E5_20260621_121531/predictions/pred_bottom.npy')
y_true_e5 = np.load(f'{OUT}/E5_20260621_121531/predictions/y_true.npy')[:,:275]
p_bot_e7 = np.load(f'{OUT}/E7_20260621_133716/predictions/pred_bottom.npy')
y_true_e7 = np.load(f'{OUT}/E7_20260621_133716/predictions/y_true.npy')[:,:275]
taus = np.arange(0.05, 1.0, 0.05)

def compute_pit(q_pred, y_true, taus):
    T, N, K = q_pred.shape
    q_flat = q_pred.reshape(-1, K)
    y_flat = y_true.ravel()
    pit = np.zeros(len(y_flat))
    for i in range(len(y_flat)):
        pit[i] = np.interp(y_flat[i], q_flat[i, :], taus)
    return np.clip(pit[~np.isnan(pit)], 0.001, 0.999)

pit_e5 = compute_pit(p_bot_e5, y_true_e5, taus)
pit_e7 = compute_pit(p_bot_e7, y_true_e7, taus)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, pit, title in [(axes[0], pit_e5, 'E5: Behav+Behav'), (axes[1], pit_e7, 'E7: Behav+Admin')]:
    ax.hist(pit, bins=20, range=(0,1), density=True, color='#2C7BB6', alpha=0.8, edgecolor='white')
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.2, label='Uniform (ideal)')
    ax.set_xlabel('PIT'); ax.set_ylabel('Density'); ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(ax.get_ylim()[1], 1.5))

plt.tight_layout()
plt.savefig(f'{FIG}/Fig_PIT_comparison.png', dpi=300)
plt.close()
print(f'Saved: {FIG}/Fig_PIT_comparison.png')

print('\nDone.')
