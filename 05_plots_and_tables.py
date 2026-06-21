"""
第四章图表生成: 表4 + 图1-10
所有数据来自 output/ 中的实验结果
"""
import numpy as np, pandas as pd, json, os, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import norm

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 11, 'axes.labelsize': 10,
                     'legend.fontsize': 8, 'figure.dpi': 150, 'savefig.dpi': 300})

OUT = r'E:/Desktop/毕业论文/code/output'
FIG = r'E:/Desktop/毕业论文/图'
os.makedirs(FIG, exist_ok=True)

# ======== 加载基础数据 ========
RAW   = r'E:/Desktop/毕业论文/data/UrbanEV-main/UrbanEV-main/data'
with open(f'{OUT}/hierarchy_meta.json') as f: h_meta = json.load(f)
with open(f'{OUT}/data_summary.json') as f: d_sum = json.load(f)

vol = pd.read_csv(f'{RAW}/volume.csv', index_col=0)
vol.index = pd.to_datetime(vol.index)
monthly_avg = vol.resample('ME').sum().mean(axis=0)  # 各TAZ月均充电量

# G 组结果
def load_json(name):
    with open(f'{OUT}/{name}_results.json') as f: return json.load(f)
g1=load_json('G1'); g2=load_json('G2'); g4=load_json('G4')
g5=load_json('G5'); g7=load_json('G7'); g8=load_json('G8')
g7p=load_json('G7_prob'); g7bv2=load_json('G7_bootstrap_v2')

# 预测数据
p_bot_g5 = np.load(f'{OUT}/G5_pred_bottom.npy')  # (744,275,19)
p_bot_g7 = np.load(f'{OUT}/G7_pred_bottom.npy')
p_bot_g2 = np.load(f'{OUT}/G2_pred_bottom.npy')
p_bot_g4 = np.load(f'{OUT}/G4_pred_bottom.npy')
p_mid_g2 = np.load(f'{OUT}/G2_pred_middle.npy')  # (744,8,19)
p_mid_g5 = np.load(f'{OUT}/G5_pred_middle.npy')  # (744,10,19)
y_t_g2 = np.load(f'{OUT}/G2_y_true.npy')
y_t_g5 = np.load(f'{OUT}/G5_y_true.npy')
y_t_g7 = np.load(f'{OUT}/G7_y_true.npy')
y_r_g1 = np.load(f'{OUT}/G1_y_rec_shrink.npy')
y_r_g2 = np.load(f'{OUT}/G2_y_rec_shrink.npy')
y_r_g7 = np.load(f'{OUT}/G7_y_rec_shrink.npy')
y_r_bv2 = np.load(f'{OUT}/G7_y_rec_bootstrap_v2.npy')  # (744,286,500)

n_bottom = 275; n_test = p_bot_g7.shape[0]
TAUS = np.arange(0.05, 1.00, 0.05)

# 辅助
def rmse_per_node(yp, yt, axis=0):
    return np.sqrt(np.mean((yp-yt)**2, axis=axis))

# ================================================================
# 表4: TAZ分档 RMSE
# ================================================================
print('=== Table 4: TAZ stratification ===')
taz_monthly = monthly_avg[monthly_avg.index.astype(str).isin(
    [str(c) for c in range(len(monthly_avg))])].values  # won't match, need proper alignment
# 直接用已经加载的volume数据
taz_vols = []
for j in range(n_bottom):
    taz_vols.append(monthly_avg.iloc[j] if j < len(monthly_avg) else 0)
taz_vols = np.array(taz_vols)

# 正确对齐: monthly_avg index 是 TAZ ID (102, 104, ...)
# p_bot 列索引 0..274 对应 zone_cols 顺序
occ = pd.read_csv(f'{RAW}/occupancy.csv'); occ['time']=pd.to_datetime(occ['time']); occ=occ.set_index('time')
zone_cols = [c for c in occ.columns]
taz_vols_aligned = np.array([monthly_avg.get(z, 0) for z in zone_cols])  # zone_cols 是字符串 '102'

y_tb_g7 = y_t_g7[:, :n_bottom]
y_tb_g2 = y_t_g2[:, :n_bottom]
rmse_g5_per_taz = rmse_per_node(p_bot_g5[:,:,9], y_tb_g2, axis=0)  # (275,)
rmse_g7_per_taz = rmse_per_node(p_bot_g7[:,:,9], y_tb_g7, axis=0)

p33, p67 = np.percentile(taz_vols_aligned, [33.3, 66.7])
lo = taz_vols_aligned <= p33; mi = (taz_vols_aligned > p33) & (taz_vols_aligned <= p67)
hi = taz_vols_aligned > p67

for label, mask in [('Low', lo), ('Mid', mi), ('High', hi)]:
    r5 = np.sqrt(np.mean(rmse_g5_per_taz[mask]**2))
    r7 = np.sqrt(np.mean(rmse_g7_per_taz[mask]**2))
    impr = (r5-r7)/r5*100
    vol_range = f'{taz_vols_aligned[mask].min():.0f}-{taz_vols_aligned[mask].max():.0f}'
    print(f'  {label}: vol={vol_range}, n={mask.sum()}, G5_RMSE={r5:.2f}, G7_RMSE={r7:.2f}, impr={impr:.1f}%')

# ================================================================
# 图1: 275 TAZ RMSE 箱线图 (独立 vs 分组)
# ================================================================
print('=== Fig 1: RMSE boxplot ===')
fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot([rmse_g5_per_taz, rmse_g7_per_taz], labels=['Independent\n(G5)', 'Grouped\n(G7)'],
           patch_artist=True, widths=0.4,
           boxprops=dict(facecolor='lightblue'),
           medianprops=dict(color='red', linewidth=1.5))
ax.set_ylabel('RMSE (occupancy %)')
ax.set_title('Per-TAZ RMSE Distribution: Independent vs Grouped Modeling')
ax.text(0.02, 0.95, f'Median: {np.median(rmse_g5_per_taz):.2f} → {np.median(rmse_g7_per_taz):.2f}',
        transform=ax.transAxes, fontsize=9, verticalalignment='top')
fig.savefig(f'{FIG}/fig1_rmse_boxplot.png')
plt.close()

# ================================================================
# 图2: 稀疏度 vs RMSE 改善散点图
# ================================================================
print('=== Fig 2: Sparsity vs improvement ===')
improvement = (rmse_g5_per_taz - rmse_g7_per_taz) / (rmse_g5_per_taz + 1e-8) * 100
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(taz_vols_aligned, improvement, alpha=0.5, s=15, c='steelblue', edgecolors='none')
ax.set_xscale('log')
ax.set_xlabel('Avg Monthly Volume (kWh, log scale)')
ax.set_ylabel('RMSE Improvement by Grouped (%)')
ax.set_title('TAZ Sparsity vs Grouped Modeling Benefit')
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
# 标注低活跃区域
ax.axvline(x=np.percentile(taz_vols_aligned, 33.3), color='red', linestyle=':', alpha=0.5,
           label='33rd percentile')
ax.legend()
fig.savefig(f'{FIG}/fig2_sparsity_improvement.png')
plt.close()

# ================================================================
# 图3: 簇内相关 vs 中层 RMSE
# ================================================================
print('=== Fig 3: Intra-corr vs middle RMSE ===')
# 行政8区 + 行为10簇 各自的簇内相关和中层RMSE
admin_sizes = h_meta['hierarchies']['admin']['district_sizes']
behav_sizes = h_meta['hierarchies']['behavior']['district_sizes']

# 每个中层节点的 RMSE (从G2和G5的中层预测计算)
y_tm_g2 = y_t_g2[:, n_bottom:n_bottom+8]  # (744,8)
y_tm_g5 = y_t_g5[:, n_bottom:n_bottom+10] # (744,10)

mid_rmse_g2 = rmse_per_node(p_mid_g2[:,:,9], y_tm_g2, axis=0)  # (8,)
mid_rmse_g5 = rmse_per_node(p_mid_g5[:,:,9], y_tm_g5, axis=0)  # (10,)

# 每个中层节点内的簇内相关 (需从原始数据计算)
# 用S矩阵还原各节点包含哪些TAZ, 然后算簇内平均相关
S_admin = np.load(f'{OUT}/S_admin.npy')
S_behav = np.load(f'{OUT}/S_behavior.npy')

def per_node_intra_corr(S, occ_wide, zone_cols, n_mid):
    cors = []
    for d in range(n_mid):
        members = np.where(S[n_bottom+d, :]==1)[0]
        if len(members) > 1:
            sub = occ_wide.iloc[:, members]
            cm = sub.corr().values
            upper = cm[np.triu_indices_from(cm, k=1)]
            cors.append(np.mean(upper))
        else:
            cors.append(0)
    return np.array(cors)

admin_corr = per_node_intra_corr(S_admin, occ, zone_cols, 8)
behav_corr = per_node_intra_corr(S_behav, occ, zone_cols, 10)

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(admin_corr, mid_rmse_g2, c='#d62728', label='Admin (G2)', s=40, edgecolors='black', linewidth=0.5)
ax.scatter(behav_corr, mid_rmse_g5, c='#1f77b4', label='Behavior (G5)', s=40, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Intra-cluster Mean Pearson Correlation')
ax.set_ylabel('Middle Node RMSE')
ax.set_title('Intra-cluster Homogeneity vs Middle-Level Forecast Accuracy')
ax.legend()
fig.savefig(f'{FIG}/fig3_intracorr_vs_rmse.png')
plt.close()

# ================================================================
# 图4: Δ_structure + Δ_quality 堆叠条形图
# ================================================================
print('=== Fig 4: Decomposition bar ===')
# 用city RMSE
indep_total = g1['city']['rmse_after'] - g5['city']['rmse_after']
indep_struct = g1['city']['rmse_after'] - g2['city']['rmse_after']
indep_qual   = g2['city']['rmse_after'] - g5['city']['rmse_after']

group_total = g1['city']['rmse_after'] - g7['city']['rmse_after']
group_struct = g1['city']['rmse_after'] - g4['city']['rmse_after']
group_qual   = g4['city']['rmse_after'] - g7['city']['rmse_after']

fig, ax = plt.subplots(figsize=(7, 5))
x = [0, 1]
w = 0.5
ax.bar(x, [indep_struct, group_struct], w, label=r'$\Delta_{structure}$', color='#2c3e50')
ax.bar(x, [indep_qual, group_qual], w, bottom=[indep_struct, group_struct],
       label=r'$\Delta_{quality}$', color='#3498db')
# 标注百分比
for i, (s, q, t) in enumerate([(indep_struct, indep_qual, indep_total),
                                (group_struct, group_qual, group_total)]):
    if t > 0:
        ax.text(x[i], s/2, f'{s/t*100:.0f}%', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
        ax.text(x[i], s+q/2, f'{q/t*100:.0f}%', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    ax.text(x[i], t+1, f'Total: {t:.1f}', ha='center', fontsize=9)

ax.set_xticks(x); ax.set_xticklabels(['Independent\nStrategy', 'Grouped\nStrategy'])
ax.set_ylabel('City RMSE Reduction')
ax.set_title('Decomposition of Reconciliation Gain')
ax.legend(loc='upper right')
fig.savefig(f'{FIG}/fig4_decomposition_bar.png')
plt.close()

# ================================================================
# 图5: 286节点热力图 (简化版: 分层汇总)
# ================================================================
print('=== Fig 5: Heatmap (summary) ===')
y_r_g4 = np.load(f'{OUT}/G4_y_rec_shrink.npy')
y_r_g5 = np.load(f'{OUT}/G5_y_rec_shrink.npy')

y_t_all = y_t_g7  # (744,286)

# 每层汇总RMSE
def layer_rmse(y_rec, y_true, n_b, n_m):
    bot = np.sqrt(np.mean((y_rec[:,:n_b]-y_true[:,:n_b])**2))
    mid = np.sqrt(np.mean((y_rec[:,n_b:n_b+n_m]-y_true[:,n_b:n_b+n_m])**2))
    top = np.sqrt(np.mean((y_rec[:,-1]-y_true[:,-1])**2))
    return [bot, mid, top]

layers = ['Bottom', 'Middle', 'Top']
exps = ['G1', 'G2', 'G4', 'G5', 'G7']
data = []
for y_rec, n_m, label in [(y_r_g1,1,'G1'), (y_r_g2,8,'G2'), (y_r_g4,8,'G4'),
                            (y_r_g5,10,'G5'), (y_r_g7,10,'G7')]:
    if label == 'G1':
        data.append(layer_rmse(np.column_stack([y_rec[:,:275], y_rec[:,-1:].repeat(9,axis=1), y_rec[:,-1]]),
                               y_t_all, 275, 1))
    else:
        data.append(layer_rmse(y_rec, y_t_all, 275, n_m))

data = np.array(data)
fig, ax = plt.subplots(figsize=(8, 3))
im = ax.imshow(data.T, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(exps))); ax.set_xticklabels(exps)
ax.set_yticks(range(3)); ax.set_yticklabels(layers)
for i in range(len(exps)):
    for j in range(3):
        ax.text(i, j, f'{data[i,j]:.1f}', ha='center', va='center', fontsize=10,
                color='white' if data[i,j] > np.median(data) else 'black')
ax.set_title('RMSE by Layer and Experiment')
plt.colorbar(im, ax=ax, shrink=0.8)
fig.savefig(f'{FIG}/fig5_heatmap.png')
plt.close()

# ================================================================
# 图6: 某周全市预测时序 (G1 vs G7)
# ================================================================
print('=== Fig 6: City timeseries ===')
y_tt = y_t_g7[:, -1]
city_g1 = y_r_g1[:, :n_bottom].sum(axis=1)
city_g7 = y_r_g7[:, :n_bottom].sum(axis=1)

# 测试集第一周 (168小时)
week_idx = slice(0, 168)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(168), y_tt[week_idx], 'k-', linewidth=1.5, label='True')
ax.plot(range(168), city_g1[week_idx], 'r--', linewidth=1, alpha=0.8, label='G1 (2-level)')
ax.plot(range(168), city_g7[week_idx], 'b-.', linewidth=1, alpha=0.8, label='G7 (3-level)')
ax.set_xlabel('Hour')
ax.set_ylabel('City Total Occupancy (%)')
ax.set_title('City-Level Forecast: Two-Level vs Three-Level (First Week of Test Set)')
ax.legend()
fig.savefig(f'{FIG}/fig6_city_timeseries.png')
plt.close()

# ================================================================
# 图7: 19τ ΔQL U形图
# ================================================================
print('=== Fig 7: U-shape QL ===')
ql_by_tau = g8['ql_by_tau']
deltas = [ql_by_tau[f'τ={t:.2f}']['delta'] for t in TAUS]

fig, ax = plt.subplots(figsize=(8, 4))
colors = ['#e74c3c']*6 + ['#95a5a6']*7 + ['#2980b9']*6  # low=red, mid=gray, high=blue
ax.bar(range(19), deltas, color=colors, edgecolor='white', linewidth=0.3)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.axvline(x=5.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axvline(x=12.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xticks(range(0,19,2))
ax.set_xticklabels([f'{TAUS[i]:.2f}' for i in range(0,19,2)])
ax.set_xlabel('Quantile Level τ')
ax.set_ylabel('ΔQL (G8 − G7, negative = improvement)')
ax.set_title('Quantile-Specific Reconciliation: QL Change by τ')
# 标注
ax.text(2.5, min(deltas)*1.1, 'Low τ', ha='center', fontsize=9, color='#e74c3c')
ax.text(9, max(deltas)*0.8, 'Mid τ', ha='center', fontsize=9, color='#95a5a6')
ax.text(15.5, min(deltas)*1.1, 'High τ', ha='center', fontsize=9, color='#2980b9')
fig.savefig(f'{FIG}/fig7_ushape_ql.png')
plt.close()

# ================================================================
# 图8: G7 vs G8 某TAZ的 τ=0.1 和 τ=0.9 曲线
# ================================================================
print('=== Fig 8: Quantile curves ===')
p_bot_g8 = np.load(f'{OUT}/G8_pred_bottom.npy')  # (744,275,19)
# 取一个代表性TAZ (如第50个), 取第一周
taz_idx = 50
week = slice(0, 168)

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for ax, tau_idx, tau_val, label in [(axes[0], 1, 0.10, 'τ=0.10 (Low)'),
                                      (axes[1], 17, 0.90, 'τ=0.90 (High)')]:
    ax.plot(range(168), y_tb_g7[week, taz_idx], 'k-', linewidth=1, alpha=0.7, label='True')
    ax.plot(range(168), p_bot_g7[week, taz_idx, tau_idx], 'b-', linewidth=1, alpha=0.7, label='G7 (Std MinT)')
    ax.plot(range(168), p_bot_g8[week, taz_idx, tau_idx], 'r--', linewidth=1, alpha=0.7, label='G8 (QS MinT)')
    ax.set_ylabel('Occupancy (%)')
    ax.set_title(f'TAZ {taz_idx}: {label}')
    ax.legend(fontsize=8)
axes[1].set_xlabel('Hour')
fig.suptitle('Quantile-Specific vs Standard MinT: Low and High Quantile Predictions', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f'{FIG}/fig8_quantile_curves.png')
plt.close()

# ================================================================
# 图9: PIT 直方图 (3方法)
# ================================================================
print('=== Fig 9: PIT histograms ===')
hist_prob = g7p['pit']['probabilistic_hist']
hist_det  = g7p['pit']['deterministic_hist']
# Bootstrap v2 PIT 需要重新计算
# 从 y_rec_bv2 计算PIT
y_tb = y_t_g7[:, :n_bottom]
pit_bv2_all = []
for j in range(n_bottom):
    sc_j = y_r_bv2[:, j, :]  # (744, 500)
    for t in range(len(sc_j)):
        pit_bv2_all.append(np.mean(sc_j[t,:] <= y_tb[t, j]))
pit_bv2_all = np.array(pit_bv2_all)
hist_bv2, _ = np.histogram(pit_bv2_all, bins=10, range=(0,1))
hist_bv2 = (hist_bv2 / len(pit_bv2_all)).tolist()

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
bins = np.linspace(0, 1, 11)
for ax, hist, title, ks_val in [
    (axes[0], hist_det, 'Deterministic + Gaussian', g7p['pit']['ks_deterministic']['statistic']),
    (axes[1], hist_prob, 'τ-level + Indep Sampling', g7p['pit']['ks_probabilistic']['statistic']),
    (axes[2], hist_bv2, 'Bootstrap + MinT', g7bv2['interval_90_bottom']['bootstrap_v2']['ks']),
]:
    ax.bar(bins[:-1], hist, width=0.1, align='edge', edgecolor='white', color='steelblue', alpha=0.8)
    ax.axhline(y=0.1, color='red', linestyle='--', linewidth=0.8)
    ax.set_title(f'{title}\nKS={ks_val:.3f}', fontsize=9)
    ax.set_xlabel('PIT'); ax.set_xticks([0, 0.5, 1])
axes[0].set_ylabel('Frequency')
fig.suptitle('PIT Histograms: Calibration of 90% Prediction Intervals', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(f'{FIG}/fig9_pit_histograms.png')
plt.close()

# ================================================================
# 图10: 某TAZ的区间时序 (确定性 vs Bootstrap)
# ================================================================
print('=== Fig 10: Interval timeseries ===')
taz_idx = 50; week = slice(0, 168)
yt_week = y_tb[week, taz_idx]

# 确定性+高斯
y_rec_det = y_r_g7[:, :n_bottom]
resid = y_rec_det[:, taz_idx] - y_tb[:, taz_idx]
sigma = np.std(resid)
det_lo = y_rec_det[week, taz_idx] - 1.645*sigma
det_hi = y_rec_det[week, taz_idx] + 1.645*sigma
det_mid = y_rec_det[week, taz_idx]

# Bootstrap
boot_sc = y_r_bv2[week, taz_idx, :]  # (168,500)
boot_lo = np.percentile(boot_sc, 5, axis=1)
boot_hi = np.percentile(boot_sc, 95, axis=1)
boot_mid = np.median(boot_sc, axis=1)

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for ax, lo, hi, mid, title in [
    (axes[0], det_lo, det_hi, det_mid, 'Deterministic + Gaussian'),
    (axes[1], boot_lo, boot_hi, boot_mid, 'Bootstrap + MinT'),
]:
    ax.fill_between(range(168), lo, hi, alpha=0.25, color='steelblue', label='90% PI')
    ax.plot(range(168), mid, 'b-', linewidth=1, alpha=0.8)
    ax.plot(range(168), yt_week, 'k.', markersize=3, alpha=0.6)
    ax.set_ylabel('Occupancy (%)')
    ax.set_title(title, fontsize=10)
    cov = np.mean((yt_week>=lo) & (yt_week<=hi))
    ax.text(0.98, 0.05, f'Coverage: {cov:.2f}', transform=ax.transAxes,
            ha='right', fontsize=9)
axes[1].set_xlabel('Hour')
fig.suptitle(f'90% Prediction Intervals: TAZ {taz_idx} (First Week)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f'{FIG}/fig10_interval_timeseries.png')
plt.close()

print(f'\n[Done] All figures saved to {FIG}/')
print('Table 4 data computed above.')
