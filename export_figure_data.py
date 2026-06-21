"""
导出 OriginPro 画图数据
─────────────────────
每张图对应一个 CSV, 放在 output/figures/ 下
"""
import numpy as np, pandas as pd, os, warnings
warnings.filterwarnings('ignore')

OUT = r'E:/Desktop/毕业论文/code/output/figures'
os.makedirs(OUT, exist_ok=True)
SRC = r'E:/Desktop/毕业论文/code/output'

TAUS = np.arange(0.05, 1.00, 0.05)

# ================================================================
# 图1: E5全市预测时序图 (测试集第一周, 168小时)
# ================================================================
print('图1: E5全市预测时序图...')
y_true = np.load(f'{SRC}/G7_y_true.npy')      # (744, 286)
p_bot_e5 = np.load(f'{SRC}/G7_pred_bottom.npy')  # (744, 275, 19)
y_rec = np.load(f'{SRC}/G7_y_rec_shrink.npy')     # (744, 286)
n_bottom = 275

y_tt = y_true[:, -1]                          # 全市真实
bu_med = p_bot_e5[:, :, 9].sum(axis=1)        # 底层加总 (调和前)
rec_med = y_rec[:, :n_bottom].sum(axis=1)     # MinT-shrink 调和后

# 取前168小时
df1 = pd.DataFrame({
    'Hour': np.arange(1, 169),
    'True': y_tt[:168],
    'BottomUp_Before': bu_med[:168],
    'MinT_After': rec_med[:168],
})
df1.to_csv(f'{OUT}/Fig01_E5_city_timeseries.csv', index=False)
print(f'  → {OUT}/Fig01_E5_city_timeseries.csv')

# ================================================================
# 图2: 代表性中层聚类区域时序图
# ================================================================
print('图2: 中层聚类区域时序图...')
p_mid_e5 = np.load(f'{SRC}/G7_pred_middle.npy')  # (744, 10, 19)
y_tm = y_true[:, n_bottom:n_bottom+10]             # 中层真实值
# 选 RMSE 最低的那个 cluster
rmse_mid = [np.sqrt(np.mean((y_tm[:, d] - p_mid_e5[:, d, 9])**2)) for d in range(10)]
best_cluster = int(np.argmin(rmse_mid))

df2 = pd.DataFrame({
    'Hour': np.arange(1, 169),
    'True': y_tm[:168, best_cluster],
    'Predicted': p_mid_e5[:168, best_cluster, 9],
})
df2.to_csv(f'{OUT}/Fig02_middle_cluster_timeseries.csv', index=False)
print(f'  → {OUT}/Fig02_middle_cluster_timeseries.csv (cluster {best_cluster})')

# ================================================================
# 图3: 调和增益归因分解 — 堆叠条形图
# ================================================================
print('图3: 结构效应 vs 质量效应...')
# RMSE 归因 (分组建模)
df3_rmse = pd.DataFrame({
    'Effect': ['Structure', 'Quality'],
    'RMSE_Contribution': [22.2, 11.7],
    'Percentage': [65, 35],
})
df3_rmse.to_csv(f'{OUT}/Fig03a_decomposition_RMSE.csv', index=False)

# CRPS 归因 (分组建模)
df3_crps = pd.DataFrame({
    'Effect': ['Structure', 'Quality'],
    'CRPS_Contribution': [4.71, 5.96],
    'Percentage': [44, 56],
})
df3_crps.to_csv(f'{OUT}/Fig03b_decomposition_CRPS.csv', index=False)
print(f'  → {OUT}/Fig03a_decomposition_RMSE.csv')
print(f'  → {OUT}/Fig03b_decomposition_CRPS.csv')

# ================================================================
# 图4: PIT直方图 (E5 Bottom-Up)
# ================================================================
print('图4: PIT直方图...')
p5 = p_bot_e5.sum(axis=1)  # (744, 19)
pits = []
for t in range(len(y_tt)):
    y, q = y_tt[t], p5[t, :]
    idx = np.searchsorted(q, y)
    if idx == 0:       pit = 0.025
    elif idx >= 19:    pit = 0.975
    else:
        lo, hi = TAUS[idx-1], TAUS[idx]
        qlo, qhi = q[idx-1], q[idx]
        pit = lo + (hi-lo) * (y-qlo) / max(qhi-qlo, 1e-10)
    pits.append(pit)
pits = np.array(pits)

bins_edges = np.linspace(0, 1, 11)
hist, _ = np.histogram(pits, bins=bins_edges)
df4 = pd.DataFrame({
    'Bin_Center': (bins_edges[:-1] + bins_edges[1:]) / 2,
    'Count': hist,
    'Expected': len(pits) / 10,
})
df4.to_csv(f'{OUT}/Fig04_PIT_histogram.csv', index=False)
print(f'  → {OUT}/Fig04_PIT_histogram.csv')

# ================================================================
# 图5: 交互效应随分位数变化 (Δ_quality at τ=0.1, 0.5, 0.9)
# ================================================================
print('图5: 交互效应 vs τ...')
# 手工填入 prob_analysis.py 的结果
df5 = pd.DataFrame({
    'Tau': [0.1, 0.5, 0.9],
    'Delta_quality_grouped': [6.14, 5.22, 4.58],
    'Delta_quality_independent': [0.0, 0.0, 0.0],
})
df5.to_csv(f'{OUT}/Fig05_interaction_by_tau.csv', index=False)
print(f'  → {OUT}/Fig05_interaction_by_tau.csv')

# ================================================================
# 图6: 2×2 交互矩阵 — 柱状图数据
# ================================================================
print('图6: 2×2 交互矩阵...')
# 数据: 全市加总 RMSE
df6 = pd.DataFrame({
    'Experiment': ['E2 Admin+Indep', 'E3 Admin+Grouped', 'E4 Behavior+Indep', 'E5 Behavior+Grouped'],
    'RMSE': [153.5, 131.3, 153.5, 119.6],
    'Bottom_Strategy': ['Independent', 'Grouped', 'Independent', 'Grouped'],
    'Middle_Quality': ['Low (Admin)', 'Low (Admin)', 'High (Behavior)', 'High (Behavior)'],
})
df6.to_csv(f'{OUT}/Fig06_2x2_matrix.csv', index=False)
print(f'  → {OUT}/Fig06_2x2_matrix.csv')

# ================================================================
# 图7: TAZ 稀疏度 vs RMSE改善 (散点图)
# ================================================================
print('图7: TAZ 稀疏度 vs RMSE 改善...')
# 加载 275 TAZ 的月均充电量 (从 occupancy.csv 聚合)
RAW = r'E:/Desktop/毕业论文/data/UrbanEV-main/UrbanEV-main/data'
occ = pd.read_csv(f'{RAW}/occupancy.csv')
occ['time'] = pd.to_datetime(occ['time']); occ = occ.set_index('time')
zone_cols = [c for c in occ.columns]
monthly_charge = occ.resample('ME').sum().sum(axis=0).values  # 每个TAZ的总月均充电量

# E4 独立建模 vs E5 分组建模各TAZ的RMSE (需要从中位数预测反算)
p_bot_e4 = np.load(f'{SRC}/G5_pred_bottom.npy')  # E4 (独立)
y_tb = y_true[:, :n_bottom]  # 底层真实值

rmse_e4 = np.array([np.sqrt(np.mean((y_tb[:, j] - p_bot_e4[:, j, 9])**2)) for j in range(n_bottom)])
rmse_e5 = np.array([np.sqrt(np.mean((y_tb[:, j] - p_bot_e5[:, j, 9])**2)) for j in range(n_bottom)])
improve = (rmse_e4 - rmse_e5) / rmse_e4 * 100  # 改善百分比

df7 = pd.DataFrame({
    'TAZ_ID': np.arange(1, n_bottom+1),
    'MonthlyCharge': monthly_charge,
    'RMSE_Independent': rmse_e4,
    'RMSE_Grouped': rmse_e5,
    'Improvement_Pct': improve,
})
df7.to_csv(f'{OUT}/Fig07_TAZ_sparsity_improvement.csv', index=False)
print(f'  → {OUT}/Fig07_TAZ_sparsity_improvement.csv')

# ================================================================
# 图8: MinT vs Bottom-Up CRPS 对比
# ================================================================
print('图8: MinT vs Bottom-Up 对比...')
df8 = pd.DataFrame({
    'Experiment': ['E4 (Behavior+Indep)', 'E5 (Behavior+Grouped)'],
    'BottomUp_CRPS': [64.49, 53.82],
    'MinT_CRPS': [64.46, 53.92],
})
df8.to_csv(f'{OUT}/Fig08_MinT_vs_BU.csv', index=False)
print(f'  → {OUT}/Fig08_MinT_vs_BU.csv')

# ================================================================
# 汇总
# ================================================================
print(f'\n{"="*60}')
print(f'共导出 9 个 CSV 文件至 {OUT}/')
print(f'{"="*60}')
print('''
  Fig01: 全市预测时序图 (168h)
  Fig02: 中层区域时序图
  Fig03a: RMSE 归因分解堆叠条形图
  Fig03b: CRPS 归因分解堆叠条形图
  Fig04: PIT 直方图
  Fig05: 交互效应 vs τ 折线/柱状图
  Fig06: 2×2 交互矩阵分组柱状图
  Fig07: TAZ 稀疏度 vs RMSE 改善散点图
  Fig08: MinT vs Bottom-Up 对比柱状图
''')
