"""
导出: 数据描述图 + 预测对比表 (v2)
"""
import numpy as np, pandas as pd, os, warnings
warnings.filterwarnings('ignore')

OUT = r'E:/Desktop/毕业论文/code/output/figures'
SRC = r'E:/Desktop/毕业论文/code/output'
RAW = r'E:/Desktop/毕业论文/data/UrbanEV-main/UrbanEV-main/data'
os.makedirs(OUT, exist_ok=True)

N_BOTTOM = 275

def metrics(yt, yp):
    err = (yt - yp).flatten()
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    denom = np.where(np.abs(yt.flatten()) > 1e-6, np.abs(yt.flatten()), np.nan)
    mape = float(np.nanmean(np.abs(err) / denom) * 100)
    return rmse, mae, mape

# ================================================================
# A. 数据描述图 (不变)
# ================================================================
print('=== A. 数据描述图 ===')
occ = pd.read_csv(f'{RAW}/occupancy.csv')
occ['time'] = pd.to_datetime(occ['time']); occ = occ.set_index('time')
zone_cols = [c for c in occ.columns]

# A1
total_by_zone = occ[zone_cols].sum(axis=0).sort_values(ascending=False)
cum_pct = total_by_zone.cumsum() / total_by_zone.sum() * 100
pd.DataFrame({
    'TAZ_Rank': range(1, N_BOTTOM+1),
    'Total_Occupancy': total_by_zone.values,
    'Cumulative_Pct': cum_pct.values,
    'Top20_Flag': ['Top 20%' if i <= 55 else 'Bottom 80%' for i in range(1, N_BOTTOM+1)],
}).to_csv(f'{OUT}/FigA1_TAZ_distribution.csv', index=False)
print('  A1 done')

# A2
occ['hour'], occ['is_wend'] = occ.index.hour, (occ.index.dayofweek >= 5).astype(int)
wd = occ[occ['is_wend']==0].groupby('hour')[zone_cols].mean().mean(axis=1)
we = occ[occ['is_wend']==1].groupby('hour')[zone_cols].mean().mean(axis=1)
pd.DataFrame({'Hour': range(24), 'Weekday': wd.values, 'Weekend': we.values}).to_csv(f'{OUT}/FigA2_daily_profile.csv', index=False)
print('  A2 done')

# A3
sw = occ.iloc[168*4:168*5][zone_cols].mean(axis=1).values
pd.DataFrame({'Hour_in_Week': range(168), 'Occupancy': sw}).to_csv(f'{OUT}/FigA3_weekly_profile.csv', index=False)
print('  A3 done')

# A4
occ['month'] = occ.index.month
mv = occ.groupby('month')[zone_cols].mean().mean(axis=1)
ms = occ.groupby('month')[zone_cols].mean().std(axis=1)
pd.DataFrame({
    'Month': [9,10,11,12,1,2], 'Label': ['Sep','Oct','Nov','Dec','Jan','Feb'],
    'Mean': mv.values, 'Std': ms.values,
}).to_csv(f'{OUT}/FigA4_monthly.csv', index=False)
print('  A4 done')

# A5
pd.DataFrame({
    'Hierarchy': ['Admin (8)', 'Spatial KM (10)', 'Behavioral W (10)'],
    'Correlation': [0.138, 0.191, 0.304],
}).to_csv(f'{OUT}/FigA5_middle_corr.csv', index=False)
print('  A5 done')

# ================================================================
# B. 预测对比表
# ================================================================
print('\n=== B. 预测对比表 ===')
label_map = {'E1':'G1','E2':'G2','E3':'G4','E4':'G5','E5':'G7','E6':'G8'}
y_true_all = np.load(f'{SRC}/G7_y_true.npy')

# B1: E1-E6 核心表
rows_b1 = []
for e, g in label_map.items():
    try:
        p_b = np.load(f'{SRC}/{g}_pred_bottom.npy')[:,:,9]
        p_m = np.load(f'{SRC}/{g}_pred_middle.npy')[:,:,9]
        p_t = np.load(f'{SRC}/{g}_pred_top.npy')[:,9]
    except FileNotFoundError:
        continue

    n_mid_exp = p_m.shape[1]
    y_tm = y_true_all[:, N_BOTTOM:N_BOTTOM+n_mid_exp]

    rmse_b, mae_b, mape_b = metrics(y_true_all[:,:N_BOTTOM], p_b)
    rmse_m, mae_m, mape_m = metrics(y_tm, p_m)
    rmse_t, mae_t, mape_t = metrics(y_true_all[:,-1], p_t)
    bu = p_b.sum(axis=1)
    rmse_bu, _, _ = metrics(y_true_all[:,-1], bu)

    try:
        y_rec = np.load(f'{SRC}/{g}_y_rec_shrink.npy')
        rc = y_rec[:,:N_BOTTOM].sum(axis=1)
        rmse_rec, _, _ = metrics(y_true_all[:,-1], rc)
        imp = round((rmse_bu-rmse_rec)/rmse_bu*100,1)
    except:
        rmse_rec, imp = np.nan, np.nan

    rows_b1.append({
        'Exp': e, 'Bottom_RMSE': round(rmse_b,2), 'Bottom_MAE': round(mae_b,2),
        'Bottom_MAPE': round(mape_b,1), 'Middle_RMSE': round(rmse_m,1),
        'Middle_MAE': round(mae_m,1), 'Middle_MAPE': round(mape_m,1),
        'Top_RMSE': round(rmse_t,1), 'City_BU_RMSE': round(rmse_bu,1),
        'City_MinT_RMSE': round(rmse_rec,1) if not np.isnan(rmse_rec) else '—',
        'Improve_Pct': imp if not np.isnan(imp) else '—',
    })

pd.DataFrame(rows_b1).to_csv(f'{OUT}/TableB1_all_exp.csv', index=False, encoding='utf-8-sig')
print(f'  B1: {len(rows_b1)} experiments')

# B2: E5 验证集 vs 测试集
p_bot_e5 = np.load(f'{SRC}/G7_pred_bottom.npy')
p_val_e5 = np.load(f'{SRC}/G7_val_pred_bottom.npy')
y_val = np.load(f'{SRC}/G7_y_val_true.npy')
rmse_v, mae_v, mape_v = metrics(y_val[:,-1], p_val_e5[:,:,9].sum(axis=1))
rmse_t, mae_t, mape_t = metrics(y_true_all[:,-1], p_bot_e5[:,:,9].sum(axis=1))
pd.DataFrame([
    {'Set':'Validation (Jan)','RMSE':round(rmse_v,1),'MAE':round(mae_v,1),'MAPE':round(mape_v,1)},
    {'Set':'Test (Feb)','RMSE':round(rmse_t,1),'MAE':round(mae_t,1),'MAPE':round(mape_t,1)},
]).to_csv(f'{OUT}/TableB2_E5_sets.csv', index=False, encoding='utf-8-sig')
print('  B2 done')

# B3: 稀疏度分档
p_bot_e4 = np.load(f'{SRC}/G5_pred_bottom.npy')[:,:,9]
monthly = occ[zone_cols].resample('ME').sum().sum(axis=0).values
tiers = pd.cut(monthly, bins=np.percentile(monthly,[0,33,67,100]),
               labels=['Low','Medium','High'], include_lowest=True)
rows_b3 = []
for tier in ['Low','Medium','High']:
    idx = np.where(tiers == tier)[0]
    r4, _, _ = metrics(y_true_all[:,:N_BOTTOM][:,idx], p_bot_e4[:,idx])
    r5, _, _ = metrics(y_true_all[:,:N_BOTTOM][:,idx], p_bot_e5[:,idx,9])
    rows_b3.append({
        'Tier': tier, 'N': len(idx), 'Range': f'{monthly[idx].min():.0f}-{monthly[idx].max():.0f}',
        'E4_RMSE': round(r4,2), 'E5_RMSE': round(r5,2),
        'Improve_Pct': round((r4-r5)/r4*100,1),
    })
pd.DataFrame(rows_b3).to_csv(f'{OUT}/TableB3_sparsity.csv', index=False, encoding='utf-8-sig')
print('  B3 done')

# B4: 概率预测汇总
pd.DataFrame([
    {'Metric':'PIT Mean (E5)','Value':0.537,'Note':'ideal 0.500'},
    {'Metric':'PIT Std (E5)','Value':0.092,'Note':'ideal 0.289'},
    {'Metric':'90% Coverage (MinT+Gauss)','Value':0.918,'Note':'target 0.90'},
    {'Metric':'90% Coverage (Bootstrap)','Value':0.766,'Note':'target 0.90'},
    {'Metric':'CRPS E2 (Admin+Indep)','Value':64.49,'Note':''},
    {'Metric':'CRPS E3 (Admin+Grouped)','Value':59.79,'Note':''},
    {'Metric':'CRPS E4 (Behav+Indep)','Value':64.49,'Note':''},
    {'Metric':'CRPS E5 (Behav+Grouped)','Value':53.82,'Note':'Best'},
]).to_csv(f'{OUT}/TableB4_prob.csv', index=False, encoding='utf-8-sig')
print('  B4 done')

# B5: 消融汇总
pd.DataFrame([
    {'Component':'Grouped Modeling','Ablation':'E5->E4','RMSE_Change':'119.6->153.5','Contribution':33.9},
    {'Component':'Behavioral Clustering','Ablation':'E5->E3','RMSE_Change':'119.6->131.3','Contribution':11.7},
    {'Component':'Three-Level Structure','Ablation':'E5->E1','RMSE_Change':'119.6->153.5','Contribution':33.9},
    {'Component':'Quantile-Specific Cov','Ablation':'E6->E5','RMSE_Change':'~same','Contribution':'~0'},
]).to_csv(f'{OUT}/TableB5_ablation.csv', index=False, encoding='utf-8-sig')
print('  B5 done')

print(f'\nDone: 5 数据图 + 5 对比表 -> {OUT}/')
