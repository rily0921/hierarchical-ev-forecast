"""
步骤 4：为 3.1 节生成所有描述性统计数字
输出：data_summary.json → 直接用于论文 3.1 节的数字
"""
import pandas as pd, numpy as np, json, os, warnings
warnings.filterwarnings('ignore')

RAW = r'E:/Desktop/毕业论文/data/UrbanEV-main/UrbanEV-main/data'
OUT = r'E:/Desktop/毕业论文/code/output'
os.makedirs(OUT, exist_ok=True)

# ========== 加载数据 ==========
occ = pd.read_csv(f'{RAW}/occupancy.csv')
occ['time'] = pd.to_datetime(occ['time']); occ = occ.set_index('time')

vol = pd.read_csv(f'{RAW}/volume.csv')
vol['time'] = pd.to_datetime(vol['time']); vol = vol.set_index('time')

inf = pd.read_csv(f'{RAW}/inf.csv')

zone_cols = [c for c in occ.columns]
n_taz = len(zone_cols)
n_hours = len(occ)

# ========== 已加载的层次信息 ==========
with open(f'{OUT}/hierarchy_meta.json') as f:
    h_meta = json.load(f)

# ========== 1. 基本时空信息 ==========
basic = {
    'time_start': str(occ.index.min()),
    'time_end': str(occ.index.max()),
    'total_hours': int(n_hours),
    'n_taz': n_taz,
    'n_stations_total': len(inf),
    'n_piles': int(inf['charge_count'].sum()),
    'train_hours': 24 * 30 * 4,   # 2880
    'val_hours':   24 * 30,        # 720
    'test_hours':  n_hours - 24 * 30 * 5,  # 744
    'test_covers_spring_festival': '2023-01-22' in str(occ.index[24*30*5:].values),
}

# ========== 2. 零值比例统计 ==========
zero_pct = (occ == 0).sum(axis=0) / n_hours  # 每个 TAZ 的零值比例
zero_stats = {
    'mean': float(zero_pct.mean()),
    'median': float(zero_pct.median()),
    'min': float(zero_pct.min()),
    'max': float(zero_pct.max()),
    'p25': float(zero_pct.quantile(0.25)),
    'p75': float(zero_pct.quantile(0.75)),
}

# 按零值比例分档统计
bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
zero_bins = {}
for lo, hi in bins:
    cnt = ((zero_pct >= lo) & (zero_pct < hi)).sum()
    zero_bins[f'{lo:.1f}-{hi:.1f}'] = int(cnt)
print(f"零值比例分布: {zero_bins}")

# ========== 3. TAZ 充电量分布 ==========
# 月均充电量
monthly_vol = vol.resample('ME').sum()  # 月度汇总
monthly_avg = monthly_vol.mean(axis=0)   # 各 TAZ 月均充电量
vol_stats = {
    'mean': float(monthly_avg.mean()),
    'median': float(monthly_avg.median()),
    'min': float(monthly_avg.min()),
    'max': float(monthly_avg.max()),
    'p10': float(monthly_avg.quantile(0.10)),
    'p90': float(monthly_avg.quantile(0.90)),
}

# 集中度：前 20% TAZ 贡献了多少充电量
sorted_vol = monthly_avg.sort_values(ascending=False)
top20_pct = sorted_vol.iloc[:int(n_taz * 0.2)].sum() / sorted_vol.sum() * 100
top50_pct = sorted_vol.iloc[:int(n_taz * 0.5)].sum() / sorted_vol.sum() * 100

# ========== 4. 占用率基本统计 ==========
occ_stats = {
    'city_mean': float(occ.values.mean()),
    'city_std': float(occ.values.std()),
    'city_max': float(occ.values.max()),
    'taz_mean_mean': float(occ.mean(axis=0).mean()),
    'taz_mean_median': float(occ.mean(axis=0).median()),
}

# 日周期特征：全市每小时平均值
hourly_avg = occ.mean(axis=1).groupby(occ.index.hour).mean()
peak_hour = int(hourly_avg.idxmax())
peak_val  = float(hourly_avg.max())
valley_hour = int(hourly_avg.idxmin())
valley_val  = float(hourly_avg.min())

# ========== 5. 层次结构表 ==========
hier_table = []
for htype in ['admin', 'spatial', 'behavior']:
    h = h_meta['hierarchies'][htype]
    sizes = h['district_sizes']
    hier_table.append({
        'name': htype,
        'description': h['description'],
        'n_middle': h['n_middle'],
        'n_total': h['n_total'],
        'size_range': f"{min(sizes)} – {max(sizes)}",
        'size_mean': round(np.mean(sizes), 1),
        'intra_corr': h['intra_corr'],
    })

# ========== 6. 汇总保存 ==========
summary = {
    'basic': basic,
    'zero_pct': zero_stats,
    'zero_bins': zero_bins,
    'monthly_vol': vol_stats,
    'concentration': {
        'top20_pct_contribution': round(top20_pct, 1),
        'top50_pct_contribution': round(top50_pct, 1),
    },
    'occupancy': occ_stats,
    'daily_pattern': {
        'peak_hour': peak_hour,
        'peak_val': round(peak_val, 2),
        'valley_hour': valley_hour,
        'valley_val': round(valley_val, 2),
    },
    'hierarchies': hier_table,
}

with open(f'{OUT}/data_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"\n[Done] Saved to {OUT}/data_summary.json")
