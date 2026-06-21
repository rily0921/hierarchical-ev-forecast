"""
步骤 2：构建三套中层层次结构
  1. 行政层次 (8 区)    — 已有，直接加载 district_mapping.json
  2. 空间聚类 (10 组)    — AgglomerativeClustering on TAZ 坐标
  3. 行为聚类 (10 组)    — K-Means on 充电行为特征

输出：
  S_admin.npy      (284, 275)  — 8 行政区
  S_spatial.npy    (286, 275)  — 10 空间组
  S_behavior.npy   (286, 275)  — 10 行为组
  hierarchy_meta.json         — 三套层次的元信息
"""
import pandas as pd
import numpy as np
import json, os, warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler

# ========== 路径 ==========
RAW   = r'E:/Desktop/毕业论文/data/UrbanEV-main/UrbanEV-main/data'
ADMIN = r'E:/Desktop/毕业论文/code/chapter3/data/processed/district_mapping.json'
OUT   = r'E:/Desktop/毕业论文/code/output'

os.makedirs(OUT, exist_ok=True)

# ========== 加载数据 ==========
print("Loading data...")
occ = pd.read_csv(f'{RAW}/occupancy.csv')
occ['time'] = pd.to_datetime(occ['time'])
occ = occ.set_index('time')

inf = pd.read_csv(f'{RAW}/inf.csv')
zone_cols = [c for c in occ.columns]
n_bottom = len(zone_cols)                # 275

# ========== 层次 1: 行政层次 (8 区) ==========
print("\n--- Hierarchy 1: Administrative (8 districts) ---")
with open(ADMIN) as f:
    dmap = json.load(f)
zone_to_admin = dmap['zone_to_district']
admin_names = dmap['district_names']
n_admin = len(admin_names)

S_admin = np.zeros((n_bottom + n_admin + 1, n_bottom))
S_admin[:n_bottom, :] = np.eye(n_bottom)
for d_idx, dname in enumerate(admin_names):
    for j, zid in enumerate(zone_cols):
        if str(zid) in zone_to_admin and zone_to_admin[str(zid)] == dname:
            S_admin[n_bottom + d_idx, j] = 1
S_admin[-1, :] = 1
assert (S_admin.sum(axis=1) > 0).all(), "S_admin has empty row!"

admin_sizes = [int(S_admin[n_bottom + d, :].sum()) for d in range(n_admin)]
print(f"S_admin: {S_admin.shape}  n_middle={n_admin}")
for i, (dn, sz) in enumerate(zip(admin_names, admin_sizes)):
    print(f"  {dn}: {sz} TAZs")

# ========== 层次 2: 空间聚类 (10 组) ==========
print("\n--- Hierarchy 2: Spatial Clustering (10 groups) ---")
taz_coords = inf.groupby('TAZID')[['longitude', 'latitude']].mean()
taz_coords = taz_coords[taz_coords.index.isin([int(z) for z in zone_cols])]
coords_arr = taz_coords[['longitude', 'latitude']].values

spatial_clust = AgglomerativeClustering(n_clusters=10, linkage='ward')
spatial_labels = spatial_clust.fit_predict(coords_arr)
spatial_groups = dict(zip(taz_coords.index.astype(str), spatial_labels))

n_spatial = 10
S_spatial = np.zeros((n_bottom + n_spatial + 1, n_bottom))
S_spatial[:n_bottom, :] = np.eye(n_bottom)
for j, zid in enumerate(zone_cols):
    S_spatial[n_bottom + spatial_groups[zid], j] = 1
S_spatial[-1, :] = 1
assert (S_spatial.sum(axis=1) > 0).all(), "S_spatial has empty row!"

spatial_sizes = [int(S_spatial[n_bottom + d, :].sum()) for d in range(n_spatial)]
print(f"S_spatial: {S_spatial.shape}  n_middle={n_spatial}")
print(f"  TAZ per group: {spatial_sizes}")

# ========== 层次 3: 行为聚类 (10 组) ==========
print("\n--- Hierarchy 3: Behavioral Clustering (10 groups) ---")
# 构造行为特征：工作日/周末 × 24h profile + 总量 + 变异性
occ_bottom = occ[zone_cols].values
is_weekday = occ.index.dayofweek < 5

profile_wd = occ.loc[is_weekday].groupby(occ.loc[is_weekday].index.hour).mean()   # 24 × 275
profile_we = occ.loc[~is_weekday].groupby(occ.loc[~is_weekday].index.hour).mean()  # 24 × 275

total_vol   = occ.sum(axis=0).values.reshape(-1, 1)
mean_vol    = occ.mean(axis=0).values.reshape(-1, 1)
nonzero_pct = (occ > 0).sum(axis=0).values.reshape(-1, 1) / len(occ)
cv_hourly   = (occ.std(axis=0) / (occ.mean(axis=0) + 1e-6)).values.reshape(-1, 1)

features = np.hstack([
    profile_wd.T.values,   # 275 × 24
    profile_we.T.values,   # 275 × 24
    total_vol, nonzero_pct, cv_hourly,  # 275 × 3
])
print(f"  Feature dim: {features.shape}")
features_std = StandardScaler().fit_transform(features)

behavior_clust = KMeans(n_clusters=10, random_state=42, n_init=20, max_iter=500)
behavior_labels = behavior_clust.fit_predict(features_std)
behavior_groups = dict(zip(zone_cols, behavior_labels))

n_behavior = 10
S_behavior = np.zeros((n_bottom + n_behavior + 1, n_bottom))
S_behavior[:n_bottom, :] = np.eye(n_bottom)
for j, zid in enumerate(zone_cols):
    S_behavior[n_bottom + behavior_groups[zid], j] = 1
S_behavior[-1, :] = 1
assert (S_behavior.sum(axis=1) > 0).all(), "S_behavior has empty row!"

behavior_sizes = [int(S_behavior[n_bottom + d, :].sum()) for d in range(n_behavior)]
print(f"S_behavior: {S_behavior.shape}  n_middle={n_behavior}")
print(f"  TAZ per group: {behavior_sizes}")

# ========== 簇内同质性计算 ==========
print("\n--- Intra-cluster Correlation ---")
def avg_intra_corr(data_wide, groups):
    """簇内 TAZ 对的平均 Pearson 相关系数"""
    cors = []
    for g in set(groups.values()):
        members = [z for z, grp in groups.items() if grp == g]
        if len(members) > 1:
            corr_mat = data_wide[members].corr().values
            upper = corr_mat[np.triu_indices_from(corr_mat, k=1)]
            cors.append(np.mean(upper))
    return np.mean(cors)

corr_admin   = avg_intra_corr(occ, zone_to_admin)
corr_spatial = avg_intra_corr(occ, spatial_groups)
corr_behavior = avg_intra_corr(occ, behavior_groups)

print(f"  Admin intra-correlation:     {corr_admin:.4f}")
print(f"  Spatial intra-correlation:   {corr_spatial:.4f}")
print(f"  Behavioral intra-correlation: {corr_behavior:.4f}")

# ========== 保存 ==========
print("\n--- Saving ---")
np.save(f'{OUT}/S_admin.npy', S_admin)
np.save(f'{OUT}/S_spatial.npy', S_spatial)
np.save(f'{OUT}/S_behavior.npy', S_behavior)

meta = {
    'n_bottom': n_bottom,
    'hierarchies': {
        'admin': {
            'n_middle': n_admin,
            'n_total': n_bottom + n_admin + 1,
            'district_names': admin_names,
            'district_sizes': admin_sizes,
            'intra_corr': round(corr_admin, 4),
            'description': '8 administrative districts of Shenzhen'
        },
        'spatial': {
            'n_middle': n_spatial,
            'n_total': n_bottom + n_spatial + 1,
            'district_sizes': spatial_sizes,
            'intra_corr': round(corr_spatial, 4),
            'description': '10 spatial clusters (Ward on coordinates)'
        },
        'behavior': {
            'n_middle': n_behavior,
            'n_total': n_bottom + n_behavior + 1,
            'district_sizes': behavior_sizes,
            'intra_corr': round(corr_behavior, 4),
            'description': '10 behavioral clusters (K-Means on charging features)'
        }
    }
}
with open(f'{OUT}/hierarchy_meta.json', 'w') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print(f"Heirarchy meta saved to {OUT}/hierarchy_meta.json")
print("[Done] All 3 hierarchies built.")
