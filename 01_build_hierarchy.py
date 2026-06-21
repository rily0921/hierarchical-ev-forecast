"""
步骤 1：构建两套三层层次结构（空间行政层次 & 行为聚类层次）
输出：两套 S 矩阵（286 × 275）
- 275 个底层 TAZ + 10 个中层区域 + 1 个全市顶层 = 286 节点
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist

# ========== 1. 加载数据 ==========
data_dir = r"E:\Desktop\毕业论文\data\UrbanEV-main\UrbanEV-main\data"

inf  = pd.read_csv(f"{data_dir}/inf.csv")
vol  = pd.read_csv(f"{data_dir}/volume.csv", index_col=0)
vol.index = pd.to_datetime(vol.index)

taz_ids = sorted(vol.columns.astype(int).tolist())
print(f"底层 TAZ 数量: {len(taz_ids)}")

# TAZ 坐标
inf_u = inf.groupby("TAZID").agg(lon=("longitude", "mean"), lat=("latitude", "mean"))
inf_u = inf_u[inf_u.index.isin(taz_ids)]
coord = inf_u  # TAZID → (lon, lat)

# ========== 2. 空间行政层次：K-Means 坐标聚类 → 10 个空间组 ==========
# （在没有行政区 shapefile 时，用坐标聚类近似空间行政区划）
coords_std = StandardScaler().fit_transform(coord[["lon", "lat"]])
kmeans_spatial = KMeans(n_clusters=10, random_state=42, n_init=10)
spatial_labels = kmeans_spatial.fit_predict(coords_std)

spatial_groups = {taz: int(spatial_labels[i]) + 1 for i, taz in enumerate(coord.index)}
spatial_counts = pd.Series(spatial_groups).value_counts().sort_index()
print("\n[空间层次] 各组 TAZ 数量:")
print(spatial_counts)

# ========== 3. 行为聚类层次：丰富特征 + Ward 聚类 → 10 个簇 ==========
# 特征构建：不只 24h 均值，加入工作日/周末差异、总量、变异性
vol_t = vol.T  # 275 TAZ × N 小时

# 3a. 工作日 vs 周末的 24h profile
is_weekday = vol.index.dayofweek < 5
vol_wd = vol.loc[is_weekday]    # 工作日
vol_we = vol.loc[~is_weekday]   # 周末

profile_wd = vol_wd.groupby(vol_wd.index.hour).mean().T  # 275 × 24
profile_we = vol_we.groupby(vol_we.index.hour).mean().T  # 275 × 24

# 3b. 总量特征
total_vol  = vol.sum(axis=0).values.reshape(-1, 1)          # 275 × 1
mean_vol   = vol.mean(axis=0).values.reshape(-1, 1)         # 275 × 1
nonzero_pct = (vol > 0).sum(axis=0).values.reshape(-1, 1) / len(vol)  # 275 × 1

# 3c. 变异性特征
cv_hourly = (vol.std(axis=0) / (vol.mean(axis=0) + 1e-6)).values.reshape(-1, 1)  # 275 × 1

# 拼接所有特征
features = np.hstack([
    profile_wd.values,     # 24
    profile_we.values,     # 24
    total_vol,             # 1
    mean_vol,              # 1
    nonzero_pct,           # 1
    cv_hourly,             # 1
])
print(f"\n聚类特征维度: {features.shape}")  # (275, 52)

features_std = StandardScaler().fit_transform(features)

# K-Means 聚类 → 10 个行为簇（自然分布，不做强制平衡）
# 注：充电行为天然存在主导模式和少量离群模式，这是数据特征
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=500)
cluster_labels = kmeans.fit_predict(features_std)

cluster_groups = {taz: int(cluster_labels[i]) + 1 for i, taz in enumerate(taz_ids)}
cluster_counts = pd.Series(cluster_groups).value_counts().sort_index()
print("\n[聚类层次] 各组 TAZ 数量:")
print(cluster_counts)

# ========== 4. 构建 S 矩阵（286 × 275）==========
def build_S_matrix(taz_ids, groups, n_mid):
    n_bottom = len(taz_ids)
    n_total = n_bottom + n_mid + 1
    S = np.zeros((n_total, n_bottom))
    S[:n_bottom, :] = np.eye(n_bottom)                   # 底层对角
    for j, taz in enumerate(taz_ids):
        S[n_bottom + groups[taz] - 1, j] = 1             # 中层
    S[-1, :] = 1                                          # 顶层
    return S

S_admin   = build_S_matrix(taz_ids, spatial_groups, n_mid=10)
S_cluster = build_S_matrix(taz_ids, cluster_groups, n_mid=10)

print(f"\nS_spatial 形状:  {S_admin.shape}")
print(f"S_cluster 形状:  {S_cluster.shape}")

# ========== 5. 簇内相关系数（用于 4.2 的 "质量效应" 论证）==========
def avg_intra_corr(vol, groups):
    """计算簇内 TAZ 时间序列的平均 Pearson 相关系数"""
    cors = []
    for g in set(groups.values()):
        members = [t for t, grp in groups.items() if grp == g]
        if len(members) > 1:
            sub = vol[[str(m) for m in members]]
            corr_mat = sub.corr().values
            # 上三角均值（排除对角线）
            upper = corr_mat[np.triu_indices_from(corr_mat, k=1)]
            cors.append(np.mean(upper))
    return np.mean(cors)

corr_spatial = avg_intra_corr(vol, spatial_groups)
corr_cluster = avg_intra_corr(vol, cluster_groups)
print(f"\n簇内平均相关系数:")
print(f"  空间层次: {corr_spatial:.4f}")
print(f"  聚类层次: {corr_cluster:.4f}")

# ========== 6. 保存 ==========
import os
out_dir = r"E:\Desktop\毕业论文\code\output"
os.makedirs(out_dir, exist_ok=True)

np.save(f"{out_dir}/S_admin.npy",    S_admin)
np.save(f"{out_dir}/S_cluster.npy",  S_cluster)
np.save(f"{out_dir}/taz_ids.npy",    np.array(taz_ids))
pd.Series(spatial_groups, name="spatial_group").to_csv(f"{out_dir}/admin_groups.csv")
pd.Series(cluster_groups, name="cluster_group").to_csv(f"{out_dir}/cluster_groups.csv")

print("\n[Done] S matrices saved to code/output/")
