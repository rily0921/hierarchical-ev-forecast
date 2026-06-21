"""
步骤 3：聚类方法对比
  对空间聚类和行为聚类各比较 4-5 种方法
  评价标准：簇内相关系数（越高越好）+ 簇大小标准差（越低越平衡）
  输出：对比表 → 支撑论文 2.1 节的方法选择论证
"""
import pandas as pd, numpy as np, json, os, warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform

# ========== 路径 ==========
RAW = r'E:/Desktop/毕业论文/data/UrbanEV-main/UrbanEV-main/data'
OUT = r'E:/Desktop/毕业论文/code/output'
os.makedirs(OUT, exist_ok=True)

# ========== 加载数据 ==========
occ = pd.read_csv(f'{RAW}/occupancy.csv')
occ['time'] = pd.to_datetime(occ['time']); occ = occ.set_index('time')

inf = pd.read_csv(f'{RAW}/inf.csv')
taz_coords = inf.groupby('TAZID')[['longitude', 'latitude']].mean()
taz_coords = taz_coords[taz_coords.index.isin([int(z) for z in occ.columns])]
zone_cols = list(occ.columns)
n_bottom = len(zone_cols)

# ========== 特征构建 ==========
# 空间特征
coords = taz_coords[['longitude', 'latitude']].values
coords_std = StandardScaler().fit_transform(coords)

# 行为特征（同 02 脚本，51维）
is_wd = occ.index.dayofweek < 5
p_wd = occ.loc[is_wd].groupby(occ.loc[is_wd].index.hour).mean()   # 24 × 275
p_we = occ.loc[~is_wd].groupby(occ.loc[~is_wd].index.hour).mean()  # 24 × 275
total_vol = occ.sum(axis=0).values.reshape(-1, 1)
mean_vol  = occ.mean(axis=0).values.reshape(-1, 1)
nonzero_pct = (occ > 0).sum(axis=0).values.reshape(-1, 1) / len(occ)
cv_h = (occ.std(axis=0) / (occ.mean(axis=0) + 1e-6)).values.reshape(-1, 1)

behav = np.hstack([
    p_wd.T.values, p_we.T.values, total_vol, nonzero_pct, cv_h
])
behav_std = StandardScaler().fit_transform(behav)

# ========== 评价函数 ==========
N_CLUSTERS = 10

def evaluate_clustering(labels, occ_wide, zone_cols, method_name):
    """评估一次聚类结果"""
    # 排除噪声点 (DBSCAN 的 -1)
    valid_labels = labels[labels >= 0]
    n_noise = (labels == -1).sum()
    if len(np.unique(valid_labels)) < 2:
        return None

    # 簇内平均相关系数
    groups = dict(zip(zone_cols, labels))
    cors = []
    for g in np.unique(labels):
        if g == -1: continue
        members = [z for z, grp in groups.items() if grp == g]
        if len(members) > 1:
            corr_mat = occ_wide[members].corr().values
            upper = corr_mat[np.triu_indices_from(corr_mat, k=1)]
            cors.append(np.mean(upper))
    avg_corr = np.mean(cors) if cors else 0

    # 簇大小平衡性 (标准差/均值，越小越平衡)
    sizes = [np.sum(labels == g) for g in np.unique(labels) if g != -1]
    cv_size = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 999

    return {
        'method': method_name,
        'n_clusters': len(np.unique(valid_labels)),
        'n_noise': n_noise,
        'intra_corr': round(avg_corr, 4),
        'size_cv': round(cv_size, 2),
        'sizes': sizes
    }

def safe_cluster(func, X, **kwargs):
    """安全调用聚类，返回 labels"""
    try:
        return func.fit_predict(X)
    except Exception as e:
        print(f"    Failed: {e}")
        return np.full(X.shape[0], -1)

results = []

# ================================================
# 一、空间聚类方法对比
# ================================================
print("=" * 60)
print("SPATIAL CLUSTERING METHODS COMPARISON")
print("=" * 60)

spatial_methods = [
    # (name, clusterer, use_coords or use_dist_matrix)
    ('Ward (Agglomerative)', AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='ward'), 'coords'),
    ('Complete Linkage',     AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='complete'), 'coords'),
    ('Average Linkage',      AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='average'), 'coords'),
    ('Single Linkage',       AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='single'), 'coords'),
    ('K-Means',              KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20), 'coords'),
    ('GMM',                  GaussianMixture(n_components=N_CLUSTERS, random_state=42), 'coords'),
    ('Spectral',             SpectralClustering(n_clusters=N_CLUSTERS, random_state=42, affinity='nearest_neighbors'), 'coords'),
]

for name, model, mode in spatial_methods:
    print(f"  {name}...", end=" ")
    if mode == 'dist':
        labels = safe_cluster(model, dist_matrix)
    else:
        labels = safe_cluster(model, coords_std)
    r = evaluate_clustering(labels, occ, zone_cols, name)
    if r:
        r['type'] = 'spatial'
        results.append(r)
        print(f"ok: corr={r['intra_corr']:.4f}, cv={r['size_cv']:.2f}, sizes={r['sizes']}")
    else:
        print(f"failed (all noise or single cluster)")

# ================================================
# 二、行为聚类方法对比
# ================================================
print(f"\n{'='*60}")
print("BEHAVIORAL CLUSTERING METHODS COMPARISON")
print("="*60)

behavior_methods = [
    ('K-Means',              KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20), 'features'),
    ('Ward (Agglomerative)', AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='ward'), 'features'),
    ('Complete Linkage',     AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='complete'), 'features'),
    ('Average Linkage',      AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='average'), 'features'),
    ('GMM',                  GaussianMixture(n_components=N_CLUSTERS, random_state=42), 'features'),
    ('Spectral',             SpectralClustering(n_clusters=N_CLUSTERS, random_state=42, affinity='nearest_neighbors'), 'features'),
]

for name, model, mode in behavior_methods:
    print(f"  {name}...", end=" ")
    labels = safe_cluster(model, behav_std)
    r = evaluate_clustering(labels, occ, zone_cols, name)
    if r:
        r['type'] = 'behavior'
        results.append(r)
        print(f"ok: corr={r['intra_corr']:.4f}, cv={r['size_cv']:.2f}, sizes={r['sizes']}")
    else:
        print("failed")

# ================================================
# 三、汇总对比表
# ================================================
print(f"\n{'='*60}")
print("COMPARISON TABLE")
print("="*60)

df = pd.DataFrame(results)

for ctype in ['spatial', 'behavior']:
    subset = df[df['type'] == ctype].sort_values('intra_corr', ascending=False)
    print(f"\n  [{ctype.upper()}]  sorted by intra_corr (descending)")
    print(f"  {'Method':<25} {'corr':>7} {'size_cv':>8} {'sizes'}")
    print(f"  {'-'*25} {'-'*7} {'-'*8} {'-'*30}")
    for _, row in subset.iterrows():
        print(f"  {row['method']:<25} {row['intra_corr']:>7.4f} {row['size_cv']:>8.2f} {str(row['sizes'])}")

# ================================================
# 四、推荐与保存
# ================================================
print(f"\n{'='*60}")
print("RECOMMENDATIONS")
print("="*60)

for ctype in ['spatial', 'behavior']:
    subset = df[df['type'] == ctype].copy()
    # 综合评分：簇内相关越高越好，size_cv越低越好
    # 归一化后等权加权
    c_max = subset['intra_corr'].max()
    c_min = subset['intra_corr'].min()
    s_max = subset['size_cv'].max()
    s_min = subset['size_cv'].min()

    if c_max > c_min:
        subset['score_corr'] = (subset['intra_corr'] - c_min) / (c_max - c_min)
    else:
        subset['score_corr'] = 1
    if s_max > s_min:
        subset['score_size'] = 1 - (subset['size_cv'] - s_min) / (s_max - s_min)  # 越小越好
    else:
        subset['score_size'] = 1
    subset['score_total'] = (subset['score_corr'] + subset['score_size']) / 2

    best = subset.sort_values('score_total', ascending=False).iloc[0]
    print(f"\n  {ctype.upper()}:")
    print(f"    Best overall: {best['method']}")
    print(f"    Intra-corr:   {best['intra_corr']:.4f}")
    print(f"    Size CV:      {best['size_cv']:.2f}")
    print(f"    Sizes:        {best['sizes']}")

# 保存对比表（numpy int → python int 转换）
out_csv = f'{OUT}/clustering_comparison.csv'
df_display = df.drop(columns=['sizes']).copy()
df_display.to_csv(out_csv, index=False)

# 完整结果存 json
out_json = f'{OUT}/clustering_comparison.json'
records = []
for _, row in df.iterrows():
    r = row.to_dict()
    r['sizes'] = [int(s) for s in r['sizes']]
    records.append(r)
with open(out_json, 'w') as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to {out_csv} and {out_json}")
print("[Done]")
