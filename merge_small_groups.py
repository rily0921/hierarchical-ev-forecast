"""
改进3: 单节点组合并 — 将<MIN_SIZE的组自动并入特征距离最近的大组
对比合并前后的 K=10 聚类效果
"""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, json, time, warnings, os
warnings.filterwarnings('ignore')
os.environ['LGB_VERBOSITY'] = '-1'

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from src.data.loader import DataLoader
from src.data.features import FeatureBuilder
from src.models.lgbm_quantile import IndependentQuantileTrainer
from src.models.grouped_lgbm import GroupedQuantileTrainer
from src.evaluation.metrics import point_metrics

OUT = 'output'
RAW = 'E:/Desktop/毕业论文/data/UrbanEV-main/UrbanEV-main/data'
MIN_SIZE = 5  # 最小组大小阈值

# ---- 加载数据 ----
print('Loading data...')
loader = DataLoader(RAW); data = loader.load()
fb = FeatureBuilder(data['global_feat'])
occ_bot = data['occ_bottom']
n_train = 2880 - 168; n_val = 720; n_test = data['T_test']; n_b = 275

occ = pd.read_csv(RAW + '/occupancy.csv')
occ['time'] = pd.to_datetime(occ['time']); occ = occ.set_index('time')
is_wd = occ.index.dayofweek < 5
p_wd = occ.loc[is_wd].groupby(occ.loc[is_wd].index.hour).mean()
p_we = occ.loc[~is_wd].groupby(occ.loc[~is_wd].index.hour).mean()
total_vol = occ.sum(axis=0).values.reshape(-1, 1)
nonzero_pct = (occ > 0).sum(axis=0).values.reshape(-1, 1) / len(occ)
cv_hourly = (occ.std(axis=0) / (occ.mean(axis=0) + 1e-6)).values.reshape(-1, 1)
features = np.hstack([p_wd.T.values, p_we.T.values, total_vol, nonzero_pct, cv_hourly])
features_std = StandardScaler().fit_transform(features)

# ---- 模型参数 (仅 tau=0.5) ----
TAUS_MED = [0.5]
LGB_BOTTOM = dict(n_estimators=300, learning_rate=0.08, num_leaves=31,
                  min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                  reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1, n_jobs=4)
LGB_TOP = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
               min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
               reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1, n_jobs=4)


def merge_small_groups(labels, features_std, min_size=MIN_SIZE):
    """
    将小于 min_size 的组合并到特征距离最近的大组
    返回: merged_labels (长度 n_b, 值 0..K'-1)
    """
    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)
    K = len(unique)

    # 计算聚类中心
    centroids = np.array([features_std[labels == g].mean(axis=0) for g in unique])

    # 找小组和大组
    small_groups = [g for g, c in zip(unique, counts) if c < min_size]
    large_groups = [g for g, c in zip(unique, counts) if c >= min_size]

    if not small_groups:
        return labels  # 没有小组, 无需合并

    print(f'  Small groups (size < {min_size}): {len(small_groups)}')
    for g in small_groups:
        c = counts[list(unique).index(g)]
        print(f'    Group {g}: {c} TAZs')

    # 合并: 每个小组找最近的大组
    merged = labels.copy()
    for sg in small_groups:
        sg_idx = list(unique).index(sg)
        # 到各大组的中心距离
        lg_indices = [list(unique).index(lg) for lg in large_groups]
        dists = cdist([centroids[sg_idx]], centroids[lg_indices])[0]
        nearest_lg = large_groups[np.argmin(dists)]
        merged[labels == sg] = nearest_lg
        print(f'    -> merged into Group {nearest_lg} (dist={np.min(dists):.2f})')

    # 重新编号为 0..K'-1
    new_unique = np.unique(merged)
    remap = {old: new for new, old in enumerate(new_unique)}
    merged = np.array([remap[g] for g in merged])

    new_sizes = [int((merged == g).sum()) for g in range(len(new_unique))]
    print(f'  After merge: {len(new_unique)} groups, sizes={new_sizes}')
    return merged


def run_experiment(name, labels, K):
    """用给定分组标签跑完整实验管线"""
    t0 = time.time()

    S = np.zeros((n_b + K + 1, n_b))
    S[:n_b, :] = np.eye(n_b)
    for d in range(K):
        S[n_b + d, np.where(labels == d)[0]] = 1
    S[-1, :] = 1

    occ_mid = np.zeros((len(occ_bot), K))
    for d in range(K):
        occ_mid[:, d] = occ_bot[:, labels == d].sum(axis=1)
    occ_top = occ_bot.sum(axis=1)

    group_members = [np.where(labels == g)[0].tolist() for g in range(K)]
    trainer_g = GroupedQuantileTrainer(TAUS_MED, LGB_BOTTOM, taz_categorical=True, early_stop=30)
    p_bot_val, p_bot = trainer_g.train_all_groups(
        group_members, occ_bot, fb, n_train, n_val, n_jobs_parallel=min(K, 10)
    )

    trainer_i = IndependentQuantileTrainer(TAUS_MED, LGB_TOP, early_stop=30)
    p_mid_val = np.zeros((n_val, K, 1)); p_mid = np.zeros((n_test, K, 1))
    for d in range(K):
        p_mid_val[:, d, :], p_mid[:, d, :] = trainer_i.train(occ_mid[:, d], fb, n_train, n_val)
    p_top_val, p_top = trainer_i.train(occ_top, fb, n_train, n_val)

    m_bot = point_metrics(occ_bot[2880:3600, :], p_bot_val[:, :, 0])
    bu_sum = p_bot[:, :, 0].sum(axis=1)
    m_city = point_metrics(occ_bot[3600:, :].sum(axis=1), bu_sum)

    elapsed = time.time() - t0
    group_sizes = [len(m) for m in group_members]
    print(f'  {name}: K={K}, sizes={group_sizes}')
    print(f'    Bot Val RMSE={m_bot["rmse"]:.3f}, City BU RMSE={m_city["rmse"]:.1f}, Time={elapsed:.0f}s')
    return m_bot, m_city


# ---- 主流程 ----
print('\n' + '=' * 60)
print('Small Group Merge Experiment (MIN_SIZE = {})'.format(MIN_SIZE))
print('=' * 60)

# 原始 KMeans K=10 (当前论文使用)
kmeans10 = KMeans(n_clusters=10, random_state=42, n_init=20)
labels_orig = kmeans10.fit_predict(features_std)
print('\n[Original KMeans K=10]')
m_bot_orig, m_city_orig = run_experiment('Original', labels_orig, 10)

# 合并后
print('\n[Merged KMeans K=10]')
labels_merged = merge_small_groups(labels_orig, features_std, MIN_SIZE)
K_merged = len(np.unique(labels_merged))
m_bot_merged, m_city_merged = run_experiment('Merged', labels_merged, K_merged)

# 直接 KMeans K=8 做对比 (K=10 并掉 2 个小组后≈8-9 组)
print('\n[Direct KMeans K=8 (for comparison)]')
kmeans8 = KMeans(n_clusters=8, random_state=42, n_init=20)
labels_k8 = kmeans8.fit_predict(features_std)
m_bot_k8, m_city_k8 = run_experiment('KMeans-K8', labels_k8, 8)

# 汇总
print('\n' + '=' * 60)
print('SUMMARY')
print('=' * 60)
print('  Original K=10     Bot RMSE={0:.3f}  City BU={1:.1f}'.format(
    m_bot_orig['rmse'], m_city_orig['rmse']))
print('  Merged  K={0}       Bot RMSE={1:.3f}  City BU={2:.1f}'.format(
    K_merged, m_bot_merged['rmse'], m_city_merged['rmse']))
print('  Direct  K=8       Bot RMSE={0:.3f}  City BU={1:.1f}'.format(
    m_bot_k8['rmse'], m_city_k8['rmse']))
