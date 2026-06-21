"""
优化2+3: 预测导向聚类 + 软分配 S 矩阵
- 在聚类特征中加入内在可预测性特征 (谱熵/峰度/自相关等)
- 用软分配替代硬分配, 扫描温度参数 tau
"""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, json, time, warnings, os
warnings.filterwarnings('ignore')
os.environ['LGB_VERBOSITY'] = '-1'

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import kurtosis
from scipy.spatial.distance import cdist
from src.data.loader import DataLoader
from src.data.features import FeatureBuilder
from src.models.lgbm_quantile import IndependentQuantileTrainer
from src.models.grouped_lgbm import GroupedQuantileTrainer
from src.reconciliation.mint import MinTShrink
from src.evaluation.metrics import point_metrics, reconciliation_gain

OUT = 'output'
RAW = 'E:/Desktop/毕业论文/data/UrbanEV-main/UrbanEV-main/data'
K_OPT = 12  # 用 K-sweep 的最优值
MIN_SIZE = 5
TAUS_MED = [0.5]
n_b = 275

LGB_BOTTOM = dict(n_estimators=300, learning_rate=0.08, num_leaves=31,
                  min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                  reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1, n_jobs=4)
LGB_TOP = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
               min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
               reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1, n_jobs=4)

# ---- 加载数据 ----
print('Loading data...')
loader = DataLoader(RAW); data = loader.load()
fb = FeatureBuilder(data['global_feat'])
occ_bot = data['occ_bottom']
n_train = 2880 - 168; n_val = 720; n_test = data['T_test']

occ = pd.read_csv(RAW + '/occupancy.csv')
occ['time'] = pd.to_datetime(occ['time']); occ = occ.set_index('time')

# ---- 构建特征: 行为 + 内在可预测性 ----
print('Building features (behavioral + intrinsic predictability)...')

# 行为特征 (现有, 51维)
is_wd = occ.index.dayofweek < 5
p_wd = occ.loc[is_wd].groupby(occ.loc[is_wd].index.hour).mean()
p_we = occ.loc[~is_wd].groupby(occ.loc[~is_wd].index.hour).mean()
total_vol = occ.sum(axis=0).values
mean_vol = occ.mean(axis=0).values
nonzero_pct = (occ > 0).sum(axis=0).values / len(occ)
cv_hourly = occ.std(axis=0).values / (occ.mean(axis=0).values + 1e-6)

# 内在可预测性特征 (新增, 每种一个值 per TAZ)
print('  Computing intrinsic predictability features...')
vals = occ.values  # (4344, 275)

# 谱熵 (频域能量分散度)
n_fft = len(vals) // 2 + 1
spectral_entropy = np.zeros(n_b)
for j in range(n_b):
    fft = np.abs(np.fft.rfft(vals[:, j])) ** 2
    fft_norm = fft / (fft.sum() + 1e-12)
    spectral_entropy[j] = -np.sum(fft_norm * np.log(fft_norm + 1e-12))
# 归一化到 [0,1]
spectral_entropy = spectral_entropy / np.log(n_fft)

# lag-1 自相关
lag1_acf = np.array([np.corrcoef(vals[1:, j], vals[:-1, j])[0, 1]
                     for j in range(n_b)])
lag1_acf = np.nan_to_num(lag1_acf, 0)

# 峰度
kurt = np.array([kurtosis(vals[:, j]) for j in range(n_b)])
kurt = np.nan_to_num(kurt, 0)

# 周末/工作日差异 (归一化)
wd_mean = occ.loc[is_wd].mean(axis=0).values
we_mean = occ.loc[~is_wd].mean(axis=0).values
wd_we_diff = np.abs(wd_mean - we_mean) / (mean_vol + 1e-6)

# 活跃小时比例 (充电>0 的小时)
active_ratio = (vals > 0).mean(axis=0)

# 组装特征
feat_behavior = np.column_stack([p_wd.T.values, p_we.T.values,
                                  total_vol.reshape(-1,1),
                                  nonzero_pct.reshape(-1,1),
                                  cv_hourly.reshape(-1,1)])  # 51维

feat_intrinsic = np.column_stack([
    spectral_entropy.reshape(-1,1),   # 谱熵
    lag1_acf.reshape(-1,1),           # 自相关
    kurt.reshape(-1,1),               # 峰度
    wd_we_diff.reshape(-1,1),         # 工作日/周末差异
    active_ratio.reshape(-1,1),       # 活跃比例
])  # 5维

feat_all = np.hstack([feat_behavior, feat_intrinsic])  # 56维
feat_all_std = StandardScaler().fit_transform(feat_all)

print(f'  Behavioral: {feat_behavior.shape[1]}d, Intrinsic: {feat_intrinsic.shape[1]}d')
print(f'  Total: {feat_all_std.shape[1]}d')

# ---- 合并小组的函数 ----
def merge_small(labels, features_std, min_size=MIN_SIZE):
    labels = np.array(labels); unique, counts = np.unique(labels, return_counts=True)
    centroids = np.array([features_std[labels == g].mean(axis=0) for g in unique])
    small = [g for g, c in zip(unique, counts) if c < min_size]
    if not small:
        return labels
    large = [g for g, c in zip(unique, counts) if c >= min_size]
    merged = labels.copy()
    for sg in small:
        si = list(unique).index(sg)
        li = [list(unique).index(lg) for lg in large]
        nearest = large[np.argmin(cdist([centroids[si]], centroids[li])[0])]
        merged[labels == sg] = nearest
    new_u = np.unique(merged)
    remap = {old: new for new, old in enumerate(new_u)}
    return np.array([remap[g] for g in merged])


def run_pipeline(name, labels, K, S_mid=None):
    """
    labels: 分组标签
    K: 中层节点数
    S_mid: 如果提供, 使用软分配的 S 中层矩阵 (K x n_b);
           如果 None, 使用硬分配
    """
    t0 = time.time()

    # 构建 S 矩阵
    S = np.zeros((n_b + K + 1, n_b))
    S[:n_b, :] = np.eye(n_b)
    if S_mid is not None:
        # 软分配
        S[n_b:n_b+K, :] = S_mid
    else:
        # 硬分配
        for d in range(K):
            S[n_b + d, np.where(labels == d)[0]] = 1
    S[-1, :] = 1

    # 聚合中层 (软分配用加权求和)
    occ_mid = np.zeros((len(occ_bot), K))
    for d in range(K):
        row = S[n_b + d, :]
        if S_mid is not None:
            occ_mid[:, d] = occ_bot @ row  # 加权求和
        else:
            occ_mid[:, d] = occ_bot[:, row == 1].sum(axis=1)
    occ_top = occ_bot.sum(axis=1)

    # 分组建模底层 (底层的分组用硬分配 labels)
    group_members = [np.where(labels == g)[0].tolist() for g in range(K)]
    trainer_g = GroupedQuantileTrainer(TAUS_MED, LGB_BOTTOM, embed_dim=4, early_stop=30)
    p_bot_val, p_bot = trainer_g.train_all_groups(
        group_members, occ_bot, fb, n_train, n_val, n_jobs_parallel=min(K, 10)
    )

    # 中层独立建模
    trainer_i = IndependentQuantileTrainer(TAUS_MED, LGB_TOP, early_stop=30)
    p_mid_val = np.zeros((n_val, K, 1)); p_mid = np.zeros((n_test, K, 1))
    for d in range(K):
        p_mid_val[:, d, :], p_mid[:, d, :] = trainer_i.train(occ_mid[:, d], fb, n_train, n_val)

    # 顶层
    p_top_val, p_top = trainer_i.train(occ_top, fb, n_train, n_val)

    # 基预测
    y_hat = np.column_stack([p_bot[:, :, 0], p_mid[:, :, 0], p_top[:, 0]])
    y_hat_v = np.column_stack([p_bot_val[:, :, 0], p_mid_val[:, :, 0], p_top_val[:, 0]])
    y_val_full = np.column_stack([occ_bot[2880:3600, :], occ_mid[2880:3600, :], occ_top[2880:3600]])

    # 调和
    rec = MinTShrink(S); rec.fit(y_hat_v - y_val_full)
    y_r = rec.reconcile(y_hat)
    city_rec = y_r[:, :n_b].sum(axis=1)

    y_top_test = occ_bot[3600:, :].sum(axis=1)
    m_bu = point_metrics(y_top_test, p_bot[:, :, 0].sum(axis=1))
    m_city = point_metrics(y_top_test, city_rec)
    gain = reconciliation_gain(m_bu['rmse'], m_city['rmse'])

    elapsed = time.time() - t0
    return m_bu, m_city, gain, elapsed


# ---- 主实验 ----
results = []

# 1. Baseline: 行为特征 + 硬分配 K=12
print('\n' + '=' * 60)
print('Experiment: Hard vs Soft S matrix, Behavioral vs Enhanced features')
print('=' * 60)

kmeans = KMeans(n_clusters=K_OPT, random_state=42, n_init=20)
labels_base = kmeans.fit_predict(feat_all_std)
labels_base = merge_small(labels_base, feat_all_std)
K_base = len(np.unique(labels_base))

print('\n[1] Hard S, Behavioral features only, K={}'.format(K_base))
m_bu, m_city, gain, et = run_pipeline('hard-behav', labels_base, K_base)
print('  BU={:.1f}, MinT={:.1f}, Delta={:+.1f}% ({:.0f}s)'.format(
    m_bu['rmse'], m_city['rmse'], gain['delta_rmse_pct'], et))
results.append({'name': 'Hard-Behav', 'BU': m_bu['rmse'], 'MinT': m_city['rmse'],
                'delta': gain['delta_rmse_pct'], 'time': et})

# 2. 软分配 S, 扫描 tau (基于合并后的分组计算组中心)
print('\n[2] Soft S, scanning tau...')
# 计算合并后各组的中心
K_merged = len(np.unique(labels_base))
centroids_merged = np.array([feat_all_std[labels_base == g].mean(axis=0) for g in range(K_merged)])
dist_merged = cdist(feat_all_std, centroids_merged)  # (275, K_merged)
sigma_d = np.std(dist_merged)

for tau_factor in [0.05, 0.1, 0.2, 0.5, 1.0]:
    tau = tau_factor * sigma_d
    weights = np.exp(-dist_merged / tau)
    S_soft_mid = (weights / weights.sum(axis=1, keepdims=True)).T  # (K_merged, n_b)

    m_bu, m_city, gain, et = run_pipeline(
        'soft-tau{:.2f}'.format(tau_factor), labels_base, K_merged, S_mid=S_soft_mid
    )
    print('  tau={:.3f} ({}*sigma): BU={:.1f}, MinT={:.1f}, Delta={:+.1f}% ({:.0f}s)'.format(
        tau, tau_factor, m_bu['rmse'], m_city['rmse'], gain['delta_rmse_pct'], et))
    results.append({'name': 'Soft-tau{:.2f}'.format(tau_factor),
                    'BU': m_bu['rmse'], 'MinT': m_city['rmse'],
                    'delta': gain['delta_rmse_pct'], 'time': et})

# 3. 增强特征 (行为+可预测性) + 硬分配
print('\n[3] Hard S, Enhanced features (behav+intrinsic), K={}'.format(K_OPT))
# 用仅预测性特征的增强聚类? 已经在用 feat_all_std
# 这里对比的是 仅行为 vs 行为+可预测
kmeans_behav = KMeans(n_clusters=K_OPT, random_state=42, n_init=20)
feat_behav_std = StandardScaler().fit_transform(feat_behavior)
labels_behav = kmeans_behav.fit_predict(feat_behav_std)
labels_behav = merge_small(labels_behav, feat_behav_std)
K_behav = len(np.unique(labels_behav))

m_bu, m_city, gain, et = run_pipeline('hard-enhanced', labels_base, K_base)
print('  Enhanced: BU={:.1f}, MinT={:.1f}, Delta={:+.1f}% ({:.0f}s)'.format(
    m_bu['rmse'], m_city['rmse'], gain['delta_rmse_pct'], et))
results.append({'name': 'Hard-Enhanced', 'BU': m_bu['rmse'], 'MinT': m_city['rmse'],
                'delta': gain['delta_rmse_pct'], 'time': et})

# 对比: 仅行为特征的硬分配
m_bu, m_city, gain, et = run_pipeline('hard-behav-only', labels_behav, K_behav)
print('  Behav-only: BU={:.1f}, MinT={:.1f}, Delta={:+.1f}% ({:.0f}s)'.format(
    m_bu['rmse'], m_city['rmse'], gain['delta_rmse_pct'], et))
results.append({'name': 'Hard-BehavOnly', 'BU': m_bu['rmse'], 'MinT': m_city['rmse'],
                'delta': gain['delta_rmse_pct'], 'time': et})

# 汇总
print('\n' + '=' * 60)
print('SUMMARY')
print('=' * 60)
print('  Method              BU RMSE  MinT RMSE   Delta%')
print('  ' + '-' * 52)
for r in results:
    print('  {:<20} {:>7.1f}  {:>9.1f}  {:>+6.1f}%'.format(
        r['name'], r['BU'], r['MinT'], r['delta']))

# 保存
with open(OUT + '/soft_clustering_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved to', OUT + '/soft_clustering_results.json')
