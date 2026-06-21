"""
K值扫描: 找出行为聚类的最优组数
对 K in {5,6,7,8,9,10,12,15}, 扫描验证集 RMSE
"""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, json, time, warnings, os
warnings.filterwarnings('ignore')
os.environ['LGB_VERBOSITY'] = '-1'

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from src.data.loader import DataLoader
from src.data.features import FeatureBuilder
from src.models.lgbm_quantile import IndependentQuantileTrainer
from src.models.grouped_lgbm import GroupedQuantileTrainer
from src.evaluation.metrics import point_metrics

OUT = 'output'
RAW = 'E:/Desktop/毕业论文/data/UrbanEV-main/UrbanEV-main/data'

# ---- 加载数据 ----
print('Loading data...')
loader = DataLoader(RAW)
data = loader.load()
fb = FeatureBuilder(data['global_feat'])
occ_bot = data['occ_bottom']
n_train = 2880 - 168
n_val = 720
n_test = data['T_test']
n_b = 275

# ---- 行为特征 ----
occ = pd.read_csv(RAW + '/occupancy.csv')
occ['time'] = pd.to_datetime(occ['time'])
occ = occ.set_index('time')
is_wd = occ.index.dayofweek < 5
p_wd = occ.loc[is_wd].groupby(occ.loc[is_wd].index.hour).mean()
p_we = occ.loc[~is_wd].groupby(occ.loc[~is_wd].index.hour).mean()
total_vol = occ.sum(axis=0).values.reshape(-1, 1)
nonzero_pct = (occ > 0).sum(axis=0).values.reshape(-1, 1) / len(occ)
cv_hourly = (occ.std(axis=0) / (occ.mean(axis=0) + 1e-6)).values.reshape(-1, 1)
features = np.hstack([p_wd.T.values, p_we.T.values, total_vol, nonzero_pct, cv_hourly])
features_std = StandardScaler().fit_transform(features)

# ---- 模型参数 (仅 tau=0.5 加速) ----
TAUS_MED = [0.5]
LGB_BOTTOM = dict(n_estimators=300, learning_rate=0.08, num_leaves=31,
                  min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                  reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1, n_jobs=4)
LGB_TOP = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
               min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
               reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1, n_jobs=4)

# ---- K 值扫描 ----
K_LIST = [5, 6, 7, 8, 9, 10, 12, 15]

print()
print('=' * 60)
print('K-Value Sweep: K in', K_LIST)
print('Training with tau=0.5 for speed...')
print('=' * 60)
print('   K     Groups  BotRMSE  CityBU   Time')
print('-' * 45)

results = []
for K in K_LIST:
    t0 = time.time()

    # 1. K-Means
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=20, max_iter=500)
    labels = kmeans.fit_predict(features_std)
    group_sizes = [int((labels == d).sum()) for d in range(K)]
    sz_str = ','.join(str(s) for s in group_sizes[:5])
    if K > 5:
        sz_str += ',...'

    # 2. S 矩阵
    S = np.zeros((n_b + K + 1, n_b))
    S[:n_b, :] = np.eye(n_b)
    for d in range(K):
        S[n_b + d, np.where(labels == d)[0]] = 1
    S[-1, :] = 1

    # 3. 聚合中层和顶层
    occ_mid = np.zeros((len(occ_bot), K))
    for d in range(K):
        occ_mid[:, d] = occ_bot[:, labels == d].sum(axis=1)
    occ_top = occ_bot.sum(axis=1)

    # 4. 分组建模底层
    group_members = [np.where(labels == g)[0].tolist() for g in range(K)]
    trainer_g = GroupedQuantileTrainer(TAUS_MED, LGB_BOTTOM, taz_categorical=True, early_stop=30)
    p_bot_val_g, p_bot_g = trainer_g.train_all_groups(
        group_members, occ_bot, fb, n_train, n_val, n_jobs_parallel=min(K, 10)
    )

    # 5. 中层独立建模
    trainer_i = IndependentQuantileTrainer(TAUS_MED, LGB_TOP, early_stop=30)
    p_mid_val = np.zeros((n_val, K, 1))
    p_mid = np.zeros((n_test, K, 1))
    for d in range(K):
        p_mid_val[:, d, :], p_mid[:, d, :] = trainer_i.train(
            occ_mid[:, d], fb, n_train, n_val
        )

    # 6. 顶层
    p_top_val, p_top = trainer_i.train(occ_top, fb, n_train, n_val)

    # 7. 评估
    y_bot_val = occ_bot[2880:3600, :]
    m_bot = point_metrics(y_bot_val, p_bot_val_g[:, :, 0])

    bu_sum = p_bot_g[:, :, 0].sum(axis=1)
    y_top_test = occ_bot[3600:, :].sum(axis=1)
    m_city = point_metrics(y_top_test, bu_sum)

    elapsed = time.time() - t0
    print(f'  {K:>2}   {sz_str:>8}   {m_bot["rmse"]:>6.2f}   {m_city["rmse"]:>6.1f}   {elapsed:>4.0f}s')

    results.append({
        'K': K, 'groups': group_sizes,
        'bot_rmse_val': round(m_bot['rmse'], 3),
        'bot_mae_val': round(m_bot['mae'], 3),
        'city_rmse_test': round(m_city['rmse'], 1),
        'time_s': round(elapsed, 1),
    })

# ---- 最优 K ----
best = min(results, key=lambda r: r['bot_rmse_val'])
print()
print('Best K =', best['K'], '(Bot Val RMSE =', best['bot_rmse_val'], ')')

with open(OUT + '/k_sweep_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Results saved to', OUT + '/k_sweep_results.json')
