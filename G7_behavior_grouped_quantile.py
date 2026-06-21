"""
G7: 行为聚类层次 + 分组建模(10个模型) + 标准MinT + 多τ分位数回归
─────────────────────────────────────────────────
核心差异 vs G5:
  底层建模策略: 独立(275个模型) → 分组(10个模型, 每组内TAZ共享参数 + TAZ embedding)
  每组内的TAZ使用 categorical embedding (dim=4) 标识个体身份
输出: output/G7_*.npy + output/G7_results.json
"""
import pandas as pd, numpy as np, json, os, time, warnings
warnings.filterwarnings('ignore')
os.environ['LGB_VERBOSITY'] = '-1'
import lightgbm as lgb
from joblib import Parallel, delayed

# ========== 参数 ==========
RAW     = r'E:/Desktop/毕业论文/data/UrbanEV-main/UrbanEV-main/data'
S_BEHAV = r'E:/Desktop/毕业论文/code/output/S_behavior.npy'
OUT     = r'E:/Desktop/毕业论文/code/output'
os.makedirs(OUT, exist_ok=True)

TAUS = np.arange(0.05, 1.00, 0.05)
N_JOBS_LGB = 4; N_JOBS_PARALLEL = 10  # 10 clusters can run in parallel
EMBED_DIM = 4                            # TAZ embedding dimension
LAG_HOURS = [1, 2, 3, 24, 48, 168]; ROLL_HOURS = [6, 24]
MAX_LAG = 168; EARLY_STOP = 30; M_SCENARIOS = 500

LGB_GROUPED = dict(n_estimators=300, learning_rate=0.08, num_leaves=31,
                   min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                   reg_alpha=0.1, reg_lambda=0.1, random_state=42,
                   verbose=-1, n_jobs=N_JOBS_LGB)
LGB_TOP = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
               min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
               reg_alpha=0.1, reg_lambda=0.1, random_state=42,
               verbose=-1, n_jobs=N_JOBS_LGB)

# ========== 1. 加载数据 ==========
print('='*60)
print('G7: Behavior + Grouped(10) + Standard MinT + Quantile')
print(f'Parallel: {N_JOBS_PARALLEL} clusters')
print('='*60)

occ = pd.read_csv(f'{RAW}/occupancy.csv')
occ['time'] = pd.to_datetime(occ['time']); occ = occ.set_index('time')
zone_cols = [c for c in occ.columns]; n_bottom = len(zone_cols)

S = np.load(S_BEHAV)
n_middle = S.shape[0] - n_bottom - 1  # 10

# 提取每个 cluster 的 TAZ 成员 (0-indexed column indices)
cluster_members = []
for d in range(n_middle):
    members = np.where(S[n_bottom + d, :] == 1)[0].tolist()
    cluster_members.append(members)
    print(f'  Cluster {d}: {len(members)} TAZs (indices {members[:3]}...)')

# 聚合层级数据
occ_bottom = occ[zone_cols].values
occ_middle = np.zeros((len(occ), n_middle))
for d in range(n_middle):
    occ_middle[:, d] = occ_bottom[:, cluster_members[d]].sum(axis=1)
occ_top = occ_bottom.sum(axis=1)

T_total = len(occ); T_train = 2880; T_val = 720

# 时间 & 天气特征 (与 G2、G5 完全一致)
time_idx = occ.index
time_feat = pd.DataFrame({
    'hour_sin': np.sin(2*np.pi*time_idx.hour/24), 'hour_cos': np.cos(2*np.pi*time_idx.hour/24),
    'wday_sin': np.sin(2*np.pi*time_idx.dayofweek/7), 'wday_cos': np.cos(2*np.pi*time_idx.dayofweek/7),
    'is_wend': (time_idx.dayofweek >= 5).astype(float), 'month': time_idx.month,
}, index=time_idx)
wc = pd.read_csv(f'{RAW}/weather_central.csv')
wc['time'] = pd.to_datetime(wc['time'], format='%Y/%m/%d %H:%M'); wc = wc.set_index('time')
wa = pd.read_csv(f'{RAW}/weather_airport.csv')
wa['time'] = pd.to_datetime(wa['time'], format='%Y/%m/%d %H:%M'); wa = wa.set_index('time')
weather_feat = pd.DataFrame({
    'T': (wc['T']+wa['T'])/2, 'U': (wc['U']+wa['U'])/2,
    'P': (wc['P']+wa['P'])/2, 'nRAIN': wc['nRAIN'],
}, index=occ.index)
global_feat = pd.concat([time_feat, weather_feat], axis=1).values  # (4344, 10)

# ========== 2. 特征构建 ==========
nt = T_train - MAX_LAG; nv = T_val

def build_features(ts):
    """对单一 TAZ 时间序列构造特征矩阵 (不含 TAZ ID)"""
    T = len(ts); X = []
    for t in range(MAX_LAG, T):
        f = list(global_feat[t, :])
        for lag in LAG_HOURS: f.append(ts[t-lag])
        for roll in ROLL_HOURS:
            f.append(np.mean(ts[t-roll:t])); f.append(np.std(ts[t-roll:t]))
        X.append(f)
    return np.array(X, dtype=np.float32), ts[MAX_LAG:].astype(np.float32)

# ========== 3. 分组建模训练 ==========
# Top & Middle: 独立建模 (与 G5 一致)
print('\n--- Top ---')
def train_single(ts, lgb_params):
    X_t, y_t = build_features(ts)
    ntest = len(X_t) - nt - nv
    p_val = np.zeros((nv, len(TAUS)))
    p = np.zeros((ntest, len(TAUS)))
    cb = lgb.early_stopping(EARLY_STOP)
    for i, tau in enumerate(TAUS):
        m = lgb.LGBMRegressor(objective='quantile', alpha=tau, **lgb_params)
        m.fit(X_t[:nt], y_t[:nt],
              eval_set=[(X_t[nt:nt+nv], y_t[nt:nt+nv])], callbacks=[cb])
        p_val[:, i] = m.predict(X_t[nt:nt+nv])
        p[:, i] = m.predict(X_t[nt+nv:])
    p_val.sort(axis=1)
    p.sort(axis=1)
    return p_val, p

p_top_val, p_top = train_single(occ_top, LGB_TOP)
p_mid_val = np.zeros((p_top_val.shape[0], n_middle, len(TAUS)))
p_mid = np.zeros((p_top.shape[0], n_middle, len(TAUS)))
for d in range(n_middle):
    p_mid_val[:, d, :], p_mid[:, d, :] = train_single(occ_middle[:, d], LGB_TOP)

# Bottom: 分组建模 (10个模型, TAZ embedding)
print(f'\n--- Bottom: Grouped Modeling (10 clusters) ---')
t0 = time.time()

def train_one_cluster(d):
    """训练一个 cluster 的 19τ 模型 (组内 TAZ 共享参数 + embedding)"""
    members = cluster_members[d]
    n_mem = len(members)

    if n_mem == 1:
        # 单 TAZ cluster: 等价于独立建模, 无需 embedding
        p_cluster_val, p_cluster = train_single(occ_bottom[:, members[0]], LGB_GROUPED)
        return p_cluster_val[:, np.newaxis, :], p_cluster[:, np.newaxis, :]  # (nval/ntest, 1, 19)

    # 构建训练数据: 组内所有 TAZ 拼接, 加上 TAZ ID
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list = []

    for local_id, j in enumerate(members):
        X_all, y_all = build_features(occ_bottom[:, j])
        taz_id_col = np.full((len(X_all), 1), local_id, dtype=np.float32)

        # 训练集
        X_train_list.append(np.hstack([X_all[:nt], taz_id_col[:nt]]))
        y_train_list.append(y_all[:nt])
        # 验证集
        X_val_list.append(np.hstack([X_all[nt:nt+nv], taz_id_col[nt:nt+nv]]))
        y_val_list.append(y_all[nt:nt+nv])
        # 测试集 (稍后用于预测, 每个 TAZ 分别)
        ntest = len(X_all) - nt - nv
        X_test_list.append(np.hstack([X_all[nt+nv:], taz_id_col[nt+nv:]]))

    X_train = np.vstack(X_train_list).astype(np.float32)
    y_train = np.hstack(y_train_list).astype(np.float32)
    X_val   = np.vstack(X_val_list).astype(np.float32)
    y_val   = np.hstack(y_val_list).astype(np.float32)

    # 训练 19τ
    cb = lgb.early_stopping(EARLY_STOP)
    p_cluster_val = np.zeros((nv, n_mem, len(TAUS)))
    p_cluster = np.zeros((p_top.shape[0], n_mem, len(TAUS)))
    for i, tau in enumerate(TAUS):
        m = lgb.LGBMRegressor(
            objective='quantile', alpha=tau,
            categorical_feature=[X_train.shape[1]-1],  # TAZ ID 在最后一列
            **{k:v for k,v in LGB_GROUPED.items() if k != 'categorical_feature'}
        )
        m.fit(X_train, y_train,
              eval_set=[(X_val, y_val)], callbacks=[cb])

        # 预测: 每个 TAZ 分别输入其 TAZ ID
        for local_id, j in enumerate(members):
            X_v = X_val_list[local_id]
            X_t = X_test_list[local_id]
            p_cluster_val[:, local_id, i] = m.predict(X_v)
            p_cluster[:, local_id, i] = m.predict(X_t)

    p_cluster.sort(axis=2)  # 单调校正 (对每个 TAZ)
    p_cluster_val.sort(axis=2)
    return p_cluster_val, p_cluster

# 并行训练 10 个 cluster
results = Parallel(n_jobs=N_JOBS_PARALLEL, verbose=10)(
    delayed(train_one_cluster)(d) for d in range(n_middle)
)

# 将 10 个 cluster 的结果按 TAZ 原始顺序拼接回 (ntest, 275, 19)
p_bot_val = np.zeros((p_top_val.shape[0], n_bottom, len(TAUS)))
p_bot = np.zeros((p_top.shape[0], n_bottom, len(TAUS)))
for d, (p_cluster_val, p_cluster) in enumerate(results):
    for local_id, j in enumerate(cluster_members[d]):
        p_bot_val[:, j, :] = p_cluster_val[:, local_id, :]
        p_bot[:, j, :] = p_cluster[:, local_id, :]

print(f'Bottom done in {time.time()-t0:.0f}s')

# ========== 4. 场景生成 ==========
print('\n--- Scenario Generation ---')
def quantiles_to_scenarios(q_pred):
    T = q_pred.shape[0]; u = np.random.RandomState(42).uniform(0, 1, (T, M_SCENARIOS))
    sc = np.array([np.interp(u[t,:], TAUS, q_pred[t,:]) for t in range(T)])
    return np.maximum(sc, 0)

sc_bot = np.array([[quantiles_to_scenarios(p_bot[:, j, :]) for j in range(n_bottom)]])
sc_bot = sc_bot.reshape(p_bot.shape[0], n_bottom, M_SCENARIOS)

# ========== 5. 标准 MinT 调和 ==========
print('\n--- Standard MinT Reconciliation ---')
val_start = T_train
test_start = T_train + T_val
y_vb = occ_bottom[val_start:test_start, :]; y_vm = occ_middle[val_start:test_start, :]; y_vt = occ_top[val_start:test_start]
y_tb = occ_bottom[test_start:, :]; y_tm = occ_middle[test_start:, :]; y_tt = occ_top[test_start:]

p_bot_val_med = p_bot_val[:, :, 9]; p_mid_val_med = p_mid_val[:, :, 9]; p_top_val_med = p_top_val[:, 9]
p_bot_med = p_bot[:, :, 9]; p_mid_med = p_mid[:, :, 9]; p_top_med = p_top[:, 9]

y_hat_val = np.column_stack([p_bot_val_med, p_mid_val_med, p_top_val_med])
y_true_val = np.column_stack([y_vb, y_vm, y_vt.reshape(-1, 1)])
y_hat = np.column_stack([p_bot_med, p_mid_med, p_top_med])
y_true_all = np.column_stack([y_tb, y_tm, y_tt.reshape(-1, 1)])

residuals = y_hat_val - y_true_val
pre_inc = float(np.mean(np.abs(p_bot_med.sum(axis=1) - p_top_med)))

from sklearn.covariance import LedoitWolf
W_shrink = LedoitWolf().fit(residuals).covariance_
W_diag = np.diag(np.var(residuals, axis=0))

def G_mat(S, W):
    Wi = np.linalg.inv(W + np.eye(W.shape[0])*1e-8)
    return np.linalg.inv(S.T @ Wi @ S) @ S.T @ Wi

# MinT-shrink (已有)
G_shrink = G_mat(S, W_shrink)
y_rec_shrink = (S @ G_shrink @ y_hat.T).T

# MinT-diag (新增)
G_diag = G_mat(S, W_diag)
y_rec_diag = (S @ G_diag @ y_hat.T).T

post_inc_shrink = float(np.mean(np.abs(y_rec_shrink[:, :n_bottom].sum(axis=1) - y_rec_shrink[:, -1])))
post_inc_diag = float(np.mean(np.abs(y_rec_diag[:, :n_bottom].sum(axis=1) - y_rec_diag[:, -1])))

# ========== 6. 评估 ==========
print('\n--- Metrics ---')

def metrics(yt, yp):
    err = yt - yp; rmse = np.sqrt(np.mean(err**2)); mae = np.mean(np.abs(err))
    d = np.where(np.abs(yt)>1e-6, np.abs(yt), np.nan)
    mape = np.nanmean(np.abs(err)/d)*100
    return rmse, mae, mape

def qloss(yt, yp_tau, tau):
    err = yt - yp_tau
    return float(np.mean(np.where(err>=0, tau*err, (tau-1)*err)))

m_bot = metrics(y_tb.flatten(), p_bot_med.flatten())
m_mid = metrics(y_tm.flatten(), p_mid_med.flatten())
m_top = metrics(y_tt, p_top_med)

bu_sum = p_bot_med.sum(axis=1)
m_city_before = metrics(y_tt, bu_sum)
m_city_after_shrink = metrics(y_tt, y_rec_shrink[:, :n_bottom].sum(axis=1))
m_city_after_diag = metrics(y_tt, y_rec_diag[:, :n_bottom].sum(axis=1))

ql = {f'τ={t:.1f}': qloss(y_tb.flatten(), p_bot[:,:,idx].flatten(), t)
      for t, idx in [(0.1,1), (0.5,9), (0.9,17)]}

# ========== 7. 保存 ==========
from collections import Counter
cluster_sizes = [len(m) for m in cluster_members]

imp_shrink = round((m_city_before[0]-m_city_after_shrink[0])/m_city_before[0]*100, 1)
imp_diag = round((m_city_before[0]-m_city_after_diag[0])/m_city_before[0]*100, 1)

results_out = {
    'experiment': 'G7',
    'description': 'Behavior + Grouped(10 models, TAZ embedding) + MinT-shrink + MinT-diag + Multi-τ Quantile',
    'covariance_source': 'validation_residuals',
    'cluster_sizes': dict(Counter({i: s for i, s in enumerate(cluster_sizes)})),
    'consistency': {
        'before': round(pre_inc, 2),
        'after_shrink': round(post_inc_shrink, 10),
        'after_diag': round(post_inc_diag, 10),
    },
    'per_level': {
        'top':    {'rmse':round(m_top[0],2),'mae':round(m_top[1],2),'mape':round(m_top[2],1)},
        'middle': {'rmse':round(m_mid[0],2),'mae':round(m_mid[1],2),'mape':round(m_mid[2],1)},
        'bottom': {'rmse':round(m_bot[0],2),'mae':round(m_bot[1],2),'mape':round(m_bot[2],1)},
    },
    'city': {
        'rmse_before': round(m_city_before[0],1),
        'rmse_after_shrink': round(m_city_after_shrink[0],1),
        'rmse_after_diag': round(m_city_after_diag[0],1),
        'improvement_shrink_pct': imp_shrink,
        'improvement_diag_pct': imp_diag,
    },
    'quantile_loss_bottom': ql,
}

with open(f'{OUT}/G7_results.json','w') as f:
    json.dump(results_out, f, indent=2, ensure_ascii=False)

for k,v in {'pred_bottom':p_bot, 'pred_middle':p_mid, 'pred_top':p_top,
            'val_pred_bottom':p_bot_val, 'val_pred_middle':p_mid_val, 'val_pred_top':p_top_val,
            'scenarios_bottom':sc_bot, 'y_true':y_true_all, 'y_val_true':y_true_val,
            'y_rec_shrink':y_rec_shrink, 'y_rec_diag':y_rec_diag}.items():
    np.save(f'{OUT}/G7_{k}.npy', v)

print(f'\n{"="*60}')
print(f'G7 Results Summary')
print(f'{"="*60}')
print(f'  Top:     RMSE={m_top[0]:.1f}, MAE={m_top[1]:.1f}, MAPE={m_top[2]:.1f}%')
print(f'  Middle:  RMSE={m_mid[0]:.1f}, MAE={m_mid[1]:.1f}, MAPE={m_mid[2]:.1f}%')
print(f'  Bottom:  RMSE={m_bot[0]:.2f}, MAE={m_bot[1]:.2f}, MAPE={m_bot[2]:.1f}%')
print(f'  City (before): {m_city_before[0]:.0f}')
print(f'  City (shrink): {m_city_after_shrink[0]:.0f} ({imp_shrink:+.1f}%)')
print(f'  City (diag):   {m_city_after_diag[0]:.0f} ({imp_diag:+.1f}%)')
print(f'  Consist (shrink): {pre_inc:.1f} → {post_inc_shrink:.1e}')
print(f'  Consist (diag):   {pre_inc:.1f} → {post_inc_diag:.1e}')
print(f'  QL (τ=0.5): {ql["τ=0.5"]:.4f}')
print(f'\n[Done] Saved to {OUT}/G7_*')

# --- 顶层诊断 ---
import matplotlib.pyplot as plt
# 对比：顶层模型预测 vs 底层加总 vs 真实值（取测试集前 168 小时）
plt.figure(figsize=(12,4))
plt.plot(y_tt[:168], 'k-', label='True', linewidth=1.5)
plt.plot(p_top_med[:168], 'r--', label=f'Top Model (RMSE={m_top[0]:.0f})', alpha=0.7)
plt.plot(bu_sum[:168], 'b--', label=f'Bottom-Up Sum (RMSE={m_city_before[0]:.0f})', alpha=0.7)
plt.legend(); plt.title('City-Level: Top Model vs Bottom-Up Sum')
plt.savefig(f'{OUT}/G7_top_diagnosis.png', dpi=150)
plt.close()


# 顶层模型 vs 简单 baseline（直接用前一小时的值作为预测）
naive_pred = np.roll(y_tt, 1)
naive_pred[0] = naive_pred[1]
m_naive = metrics(y_tt, naive_pred)
print(f'  Naive (lag-1): RMSE={m_naive[0]:.0f}')

