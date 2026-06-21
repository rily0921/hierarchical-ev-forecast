"""
G5: 行为聚类层次 + 独立建模(275个) + 标准MinT + 多τ分位数回归
─────────────────────────────────────────────────
与 G2 的唯一差异: S_admin → S_behavior
可变因素: 中层结构类型 (行政 vs 行为聚类)
固定因素: 独立建模(275), 标准MinT, 19τ 分位数
输出: output/G5_*.npy + output/G5_results.json
"""
import pandas as pd, numpy as np, json, os, time, warnings
warnings.filterwarnings('ignore')
os.environ['LGB_VERBOSITY'] = '-1'
import lightgbm as lgb
from joblib import Parallel, delayed

# ========== 参数 ==========
RAW       = r'E:/Desktop/毕业论文/data/UrbanEV-main/UrbanEV-main/data'
S_BEHAV   = r'E:/Desktop/毕业论文/code/output/S_behavior.npy'
META      = r'E:/Desktop/毕业论文/code/output/hierarchy_meta.json'
OUT       = r'E:/Desktop/毕业论文/code/output'
os.makedirs(OUT, exist_ok=True)

TAUS = np.arange(0.05, 1.00, 0.05)
N_JOBS_LGB = 4; N_JOBS_PARALLEL = 8
LAG_HOURS = [1, 2, 3, 24, 48, 168]; ROLL_HOURS = [6, 24]
MAX_LAG = 168; EARLY_STOP = 30; M_SCENARIOS = 500

LGB_BOTTOM = dict(n_estimators=300, learning_rate=0.08, num_leaves=31,
                  min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                  reg_alpha=0.1, reg_lambda=0.1, random_state=42,
                  verbose=-1, n_jobs=N_JOBS_LGB)
LGB_TOP = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
               min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
               reg_alpha=0.1, reg_lambda=0.1, random_state=42,
               verbose=-1, n_jobs=N_JOBS_LGB)

# ========== 1. 加载数据 ==========
print('='*60)
print('G5: Behavior + Independent(275) + Standard MinT + Quantile')
print(f'Parallel: {N_JOBS_PARALLEL} proc × {N_JOBS_LGB} threads')
print('='*60)

occ = pd.read_csv(f'{RAW}/occupancy.csv')
occ['time'] = pd.to_datetime(occ['time']); occ = occ.set_index('time')
zone_cols = [c for c in occ.columns]; n_bottom = len(zone_cols)

# 加载 S_behavior
S = np.load(S_BEHAV)
n_middle = S.shape[0] - n_bottom - 1
print(f'S_behavior: {S.shape}, n_middle={n_middle}')

# 聚合层级数据
occ_bottom = occ[zone_cols].values
occ_middle = np.zeros((len(occ), n_middle))
for d in range(n_middle):
    occ_middle[:, d] = occ_bottom[:, S[n_bottom + d, :] == 1].sum(axis=1)
occ_top = occ_bottom.sum(axis=1)

T_total = len(occ); T_train = 2880; T_val = 720

# 时间 & 天气特征
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
global_feat = pd.concat([time_feat, weather_feat], axis=1).values

# ========== 2. 特征构建 & 训练函数 ==========
def build_features(ts):
    T = len(ts); X = []
    for t in range(MAX_LAG, T):
        f = list(global_feat[t, :])
        for lag in LAG_HOURS: f.append(ts[t-lag])
        for roll in ROLL_HOURS:
            f.append(np.mean(ts[t-roll:t])); f.append(np.std(ts[t-roll:t]))
        X.append(f)
    return np.array(X, dtype=np.float32), ts[MAX_LAG:].astype(np.float32)

nt = T_train - MAX_LAG; nv = T_val

def train_one_node(ts, lgb_params):
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

# ========== 3. 训练 ==========
print('\n--- Top & Middle ---')
p_top_val, p_top = train_one_node(occ_top, LGB_TOP)
p_mid_val = np.zeros((p_top_val.shape[0], n_middle, len(TAUS)))
p_mid = np.zeros((p_top.shape[0], n_middle, len(TAUS)))
for d in range(n_middle):
    p_mid_val[:, d, :], p_mid[:, d, :] = train_one_node(occ_middle[:, d], LGB_TOP)
    if d % 3 == 0: print(f'  Cluster {d+1}/{n_middle} done')

print(f'\n--- Bottom (275 TAZs, {N_JOBS_PARALLEL} parallel) ---')
t0 = time.time()
ts_list = [occ_bottom[:, j] for j in range(n_bottom)]
results = Parallel(n_jobs=N_JOBS_PARALLEL, verbose=10)(
    delayed(train_one_node)(ts, LGB_BOTTOM) for ts in ts_list
)
p_bot_val = np.stack([r[0] for r in results], axis=1)
p_bot = np.stack([r[1] for r in results], axis=1)
print(f'Done in {time.time()-t0:.0f}s')

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

def G_mat(S, W):
    Wi = np.linalg.inv(W + np.eye(W.shape[0])*1e-8)
    return np.linalg.inv(S.T @ Wi @ S) @ S.T @ Wi

G = G_mat(S, W_shrink)
y_rec = (S @ G @ y_hat.T).T
post_inc = float(np.mean(np.abs(y_rec[:, :n_bottom].sum(axis=1) - y_rec[:, -1])))

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
m_city_after  = metrics(y_tt, y_rec[:, :n_bottom].sum(axis=1))

ql = {f'τ={t:.1f}': qloss(y_tb.flatten(), p_bot[:,:,idx].flatten(), t)
      for t, idx in [(0.1,1), (0.5,9), (0.9,17)]}

# ========== 7. 保存 ==========
results = {
    'experiment': 'G5',
    'description': 'Behavioral hierarchy + Independent(275) + Standard MinT + Multi-τ Quantile',
    'covariance_source': 'validation_residuals',
    'S_shape': list(S.shape), 'n_middle': n_middle,
    'consistency': {'before': round(pre_inc,2), 'after': round(post_inc,10)},
    'per_level': {
        'top':    {'rmse':round(m_top[0],2),'mae':round(m_top[1],2),'mape':round(m_top[2],1)},
        'middle': {'rmse':round(m_mid[0],2),'mae':round(m_mid[1],2),'mape':round(m_mid[2],1)},
        'bottom': {'rmse':round(m_bot[0],2),'mae':round(m_bot[1],2),'mape':round(m_bot[2],1)},
    },
    'city': {
        'rmse_before': round(m_city_before[0],1), 'rmse_after': round(m_city_after[0],1),
        'improvement_pct': round((m_city_before[0]-m_city_after[0])/m_city_before[0]*100, 1),
    },
    'quantile_loss_bottom': ql,
}

with open(f'{OUT}/G5_results.json','w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

for k,v in {'pred_bottom':p_bot, 'pred_middle':p_mid, 'pred_top':p_top,
            'val_pred_bottom':p_bot_val, 'val_pred_middle':p_mid_val, 'val_pred_top':p_top_val,
            'scenarios_bottom':sc_bot, 'y_true':y_true_all, 'y_val_true':y_true_val, 'y_rec_shrink':y_rec}.items():
    np.save(f'{OUT}/G5_{k}.npy', v)

print(f'\n{"="*60}')
print(f'G5 Results Summary')
print(f'{"="*60}')
print(f'  Top:     RMSE={m_top[0]:.1f}, MAE={m_top[1]:.1f}, MAPE={m_top[2]:.1f}%')
print(f'  Middle:  RMSE={m_mid[0]:.1f}, MAE={m_mid[1]:.1f}, MAPE={m_mid[2]:.1f}%')
print(f'  Bottom:  RMSE={m_bot[0]:.2f}, MAE={m_bot[1]:.2f}, MAPE={m_bot[2]:.1f}%')
print(f'  City:    {m_city_before[0]:.0f} → {m_city_after[0]:.0f} ({results["city"]["improvement_pct"]:.1f}%)')
print(f'  Consist: {pre_inc:.1f} → {post_inc:.1e}')
print(f'  QL (τ=0.5): {ql["τ=0.5"]:.4f}')
print(f'\n[Done] Saved to {OUT}/G5_*')
