"""
G8: 行为聚类 + 分组建模(10组) + 分位数特定MinT + 多τ分位数回归
─────────────────────────────────────────────────
与 G7 的唯一差异: 标准MinT → 分位数特定MinT (按τ分三档估计Σ)
  - 低 τ (≤0.3): Σ_low
  - 中 τ (0.3-0.7): Σ_mid
  - 高 τ (>0.7): Σ_high
  调和时每个τ选用对应档的Σ, 其余步骤与G7完全一致
对比组: G7 (同条件, 标准MinT)
输出: output/G8_*.npy + output/G8_results.json
"""
import pandas as pd, numpy as np, json, os, time, warnings
warnings.filterwarnings('ignore')
os.environ['LGB_VERBOSITY'] = '-1'
import lightgbm as lgb
from joblib import Parallel, delayed
from sklearn.covariance import LedoitWolf

RAW     = r'E:/Desktop/毕业论文/data/UrbanEV-main/UrbanEV-main/data'
S_BEHAV = r'E:/Desktop/毕业论文/code/output/S_behavior.npy'
OUT     = r'E:/Desktop/毕业论文/code/output'
os.makedirs(OUT, exist_ok=True)

TAUS = np.arange(0.05, 1.00, 0.05)
TAU_BANDS = {'low': (TAUS <= 0.3), 'mid': ((TAUS > 0.3) & (TAUS <= 0.7)), 'high': (TAUS > 0.7)}
N_JOBS_LGB = 4; N_JOBS_PARALLEL = 10
LAG_HOURS = [1,2,3,24,48,168]; ROLL_HOURS = [6,24]; MAX_LAG = 168
EARLY_STOP = 30; M_SCENARIOS = 500

LGB_GROUPED = dict(n_estimators=300, learning_rate=0.08, num_leaves=31,
                   min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                   reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1, n_jobs=N_JOBS_LGB)
LGB_TOP = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
               min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
               reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1, n_jobs=N_JOBS_LGB)

print('='*60)
print('G8: Behavior + Grouped(10) + Quantile-Specific MinT')
print('='*60)

# ======== 1. 加载数据 & 架构 (与 G7 相同) ========
occ = pd.read_csv(f'{RAW}/occupancy.csv')
occ['time'] = pd.to_datetime(occ['time']); occ = occ.set_index('time')
zone_cols = [c for c in occ.columns]; n_bottom = len(zone_cols)

S = np.load(S_BEHAV)
n_middle = S.shape[0] - n_bottom - 1

cluster_members = []
for d in range(n_middle):
    members = np.where(S[n_bottom + d, :] == 1)[0].tolist()
    cluster_members.append(members)

occ_bottom = occ[zone_cols].values
occ_middle = np.zeros((len(occ), n_middle))
for d in range(n_middle):
    occ_middle[:, d] = occ_bottom[:, cluster_members[d]].sum(axis=1)
occ_top = occ_bottom.sum(axis=1)

T_total = len(occ); T_train = 2880; T_val = 720

time_idx = occ.index
time_feat = pd.DataFrame({
    'hour_sin': np.sin(2*np.pi*time_idx.hour/24), 'hour_cos': np.cos(2*np.pi*time_idx.hour/24),
    'wday_sin': np.sin(2*np.pi*time_idx.dayofweek/7), 'wday_cos': np.cos(2*np.pi*time_idx.dayofweek/7),
    'is_wend': (time_idx.dayofweek>=5).astype(float), 'month': time_idx.month,
}, index=time_idx)
wc = pd.read_csv(f'{RAW}/weather_central.csv'); wc['time'] = pd.to_datetime(wc['time'],format='%Y/%m/%d %H:%M'); wc=wc.set_index('time')
wa = pd.read_csv(f'{RAW}/weather_airport.csv'); wa['time'] = pd.to_datetime(wa['time'],format='%Y/%m/%d %H:%M'); wa=wa.set_index('time')
weather_feat = pd.DataFrame({
    'T':(wc['T']+wa['T'])/2,'U':(wc['U']+wa['U'])/2,'P':(wc['P']+wa['P'])/2,'nRAIN':wc['nRAIN'],
}, index=occ.index)
global_feat = pd.concat([time_feat, weather_feat], axis=1).values

nt = T_train - MAX_LAG; nv = T_val

def build_features(ts):
    T=len(ts); X=[]
    for t in range(MAX_LAG, T):
        f=list(global_feat[t,:])
        for lag in LAG_HOURS: f.append(ts[t-lag])
        for roll in ROLL_HOURS:
            f.append(np.mean(ts[t-roll:t])); f.append(np.std(ts[t-roll:t]))
        X.append(f)
    return np.array(X,dtype=np.float32), ts[MAX_LAG:].astype(np.float32)

def train_single(ts, params):
    X_t,y_t=build_features(ts); ntest=len(X_t)-nt-nv; p_val=np.zeros((nv,len(TAUS))); p=np.zeros((ntest,len(TAUS)))
    cb=lgb.early_stopping(EARLY_STOP)
    for i,tau in enumerate(TAUS):
        m=lgb.LGBMRegressor(objective='quantile',alpha=tau,**params)
        m.fit(X_t[:nt],y_t[:nt],eval_set=[(X_t[nt:nt+nv],y_t[nt:nt+nv])],callbacks=[cb])
        p_val[:,i]=m.predict(X_t[nt:nt+nv])
        p[:,i]=m.predict(X_t[nt+nv:])
    p_val.sort(axis=1); p.sort(axis=1); return p_val, p

# ======== 2. 训练 (与 G7 相同的基础预测) ========
p_top_val, p_top = train_single(occ_top, LGB_TOP)
p_mid_val = np.zeros((p_top_val.shape[0], n_middle, len(TAUS)))
p_mid = np.zeros((p_top.shape[0], n_middle, len(TAUS)))
for d in range(n_middle):
    p_mid_val[:, d, :], p_mid[:, d, :] = train_single(occ_middle[:, d], LGB_TOP)

def train_one_cluster(d):
    members = cluster_members[d]; n_mem = len(members)
    if n_mem == 1:
        p_val, p = train_single(occ_bottom[:, members[0]], LGB_GROUPED)
        return p_val[:, np.newaxis, :], p[:, np.newaxis, :]
    X_tr,y_tr,X_va,y_va,X_te=[],[],[],[],[]
    for lid,j in enumerate(members):
        Xa,ya=build_features(occ_bottom[:,j])
        tid=np.full((len(Xa),1),lid,dtype=np.float32)
        X_tr.append(np.hstack([Xa[:nt],tid[:nt]])); y_tr.append(ya[:nt])
        X_va.append(np.hstack([Xa[nt:nt+nv],tid[nt:nt+nv]])); y_va.append(ya[nt:nt+nv])
        X_te.append(np.hstack([Xa[nt+nv:],tid[nt+nv:]]))
    Xt=np.vstack(X_tr).astype(np.float32); yt=np.hstack(y_tr).astype(np.float32)
    Xv=np.vstack(X_va).astype(np.float32); yv=np.hstack(y_va).astype(np.float32)
    cb=lgb.early_stopping(EARLY_STOP); pc_val=np.zeros((nv,n_mem,len(TAUS))); pc=np.zeros((p_top.shape[0],n_mem,len(TAUS)))
    cat_idx = Xt.shape[1] - 1
    for i,tau in enumerate(TAUS):
        m=lgb.LGBMRegressor(objective='quantile',alpha=tau,
                            **{k:v for k,v in LGB_GROUPED.items()})
        m.fit(Xt,yt,eval_set=[(Xv,yv)],callbacks=[cb],
              categorical_feature=[cat_idx])
        for lid in range(n_mem):
            pc_val[:,lid,i]=m.predict(X_va[lid])
            pc[:,lid,i]=m.predict(X_te[lid])
    pc_val.sort(axis=2); pc.sort(axis=2); return pc_val, pc

t0=time.time()
results = Parallel(n_jobs=N_JOBS_PARALLEL, verbose=10)(
    delayed(train_one_cluster)(d) for d in range(n_middle))
p_bot_val = np.zeros((p_top_val.shape[0], n_bottom, len(TAUS)))
p_bot = np.zeros((p_top.shape[0], n_bottom, len(TAUS)))
for d, (pc_val, pc) in enumerate(results):
    for lid, j in enumerate(cluster_members[d]):
        p_bot_val[:, j, :] = pc_val[:, lid, :]
        p_bot[:, j, :] = pc[:, lid, :]
print(f'Training done in {time.time()-t0:.0f}s')

# ======== 3. 场景生成 ========
def quantiles_to_scenarios(q_pred):
    T=q_pred.shape[0]; u=np.random.RandomState(42).uniform(0,1,(T,M_SCENARIOS))
    sc=np.array([np.interp(u[t,:],TAUS,q_pred[t,:]) for t in range(T)])
    return np.maximum(sc,0)
sc_bot = np.array([[quantiles_to_scenarios(p_bot[:,j,:]) for j in range(n_bottom)]])
sc_bot = sc_bot.reshape(p_bot.shape[0], n_bottom, M_SCENARIOS)

# ======== 4. 分位数特定 MinT (核心区别 vs G7) ========
print('\n--- Quantile-Specific MinT Reconciliation ---')
val_start = T_train
test_start = T_train + T_val
y_vb = occ_bottom[val_start:test_start, :]; y_vm = occ_middle[val_start:test_start, :]; y_vt = occ_top[val_start:test_start]
y_tb = occ_bottom[test_start:, :]; y_tm = occ_middle[test_start:, :]; y_tt = occ_top[test_start:]

def G_mat(S, W):
    Wi = np.linalg.inv(W + np.eye(W.shape[0])*1e-8)
    return np.linalg.inv(S.T @ Wi @ S) @ S.T @ Wi

# 标准 MinT: 用验证集中位数残差估计一套 W, 并应用到所有分位数
y_hat_val_med = np.column_stack([p_bot_val[:, :, 9], p_mid_val[:, :, 9], p_top_val[:, 9]])
y_true_val = np.column_stack([y_vb, y_vm, y_vt.reshape(-1, 1)])
W_std = LedoitWolf().fit(y_hat_val_med - y_true_val).covariance_
G_std = G_mat(S, W_std)
y_rec_std_by_tau = np.zeros((p_bot.shape[0], S.shape[0], len(TAUS)))
for i in range(len(TAUS)):
    y_hat_tau_test = np.column_stack([p_bot[:, :, i], p_mid[:, :, i], p_top[:, i]])
    y_rec_std_by_tau[:, :, i] = (S @ G_std @ y_hat_tau_test.T).T

# 对每个τ分档, 估计独立的Σ并调和
y_rec_by_tau = np.zeros((p_bot.shape[0], S.shape[0], len(TAUS)))  # (ntest, 286, 19)

for band_name, band_mask in TAU_BANDS.items():
    band_indices = np.where(band_mask)[0]
    print(f'  Band {band_name}: τ ∈ {TAUS[band_indices[0]]:.2f} ~ {TAUS[band_indices[-1]]:.2f} ({len(band_indices)} τ)')

    for i in band_indices:
        tau = TAUS[i]
        # 拼接该τ的全层级预测
        y_hat_tau_val = np.column_stack([p_bot_val[:, :, i], p_mid_val[:, :, i], p_top_val[:, i]])
        y_hat_tau = np.column_stack([p_bot[:, :, i], p_mid[:, :, i], p_top[:, i]])
        resid = y_hat_tau_val - y_true_val

        # 为该τ的残差估计 Σ (存入 band 级别稍后合并)
        W = LedoitWolf().fit(resid).covariance_
        G = G_mat(S, W)
        y_rec_by_tau[:, :, i] = (S @ G @ y_hat_tau.T).T

idx_med = 9
y_rec_med = y_rec_by_tau[:, :, idx_med]
p_bot_med = p_bot[:, :, idx_med]; p_top_med = p_top[:, idx_med]
pre_inc = float(np.mean(np.abs(p_bot_med.sum(axis=1) - p_top_med)))
post_inc = float(np.mean(np.abs(y_rec_med[:, :n_bottom].sum(axis=1) - y_rec_med[:, -1])))

# ======== 5. 评估 ========
def metrics(yt, yp):
    err = yt - yp; rmse = np.sqrt(np.mean(err**2)); mae = np.mean(np.abs(err))
    d = np.where(np.abs(yt)>1e-6, np.abs(yt), np.nan); mape=np.nanmean(np.abs(err)/d)*100
    return rmse, mae, mape

def qloss(yt, yp_tau, tau):
    err = yt - yp_tau
    return float(np.mean(np.where(err>=0, tau*err, (tau-1)*err)))

# 中位数评估
m_bot = metrics(y_tb.flatten(), p_bot_med.flatten())
m_mid = metrics(y_tm.flatten(), p_mid[:,:,idx_med].flatten())
bu_sum = p_bot_med.sum(axis=1)
m_city_before = metrics(y_tt, bu_sum)
m_city_after  = metrics(y_tt, y_rec_med[:, :n_bottom].sum(axis=1))

# 分τ QL 对比 (标准MinT vs 分位数特定MinT 在各τ上的QL差异)
ql_by_tau = {}
for i, tau in enumerate(TAUS):
    ql_std = qloss(y_tb.flatten(), y_rec_std_by_tau[:, :n_bottom, i].flatten(), tau)
    ql_qs  = qloss(y_tb.flatten(), y_rec_by_tau[:, :n_bottom, i].flatten(), tau)
    ql_by_tau[f'τ={tau:.2f}'] = {'standard': round(ql_std,4), 'quantile_specific': round(ql_qs,4),
                                  'delta': round(ql_qs-ql_std,4), 'improved': ql_qs < ql_std}

# 按分档汇总
for band_name in ['low', 'mid', 'high']:
    band_taus = [f'τ={TAUS[i]:.2f}' for i in np.where(TAU_BANDS[band_name])[0]]
    avg_delta = np.mean([ql_by_tau[t]['delta'] for t in band_taus])
    print(f'  {band_name} band avg ΔQL: {avg_delta:.4f}')

result = {
    'experiment': 'G8',
    'description': 'Behavior + Grouped(10) + Quantile-Specific MinT + Multi-τ Quantile',
    'covariance_source': 'validation_residuals',
    'standard_mint_reference': 'single_G_from_validation_median_residuals',
    'tau_bands': {'low': 'τ≤0.3', 'mid': '0.3<τ≤0.7', 'high': 'τ>0.7'},
    'consistency': {'before': round(pre_inc,2), 'after': round(post_inc,10)},
    'per_level_median': {
        'bottom': {'rmse':round(m_bot[0],2),'mae':round(m_bot[1],2),'mape':round(m_bot[2],1)},
        'middle': {'rmse':round(m_mid[0],2),'mae':round(m_mid[1],2),'mape':round(m_mid[2],1)},
    },
    'city_median': {
        'rmse_before': round(m_city_before[0],1), 'rmse_after': round(m_city_after[0],1),
        'improvement_pct': round((m_city_before[0]-m_city_after[0])/m_city_before[0]*100, 1),
    },
    'ql_by_tau': ql_by_tau,
}

with open(f'{OUT}/G8_results.json','w') as f: json.dump(result, f, indent=2, ensure_ascii=False)
for k,v in {'pred_bottom':p_bot, 'pred_middle':p_mid, 'pred_top':p_top,
            'val_pred_bottom':p_bot_val, 'val_pred_middle':p_mid_val, 'val_pred_top':p_top_val,
            'scenarios_bottom':sc_bot, 'y_val_true':y_true_val,
            'y_rec_std_tau':y_rec_std_by_tau, 'y_rec_tau':y_rec_by_tau}.items():
    np.save(f'{OUT}/G8_{k}.npy', v)

print(f'\n{"="*60}')
print(f'G8 Results Summary')
print(f'{"="*60}')
print(f'  City median: {m_city_before[0]:.0f} → {m_city_after[0]:.0f} ({result["city_median"]["improvement_pct"]:.1f}%)')
# 各档改善
for band_name in ['low', 'mid', 'high']:
    band_taus = [f'τ={TAUS[i]:.2f}' for i in np.where(TAU_BANDS[band_name])[0]]
    avg_delta = np.mean([ql_by_tau[t]['delta'] for t in band_taus])
    n_improved = sum([ql_by_tau[t]['improved'] for t in band_taus])
    print(f'  {band_name}: avg ΔQL={avg_delta:.4f} ({n_improved}/{len(band_taus)} improved)')
print(f'\n[Done] Saved to {OUT}/G8_*')
