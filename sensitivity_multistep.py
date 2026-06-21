"""
敏感性分析 1: 预测时长扩展 (Multi-step forecasting)
Direct multi-step: h=1, 24, 48, 72 小时
对比 G5 (独立) vs G7 (分组) 在各预测视域下的趋势
仅 τ=0.5 以控制计算量
"""
import pandas as pd, numpy as np, os, time, json, warnings
warnings.filterwarnings('ignore')
os.environ['LGB_VERBOSITY'] = '-1'
import lightgbm as lgb
from joblib import Parallel, delayed

RAW     = r'E:/Desktop/毕业论文/data/UrbanEV-main/UrbanEV-main/data'
S_BEHAV = r'E:/Desktop/毕业论文/code/output/S_behavior.npy'
OUT     = r'E:/Desktop/毕业论文/code/output'

HORIZONS = [1, 24, 48, 72]
N_JOBS_LGB = 4; N_JOBS_PARALLEL = 10
LAG_HOURS = [1,2,3,24,48,168]; ROLL_HOURS = [6,24]; MAX_LAG = 168
EARLY_STOP = 30

LGB_GROUPED = dict(n_estimators=300, learning_rate=0.08, num_leaves=31,
                   min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                   reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1, n_jobs=N_JOBS_LGB)
LGB_TOP = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
               min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
               reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1, n_jobs=N_JOBS_LGB)

print("Loading data...")
occ = pd.read_csv(f'{RAW}/occupancy.csv'); occ['time']=pd.to_datetime(occ['time']); occ=occ.set_index('time')
zone_cols=list(occ.columns); n_bottom=len(zone_cols)

S=np.load(S_BEHAV); n_mid=S.shape[0]-n_bottom-1
cluster_members=[]
for d in range(n_mid): cluster_members.append(np.where(S[n_bottom+d,:]==1)[0].tolist())

occ_bottom=occ[zone_cols].values
occ_mid=np.zeros((len(occ),n_mid))
for d in range(n_mid): occ_mid[:,d]=occ_bottom[:,cluster_members[d]].sum(axis=1)
occ_top=occ_bottom.sum(axis=1)

T_train=2880; T_val=720
T_total=len(occ); test_start=T_train+T_val

ti=occ.index; tf=pd.DataFrame({
    'h_s':np.sin(2*np.pi*ti.hour/24),'h_c':np.cos(2*np.pi*ti.hour/24),
    'w_s':np.sin(2*np.pi*ti.dayofweek/7),'w_c':np.cos(2*np.pi*ti.dayofweek/7),
    'we':(ti.dayofweek>=5).astype(float),'mo':ti.month},index=ti)
wc=pd.read_csv(f'{RAW}/weather_central.csv');wc['time']=pd.to_datetime(wc['time'],format='%Y/%m/%d %H:%M');wc=wc.set_index('time')
wa=pd.read_csv(f'{RAW}/weather_airport.csv');wa['time']=pd.to_datetime(wa['time'],format='%Y/%m/%d %H:%M');wa=wa.set_index('time')
wf=pd.DataFrame({'T':(wc['T']+wa['T'])/2,'U':(wc['U']+wa['U'])/2,'P':(wc['P']+wa['P'])/2,'nR':wc['nRAIN']},index=occ.index)
gf=pd.concat([tf,wf],axis=1).values

all_results = {}

for hor in HORIZONS:
    print(f'\n{"="*60}')
    print(f'HORIZON h={hor}')
    print(f'{"="*60}')

    nt = T_train - MAX_LAG - hor + 1  # 训练集特征数量: 留出 hor 步给 target
    nv = T_val

    def build_features_multistep(ts):
        """特征在 t, 目标在 t+hor"""
        T = len(ts); X = []; y = []
        for t in range(MAX_LAG, T - hor):
            f = list(gf[t, :])
            [f.append(ts[t-l]) for l in LAG_HOURS]
            for r in ROLL_HOURS: f.append(np.mean(ts[t-r:t])); f.append(np.std(ts[t-r:t]))
            X.append(f); y.append(ts[t+hor])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def train_single(ts, lgb_params):
        X_t, y_t = build_features_multistep(ts)
        ntest = len(X_t) - nt - nv
        m = lgb.LGBMRegressor(objective='quantile', alpha=0.5, **lgb_params)
        m.fit(X_t[:nt], y_t[:nt],
              eval_set=[(X_t[nt:nt+nv], y_t[nt:nt+nv])],
              callbacks=[lgb.early_stopping(EARLY_STOP)])
        return m.predict(X_t[nt+nv:])

    def train_one_cluster(d):
        members=cluster_members[d]; nm=len(members)
        if nm==1:
            p = train_single(occ_bottom[:, members[0]], LGB_GROUPED)
            return p[:, np.newaxis]

        Xtr,ytr,Xva,yva,Xte=[],[],[],[],[]
        for lid,j in enumerate(members):
            Xa,ya=build_features_multistep(occ_bottom[:,j])
            tid=np.full((len(Xa),1),lid,dtype=np.float32)
            Xtr.append(np.hstack([Xa[:nt],tid[:nt]])); ytr.append(ya[:nt])
            Xva.append(np.hstack([Xa[nt:nt+nv],tid[nt:nt+nv]])); yva.append(ya[nt:nt+nv])
            Xte.append(np.hstack([Xa[nt+nv:],tid[nt+nv:]]))
        Xt=np.vstack(Xtr).astype(np.float32); yt=np.hstack(ytr).astype(np.float32)
        Xv=np.vstack(Xva).astype(np.float32); yv=np.hstack(yva).astype(np.float32)
        m=lgb.LGBMRegressor(objective='quantile',alpha=0.5,**LGB_GROUPED)
        m.fit(Xt,yt,eval_set=[(Xv,yv)],callbacks=[lgb.early_stopping(EARLY_STOP)])
        pc=np.zeros((len(Xte[0]),nm))
        for lid in range(nm): pc[:,lid]=m.predict(Xte[lid])
        return pc

    def metrics(yt,yp):
        err=yt-yp; rmse=np.sqrt(np.mean(err**2)); mae=np.mean(np.abs(err))
        d=np.where(np.abs(yt)>1e-6,np.abs(yt),np.nan)
        return rmse,mae,float(np.nanmean(np.abs(err)/d)*100)

    t0=time.time()

    # --- G7: 分组建模 ---
    p_top= train_single(occ_top, LGB_TOP)
    p_mid=np.zeros((len(p_top),n_mid))
    for d in range(n_mid): p_mid[:,d]=train_single(occ_mid[:,d], LGB_TOP)

    results=Parallel(n_jobs=N_JOBS_PARALLEL,verbose=0)(
        delayed(train_one_cluster)(d) for d in range(n_mid))
    p_bot=np.zeros((len(p_top),n_bottom))
    for d,pc in enumerate(results):
        for lid,j in enumerate(cluster_members[d]): p_bot[:,j]=pc[:,lid]

    # G7 评估 — 取测试集最后 len(p_top) 个时点
    n_test_pts = len(p_top)
    y_tb = occ_bottom[-n_test_pts:, :]
    y_tm = occ_mid[-n_test_pts:, :]
    y_tt = occ_top[-n_test_pts:]

    m_bot7 = metrics(y_tb.flatten(), p_bot.flatten())
    m_mid7 = metrics(y_tm.flatten(), p_mid.flatten())
    city_b7 = metrics(y_tt, p_bot.sum(axis=1))

    # --- G5: 独立建模 ---
    ts_list = [occ_bottom[:, j] for j in range(n_bottom)]
    results5 = Parallel(n_jobs=N_JOBS_PARALLEL, verbose=0)(
        delayed(train_single)(ts, LGB_GROUPED) for ts in ts_list)
    p_bot5 = np.column_stack(results5)
    m_bot5 = metrics(y_tb.flatten(), p_bot5.flatten())
    city_b5 = metrics(y_tt, p_bot5.sum(axis=1))

    elapsed=time.time()-t0
    print(f'  h={hor}: G5@Rmse={m_bot5[0]:.2f},MaPE={m_bot5[2]:.1f}%; '
          f'G7@Rmse={m_bot7[0]:.2f},MaPE={m_bot7[2]:.1f}%; '
          f'City G5={city_b5[0]:.0f} G7={city_b7[0]:.0f}; {elapsed:.0f}s')

    all_results[str(hor)]={
        'G5':{'bottom_rmse':round(m_bot5[0],2),'bottom_mae':round(m_bot5[1],2),'bottom_mape':round(m_bot5[2],1),
              'city_rmse':round(city_b5[0],1)},
        'G7':{'bottom_rmse':round(m_bot7[0],2),'bottom_mae':round(m_bot7[1],2),'bottom_mape':round(m_bot7[2],1),
              'middle_rmse':round(m_mid7[0],2),'city_rmse':round(city_b7[0],1)},
    }

# 汇总
print(f'\n{"="*60}')
print('MULTI-STEP SENSITIVITY SUMMARY')
print(f'{"="*60}')
print(f'{"h":<6} {"G5_RMSE":>10} {"G5_MAPE":>8} {"G7_RMSE":>10} {"G7_MAPE":>8} {"G7_Mid":>8} {"G7_City":>8}')
for hor in HORIZONS:
    r=all_results[str(hor)]
    print(f'{hor:<6} {r["G5"]["bottom_rmse"]:>10.2f} {r["G5"]["bottom_mape"]:>7.1f}% '
          f'{r["G7"]["bottom_rmse"]:>10.2f} {r["G7"]["bottom_mape"]:>7.1f}% '
          f'{r["G7"]["middle_rmse"]:>8.2f} {r["G7"]["city_rmse"]:>8.1f}')

with open(f'{OUT}/sensitivity_multistep.json','w') as f: json.dump(all_results,f,indent=2)
print(f'\n[Done] Saved to {OUT}/sensitivity_multistep.json')
