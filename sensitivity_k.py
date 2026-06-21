"""
敏感性分析 2: 中层节点数 K
对行为聚类分别取 K=5, 10, 15, 20, 运行 G7 流水线, 仅 τ=0.5
"""
import pandas as pd, numpy as np, json, os, time, warnings
warnings.filterwarnings('ignore')
os.environ['LGB_VERBOSITY'] = '-1'
import lightgbm as lgb
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf

RAW     = r'E:/Desktop/毕业论文/data/UrbanEV-main/UrbanEV-main/data'
OUT     = r'E:/Desktop/毕业论文/code/output'

K_LIST = [5, 10, 15, 20]
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
occ=pd.read_csv(f'{RAW}/occupancy.csv'); occ['time']=pd.to_datetime(occ['time']); occ=occ.set_index('time')
zone_cols=list(occ.columns); n_bottom=len(zone_cols)

vol=pd.read_csv(f'{RAW}/volume.csv',index_col=0); vol.index=pd.to_datetime(vol.index)
taz_ids=sorted(vol.columns.astype(int).tolist())

# 行为特征 (与 01_build_hierarchy.py 一致)
is_weekday=vol.index.dayofweek<5
vol_wd=vol.loc[is_weekday]; vol_we=vol.loc[~is_weekday]
profile_wd=vol_wd.groupby(vol_wd.index.hour).mean().T
profile_we=vol_we.groupby(vol_we.index.hour).mean().T
total_vol=vol.sum(axis=0).values.reshape(-1,1)
mean_vol=vol.mean(axis=0).values.reshape(-1,1)
nonzero_pct=(vol>0).sum(axis=0).values.reshape(-1,1)/len(vol)
cv_hourly=(vol.std(axis=0)/(vol.mean(axis=0)+1e-6)).values.reshape(-1,1)
features=np.hstack([profile_wd.values,profile_we.values,total_vol,mean_vol,nonzero_pct,cv_hourly])
features_std=StandardScaler().fit_transform(features)

occ_bottom=occ[zone_cols].values
occ_top=occ_bottom.sum(axis=1)

T_train=2880; T_val=720; test_start=T_train+T_val
nt=T_train-MAX_LAG; nv=T_val

ti=occ.index; tf=pd.DataFrame({
    'h_s':np.sin(2*np.pi*ti.hour/24),'h_c':np.cos(2*np.pi*ti.hour/24),
    'w_s':np.sin(2*np.pi*ti.dayofweek/7),'w_c':np.cos(2*np.pi*ti.dayofweek/7),
    'we':(ti.dayofweek>=5).astype(float),'mo':ti.month},index=ti)
wc=pd.read_csv(f'{RAW}/weather_central.csv');wc['time']=pd.to_datetime(wc['time'],format='%Y/%m/%d %H:%M');wc=wc.set_index('time')
wa=pd.read_csv(f'{RAW}/weather_airport.csv');wa['time']=pd.to_datetime(wa['time'],format='%Y/%m/%d %H:%M');wa=wa.set_index('time')
wf=pd.DataFrame({'T':(wc['T']+wa['T'])/2,'U':(wc['U']+wa['U'])/2,'P':(wc['P']+wa['P'])/2,'nR':wc['nRAIN']},index=occ.index)
gf=pd.concat([tf,wf],axis=1).values

def build_features(ts):
    T=len(ts);X=[]
    for t in range(MAX_LAG,T):
        f=list(gf[t,:]);[f.append(ts[t-l]) for l in LAG_HOURS]
        for r in ROLL_HOURS:f.append(np.mean(ts[t-r:t]));f.append(np.std(ts[t-r:t]))
        X.append(f)
    return np.array(X,dtype=np.float32),ts[MAX_LAG:].astype(np.float32)

# 先聚类, 然后对每个 K 跑 G7
all_results={}

for K in K_LIST:
    print(f'\n{"="*60}')
    print(f'K = {K}')
    print(f'{"="*60}')
    t_total0=time.time()

    # 聚类
    kmeans=KMeans(n_clusters=K,random_state=42,n_init=20,max_iter=500)
    labels=kmeans.fit_predict(features_std)
    cluster_groups={taz:int(labels[i])+1 for i,taz in enumerate(taz_ids)}
    counts=pd.Series(cluster_groups).value_counts().sort_index()
    print(f'  Cluster sizes: {counts.tolist()}')

    # 建 S 矩阵
    n_mid=K
    S=np.zeros((n_bottom+n_mid+1,n_bottom))
    S[:n_bottom,:]=np.eye(n_bottom)
    for j,taz in enumerate(taz_ids): S[n_bottom+cluster_groups[taz]-1,j]=1
    S[-1,:]=1

    # 构建 cluster_members
    cluster_members=[np.where(S[n_bottom+d,:]==1)[0].tolist() for d in range(n_mid)]

    # 聚合中层
    occ_mid=np.zeros((len(occ),n_mid))
    for d in range(n_mid): occ_mid[:,d]=occ_bottom[:,cluster_members[d]].sum(axis=1)

    def train_single(ts,lgb_params):
        X_t,y_t=build_features(ts)
        ntest=len(X_t)-nt-nv
        m=lgb.LGBMRegressor(objective='quantile',alpha=0.5,**lgb_params)
        m.fit(X_t[:nt],y_t[:nt],eval_set=[(X_t[nt:nt+nv],y_t[nt:nt+nv])],
              callbacks=[lgb.early_stopping(EARLY_STOP)])
        return m.predict(X_t[nt+nv:])

    def train_one_cluster(d):
        members=cluster_members[d];nm=len(members)
        if nm==1:
            p=train_single(occ_bottom[:,members[0]],LGB_GROUPED)
            return p[:,np.newaxis]
        Xtr,ytr,Xva,yva,Xte=[],[],[],[],[]
        for lid,j in enumerate(members):
            Xa,ya=build_features(occ_bottom[:,j])
            tid=np.full((len(Xa),1),lid,dtype=np.float32)
            Xtr.append(np.hstack([Xa[:nt],tid[:nt]]));ytr.append(ya[:nt])
            Xva.append(np.hstack([Xa[nt:nt+nv],tid[nt:nt+nv]]));yva.append(ya[nt:nt+nv])
            Xte.append(np.hstack([Xa[nt+nv:],tid[nt+nv:]]))
        Xt=np.vstack(Xtr).astype(np.float32);yt=np.hstack(ytr).astype(np.float32)
        Xv=np.vstack(Xva).astype(np.float32);yv=np.hstack(yva).astype(np.float32)
        m=lgb.LGBMRegressor(objective='quantile',alpha=0.5,**LGB_GROUPED)
        m.fit(Xt,yt,eval_set=[(Xv,yv)],callbacks=[lgb.early_stopping(EARLY_STOP)])
        pc=np.zeros((len(Xte[0]),nm))
        for lid in range(nm):pc[:,lid]=m.predict(Xte[lid])
        return pc

    t0=time.time()
    p_top=train_single(occ_top,LGB_TOP)
    p_mid=np.zeros((len(p_top),n_mid))
    for d in range(n_mid):p_mid[:,d]=train_single(occ_mid[:,d],LGB_TOP)

    results=Parallel(n_jobs=N_JOBS_PARALLEL,verbose=0)(
        delayed(train_one_cluster)(d) for d in range(n_mid))
    p_bot=np.zeros((len(p_top),n_bottom))
    for d,pc in enumerate(results):
        for lid,j in enumerate(cluster_members[d]):p_bot[:,j]=pc[:,lid]

    # MinT 调和
    y_tb=occ_bottom[test_start:,:];y_tm=occ_mid[test_start:,:];y_tt=occ_top[test_start:]
    yh=np.column_stack([p_bot,p_mid,p_top.reshape(-1,1)])
    yt_col=np.column_stack([y_tb,y_tm,y_tt.reshape(-1,1)])
    res=yh-yt_col
    W=LedoitWolf().fit(res).covariance_
    Wi=np.linalg.inv(W+np.eye(W.shape[0])*1e-8)
    G=np.linalg.inv(S.T@Wi@S)@S.T@Wi
    yr=(S@G@yh.T).T

    def metrics(yt,yp):
        err=yt-yp;rmse=np.sqrt(np.mean(err**2));mae=np.mean(np.abs(err))
        d=np.where(np.abs(yt)>1e-6,np.abs(yt),np.nan)
        return rmse,mae,float(np.nanmean(np.abs(err)/d)*100)

    m_bot=metrics(y_tb.flatten(),p_bot.flatten())
    m_mid=metrics(y_tm.flatten(),p_mid.flatten())
    city_b=metrics(y_tt,p_bot.sum(axis=1))
    city_a=metrics(y_tt,yr[:,:n_bottom].sum(axis=1))
    pre_inc=float(np.mean(np.abs(p_bot.sum(axis=1)-p_top)))
    post_inc=float(np.mean(np.abs(yr[:,:n_bottom].sum(axis=1)-yr[:,-1])))

    elapsed=time.time()-t0
    print(f'  K={K}: Bot@Rmse={m_bot[0]:.2f},MaPE={m_bot[2]:.1f}%; '
          f'Mid@Rmse={m_mid[0]:.2f}; City={city_b[0]:.0f}->{city_a[0]:.0f}; '
          f'Consist={pre_inc:.1f}->{post_inc:.1e}; {elapsed:.0f}s')

    # 簇内相关系数
    cors=[]
    for d in range(n_mid):
        taz_ids_in_cluster=[str(taz_ids[j]) for j in cluster_members[d]]
        sub=vol[taz_ids_in_cluster]
        if len(cluster_members[d])>1:
            cm=sub.corr().values
            cors.append(np.mean(cm[np.triu_indices_from(cm,k=1)]))
    cluster_corr=float(np.mean(cors)) if cors else 0

    all_results[str(K)]={
        'n_mid':K,'cluster_sizes':counts.tolist(),
        'cluster_corr':round(cluster_corr,4),
        'bottom':{'rmse':round(m_bot[0],2),'mae':round(m_bot[1],2),'mape':round(m_bot[2],1)},
        'middle':{'rmse':round(m_mid[0],2)},
        'city_before':round(city_b[0],1),'city_after':round(city_a[0],1),
        'consistency_before':round(pre_inc,1),
        'time_s':round(elapsed,0),
    }

# 汇总
print(f'\n{"="*65}')
print(f'K-VALUE SENSITIVITY SUMMARY')
print(f'{"="*65}')
print(f'{"K":<6} {"N_TAZ_range":<14} {"ClusterCorr":>10} {"BotRMSE":>8} {"BotMAPE":>7} {"MidRMSE":>8} {"City_aft":>8}')
for K in K_LIST:
    r=all_results[str(K)]
    sz=r['cluster_sizes']
    sz_range=f'{min(sz)}-{max(sz)}'
    print(f'{K:<6} {sz_range:<14} {r["cluster_corr"]:>10.3f} {r["bottom"]["rmse"]:>8.2f} '
          f'{r["bottom"]["mape"]:>6.1f}% {r["middle"]["rmse"]:>8.2f} {r["city_after"]:>8.1f}')

with open(f'{OUT}/sensitivity_k.json','w') as f: json.dump(all_results,f,indent=2)
print(f'\n[Done] Saved to {OUT}/sensitivity_k.json')
