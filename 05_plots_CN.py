"""
第四章图表中文版: 图1-10 中文标题、标签、图例
"""
import numpy as np, pandas as pd, json, os, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
                     'legend.fontsize': 9, 'figure.dpi': 150, 'savefig.dpi': 300,
                     'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
                     'axes.unicode_minus': False})

OUT = r'E:/Desktop/毕业论文/code/output'
FIG = r'E:/Desktop/毕业论文/图'
os.makedirs(FIG, exist_ok=True)

# ======== 加载数据 ========
RAW = r'E:/Desktop/毕业论文/data/UrbanEV-main/UrbanEV-main/data'
with open(f'{OUT}/hierarchy_meta.json') as f: h_meta = json.load(f)

def load_json(name):
    with open(f'{OUT}/{name}_results.json') as f: return json.load(f)
g1,g2,g4,g5,g7,g8 = load_json('G1'),load_json('G2'),load_json('G4'),load_json('G5'),load_json('G7'),load_json('G8')
g7p = load_json('G7_prob'); g7bv2 = load_json('G7_bootstrap_v2')

vol = pd.read_csv(f'{RAW}/volume.csv', index_col=0); vol.index=pd.to_datetime(vol.index)
mvol = vol.resample('ME').sum().mean(axis=0)

occ = pd.read_csv(f'{RAW}/occupancy.csv'); occ['time']=pd.to_datetime(occ['time']); occ=occ.set_index('time')
zone_cols = list(occ.columns)
n_bottom=275; TAUS=np.arange(0.05,1.00,0.05)

p_bot_g5 = np.load(f'{OUT}/G5_pred_bottom.npy')
p_bot_g7 = np.load(f'{OUT}/G7_pred_bottom.npy')
p_bot_g8 = np.load(f'{OUT}/G8_pred_bottom.npy')
p_mid_g2 = np.load(f'{OUT}/G2_pred_middle.npy')
p_mid_g5 = np.load(f'{OUT}/G5_pred_middle.npy')
y_t_g2 = np.load(f'{OUT}/G2_y_true.npy')
y_t_g5 = np.load(f'{OUT}/G5_y_true.npy')
y_t_g7 = np.load(f'{OUT}/G7_y_true.npy')
y_r_g1 = np.load(f'{OUT}/G1_y_rec_shrink.npy')
y_r_g2 = np.load(f'{OUT}/G2_y_rec_shrink.npy')
y_r_g4 = np.load(f'{OUT}/G4_y_rec_shrink.npy')
y_r_g5 = np.load(f'{OUT}/G5_y_rec_shrink.npy')
y_r_g7 = np.load(f'{OUT}/G7_y_rec_shrink.npy')
y_r_bv2 = np.load(f'{OUT}/G7_y_rec_bootstrap_v2.npy')

S_a=np.load(f'{OUT}/S_admin.npy'); S_b=np.load(f'{OUT}/S_behavior.npy')

taz_vols = np.array([mvol.get(z,0) for z in zone_cols])

def rmse_per_node(yp, yt, ax=0):
    return np.sqrt(np.mean((yp-yt)**2, axis=ax))

rmse_g5_taz = rmse_per_node(p_bot_g5[:,:,9], y_t_g2[:,:n_bottom], 0)
rmse_g7_taz = rmse_per_node(p_bot_g7[:,:,9], y_t_g7[:,:n_bottom], 0)

# ================================================================
# 图1 CN: 独立 vs 分组 RMSE 箱线图
# ================================================================
fig,ax=plt.subplots(figsize=(8,5))
bp=ax.boxplot([rmse_g5_taz, rmse_g7_taz],
              labels=['独立建模\n(G5)', '分组建模\n(G7)'],
              patch_artist=True, widths=0.4,
              boxprops=dict(facecolor='lightblue'),
              medianprops=dict(color='red', linewidth=1.5))
ax.set_ylabel('RMSE（占用率 %）')
ax.set_title('275个TAZ的RMSE分布：独立建模 vs 分组建模')
ax.text(0.02, 0.95, f'中位数: {np.median(rmse_g5_taz):.2f} → {np.median(rmse_g7_taz):.2f}',
        transform=ax.transAxes, fontsize=10, verticalalignment='top')
fig.tight_layout(); fig.savefig(f'{FIG}/fig1_rmse_boxplot_CN.png'); plt.close()

# ================================================================
# 图2 CN: 稀疏度 vs RMSE改善
# ================================================================
impr = (rmse_g5_taz-rmse_g7_taz)/(rmse_g5_taz+1e-8)*100
fig,ax=plt.subplots(figsize=(8,5))
ax.scatter(taz_vols, impr, alpha=0.5, s=15, c='steelblue', edgecolors='none')
ax.set_xscale('log')
ax.set_xlabel('月均充电量 (kWh, 对数坐标)')
ax.set_ylabel('分组建模 RMSE 改善 (%)')
ax.set_title('TAZ稀疏度与分组建模收益的关系')
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
ax.axvline(x=np.percentile(taz_vols,33.3), color='red', linestyle=':', alpha=0.5, label='33%分位')
ax.legend()
fig.tight_layout(); fig.savefig(f'{FIG}/fig2_sparsity_improvement_CN.png'); plt.close()

# ================================================================
# 图3 CN: 簇内相关 vs 中层RMSE
# ================================================================
y_tm_g2=y_t_g2[:,n_bottom:n_bottom+8]
y_tm_g5=y_t_g5[:,n_bottom:n_bottom+10]
mid_rmse_g2=rmse_per_node(p_mid_g2[:,:,9], y_tm_g2, 0)
mid_rmse_g5=rmse_per_node(p_mid_g5[:,:,9], y_tm_g5, 0)

def per_node_corr(S, occ_w, n_mid):
    cors=[]
    for d in range(n_mid):
        mb=np.where(S[n_bottom+d,:]==1)[0]
        if len(mb)>1:
            cm=occ_w.iloc[:,mb].corr().values
            cors.append(np.mean(cm[np.triu_indices_from(cm,k=1)]))
        else: cors.append(0)
    return np.array(cors)

admin_c=per_node_corr(S_a, occ, 8); behav_c=per_node_corr(S_b, occ, 10)

fig,ax=plt.subplots(figsize=(7,5))
ax.scatter(admin_c, mid_rmse_g2, c='#d62728', label='行政层次 (G2)', s=40, edgecolors='black', linewidth=0.5)
ax.scatter(behav_c, mid_rmse_g5, c='#1f77b4', label='行为聚类 (G5)', s=40, edgecolors='black', linewidth=0.5)
ax.set_xlabel('簇内平均Pearson相关系数')
ax.set_ylabel('中层节点 RMSE')
ax.set_title('簇内同质性与中层预测精度的关系')
ax.legend()
fig.tight_layout(); fig.savefig(f'{FIG}/fig3_intracorr_vs_rmse_CN.png'); plt.close()

# ================================================================
# 图4 CN: 调和增益分解堆叠条形图
# ================================================================
is_=g1['city']['rmse_after']-g2['city']['rmse_after']
iq_=g2['city']['rmse_after']-g5['city']['rmse_after']
gs_=g1['city']['rmse_after']-g4['city']['rmse_after']
gq_=g4['city']['rmse_after']-g7['city']['rmse_after']
it_=is_+iq_; gt_=gs_+gq_

fig,ax=plt.subplots(figsize=(7,5))
x=[0,1]; w=0.5
ax.bar(x, [is_, gs_], w, label='结构效应 (Δ_structure)', color='#2c3e50')
ax.bar(x, [iq_, gq_], w, bottom=[is_, gs_], label='质量效应 (Δ_quality)', color='#3498db')
for i,(s,q,t) in enumerate([(is_,iq_,it_),(gs_,gq_,gt_)]):
    if t>0 and s>0:
        ax.text(x[i], s/2, f'{s/t*100:.0f}%', ha='center', va='center', fontsize=11, color='white', fontweight='bold')
        ax.text(x[i], s+q/2, f'{q/t*100:.0f}%', ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    ax.text(x[i], t+1, f'合计: {t:.1f}', ha='center', fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(['独立策略', '分组策略'])
ax.set_ylabel('全市RMSE降低量')
ax.set_title('调和增益的归因分解')
ax.legend(loc='upper right')
fig.tight_layout(); fig.savefig(f'{FIG}/fig4_decomposition_bar_CN.png'); plt.close()

# ================================================================
# 图5 CN: 层级RMSE热力图
# ================================================================
def layer_rmse(yr,yt,n_b,n_m):
    bot=np.sqrt(np.mean((yr[:,:n_b]-yt[:,:n_b])**2))
    mid=np.sqrt(np.mean((yr[:,n_b:n_b+n_m]-yt[:,n_b:n_b+n_m])**2)) if n_m>0 else 0
    top=np.sqrt(np.mean((yr[:,-1]-yt[:,-1])**2))
    return [bot,mid,top]

data=[]
for yr,nm,lb in [(y_r_g1,1,'G1'),(y_r_g2,8,'G2'),(y_r_g4,8,'G4'),(y_r_g5,10,'G5'),(y_r_g7,10,'G7')]:
    if lb=='G1':
        data.append(layer_rmse(np.column_stack([yr[:,:275], np.zeros((yr.shape[0],10)), yr[:,-1:]]), y_t_g7, 275, 1))
    else:
        data.append(layer_rmse(yr, y_t_g7, 275, nm))
data=np.array(data)

fig,ax=plt.subplots(figsize=(8,3))
layers=['底层', '中层', '顶层']; exps=['G1','G2','G4','G5','G7']
im=ax.imshow(data.T, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(exps))); ax.set_xticklabels(exps)
ax.set_yticks(range(3)); ax.set_yticklabels(layers)
for i in range(len(exps)):
    for j in range(3):
        c='white' if data[i,j]>np.median(data) else 'black'
        ax.text(i,j,f'{data[i,j]:.1f}',ha='center',va='center',fontsize=10,color=c)
ax.set_title('各层级RMSE：按实验组')
plt.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout(); fig.savefig(f'{FIG}/fig5_heatmap_CN.png'); plt.close()

# ================================================================
# 图6 CN: 全市预测时序
# ================================================================
y_tt=y_t_g7[:,-1]; city_g1=y_r_g1[:,:n_bottom].sum(axis=1)
city_g7=y_r_g7[:,:n_bottom].sum(axis=1)
wk=slice(0,168)
fig,ax=plt.subplots(figsize=(10,4))
ax.plot(range(168), y_tt[wk], 'k-', linewidth=1.5, label='真实值')
ax.plot(range(168), city_g1[wk], 'r--', linewidth=1, alpha=0.8, label='G1 (两层直调)')
ax.plot(range(168), city_g7[wk], 'b-.', linewidth=1, alpha=0.8, label='G7 (三层调和)')
ax.set_xlabel('小时'); ax.set_ylabel('全市总占用率 (%)')
ax.set_title('全市预测对比：两层直调 vs 三层调和（测试集首周）')
ax.legend()
fig.tight_layout(); fig.savefig(f'{FIG}/fig6_city_timeseries_CN.png'); plt.close()

# ================================================================
# 图7 CN: U形QL
# ================================================================
deltas=[g8['ql_by_tau'][f'τ={t:.2f}']['delta'] for t in TAUS]
colors=['#e74c3c']*6+['#95a5a6']*7+['#2980b9']*6
fig,ax=plt.subplots(figsize=(8,4))
ax.bar(range(19), deltas, color=colors, edgecolor='white', linewidth=0.3)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.axvline(x=5.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axvline(x=12.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xticks(range(0,19,2)); ax.set_xticklabels([f'{TAUS[i]:.2f}' for i in range(0,19,2)])
ax.set_xlabel('分位数水平 τ'); ax.set_ylabel('ΔQL (G8−G7, 负值=改善)')
ax.set_title('分位数特定调和：各τ水平的QL变化')
ax.text(2.5, min(deltas)*1.15, '低τ', ha='center', fontsize=10, color='#e74c3c')
ax.text(9, max(deltas)*0.7, '中τ', ha='center', fontsize=10, color='#95a5a6')
ax.text(15.5, min(deltas)*1.15, '高τ', ha='center', fontsize=10, color='#2980b9')
fig.tight_layout(); fig.savefig(f'{FIG}/fig7_ushape_ql_CN.png'); plt.close()

# ================================================================
# 图8 CN: G7 vs G8 分位数曲线
# ================================================================
taz_idx=50; wk=slice(0,168); y_tb7=y_t_g7[:,:n_bottom]
fig,axes=plt.subplots(2,1,figsize=(10,6), sharex=True)
for ax,ti,tv,lb in [(axes[0],1,0.10,'τ=0.10 (低分位)'),(axes[1],17,0.90,'τ=0.90 (高分位)')]:
    ax.plot(range(168), y_tb7[wk,taz_idx], 'k-', linewidth=1, alpha=0.7, label='真实值')
    ax.plot(range(168), p_bot_g7[wk,taz_idx,ti], 'b-', linewidth=1, alpha=0.7, label='G7 (标准MinT)')
    ax.plot(range(168), p_bot_g8[wk,taz_idx,ti], 'r--', linewidth=1, alpha=0.7, label='G8 (分位数特定MinT)')
    ax.set_ylabel('占用率 (%)')
    ax.set_title(f'TAZ {taz_idx}: {lb}')
    ax.legend(fontsize=8)
axes[1].set_xlabel('小时')
fig.suptitle('分位数特定 vs 标准 MinT：低分位与高分位预测对比', fontsize=13)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig(f'{FIG}/fig8_quantile_curves_CN.png'); plt.close()

# ================================================================
# 图9 CN: PIT直方图
# ================================================================
hist_prob=g7p['pit']['probabilistic_hist']
hist_det=g7p['pit']['deterministic_hist']
# Bootstrap v2 PIT
ytb=y_t_g7[:,:n_bottom]; pit_bv2=[]
for j in range(n_bottom):
    sj=y_r_bv2[:,j,:]
    for t in range(len(sj)):
        pit_bv2.append(np.mean(sj[t,:]<=ytb[t,j]))
pit_bv2=np.array(pit_bv2); h_bv2,_=np.histogram(pit_bv2,bins=10,range=(0,1))
h_bv2=(h_bv2/len(pit_bv2)).tolist()

bins=np.linspace(0,1,11)
fig,axes=plt.subplots(1,3,figsize=(12,3.5), sharey=True)
for ax,h,title,ks in [
    (axes[0],hist_det,'确定性 + 高斯假设',g7p['pit']['ks_deterministic']['statistic']),
    (axes[1],hist_prob,'τ级调和 + 独立采样',g7p['pit']['ks_probabilistic']['statistic']),
    (axes[2],h_bv2,'残差Bootstrap + MinT',g7bv2['interval_90_bottom']['bootstrap_v2']['ks'])]:
    ax.bar(bins[:-1],h,width=0.1,align='edge',edgecolor='white',color='steelblue',alpha=0.8)
    ax.axhline(y=0.1, color='red', linestyle='--', linewidth=0.8)
    ax.set_title(f'{title}\nKS={ks:.3f}', fontsize=10)
    ax.set_xlabel('PIT'); ax.set_xticks([0,0.5,1])
axes[0].set_ylabel('频率')
fig.suptitle('PIT直方图：90%预测区间的校准度对比', fontsize=13)
fig.tight_layout(rect=[0,0,1,0.92])
fig.savefig(f'{FIG}/fig9_pit_histograms_CN.png'); plt.close()

# ================================================================
# 图10 CN: 区间时序 (确定性 vs Bootstrap)
# ================================================================
taz_idx=50; wk=slice(0,168); ytw=ytb[wk,taz_idx]
yrd=y_r_g7[:,:n_bottom]; resid=yrd[:,taz_idx]-ytb[:,taz_idx]; s=np.std(resid)
d_lo=yrd[wk,taz_idx]-1.645*s; d_hi=yrd[wk,taz_idx]+1.645*s; d_m=yrd[wk,taz_idx]
bs=y_r_bv2[wk,taz_idx,:]; b_lo=np.percentile(bs,5,axis=1)
b_hi=np.percentile(bs,95,axis=1); b_m=np.median(bs,axis=1)

fig,axes=plt.subplots(2,1,figsize=(10,6), sharex=True)
for ax,lo,hi,mid,title in [
    (axes[0],d_lo,d_hi,d_m,'确定性调和 + 高斯区间'),
    (axes[1],b_lo,b_hi,b_m,'残差Bootstrap + MinT')]:
    ax.fill_between(range(168),lo,hi,alpha=0.25,color='steelblue',label='90%预测区间')
    ax.plot(range(168),mid,'b-',linewidth=1,alpha=0.8)
    ax.plot(range(168),ytw,'k.',markersize=3,alpha=0.6)
    ax.set_ylabel('占用率 (%)'); ax.set_title(title,fontsize=11)
    cov=np.mean((ytw>=lo)&(ytw<=hi))
    ax.text(0.98,0.05,f'覆盖率: {cov:.2f}',transform=ax.transAxes,ha='right',fontsize=10)
axes[1].set_xlabel('小时')
fig.suptitle(f'90%预测区间对比：TAZ {taz_idx}（测试集首周）', fontsize=13)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig(f'{FIG}/fig10_interval_timeseries_CN.png'); plt.close()

print(f'[Done] All 10 Chinese figures saved to {FIG}/')
