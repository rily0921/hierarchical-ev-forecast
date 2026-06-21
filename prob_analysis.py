"""
概率延伸分析: 交互效应在概率预测层面
────────────────────────────────────
基于已有 E1-E6 预测结果 (output/G*.npy)
"""
import numpy as np, warnings
warnings.filterwarnings('ignore')

OUT = r'E:/Desktop/毕业论文/code/output'
TAUS = np.arange(0.05, 1.00, 0.05)

label_map = {'E1':'G1','E2':'G2','E3':'G4','E4':'G5','E5':'G7','E6':'G8'}

# ============================================================
# 1. 加载
# ============================================================
print('Loading predictions...')
pred_bot = {}
for e, g in label_map.items():
    try:
        p = np.load(f'{OUT}/{g}_pred_bottom.npy')
        pred_bot[e] = p  # (744, 275, 19)
        print(f'  {e} ({g}): {p.shape}')
    except FileNotFoundError:
        print(f'  {e} ({g}): MISSING')

y_true = np.load(f'{OUT}/G7_y_true.npy')
n_bottom = 275
y_tt = y_true[:, -1]  # 全市真实值 (744,)

# ============================================================
# 2. Pinball Loss
# ============================================================
def pinball(yt, yp_tau, tau):
    err = yt - yp_tau
    return float(np.mean(np.where(err>=0, tau*err, (tau-1)*err)))

# ============================================================
# 3. CRPS = mean Pinball Loss over τ
# ============================================================
def crps(yt, q_pred):
    """q_pred: (T, 19)"""
    if q_pred.ndim == 1:
        q_pred = q_pred.reshape(-1, 1)
    total = 0.0
    for i, tau in enumerate(TAUS):
        total += pinball(yt, q_pred[:, i], tau)
    return total / len(TAUS)

# ============================================================
# 4. 不同 τ 下的 2×2 交互矩阵
# ============================================================
print('\n' + '='*70)
print('2x2 Interaction Matrix at τ=0.1, 0.5, 0.9 (Pinball Loss)')
print('='*70)

tau_levels = {'τ=0.1': 1, 'τ=0.5': 9, 'τ=0.9': 17}

for label, idx in tau_levels.items():
    print(f'\n--- {label} (index={idx}) ---')
    row = {}
    for e in ['E2','E3','E4','E5']:
        if e not in pred_bot:
            continue
        bu = pred_bot[e][:, :, idx].sum(axis=1)
        ql = pinball(y_tt, bu, TAUS[idx])
        row[e] = ql
        tag = ''
        if e in ['E3','E4','E5']:
            best = min(row.get('E2',ql), row.get('E3',ql), row.get('E4',ql), ql)
            if ql == best: tag = ' ★'
        print(f'  {e}: QL={ql:.2f}{tag}')
    dq_indep = row.get('E2',0) - row.get('E4',0)
    dq_group = row.get('E3',0) - row.get('E5',0)
    print(f'  Δ_quality(独立): {dq_indep:.2f}')
    print(f'  Δ_quality(分组): {dq_group:.2f}')
    print(f'  → 交互效应: {"✅" if dq_group > 0 and dq_group > dq_indep else "✗"}')

# ============================================================
# 5. CRPS 归因分解
# ============================================================
print('\n' + '='*70)
print('CRPS Attribution Decomposition')
print('='*70)

crps_val = {}
for e in ['E2','E3','E4','E5']:
    if e not in pred_bot:
        continue
    bu_q = pred_bot[e].sum(axis=1)  # (744, 19)
    crps_val[e] = crps(y_tt, bu_q)
    print(f'  {e}: CRPS={crps_val[e]:.2f}')

if all(k in crps_val for k in ['E2','E3','E4','E5']):
    # 分组
    ds = crps_val['E2'] - crps_val['E3']
    dq = crps_val['E3'] - crps_val['E5']
    dt = ds + dq
    print(f'\n  Grouped: Δ_structure={ds:.2f}, Δ_quality={dq:.2f}, Δ_total={dt:.2f}')
    print(f'            结构占比={ds/dt*100:.0f}%, 质量占比={dq/dt*100:.0f}%' if dt>0 else '')
    # 独立
    ds_i = crps_val['E2'] - crps_val['E2']  # same middle structure different? no
    dq_i = crps_val['E2'] - crps_val['E4']
    print(f'  Independent: Δ_quality={dq_i:.2f}')

# ============================================================
# 6. MinT vs Bottom-Up in CRPS
# ============================================================
print('\n' + '='*70)
print('MinT vs Bottom-Up (CRPS)')
print('='*70)

for e in ['E1','E4','E5']:
    if e not in pred_bot:
        continue
    g = label_map[e]
    bu_q = pred_bot[e].sum(axis=1)  # (744, 19)
    bu_crps = crps(y_tt, bu_q)
    print(f'  {e} Bottom-Up CRPS={bu_crps:.2f}', end='')
    try:
        y_rec = np.load(f'{OUT}/{g}_y_rec_shrink.npy')
        rec_med = y_rec[:, :n_bottom].sum(axis=1)
        bu_med = bu_q[:, 9]
        q_rec = rec_med.reshape(-1,1) + (bu_q - bu_med.reshape(-1,1))
        rec_crps = crps(y_tt, q_rec)
        print(f'  MinT CRPS≈{rec_crps:.2f}  → 较优: {"MinT" if rec_crps < bu_crps else "Bottom-Up"}')
    except FileNotFoundError:
        print('  (no REC file)')

# ============================================================
# 7. PIT
# ============================================================
print('\n' + '='*70)
print('PIT Histogram (E5 Bottom-Up)')
print('='*70)

p5 = pred_bot['E5'].sum(axis=1)  # (744, 19)
pits = []
for t in range(len(y_tt)):
    y, q = y_tt[t], p5[t,:]
    idx = np.searchsorted(q, y)
    if idx == 0:        pit = 0.025
    elif idx >= 19:     pit = 0.975
    else:
        lo, hi = TAUS[idx-1], TAUS[idx]
        qlo, qhi = q[idx-1], q[idx]
        pit = lo + (hi-lo)*(y-qlo)/max(qhi-qlo, 1e-10)
    pits.append(pit)

pits = np.array(pits)
bins = np.linspace(0,1,11)
hist, _ = np.histogram(pits, bins=bins)
print(f'  Bins (10): {hist}')
print(f'  Mean PIT: {pits.mean():.3f}  (ideal 0.500)')
print(f'  Std PIT:  {pits.std():.3f}  (ideal 0.289)')

# Uniformity check: chi2
expected = len(pits)/10
chi2 = np.sum((hist - expected)**2 / expected)
print(f'  Chi2(9) = {chi2:.1f} (critical 16.9 at α=0.05)')
print(f'  → Uniform at α=0.05: {"✅ YES" if chi2 < 16.9 else "✗ NO"}')

print('\n[Done]')
