"""
G1: 两层直调(无中层) + 独立建模(275个) + 标准MinT + 多τ分位数回归
─────────────────────────────────────────────────
作用: 验证中间层的必要性——两层 vs 三层的调和增益差异
实现: 底层和顶层基础预测复用 G2 结果 (独立建模相同)
      仅构建两层 S 矩阵 (276×275) 并执行调和
输出: output/G1_results.json + output/G1_y_rec_shrink.npy
"""
import numpy as np, json, os, warnings
warnings.filterwarnings('ignore')

OUT = r'E:/Desktop/毕业论文/code/output'

# ========== 1. 加载 G2 的基础预测 + 真实值 ==========
print('Loading G2 base forecasts...')
p_bot = np.load(f'{OUT}/G2_pred_bottom.npy')     # (ntest, 275, 19)
p_top = np.load(f'{OUT}/G2_pred_top.npy')         # (ntest, 19)
p_bot_val = np.load(f'{OUT}/G2_val_pred_bottom.npy')  # (nval, 275, 19)
p_top_val = np.load(f'{OUT}/G2_val_pred_top.npy')      # (nval, 19)
y_true = np.load(f'{OUT}/G2_y_true.npy')           # (ntest, 284)
y_val_true = np.load(f'{OUT}/G2_y_val_true.npy')   # (nval, 284)

n_bottom = p_bot.shape[1]  # 275
n_test   = p_bot.shape[0]

# 提取真实值: 底层(前275列) + 顶层(最后一列)
y_bot = y_true[:, :n_bottom]
y_top = y_true[:, -1]
y_bot_val = y_val_true[:, :n_bottom]
y_top_val = y_val_true[:, -1]

# ========== 2. 构建两层 S 矩阵 (276 × 275) ==========
S2 = np.zeros((n_bottom + 1, n_bottom))
S2[:n_bottom, :] = np.eye(n_bottom)   # 底层=自身
S2[-1, :] = 1                          # 顶层=全部底层之和
print(f'S_twolevel: {S2.shape}')

# ========== 3. 标准 MinT 调和 ==========
# 中位数预测 (τ=0.5, index=9)
p_bot_med = p_bot[:, :, 9]
p_top_med = p_top[:, 9]
p_bot_val_med = p_bot_val[:, :, 9]
p_top_val_med = p_top_val[:, 9]

y_hat = np.column_stack([p_bot_med, p_top_med])    # (ntest, 276)
y_true_2l = np.column_stack([y_bot, y_top])
y_hat_val = np.column_stack([p_bot_val_med, p_top_val_med])
y_true_2l_val = np.column_stack([y_bot_val, y_top_val])

residuals = y_hat_val - y_true_2l_val
pre_inc = float(np.mean(np.abs(p_bot_med.sum(axis=1) - p_top_med)))

from sklearn.covariance import LedoitWolf
W = LedoitWolf().fit(residuals).covariance_

def G_mat(S, W):
    Wi = np.linalg.inv(W + np.eye(W.shape[0])*1e-8)
    return np.linalg.inv(S.T @ Wi @ S) @ S.T @ Wi

G = G_mat(S2, W)
y_rec = (S2 @ G @ y_hat.T).T
post_inc = float(np.mean(np.abs(y_rec[:, :n_bottom].sum(axis=1) - y_rec[:, -1])))

# ========== 4. 评估 ==========
def metrics(yt, yp):
    err = yt - yp
    rmse = np.sqrt(np.mean(err**2)); mae = np.mean(np.abs(err))
    d = np.where(np.abs(yt)>1e-6, np.abs(yt), np.nan)
    mape = np.nanmean(np.abs(err)/d)*100
    return rmse, mae, mape

m_bot = metrics(y_bot.flatten(), p_bot_med.flatten())
m_top = metrics(y_top, p_top_med)

# 全市: 调和前(底层直接加总) vs 调和后(顶层预测 vs 调和后底层加总)
bu_sum = p_bot_med.sum(axis=1)
m_city_before = metrics(y_top, bu_sum)          # 底层加总 vs 真实全市
m_city_after  = metrics(y_top, y_rec[:, -1])     # 调和后顶层 vs 真实全市

# ========== 5. 保存 ==========
results = {
    'experiment': 'G1',
    'description': 'Two-level (no middle) + Independent(275) + Standard MinT + Multi-τ Quantile',
    'covariance_source': 'validation_residuals',
    'S_shape': list(S2.shape),
    'consistency': {
        'before': round(pre_inc, 2),
        'after_shrink': round(post_inc, 10),
    },
    'per_level': {
        'bottom': {'rmse': round(m_bot[0], 2), 'mae': round(m_bot[1], 2), 'mape': round(m_bot[2], 1)},
        'top':    {'rmse': round(m_top[0], 1), 'mae': round(m_top[1], 1), 'mape': round(m_top[2], 1)},
    },
    'city': {
        'rmse_before': round(m_city_before[0], 1),
        'rmse_after':  round(m_city_after[0], 1),
        'improvement_pct': round((m_city_before[0] - m_city_after[0]) / m_city_before[0] * 100, 1),
    },
}

with open(f'{OUT}/G1_results.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
np.save(f'{OUT}/G1_y_rec_shrink.npy', y_rec)

print(f'{"="*60}')
print(f'G1 Results Summary (Two-Level, No Middle)')
print(f'{"="*60}')
print(f'  Bottom:  RMSE={m_bot[0]:.2f}, MAE={m_bot[1]:.2f}, MAPE={m_bot[2]:.1f}%')
print(f'  Top:     RMSE={m_top[0]:.1f}, MAE={m_top[1]:.1f}, MAPE={m_top[2]:.1f}%')
print(f'  City:    {m_city_before[0]:.0f} → {m_city_after[0]:.0f} ({results["city"]["improvement_pct"]:.1f}%)')
print(f'  Consist: {pre_inc:.1f} → {post_inc:.1e}')
print(f'\n[Done] Saved to {OUT}/G1_*')
