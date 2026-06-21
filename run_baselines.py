"""
补充基线: Top-Down调和 + OLS调和 + Seasonal Naive
在 E2/E5/E7 上运行, 对比已有 MinT/GA-MinT
"""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, json, time, warnings, os
warnings.filterwarnings('ignore')
os.environ['LGB_VERBOSITY'] = '-1'

from src.reconciliation.mint import MinTShrink, GAMinT_BD
from src.evaluation.metrics import point_metrics, reconciliation_gain, per_level_gain, diebold_mariano

OUT = 'output'
n_b = 275
TAUS = np.arange(0.05, 1.0, 0.05)
tm = len(TAUS) // 2

# ---- 基线实现 ----

class TopDownReconciler:
    """Top-Down 调和: 用历史比例将顶层预测分解到各层"""
    def __init__(self, S, proportions=None):
        self.S = S
        self.proportions = proportions  # (N_bottom,) 每个底层节点的历史占比

    def fit(self, y_train_bottom):
        """用训练集计算历史比例"""
        total = y_train_bottom.sum(axis=1, keepdims=True) + 1e-8
        self.proportions = np.mean(y_train_bottom / total, axis=0)
        self.proportions = self.proportions / self.proportions.sum()
        return self

    def reconcile(self, y_hat):
        """y_hat: (T, N_total) 顶层预测在最后一列"""
        top_hat = y_hat[:, -1]  # 使用独立顶层预测
        b_tilde = top_hat.reshape(-1, 1) * self.proportions.reshape(1, -1)
        return (self.S @ b_tilde.T).T


class SeasonalNaive:
    """季节性 Naive 基线: 168h 前 (一周前同时刻) 的值作为预测"""
    def __init__(self, period=168):
        self.period = period

    def predict(self, y, n_test):
        """y: (T_total,) 历史序列"""
        pred = np.zeros(n_test)
        for t in range(n_test):
            pred[t] = y[-self.period + t % self.period]
        return pred


# ---- 数据 ----
print('Loading data...')
occ = pd.read_csv('E:/Desktop/毕业论文/data/UrbanEV-main/UrbanEV-main/data/occupancy.csv')
occ['time'] = pd.to_datetime(occ['time']); occ = occ.set_index('time')
zone_cols = list(occ.columns)
occ_bot = occ[zone_cols].values  # (4344, 275)
y_train = occ_bot[:2880, :]
y_test = occ_bot[3600:, :]

# ---- 对每个实验运行基线 ----
results = []

for exp_dir, exp_name, S_key in [
    ('E5_20260621_121531', 'E5', 'S_behavior.npy'),
    ('E7_20260621_133716', 'E7', 'S_behavior.npy'),
    ('E2_20260621_121531', 'E2', 'S_admin.npy'),   # 用旧G2结果
]:
    S = np.load(f'{OUT}/{S_key}')
    n_m = S.shape[0] - n_b - 1

    # 加载基预测
    try:
        preds = f'{OUT}/{exp_dir}/predictions'
        p_bot = np.load(f'{preds}/pred_bottom.npy')
        p_mid = np.load(f'{preds}/pred_middle.npy')
        p_top = np.load(f'{preds}/pred_top.npy')
        y_true = np.load(f'{preds}/y_true.npy')
        p_bot_v = np.load(f'{preds}/pred_bottom_val.npy')
        p_mid_v = np.load(f'{preds}/pred_middle_val.npy')
        p_top_v = np.load(f'{preds}/pred_top_val.npy')
        y_val = np.load(f'{preds}/y_val_true.npy')
    except FileNotFoundError:
        # E2 用旧G2格式
        p_bot = np.load(f'{OUT}/G2_pred_bottom.npy')
        p_mid = np.load(f'{OUT}/G2_pred_middle.npy')
        p_top = np.load(f'{OUT}/G2_pred_top.npy')
        y_true = np.load(f'{OUT}/G2_y_true.npy')
        p_bot_v = np.load(f'{OUT}/G2_val_pred_bottom.npy')
        p_mid_v = np.load(f'{OUT}/G2_val_pred_middle.npy')
        p_top_v = np.load(f'{OUT}/G2_val_pred_top.npy')
        y_val = np.load(f'{OUT}/G2_y_val_true.npy')

    y_hat = np.column_stack([p_bot[:, :, tm], p_mid[:, :, tm] if n_m > 0 else np.zeros((744, 0)), p_top[:, tm]])
    y_hat_v = np.column_stack([p_bot_v[:, :, tm], p_mid_v[:, :, tm] if n_m > 0 else np.zeros((720, 0)), p_top_v[:, tm]])
    res = y_hat_v - y_val
    y_top = y_true[:, -1]
    bu_sum = y_hat[:, :n_b].sum(axis=1)
    m_bu = point_metrics(y_top, bu_sum)

    print(f'\n{exp_name}: BU RMSE={m_bu["rmse"]:.1f}')
    print(f'  {"Method":<16} {"RMSE":>7} {"Delta%":>7}  {"DM p":>6}  {"MidGain":>8}')
    print(f'  {"-"*52}')

    # 1. Top-Down
    td = TopDownReconciler(S)
    td.fit(y_train)
    y_td = td.reconcile(y_hat)
    city_td = y_td[:, :n_b].sum(axis=1)
    m_td = point_metrics(y_top, city_td)
    g_td = reconciliation_gain(m_bu['rmse'], m_td['rmse'])
    dm_td = diebold_mariano(y_top - bu_sum, y_top - city_td)
    pg_td = per_level_gain(y_true, y_hat, y_td, S, n_b)

    print(f'  {"Top-Down":<16} {m_td["rmse"]:>7.1f} {g_td["delta_rmse_pct"]:>+6.1f}%  '
          f'{dm_td["p_value"]:>5.3f}  {pg_td["middle"]["delta_pct"]:>+7.1f}%')

    # 2. OLS (MinT with W=I)
    from src.reconciliation.mint import MinTDiag
    ols = MinTDiag(S)
    ols.fit(res)
    y_ols = ols.reconcile(y_hat)
    city_ols = y_ols[:, :n_b].sum(axis=1)
    m_ols = point_metrics(y_top, city_ols)
    g_ols = reconciliation_gain(m_bu['rmse'], m_ols['rmse'])
    dm_ols = diebold_mariano(y_top - bu_sum, y_top - city_ols)
    pg_ols = per_level_gain(y_true, y_hat, y_ols, S, n_b)

    print(f'  {"OLS (W=I)":<16} {m_ols["rmse"]:>7.1f} {g_ols["delta_rmse_pct"]:>+6.1f}%  '
          f'{dm_ols["p_value"]:>5.3f}  {pg_ols["middle"]["delta_pct"]:>+7.1f}%')

    # 3. MinT-Shrink (已有, 用于对比)
    mint = MinTShrink(S)
    mint.fit(res)
    y_mint = mint.reconcile(y_hat)
    city_mint = y_mint[:, :n_b].sum(axis=1)
    m_mint = point_metrics(y_top, city_mint)
    g_mint = reconciliation_gain(m_bu['rmse'], m_mint['rmse'])
    dm_mint = diebold_mariano(y_top - bu_sum, y_top - city_mint)
    pg_mint = per_level_gain(y_true, y_hat, y_mint, S, n_b)

    print(f'  {"MinT-Shrink":<16} {m_mint["rmse"]:>7.1f} {g_mint["delta_rmse_pct"]:>+6.1f}%  '
          f'{dm_mint["p_value"]:>5.3f}  {pg_mint["middle"]["delta_pct"]:>+7.1f}%')

    # 4. Seasonal Naive
    sn = SeasonalNaive(period=168)
    naive_top = sn.predict(occ_bot[:3600, :].sum(axis=1), 744)
    m_naive = point_metrics(y_top, naive_top)
    g_naive = reconciliation_gain(m_bu['rmse'], m_naive['rmse'])

    print(f'  {"Seasonal Naive":<16} {m_naive["rmse"]:>7.1f} {g_naive["delta_rmse_pct"]:>+6.1f}%')

    results.append({
        'experiment': exp_name,
        'BU_rmse': m_bu['rmse'],
        'top_down': {'rmse': round(m_td['rmse'], 1), 'delta': g_td['delta_rmse_pct']},
        'ols': {'rmse': round(m_ols['rmse'], 1), 'delta': g_ols['delta_rmse_pct']},
        'mint_shrink': {'rmse': round(m_mint['rmse'], 1), 'delta': g_mint['delta_rmse_pct']},
        'seasonal_naive': {'rmse': round(m_naive['rmse'], 1), 'delta': g_naive['delta_rmse_pct']},
    })

# 汇总
print('\n' + '=' * 60)
print('BASELINE SUMMARY')
print('=' * 60)
print(f'  {"Exp":<6} {"BU":>7} {"TopDown":>8} {"OLS":>8} {"MinT":>8} {"SeasNaive":>10}')
print(f'  {"-"*47}')
for r in results:
    print(f'  {r["experiment"]:<6} {r["BU_rmse"]:>7.1f} {r["top_down"]["rmse"]:>8.1f} '
          f'{r["ols"]["rmse"]:>8.1f} {r["mint_shrink"]["rmse"]:>8.1f} {r["seasonal_naive"]["rmse"]:>10.1f}')

with open(OUT + '/baseline_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved to {OUT}/baseline_results.json')
