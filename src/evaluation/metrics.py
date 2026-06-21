"""
评估指标: 点预测 + 概率预测 + 统计检验

参考文献:
  - sMAPE: Makridakis (1993), J Forecasting
  - MASE: Hyndman & Koehler (2006), IJF
  - DM test: Diebold & Mariano (1995), JBES
"""

import numpy as np
from scipy import stats


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                  y_train: np.ndarray = None, seasonal_period: int = 168) -> dict:
    """
    点预测指标: RMSE, MAE, MAPE, sMAPE, MASE

    参数:
        y_true, y_pred: 形状兼容的数组
        y_train:        训练集真实值 (MASE 分母需要), 可选
        seasonal_period: 季节性周期 (MASE 分母的 naive 预测使用)

    返回:
        {'rmse': float, 'mae': float, 'mape': float,
         'smape': float, 'mape_excluded_pct': float,
         'mase': float or None}
    """
    yt = np.asarray(y_true).ravel().astype(np.float64)
    yp = np.asarray(y_pred).ravel().astype(np.float64)
    err = yt - yp

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))

    # MAPE: 避免除零, 报告排除比例
    abs_yt = np.abs(yt)
    valid = abs_yt > 1e-6
    excluded_pct = float(100 * (1 - np.mean(valid)))
    mape = float(np.mean(np.abs(err[valid]) / abs_yt[valid]) * 100) if np.any(valid) else np.nan

    # sMAPE: 对称 MAPE, 天然处理零值
    denom = (abs_yt + np.abs(yp)) / 2 + 1e-8
    smape = float(np.mean(np.abs(err) / denom) * 100)

    # MASE: MAE / MAE_of_seasonal_naive
    mase = None
    if y_train is not None:
        yt_train = np.asarray(y_train).ravel().astype(np.float64)
        naive_err = np.abs(yt_train[seasonal_period:] - yt_train[:-seasonal_period])
        mae_naive = np.mean(naive_err)
        mase = float(mae / mae_naive) if mae_naive > 0 else None

    return {'rmse': rmse, 'mae': mae, 'mape': mape, 'smape': smape,
            'mape_excluded_pct': excluded_pct, 'mase': mase}


def diebold_mariano(err1: np.ndarray, err2: np.ndarray,
                    loss: str = 'se', max_lag: int = None) -> dict:
    """
    Diebold-Mariano 检验: 比较两种预测方法的误差是否显著不同

    H0: E[L(e1)] = E[L(e2)]  (两种方法精度相同)
    HAC 标准误, 双边检验

    参数:
        err1, err2: (T,) 两种方法的预测误差
        loss:       'se' (squared error) 或 'ae' (absolute error)
        max_lag:    HAC 最大滞后 (默认自动: floor(4*(T/100)^(2/9)))

    返回:
        {'dm_stat': float, 'p_value': float, 'significant_5pct': bool}
    """
    T = len(err1)
    err1 = np.asarray(err1).ravel()
    err2 = np.asarray(err2).ravel()

    # 差分损失
    if loss == 'se':
        d = err1 ** 2 - err2 ** 2
    elif loss == 'ae':
        d = np.abs(err1) - np.abs(err2)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    d_mean = np.mean(d)

    # HAC 标准误 (Newey-West 类型, 截断滞后)
    if max_lag is None:
        max_lag = max(1, int(np.floor(4 * (T / 100) ** (2 / 9))))

    # 自协方差加总
    hac_var = np.var(d) / T  # 基础项
    for lag in range(1, max_lag + 1):
        acf = np.mean((d[lag:] - d_mean) * (d[:-lag] - d_mean))
        weight = 1 - lag / (max_lag + 1)  # Bartlett kernel
        hac_var += 2 * weight * acf / T
    hac_var = max(hac_var, 1e-15)

    dm_stat = d_mean / np.sqrt(hac_var)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))  # 双边
    significant = p_value < 0.05

    return {'dm_stat': float(dm_stat), 'p_value': float(p_value),
            'significant_5pct': significant}


def pinball_loss(y_true, y_pred_tau, tau):
    """Pinball Loss: QL(τ) = mean(τ·(y-ŷ)⁺ + (1-τ)·(ŷ-y)⁺)"""
    err = np.asarray(y_true).ravel() - np.asarray(y_pred_tau).ravel()
    return float(np.mean(np.where(err >= 0, tau * err, (tau - 1) * err)))


def multi_pinball(y_true, q_pred, taus):
    """多分位数 Pinball Loss"""
    total = 0.0
    result = {}
    for i, tau in enumerate(taus):
        qp = q_pred[:, i] if q_pred.ndim == 2 else q_pred[:, :, i]
        ql = pinball_loss(y_true, qp, tau)
        result[f'tau_{tau:.2f}'] = ql
        total += ql
    result['avg_pinball'] = total / len(taus)
    return result


def crps_from_quantiles(y_true, q_pred, taus):
    """CRPS ≈ mean(Pinball Loss over τ)"""
    return multi_pinball(y_true, q_pred, taus)['avg_pinball']


def reconciliation_gain(rmse_base, rmse_reconciled):
    """调和增益: 正=改善, 负=恶化"""
    return {'delta_rmse_pct': round((rmse_base - rmse_reconciled) / rmse_base * 100, 2)}


def per_level_gain(y_true, y_hat_base, y_hat_reconciled, S, n_bottom):
    """
    逐层调和增益: 底层/中层/顶层 分别计算 ΔRMSE%

    参数:
        y_true:        (T, N_total)
        y_hat_base:    (T, N_total) BU 预测 (或基预测)
        y_hat_reconciled: (T, N_total) 调和后预测
        S:             (N_total, N_bottom)
        n_bottom:      底层节点数

    返回:
        {'bottom': dict, 'middle': dict, 'top': dict}  每层含 rmse_before/after 和 delta_pct
    """
    n_total = S.shape[0]
    n_mid = n_total - n_bottom - 1

    levels = {}
    for level_name, cols in [
        ('bottom', slice(0, n_bottom)),
        ('middle', slice(n_bottom, n_bottom + n_mid)),
        ('top', slice(-1, None)),
    ]:
        yt = y_true[:, cols]
        yb = y_hat_base[:, cols]
        yr = y_hat_reconciled[:, cols]
        rmse_before = float(np.sqrt(np.mean((yt - yb) ** 2)))
        rmse_after = float(np.sqrt(np.mean((yt - yr) ** 2)))
        delta = (rmse_before - rmse_after) / rmse_before * 100
        levels[level_name] = {
            'rmse_before': round(rmse_before, 2),
            'rmse_after': round(rmse_after, 2),
            'delta_pct': round(delta, 2),
        }
    return levels


def all_point_metrics(y_true, y_pred):
    """兼容旧代码"""
    m = point_metrics(y_true, y_pred)
    return m['rmse'], m['mae'], m['mape']
