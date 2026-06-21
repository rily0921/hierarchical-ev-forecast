"""
评估指标: 点预测 + 概率预测

用法:
    from src.evaluation.metrics import point_metrics, pinball_loss, crps
"""

import numpy as np


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    点预测指标

    参数:
        y_true, y_pred: 形状兼容的数组

    返回:
        {'rmse': float, 'mae': float, 'mape': float}
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    err = yt - yp

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))

    # MAPE: 避免除零
    denom = np.where(np.abs(yt) > 1e-6, np.abs(yt), np.nan)
    mape = float(np.nanmean(np.abs(err) / denom) * 100)

    return {'rmse': rmse, 'mae': mae, 'mape': mape}


def pinball_loss(y_true: np.ndarray, y_pred_tau: np.ndarray,
                 tau: float) -> float:
    """
    单分位数的 Pinball Loss

    QL(τ) = mean( (τ - 1_{y<ŷ}) * (y - ŷ) )
    """
    err = np.asarray(y_true).ravel() - np.asarray(y_pred_tau).ravel()
    return float(np.mean(np.where(err >= 0, tau * err, (tau - 1) * err)))


def multi_pinball(y_true: np.ndarray, q_pred: np.ndarray,
                  taus: list) -> dict:
    """
    多个分位数的 Pinball Loss

    参数:
        y_true: (T,) 或 (T, N)
        q_pred: (T, n_taus) 或 (T, N, n_taus)
        taus:   分位点列表

    返回:
        {f'tau_{t:.2f}': float, ... , 'avg_pinball': float}
    """
    taus_arr = np.asarray(taus)
    result = {}
    total = 0.0
    for i, tau in enumerate(taus_arr):
        if q_pred.ndim == 2:
            qp = q_pred[:, i]
        else:
            qp = q_pred[:, :, i]
        ql = pinball_loss(y_true, qp, tau)
        result[f'tau_{tau:.2f}'] = ql
        total += ql
    result['avg_pinball'] = total / len(taus)
    return result


def crps_from_quantiles(y_true: np.ndarray, q_pred: np.ndarray,
                        taus: list) -> float:
    """
    从分位数预测近似 CRPS (Continuous Ranked Probability Score)

    CRPS = ∫ Pinball(τ) dτ  ≈  mean(Pinball over taus)
    """
    total = 0.0
    for i, tau in enumerate(taus):
        qp = q_pred[:, i] if q_pred.ndim == 2 else q_pred[:, :, i]
        total += pinball_loss(y_true, qp, tau)
    return total / len(taus)


def reconciliation_gain(rmse_base: float, rmse_reconciled: float) -> dict:
    """
    调和增益

    返回:
        {'delta_rmse_pct': float}  — 正=改善, 负=恶化
    """
    delta_pct = (rmse_base - rmse_reconciled) / rmse_base * 100
    return {'delta_rmse_pct': round(delta_pct, 2)}


def all_point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """兼容旧代码的返回值顺序 (rmse, mae, mape)"""
    m = point_metrics(y_true, y_pred)
    return m['rmse'], m['mae'], m['mape']
