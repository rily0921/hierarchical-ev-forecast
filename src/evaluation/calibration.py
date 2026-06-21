"""
概率预测校准评估: PIT 直方图 / 可靠性曲线 / 区间覆盖率

校准 (calibration) 衡量预测概率与实际频率的一致性。
结合 Pinball Loss (锐度 sharpness) 可以完整评估概率预测质量。

参考: Gneiting & Raftery (2007), Gneiting & Katzfuss (2014)

用法:
    from src.evaluation.calibration import pit_histogram, reliability_diagram, interval_coverage
"""

import numpy as np


def pit_histogram(y_true: np.ndarray, q_pred: np.ndarray,
                  taus: np.ndarray, n_bins: int = 20) -> dict:
    """
    PIT (Probability Integral Transform) 直方图

    原理:
      对每个观测 y_t，求 F(y_t) = 在预测分布中低于 y_t 的概率
      如果预测分布是真实分布的完美估计，F(y_t) ~ Uniform(0,1)
      - 均匀  → 校准良好
      - U 形   → 分散不足 (underdispersed, 预测区间太窄)
      - 倒 U 形 → 分散过度 (overdispersed, 预测区间太宽)

    参数:
        y_true: (T,) 或 (T, N)  实际值
        q_pred: (T, n_taus) 或 (T, N, n_taus)  分位数预测
        taus:   (n_taus,) 分位点
        n_bins: PIT 直方图的分箱数

    返回:
        {'counts': ndarray, 'bin_edges': ndarray, 'pit_values': ndarray}
    """
    taus = np.asarray(taus)

    # 将多维展平为 (total_points, n_taus)
    if q_pred.ndim == 3:
        T, N, K = q_pred.shape
        q_flat = q_pred.reshape(-1, K)       # (T*N, K)
        y_flat = np.asarray(y_true).ravel()  # (T*N,)
    else:
        q_flat = q_pred
        y_flat = np.asarray(y_true).ravel()

    # 线性插值求 PIT 值: F(y_t)
    pit_values = np.full(len(y_flat), np.nan)
    for i in range(len(y_flat)):
        pit_values[i] = np.interp(y_flat[i], q_flat[i, :], taus)
    pit_values = np.clip(pit_values, 0, 1)

    # 去除 NaN
    pit_values = pit_values[~np.isnan(pit_values)]

    # 直方图
    counts, bin_edges = np.histogram(pit_values, bins=n_bins, range=(0, 1))
    # 归一化到概率密度
    densities = counts / counts.sum() * n_bins

    return {
        'counts': counts,
        'densities': densities,
        'bin_edges': bin_edges,
        'pit_values': pit_values,
        'n_obs': len(pit_values),
    }


def reliability_diagram(y_true: np.ndarray, q_pred: np.ndarray,
                        taus: np.ndarray,
                        coverage_levels: np.ndarray = None) -> dict:
    """
    可靠性曲线 (Reliability Diagram)

    名义覆盖率 vs 实际覆盖率
    理想情况: y = x 对角线

    参数:
        coverage_levels: 评估的覆盖率，默认 [0.1, 0.2, ..., 0.9]

    返回:
        {'nominal': ndarray, 'actual': ndarray, 'avg_widths': ndarray}
    """
    taus = np.asarray(taus)
    if coverage_levels is None:
        coverage_levels = np.arange(0.1, 1.0, 0.1)

    if q_pred.ndim == 3:
        T, N, K = q_pred.shape
        q_flat = q_pred.reshape(-1, K)
        y_flat = np.asarray(y_true).ravel()
    else:
        q_flat = q_pred
        y_flat = np.asarray(y_true).ravel()

    actual = np.zeros(len(coverage_levels))
    avg_widths = np.zeros(len(coverage_levels))

    for i, level in enumerate(coverage_levels):
        lower_frac = (1 - level) / 2
        upper_frac = (1 + level) / 2

        lower_idx = np.searchsorted(taus, lower_frac)
        upper_idx = min(np.searchsorted(taus, upper_frac), len(taus) - 1)

        lower = q_flat[:, lower_idx]
        upper = q_flat[:, upper_idx]

        actual[i] = np.mean((y_flat >= lower) & (y_flat <= upper))
        avg_widths[i] = np.mean(upper - lower)

    return {
        'nominal': coverage_levels,
        'actual': actual,
        'avg_widths': avg_widths,
    }


def interval_coverage(y_true: np.ndarray, q_pred: np.ndarray,
                      taus: np.ndarray,
                      levels: list = None) -> dict:
    """
    各置信水平的预测区间覆盖率

    参数:
        levels: 评估的置信水平，默认 [0.50, 0.80, 0.90, 0.95, 0.99]

    返回:
        {level: {'nominal': float, 'actual': float, 'avg_width': float}, ...}
    """
    taus = np.asarray(taus)
    if levels is None:
        levels = [0.50, 0.80, 0.90, 0.95, 0.99]

    if q_pred.ndim == 3:
        T, N, K = q_pred.shape
        q_flat = q_pred.reshape(-1, K)
        y_flat = np.asarray(y_true).ravel()
    else:
        q_flat = q_pred
        y_flat = np.asarray(y_true).ravel()

    results = {}
    for level in levels:
        lower_frac = (1 - level) / 2
        upper_frac = (1 + level) / 2

        lower_idx = np.searchsorted(taus, lower_frac)
        upper_idx = min(np.searchsorted(taus, upper_frac), len(taus) - 1)

        lower = q_flat[:, lower_idx]
        upper = q_flat[:, upper_idx]

        coverage = np.mean((y_flat >= lower) & (y_flat <= upper))
        avg_width = np.mean(upper - lower)

        results[f'{level:.0%}'] = {
            'nominal_coverage': level,
            'actual_coverage': round(float(coverage), 4),
            'avg_interval_width': round(float(avg_width), 4),
        }

    return results


def calibration_summary(y_true: np.ndarray, q_pred: np.ndarray,
                        taus: np.ndarray) -> dict:
    """
    生成完整的校准评估摘要 (用于 results.json)

    包含: PIT + 可靠性 + 区间覆盖率
    """
    pit = pit_histogram(y_true, q_pred, taus)
    rel = reliability_diagram(y_true, q_pred, taus)
    cov = interval_coverage(y_true, q_pred, taus)

    # PIT 偏差指标: 偏离均匀分布的程度
    # 理想: 每个 bin 的 density = 1.0
    pit_deviation = float(np.mean(np.abs(pit['densities'] - 1.0)))

    return {
        'pit_deviation': round(pit_deviation, 4),
        'pit_max_deviation': round(float(np.max(np.abs(pit['densities'] - 1.0))), 4),
        'pit_n_obs': pit['n_obs'],
        'reliability': {
            'nominal': rel['nominal'].tolist(),
            'actual': [round(x, 4) for x in rel['actual']],
        },
        'interval_coverage': cov,
    }
