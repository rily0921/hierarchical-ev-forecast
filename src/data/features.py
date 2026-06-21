"""
特征工程：对单条时间序列构造 (滞后特征 + 滚动统计 + 全局特征)

用法:
    fb = FeatureBuilder(global_feat, lag_hours=[1,2,3,24,48,168],
                        roll_hours=[6,24], max_lag=168)
    X, y = fb.build(ts)   # ts: (T,) → X: (T-max_lag, n_feat), y: (T-max_lag,)
"""

import numpy as np


class FeatureBuilder:
    """对单条时间序列构造特征矩阵"""

    def __init__(self, global_feat: np.ndarray,
                 lag_hours: list = None,
                 roll_hours: list = None,
                 max_lag: int = 168):
        """
        global_feat: (T, n_global)  全局特征（时间+天气）
        lag_hours:   [int]          滞后阶数
        roll_hours:  [int]          滚动窗口大小
        max_lag:     int            最大滞后 (= max(lag_hours))
        """
        self.global_feat = global_feat
        self.lag_hours = lag_hours or [1, 2, 3, 24, 48, 168]
        self.roll_hours = roll_hours or [6, 24]
        self.max_lag = max_lag

        # 特征维度: 全局 + 滞后 + 2×滚动
        self.n_global = global_feat.shape[1]
        self.n_lag = len(self.lag_hours)
        self.n_roll = 2 * len(self.roll_hours)
        self.n_features = self.n_global + self.n_lag + self.n_roll

    def build(self, ts: np.ndarray) -> tuple:
        """
        对单条时间序列构造特征

        参数:
            ts: (T,) 一维时间序列

        返回:
            X: (T - max_lag, n_features)  float32
            y: (T - max_lag,)             float32
        """
        T = len(ts)
        n_samples = T - self.max_lag
        X = np.zeros((n_samples, self.n_features), dtype=np.float32)

        # 全局特征 (列 0 ~ n_global-1)
        X[:, :self.n_global] = self.global_feat[self.max_lag:T, :]

        col = self.n_global

        # 滞后特征
        for lag in self.lag_hours:
            X[:, col] = ts[self.max_lag - lag: T - lag]
            col += 1

        # 滚动统计 (mean + std)
        for roll in self.roll_hours:
            for t_idx, t in enumerate(range(self.max_lag, T)):
                window = ts[t - roll: t]
                X[t_idx, col] = np.mean(window)
                X[t_idx, col + 1] = np.std(window)
            col += 2

        y = ts[self.max_lag:].astype(np.float32)
        return X, y
