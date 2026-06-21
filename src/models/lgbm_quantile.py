"""
LightGBM 分位数回归训练器 — 独立建模
支持加权分位数损失 (方向2): 对尾部τ给极端观测更高权重
"""

import numpy as np
import lightgbm as lgb


class IndependentQuantileTrainer:
    """对单条时间序列训练 19τ 分位数回归模型"""

    def __init__(self, taus: list, lgb_params: dict, early_stop: int = 30,
                 tail_weight_alpha: float = 0.0):
        """
        taus:              分位点列表, 如 [0.05, 0.10, ..., 0.95]
        lgb_params:        LightGBM 参数字典
        early_stop:        早停轮数
        tail_weight_alpha: 尾部加权强度 (0=等权, 1=标准加权, 2=强加权)
        """
        self.taus = taus
        self.lgb_params = lgb_params
        self.early_stop = early_stop
        self.tail_weight_alpha = tail_weight_alpha

    def train(self, ts: np.ndarray, feature_builder,
              n_train: int, n_val: int) -> tuple:
        X, y = feature_builder.build(ts)
        n_test = len(X) - n_train - n_val
        n_taus = len(self.taus)

        p_val = np.zeros((n_val, n_taus), dtype=np.float32)
        p_test = np.zeros((n_test, n_taus), dtype=np.float32)

        X_tr, y_tr = X[:n_train], y[:n_train]
        X_va, y_va = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
        X_te = X[n_train + n_val:]

        cb = lgb.early_stopping(self.early_stop)

        # 预计算用于加权的统计量
        if self.tail_weight_alpha > 0:
            y_median = np.median(y_tr)
            y_std = np.std(y_tr) + 1e-6

        for i, tau in enumerate(self.taus):
            # 构造样本权重
            sample_weight = None
            if self.tail_weight_alpha > 0:
                sample_weight = self._compute_tail_weights(
                    y_tr, tau, y_median, y_std
                )

            model = lgb.LGBMRegressor(
                objective='quantile', alpha=tau, **self.lgb_params
            )
            model.fit(
                X_tr, y_tr,
                sample_weight=sample_weight,
                eval_set=[(X_va, y_va)],
                callbacks=[cb],
            )
            p_val[:, i] = model.predict(X_va)
            p_test[:, i] = model.predict(X_te)

        # 单调重排列
        p_val.sort(axis=1)
        p_test.sort(axis=1)
        return p_val, p_test

    def _compute_tail_weights(self, y: np.ndarray, tau: float,
                              median: float, std: float) -> np.ndarray:
        """
        尾部感知的样本权重:
          τ > 0.6 (上尾): 高需求观测权重更大
          τ < 0.4 (下尾): 低需求观测权重更大
          τ ∈ [0.4, 0.6]: 等权
        """
        alpha = self.tail_weight_alpha
        base = np.ones(len(y), dtype=np.float32)

        if tau > 0.6:
            # 上尾: 重点关注高需求时段
            deviation = np.maximum(0, y - median) / std
            weight = base + alpha * deviation * (tau - 0.6) / 0.35
        elif tau < 0.4:
            # 下尾: 重点关注低需求时段
            deviation = np.maximum(0, median - y) / std
            weight = base + alpha * deviation * (0.4 - tau) / 0.35
        else:
            weight = base.copy()

        return weight.astype(np.float32)
