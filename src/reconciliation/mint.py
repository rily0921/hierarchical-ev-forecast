"""
层次调和模块: MinT + GA-MinT

包含:
  - MinTDiag       : 对角协方差 MinT
  - MinTShrink     : Ledoit-Wolf 收缩 MinT (当前论文使用)
  - GAMinT_BD      : GA-MinT 块对角 (新增)
  - GAMinT_GAS     : GA-MinT 分组感知收缩 (新增)
  - QuantileMinT   : 分位数特定协方差 MinT (E6)

用法:
    rec = MinTShrink(S)
    rec.fit(residuals_val)               # 从验证集残差估计 W → 计算 G
    reconciled = rec.reconcile(y_hat)     # y_hat: (T, N_total) → (T, N_total)
"""

import numpy as np
from sklearn.covariance import LedoitWolf
from abc import ABC, abstractmethod


# ═══════════════════════════════════════════════════════════════
# 基类
# ═══════════════════════════════════════════════════════════════

class BaseReconciler(ABC):
    """调和基类"""

    def __init__(self, S: np.ndarray, min_eig: float = 1e-8):
        """
        S:       (N_total, N_bottom) 汇总矩阵
        min_eig: 矩阵求逆时的正则化项
        """
        self.S = S
        self.min_eig = min_eig
        self.G = None   # 调和矩阵 (拟合后设置)
        self.W = None   # 协方差矩阵 (拟合后设置)

    @abstractmethod
    def fit(self, residuals: np.ndarray):
        """从 (T_val, N_total) 验证集残差估计 W 并计算 G"""
        pass

    def reconcile(self, y_hat: np.ndarray) -> np.ndarray:
        """
        调和基预测

        参数:
            y_hat: (T, N_total) 或 (N_total,)  基预测向量

        返回:
            y_tilde: 同形状的调和后预测
        """
        if self.G is None:
            raise RuntimeError("Must call .fit() before .reconcile()")
        was_1d = y_hat.ndim == 1
        if was_1d:
            y_hat = y_hat.reshape(-1, 1)
        # y_tilde = S @ G @ y_hat
        result = (self.S @ self.G @ y_hat.T).T
        return result.ravel() if was_1d else result

    # ── 内部方法 ──────────────────────────────────
    def _compute_G(self, W: np.ndarray) -> np.ndarray:
        """G = (S^T W^{-1} S)^{-1} S^T W^{-1}"""
        W_reg = W + np.eye(W.shape[0]) * self.min_eig
        W_inv = np.linalg.inv(W_reg)
        STS_inv = np.linalg.inv(self.S.T @ W_inv @ self.S)
        return STS_inv @ self.S.T @ W_inv


# ═══════════════════════════════════════════════════════════════
# 标准 MinT
# ═══════════════════════════════════════════════════════════════

class MinTDiag(BaseReconciler):
    """对角协方差 MinT (假设各节点误差独立)"""

    def fit(self, residuals: np.ndarray):
        var = np.var(residuals, axis=0)
        self.W = np.diag(var)
        self.G = self._compute_G(self.W)
        return self


class MinTShrink(BaseReconciler):
    """Ledoit-Wolf 收缩 MinT (当前论文使用)"""

    def fit(self, residuals: np.ndarray):
        lw = LedoitWolf()
        lw.fit(residuals)
        self.W = lw.covariance_
        self.G = self._compute_G(self.W)
        return self


# ═══════════════════════════════════════════════════════════════
# GA-MinT
# ═══════════════════════════════════════════════════════════════

class GAMinT_BD(BaseReconciler):
    """
    GA-MinT 变体 1: 纯块对角

    假设: 不同分组节点的预测误差相互独立
    → W 为块对角矩阵
    → 每组独立估计组内协方差
    → 参数从 ~38000 降到 ~4000
    """

    def __init__(self, S: np.ndarray, group_labels: np.ndarray,
                 min_eig: float = 1e-8):
        """
        group_labels: (N_bottom,) int  [0, 0, 1, 1, 2, ...]
        """
        super().__init__(S, min_eig)
        self.group_labels = group_labels
        self.n_groups = len(np.unique(group_labels))
        self.W_blocks = []  # 每组的协方差矩阵 (用于诊断)

    def fit(self, residuals: np.ndarray):
        n_bottom = self.S.shape[1]
        n_total = residuals.shape[1]

        # 只对底层节点做块对角 (中层和顶层节点数少，用全协方差)
        W_bottom = self._estimate_block_diag(residuals[:, :n_bottom])
        # 中顶层用样本协方差 (维度小，不存在估计困难)
        if n_total > n_bottom:
            W_upper = np.cov(residuals[:, n_bottom:], rowvar=False)
            # 拼装: 块对角 (底层 + 中顶层)，组间交叉项归零
            self.W = np.zeros((n_total, n_total))
            self.W[:n_bottom, :n_bottom] = W_bottom
            self.W[n_bottom:, n_bottom:] = W_upper
        else:
            self.W = W_bottom

        self.G = self._compute_G(self.W)
        return self

    def _estimate_block_diag(self, residuals_bottom: np.ndarray) -> np.ndarray:
        """估计底层节点的块对角协方差"""
        n_bottom = residuals_bottom.shape[1]
        W = np.zeros((n_bottom, n_bottom))
        self.W_blocks = []

        for g in range(self.n_groups):
            idx = np.where(self.group_labels == g)[0]
            if len(idx) == 0:
                continue
            E_g = residuals_bottom[:, idx]                 # (T_val, n_g)
            E_g = E_g - E_g.mean(axis=0, keepdims=True)   # 中心化
            W_g = (E_g.T @ E_g) / (E_g.shape[0] - 1)      # n_g × n_g
            self.W_blocks.append(W_g)
            W[np.ix_(idx, idx)] = W_g

        return W


class GAMinT_GAS(BaseReconciler):
    """
    GA-MinT 变体 4: 向块对角收缩

    不强制组间独立，而是将 Ledoit-Wolf 的收缩目标
    从"对角矩阵"改为"块对角矩阵"。

    效果: 组间协方差被收缩但不归零，组内保持全协方差结构
    """

    def __init__(self, S: np.ndarray, group_labels: np.ndarray,
                 min_eig: float = 1e-8):
        super().__init__(S, min_eig)
        self.group_labels = group_labels
        self.n_groups = len(np.unique(group_labels))

    def fit(self, residuals: np.ndarray):
        n_bottom = self.S.shape[1]
        n_total = residuals.shape[1]

        # 样本协方差
        W_sam = np.cov(residuals, rowvar=False)

        # 收缩目标: 从样本协方差提取块对角结构
        W_target = np.zeros_like(W_sam)

        # 底层：块对角 (分组结构)
        for g in range(self.n_groups):
            idx = np.where(self.group_labels == g)[0]
            W_target[np.ix_(idx, idx)] = W_sam[np.ix_(idx, idx)]

        # 中顶层：保持全协方差 (维度小)
        if n_total > n_bottom:
            upper_idx = np.arange(n_bottom, n_total)
            W_target[np.ix_(upper_idx, upper_idx)] = W_sam[np.ix_(upper_idx, upper_idx)]

        # 计算最优收缩强度 τ (Ledoit-Wolf 公式)
        tau = self._compute_lw_tau(residuals, W_target)

        # 向块对角收缩
        self.W = tau * W_target + (1 - tau) * W_sam
        self.G = self._compute_G(self.W)
        return self

    def _compute_lw_tau(self, residuals: np.ndarray,
                        target: np.ndarray) -> float:
        """计算向 target 收缩的 Ledoit-Wolf 最优 τ"""
        n, p = residuals.shape
        S = np.cov(residuals, rowvar=False)

        # 分子: 协方差估计的方差
        # 使用 Ledoit-Wolf 的近似
        d2 = np.sum((S - target) ** 2)

        # 分母: E[||S - Σ||²] 的估计
        # 简化版本: 用 bootstrap 思想
        b2 = 0.0
        residuals_centered = residuals - residuals.mean(axis=0)
        for t in range(n):
            e_t = residuals_centered[t, :]
            diff = np.outer(e_t, e_t) - S
            b2 += np.sum(diff ** 2)
        b2 /= (n ** 2)

        return min(b2 / max(d2, 1e-12), 1.0)


# ═══════════════════════════════════════════════════════════════
# 分位数特定 MinT (E6)
# ═══════════════════════════════════════════════════════════════

class QuantileSpecificMinT:
    """
    对 low/mid/high 三档分位数分别估计协方差

    与 G8 逻辑一致:
      - low  (τ ≤ 0.3):  Σ_low
      - mid  (0.3 < τ ≤ 0.7): Σ_mid
      - high (τ > 0.7): Σ_high
    """

    def __init__(self, S: np.ndarray, tau_bands: dict = None):
        """
        tau_bands: {'low': [τ ≤ 0.3], 'mid': [0.3 < τ ≤ 0.7], 'high': [τ > 0.7]}
        """
        self.S = S
        self.tau_bands = tau_bands or {
            'low':  lambda t: t <= 0.3,
            'mid':  lambda t: 0.3 < t <= 0.7,
            'high': lambda t: t > 0.7,
        }
        self.reconcilers = {}  # {band_name: BaseReconciler}

    def fit(self, quantile_errors: dict, reconciler_cls=MinTShrink):
        """
        quantile_errors: {tau: residuals_array}  各分位数的验证集残差
        """
        for band_name, band_fn in self.tau_bands.items():
            # 聚合该档所有 τ 的残差
            band_residuals = []
            for tau, residuals in quantile_errors.items():
                if band_fn(tau):
                    band_residuals.append(residuals)
            if band_residuals:
                combined = np.vstack(band_residuals)
                rec = reconciler_cls(self.S)
                rec.fit(combined)
                self.reconcilers[band_name] = rec
        return self

    def reconcile(self, y_hat: np.ndarray, tau: float) -> np.ndarray:
        """对指定 τ 用对应档的 G 调和"""
        for band_name, band_fn in self.tau_bands.items():
            if band_fn(tau):
                if band_name in self.reconcilers:
                    return self.reconcilers[band_name].reconcile(y_hat)
        # 默认 fallback: 用 mid
        if 'mid' in self.reconcilers:
            return self.reconcilers['mid'].reconcile(y_hat)
        raise RuntimeError("No reconciler available")
