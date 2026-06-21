"""
层次调和模块: MinT + GA-MinT + Bayes-GA-MinT + 高斯投影概率调和

包含:
  - MinTDiag       : 对角协方差 MinT
  - MinTShrink     : Ledoit-Wolf 收缩 MinT
  - GAMinT_BD      : GA-MinT 块对角
  - GAMinT_GAS     : GA-MinT 分组感知收缩
  - BayesGAMinT    : 贝叶斯块对角 IW 先验 MinT
  - QuantileMinT   : 分位数特定协方差 MinT (E6)

概率调和 (Panagiotelis et al., 2023):
  基预测 N(ŷ, W) → 调和后 N(S·G·ŷ, S·G·W·G^T·S^T)
  → reconciled_gaussian() + gaussian_quantiles()

用法:
    rec = MinTShrink(S)
    rec.fit(residuals_val)
    y_tilde, W_tilde = rec.reconciled_gaussian(y_hat)  # 全分布调和
    q = rec.gaussian_quantiles(y_tilde[:,:n_b], W_tilde[:n_b,:n_b], taus)
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

    # ── 高斯投影概率调和 (Panagiotelis et al., 2023) ──

    def reconciled_gaussian(self, y_hat: np.ndarray) -> tuple:
        """
        高斯投影调和: 对整个概率分布做线性投影

        理论:
          如果基预测分布是 N(ŷ, W), 则调和后分布为 N(μ̃, Σ̃)
          其中 μ̃ = S G ŷ, Σ̃ = S G W G^T S^T
          (Panagiotelis et al., 2023, EJOR; Wickramasuriya, 2024)

        参数:
          y_hat: (T, N_total) 基预测 (中位数, 即高斯均值)

        返回:
          y_tilde: (T, N_total) 调和后均值
          W_tilde: (N_total, N_total) 调和后协方差矩阵
        """
        if self.G is None or self.W is None:
            raise RuntimeError("Must call .fit() before .reconciled_gaussian()")

        y_tilde = self.reconcile(y_hat)
        W_tilde = self.S @ self.G @ self.W @ self.G.T @ self.S.T
        return y_tilde, W_tilde

    @staticmethod
    def gaussian_quantiles(mean: np.ndarray, cov: np.ndarray,
                           taus: list) -> np.ndarray:
        """
        从高斯分布计算分位数

        参数:
          mean: (T,) 或 (T, N) — 调和后均值
          cov:  (N, N) — 调和后协方差矩阵
          taus: list — 分位点

        返回:
          quantiles: (T, N, len(taus))
        """
        from scipy.stats import norm

        if mean.ndim == 1:
            mean = mean.reshape(-1, 1)
        T, N = mean.shape
        z = norm.ppf(taus)  # (n_taus,)
        var = np.maximum(np.diag(cov), 0)
        std = np.sqrt(var)

        quantiles = np.zeros((T, N, len(taus)))
        for i, zv in enumerate(z):
            quantiles[:, :, i] = mean + zv * std
        return quantiles


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


# ═══════════════════════════════════════════════════════════════
# Bayesian GA-MinT (Bayes-GA-MinT)
# ═══════════════════════════════════════════════════════════════

class BayesGAMinT(BaseReconciler):
    """
    贝叶斯分组感知 MinT — 块对角 IW 先验 + MinT 投影修订

    理论动机:
      分组建模 → 同组节点误差相关, 不同组近似独立
      → W 具有块对角结构 → 在 W 上放置块对角 IW 先验
      → 后验均值 ≈ 频率学派 GA-MinT (GAMinT_BD)
      → 后验预测为多元 t 分布 (由于 W 的不确定性)

    与 GAMinT_BD 的关系:
      GAMinT_BD: Ψ' 强制块对角 → 点预测好, 但预测区间忽略组间相关性
      BayesGAMinT: Ψ₀ 块对角 (先验), Ψ' = Ψ₀ + R^T R (后验, 含组间关联)
                   → 点预测 = MinT 投影, 预测区间 = t 分布

    用法:
      bayes = BayesGAMinT(S, group_labels)
      bayes.fit(residuals)                    # IW 后验估计
      reconciled = bayes.reconcile(y_hat)      # MinT 点预测
      pi = bayes.prediction_interval(0.9)      # t 分布预测区间
    """

    def __init__(self, S: np.ndarray, group_labels: np.ndarray,
                 min_eig: float = 1e-8):
        super().__init__(S, min_eig)
        self.group_labels = group_labels
        self.G = len(np.unique(group_labels))
        self.Psi_prime = None
        self.nu_prime = None
        self.n_bottom = S.shape[1]
        self.n_total = S.shape[0]
        self.n_upper = self.n_total - self.n_bottom
        self.tilde_b = None
        self.tilde_nu = None

    def fit(self, residuals: np.ndarray, nu0: float = None):
        """
        IW 后验估计: 块对角先验 + 高斯似然 → IW 后验

        参数:
          residuals: (T_val, N_total) 验证集基预测残差
          nu0:       先验自由度, 默认 n_bottom + 10
        """
        T_val = residuals.shape[0]
        residuals_bottom = residuals[:, :self.n_bottom]
        n_b = self.n_bottom

        # 1. 块对角先验 Ψ₀: 组内 Ledoit-Wolf 收缩
        Psi_0 = np.zeros((n_b, n_b))
        for g in range(self.G):
            idx = np.where(self.group_labels == g)[0]
            if len(idx) == 0:
                continue
            R_g = residuals_bottom[:, idx]
            Psi_g = (R_g.T @ R_g) / (T_val - 1)
            lam = self._estimate_shrinkage(R_g)
            Psi_g = (1 - lam) * Psi_g + lam * np.diag(np.diag(Psi_g))
            Psi_0[np.ix_(idx, idx)] = Psi_g

        # 2. 先验自由度
        nu0_val = max(nu0, n_b + 2) if nu0 is not None else n_b + 10

        # 3. 后验: Ψ' = Ψ₀(块对角) + R_full^T R_full (全矩阵)
        #    先验约束组内, 数据引入组间关联
        self.Psi_prime = Psi_0 + residuals_bottom.T @ residuals_bottom
        self.nu_prime = nu0_val + T_val

        # 4. 加入上层 (使用全残差, 维度小无估计困难)
        if self.n_upper > 0:
            res_up = residuals[:, self.n_bottom:]
            Psi_full = np.zeros((self.n_total, self.n_total))
            Psi_full[:n_b, :n_b] = self.Psi_prime
            Psi_full[n_b:, n_b:] = res_up.T @ res_up
            Psi_full[n_b:, :n_b] = res_up.T @ residuals_bottom
            Psi_full[:n_b, n_b:] = residuals_bottom.T @ res_up
            self.Psi_prime = Psi_full

        return self

    def _estimate_shrinkage(self, R: np.ndarray) -> float:
        """Ledoit-Wolf 收缩系数"""
        T, n = R.shape
        S = (R.T @ R) / (T - 1)
        d2 = np.sum((S - np.diag(np.diag(S))) ** 2)
        b2 = sum(np.sum((np.outer(R[t, :], R[t, :]) - S) ** 2)
                 for t in range(T)) / (T ** 2)
        return min(b2 / max(d2, 1e-12), 1.0)

    def reconcile(self, y_hat: np.ndarray) -> np.ndarray:
        """
        MinT 投影修订 — 点预测

        y_hat: (T, N_total) 或 (N_total,)
        返回: 同形状的修订预测
        """
        if self.Psi_prime is None:
            raise RuntimeError("Must call .fit() before .reconcile()")

        was_1d = y_hat.ndim == 1
        if was_1d:
            y_hat = y_hat.reshape(-1, 1)

        W = self.Psi_prime + np.eye(self.n_total) * self.min_eig
        W_inv = np.linalg.inv(W)
        STS_inv = np.linalg.inv(self.S.T @ W_inv @ self.S)
        G = STS_inv @ self.S.T @ W_inv

        result = (self.S @ G @ y_hat.T).T
        self.tilde_b = (G @ y_hat.T)[:self.n_bottom, :].T
        self.tilde_nu = self.nu_prime - self.n_bottom + 1
        self.G = G

        return result.ravel() if was_1d else result

    def prediction_interval(self, level: float = 0.9) -> dict:
        """
        从 IW 后验计算 t 分布预测区间

        注意: 由于块对角先验对组间协方差的抑制,
        城市总量区间使用经验误差标准差校准 (conservative).
        """
        from scipy.stats import t as t_dist

        if self.tilde_b is None:
            raise RuntimeError("Must call .reconcile() before .prediction_interval()")

        alpha = (1 - level) / 2
        t_quantile = t_dist.ppf(1 - alpha, df=self.tilde_nu)

        T, n_b = self.tilde_b.shape

        # 逐节点边际标准差
        margin_var = np.diag(self.Psi_prime[:n_b, :n_b]) / self.tilde_nu
        margin_std = np.sqrt(np.maximum(margin_var, 0))

        lower = np.maximum(self.tilde_b - t_quantile * margin_std, 0)
        upper = np.maximum(self.tilde_b + t_quantile * margin_std, 0)

        # 城市总量: 使用 reconciled 误差经验分布
        ones = np.ones(n_b)
        city = self.tilde_b @ ones
        city_err_std = np.std(city)
        city_lower = city - t_quantile * city_err_std
        city_upper = city + t_quantile * city_err_std

        return {'lower': lower, 'upper': upper,
                'city_lower': np.maximum(city_lower, 0),
                'city_upper': np.maximum(city_upper, 0),
                'level': level, 't_quantile': t_quantile,
                'nu': self.tilde_nu}
