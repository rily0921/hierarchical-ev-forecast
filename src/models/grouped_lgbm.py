"""
LightGBM 分组建模训练器 — 组内共享参数 + TAZ embedding

用于: 底层分组建模 (E3: 8行政组, E5/E6/E7/E8: 10行为组或交叉组)

核心思路:
  - 同组 TAZ 的训练样本拼接在一起
  - 加入 categorical feature (TAZ ID) 区分组内不同节点
  - 一套参数服务组内多个 TAZ → 缓解稀疏节点的数据不足问题

用法:
    trainer = GroupedQuantileTrainer(taus, lgb_params, embed_dim=4, early_stop=30)
    p_val, p_test = trainer.train_one_group(
        members, occ_bottom, feature_builder, n_train, n_val
    )
    # p_val:  (n_val,  n_members, n_taus)
    # p_test: (n_test, n_members, n_taus)
"""

import numpy as np
import lightgbm as lgb

from .lgbm_quantile import IndependentQuantileTrainer


class GroupedQuantileTrainer:
    """分组建模：组内 TAZ 共享参数 + TAZ ID 标识"""

    def __init__(self, taus: list, lgb_params: dict,
                 embed_dim: int = 4, early_stop: int = 30,
                 n_jobs_lgb: int = 4):
        """
        taus:        分位点列表
        lgb_params:  LightGBM 参数字典
        embed_dim:   TAZ ID 用作 categorical feature 时的维度 (LightGBM 自动处理)
        early_stop:  早停轮数
        n_jobs_lgb:  LightGBM 内部线程数
        """
        self.taus = taus
        self.lgb_params = lgb_params
        self.embed_dim = embed_dim
        self.early_stop = early_stop
        self.n_jobs_lgb = n_jobs_lgb
        # 独立训练器 (用于单节点组降级)
        self._indep_trainer = IndependentQuantileTrainer(
            taus, lgb_params, early_stop
        )

    def train_one_group(self, members: list, occ_bottom: np.ndarray,
                        feature_builder, n_train: int, n_val: int) -> tuple:
        """
        训练一个分组的 19τ 分位数模型

        参数:
            members:        该组 TAZ 的列索引列表 [3, 17, 42, ...]
            occ_bottom:     (T, n_bottom) 底层 occupancy
            feature_builder: FeatureBuilder 实例
            n_train:        训练集样本数
            n_val:          验证集样本数

        返回:
            p_val:  (n_val, n_members, n_taus)  验证集预测
            p_test: (n_test, n_members, n_taus)  测试集预测
        """
        n_mem = len(members)
        n_taus = len(self.taus)

        # ── 单节点组：降级为独立建模 ──
        if n_mem == 1:
            pv, pt = self._indep_trainer.train(
                occ_bottom[:, members[0]], feature_builder, n_train, n_val
            )
            return pv[:, np.newaxis, :], pt[:, np.newaxis, :]

        # ── 多节点组：拼接数据 + TAZ embedding ──
        X_train_list, y_train_list = [], []
        X_val_list, y_val_list = [], []
        X_test_list = []
        n_test = None  # 将在循环中确定

        for local_id, col_idx in enumerate(members):
            X_all, y_all = feature_builder.build(occ_bottom[:, col_idx])
            # TAZ ID 作为 categorical feature (放在最后一列)
            taz_id = np.full((len(X_all), 1), local_id, dtype=np.float32)
            X_all = np.hstack([X_all, taz_id])

            X_train_list.append(X_all[:n_train])
            y_train_list.append(y_all[:n_train])
            X_val_list.append(X_all[n_train:n_train + n_val])
            y_val_list.append(y_all[n_train:n_train + n_val])
            if n_test is None:
                n_test = len(X_all) - n_train - n_val
            X_test_list.append(X_all[n_train + n_val:])

        # 拼接所有 TAZ 的训练/验证数据
        X_train = np.vstack(X_train_list).astype(np.float32)
        y_train = np.hstack(y_train_list).astype(np.float32)
        X_val   = np.vstack(X_val_list).astype(np.float32)
        y_val   = np.hstack(y_val_list).astype(np.float32)

        # ── 训练 19τ ──
        cb = lgb.early_stopping(self.early_stop)
        cat_feat_idx = X_train.shape[1] - 1  # TAZ ID 在最后一列

        p_val = np.zeros((n_val, n_mem, n_taus), dtype=np.float32)
        p_test = np.zeros((n_test, n_mem, n_taus), dtype=np.float32)

        # 清理 lgb_params: 移除可能与构造函数参数冲突的键
        clean_params = {k: v for k, v in self.lgb_params.items()
                        if k not in ('categorical_feature', 'objective', 'alpha')}

        for i, tau in enumerate(self.taus):
            model = lgb.LGBMRegressor(
                objective='quantile', alpha=tau,
                categorical_feature=[cat_feat_idx],
                **clean_params,
            )
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)], callbacks=[cb])

            # 对组内每个 TAZ 分别预测 (使用各自的 TAZ ID)
            for local_id in range(n_mem):
                p_val[:, local_id, i] = model.predict(X_val_list[local_id])
                p_test[:, local_id, i] = model.predict(X_test_list[local_id])

        # ── 单调重排列 (每个 TAZ 独立) ──
        p_val.sort(axis=2)
        p_test.sort(axis=2)

        return p_val, p_test

    def train_all_groups(self, group_members: list,
                         occ_bottom: np.ndarray,
                         feature_builder,
                         n_train: int, n_val: int,
                         n_jobs_parallel: int = 10) -> tuple:
        """
        并行训练所有分组

        参数:
            group_members:   [ [TAZ indices for group 0], [...], ... ]
            occ_bottom:      (T, n_bottom)
            feature_builder: FeatureBuilder 实例
            n_train:         训练集样本数
            n_val:           验证集样本数
            n_jobs_parallel: joblib 并行数

        返回:
            p_bot_val:  (n_val, n_bottom, n_taus)
            p_bot:      (n_test, n_bottom, n_taus)
        """
        from joblib import Parallel, delayed

        n_groups = len(group_members)
        results = Parallel(n_jobs=min(n_jobs_parallel, n_groups), verbose=10)(
            delayed(self.train_one_group)(
                members, occ_bottom, feature_builder, n_train, n_val
            )
            for members in group_members
        )

        # 将各组结果按 TAZ 原始顺序拼回
        n_bottom = occ_bottom.shape[1]
        n_taus = len(self.taus)
        n_val_actual = results[0][0].shape[0]
        n_test_actual = results[0][1].shape[0]

        p_bot_val = np.zeros((n_val_actual, n_bottom, n_taus), dtype=np.float32)
        p_bot = np.zeros((n_test_actual, n_bottom, n_taus), dtype=np.float32)

        for g, (p_val_g, p_test_g) in enumerate(results):
            for local_id, col_idx in enumerate(group_members[g]):
                p_bot_val[:, col_idx, :] = p_val_g[:, local_id, :]
                p_bot[:, col_idx, :] = p_test_g[:, local_id, :]

        return p_bot_val, p_bot
