"""
实验运行入口

用法:
    python run.py --exp E5                          # 运行单个实验
    python run.py --exp E1,E2,E3,E4,E5,E6           # 运行多个实验
    python run.py --exp E5 --reconcile ga_mint_bd   # 只跑一种调和方式

配置文件: config.yaml (同目录)
"""

import sys
import yaml
import argparse
import time
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')
import os; os.environ['LGB_VERBOSITY'] = '-1'

# 确保 src/ 在路径中
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import DataLoader
from src.data.features import FeatureBuilder
from src.data.hierarchy import HierarchyBuilder
from src.models.lgbm_quantile import IndependentQuantileTrainer
from src.models.grouped_lgbm import GroupedQuantileTrainer
from src.reconciliation.mint import (
    MinTDiag, MinTShrink, GAMinT_BD, GAMinT_GAS, QuantileSpecificMinT
)
from src.evaluation.metrics import (
    point_metrics, multi_pinball, reconciliation_gain
)
from src.evaluation.calibration import calibration_summary
from src.utils.tracker import ExperimentTracker
from joblib import Parallel, delayed


# ═══════════════════════════════════════════════════════════════
# 调和方式注册表
# ═══════════════════════════════════════════════════════════════

def _create_reconciler(name: str, S: np.ndarray, group_labels: np.ndarray):
    """根据名称创建调和方法实例"""
    registry = {
        'mint_diag':    lambda: MinTDiag(S),
        'mint_shrink':  lambda: MinTShrink(S),
        'ga_mint_bd':   lambda: GAMinT_BD(S, group_labels),
        'ga_mint_gas':  lambda: GAMinT_GAS(S, group_labels),
    }
    if name not in registry:
        raise ValueError(f"Unknown reconciliation: {name}. "
                         f"Available: {list(registry.keys())}")
    return registry[name]()


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def run_experiment(exp_name: str, cfg: dict, reconcile_filter: list = None):
    """运行单个实验"""
    exp_cfg = cfg['experiments'][exp_name]
    print(f"\n{'='*60}")
    print(f"  {exp_name}: {exp_cfg['description']}")
    print(f"{'='*60}")

    # ── 1. 加载数据 ──
    print('\n[1/6] Loading data...')
    loader = DataLoader(cfg['paths']['raw_data'],
                        cfg['data']['train_hours'],
                        cfg['data']['val_hours'])
    data = loader.load()
    print(f'  occ_bottom: {data["occ_bottom"].shape}, '
          f'T_train={data["T_train"]}, T_val={data["T_val"]}')

    # ── 2. 特征构建 ──
    print('[2/6] Building features...')
    fb = FeatureBuilder(
        data['global_feat'],
        lag_hours=cfg['features']['lag_hours'],
        roll_hours=cfg['features']['roll_hours'],
        max_lag=cfg['features']['max_lag'],
    )
    n_train = data['T_train'] - cfg['features']['max_lag']
    n_val = data['T_val']

    # ── 3. 层次结构 ──
    print('[3/6] Building hierarchy...')
    hb = HierarchyBuilder(cfg['paths'])
    hierarchy = hb.build(exp_cfg['hierarchy'], data['occ_bottom'],
                         zone_cols=data['zone_cols'])
    print(f'  Hierarchy: {hierarchy.name}, '
          f'n_middle={hierarchy.n_middle}, n_total={hierarchy.n_total}')

    # 聚合中层和顶层
    occ_mid, occ_top = hb.build_middle_top(data['occ_bottom'], hierarchy)

    # ── 4. 训练基预测 ──
    print('[4/6] Training base forecasts...')
    taus = cfg['quantile']['taus']
    t0 = time.time()

    # 顶层
    lgb_top = {**cfg['lgb_defaults'], **cfg['lgb_top'],
               'n_jobs': cfg['compute']['n_jobs_lgb'],
               'random_state': cfg['lgb_defaults']['random_state']}
    trainer_top = IndependentQuantileTrainer(taus, lgb_top, cfg['lgb_early_stop'])
    p_top_val, p_top = trainer_top.train(occ_top, fb, n_train, n_val)

    # 中层 (如果有)
    n_test = p_top.shape[0]
    if hierarchy.n_middle > 0:
        p_mid_val = np.zeros((n_val, hierarchy.n_middle, len(taus)), dtype=np.float32)
        p_mid = np.zeros((n_test, hierarchy.n_middle, len(taus)), dtype=np.float32)
        for d in range(hierarchy.n_middle):
            p_mid_val[:, d, :], p_mid[:, d, :] = trainer_top.train(
                occ_mid[:, d], fb, n_train, n_val
            )
    else:
        p_mid_val = np.zeros((n_val, 0, len(taus)), dtype=np.float32)
        p_mid = np.zeros((n_test, 0, len(taus)), dtype=np.float32)

    # 底层
    lgb_bottom = {**cfg['lgb_defaults'],
                  'n_jobs': cfg['compute']['n_jobs_lgb'],
                  'random_state': cfg['lgb_defaults']['random_state']}

    if exp_cfg['bottom_strategy'] == 'independent':
        # 逐节点独立训练
        trainer = IndependentQuantileTrainer(taus, lgb_bottom, cfg['lgb_early_stop'])
        n_bottom = data['n_bottom']
        p_bot_val = np.zeros((n_val, n_bottom, len(taus)), dtype=np.float32)
        p_bot = np.zeros((n_test, n_bottom, len(taus)), dtype=np.float32)

        results = Parallel(
            n_jobs=cfg['compute']['n_jobs_parallel'], verbose=10
        )(delayed(trainer.train)(
            data['occ_bottom'][:, i], fb, n_train, n_val
        ) for i in range(n_bottom))

        for i, (pv, pt) in enumerate(results):
            p_bot_val[:, i, :] = pv
            p_bot[:, i, :] = pt

    elif exp_cfg['bottom_strategy'] == 'grouped':
        # 分组建模
        trainer = GroupedQuantileTrainer(
            taus, lgb_bottom,
            taz_categorical=True  # TAZ categorical feature (not a learned embedding),
            early_stop=cfg['lgb_early_stop'],
            n_jobs_lgb=cfg['compute']['n_jobs_lgb'],
        )
        # 获取分组标签
        if exp_cfg.get('group_source') and exp_cfg['group_source'] != exp_cfg['hierarchy']:
            # 正交实验: 分组来源 ≠ 中层结构
            group_labels = hb.load_group_labels_from_source(
                exp_cfg['group_source'], data['zone_cols'], hierarchy.n_middle
            )
        else:
            group_labels = hierarchy.group_labels

        # 按分组组织 TAZ 成员
        group_members = []
        for g in range(len(np.unique(group_labels))):
            group_members.append(
                np.where(group_labels == g)[0].tolist()
            )

        p_bot_val, p_bot = trainer.train_all_groups(
            group_members, data['occ_bottom'], fb,
            n_train, n_val, cfg['compute']['n_jobs_parallel']
        )
        # 更新 hierarchy 的分组标签 (可能不同于中层归属)
        hierarchy.group_labels = group_labels

    print(f'  Training done in {time.time()-t0:.0f}s')

    # ── 5. 调和 + 评估 ──
    print('[5/6] Reconciliation & evaluation...')

    # 准备基预测
    p_bot_med = p_bot[:, :, len(taus)//2]   # median (τ=0.5)
    p_mid_med = p_mid[:, :, len(taus)//2] if hierarchy.n_middle > 0 else np.zeros((n_test, 0))
    p_top_med = p_top[:, len(taus)//2]

    y_hat = np.column_stack([p_bot_med, p_mid_med, p_top_med])  # (n_test, N_total)
    y_true_test = np.column_stack([
        data['occ_bottom'][data['T_train']+data['T_val']:, :],
        occ_mid[data['T_train']+data['T_val']:, :] if hierarchy.n_middle > 0 else np.zeros((n_test, 0)),
        occ_top[data['T_train']+data['T_val']:].reshape(-1, 1),
    ])

    # 验证集残差
    val_start = data['T_train']
    val_end = data['T_train'] + data['T_val']
    y_val_true = np.column_stack([
        data['occ_bottom'][val_start:val_end, :],
        occ_mid[val_start:val_end, :] if hierarchy.n_middle > 0 else np.zeros((n_val, 0)),
        occ_top[val_start:val_end].reshape(-1, 1),
    ])

    p_bot_val_med = p_bot_val[:, :, len(taus)//2]
    p_mid_val_med = p_mid_val[:, :, len(taus)//2] if hierarchy.n_middle > 0 else np.zeros((n_val, 0))
    p_top_val_med = p_top_val[:, len(taus)//2]

    y_hat_val = np.column_stack([p_bot_val_med, p_mid_val_med, p_top_val_med])
    residuals = y_hat_val - y_val_true

    # Bottom-Up 基准
    bu_sum = p_bot_med.sum(axis=1)
    m_bu = point_metrics(y_true_test[:, -1], bu_sum)

    # 初始化追踪器
    tracker = ExperimentTracker(exp_name, cfg['paths']['output'])
    tracker.save_predictions({
        'pred_bottom': p_bot, 'pred_middle': p_mid, 'pred_top': p_top,
        'y_true': y_true_test,
        'pred_bottom_val': p_bot_val, 'pred_middle_val': p_mid_val, 'pred_top_val': p_top_val,
        'y_val_true': y_val_true,
    })

    # 对每种调和方式
    methods = reconcile_filter or exp_cfg['reconciliation']
    for rec_name in methods:
        print(f'  --- {rec_name} ---')

        if rec_name == 'mint_quantile_specific':
            # 分位数特定 MinT
            qmt = QuantileSpecificMinT(hierarchy.S)
            # 构建各 τ 的残差
            q_errors = {}
            for i, tau in enumerate(taus):
                q_hat_val = np.column_stack([
                    p_bot_val[:, :, i],
                    p_mid_val[:, :, i] if hierarchy.n_middle > 0 else np.zeros((n_val, 0)),
                    p_top_val[:, i],
                ])
                q_errors[tau] = q_hat_val - y_val_true
            qmt.fit(q_errors)

            # 调和各 τ
            y_rec = np.zeros_like(y_hat)
            for i, tau in enumerate(taus):
                q_hat = np.column_stack([
                    p_bot[:, :, i],
                    p_mid[:, :, i] if hierarchy.n_middle > 0 else np.zeros((n_test, 0)),
                    p_top[:, i],
                ])
                y_rec_i = qmt.reconcile(q_hat, tau)
                y_rec[:, :] = y_rec_i  # 用对应档的 G
            rec = None  # QuantileSpecificMinT 不是 BaseReconciler
        else:
            # 标准调和方法
            rec = _create_reconciler(rec_name, hierarchy.S, hierarchy.group_labels)
            rec.fit(residuals)
            y_rec = rec.reconcile(y_hat)

        # 评估: 点预测
        m_bot = point_metrics(y_true_test[:, :data['n_bottom']], y_rec[:, :data['n_bottom']])
        m_top = point_metrics(y_true_test[:, -1], y_rec[:, -1])
        m_city = point_metrics(y_true_test[:, -1], y_rec[:, :data['n_bottom']].sum(axis=1))
        gain = reconciliation_gain(m_bu['rmse'], m_city['rmse'])

        # 评估: 概率预测
        # 对底层节点
        # (注意: 调和只对中位数做，概率评估需要分位数调和)
        # 先用基预测的分位数评估，后续可扩展
        ql = multi_pinball(
            y_true_test[:, :data['n_bottom']],
            p_bot,
            taus
        )

        # 校准评估 (基于基预测，不是调和后的)
        calib = calibration_summary(
            y_true_test[:, :data['n_bottom']],
            p_bot,
            np.array(taus)
        )

        # 中层评估
        if hierarchy.n_middle > 0:
            m_mid = point_metrics(
                y_true_test[:, data['n_bottom']:data['n_bottom']+hierarchy.n_middle],
                y_rec[:, data['n_bottom']:data['n_bottom']+hierarchy.n_middle]
            )
        else:
            m_mid = {'rmse': 0, 'mae': 0, 'mape': 0}

        # 汇总
        metrics = {
            'bottom': m_bot,
            'middle': m_mid,
            'top': m_top,
            'city': {
                'rmse_before': round(m_bu['rmse'], 2),
                'rmse_after': round(m_city['rmse'], 2),
                'improvement_pct': gain['delta_rmse_pct'],
            },
            'quantile_loss': ql,
            'calibration': calib,
        }

        tracker.save(rec_name, y_rec, metrics, cfg)

    print('[6/6] Finalizing...')
    tracker.finalize()
    return tracker.run_id


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='层次概率预测实验')
    parser.add_argument('--exp', type=str, required=True,
                        help='实验名称 (E1, E2, ..., E8 或逗号分隔)')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件路径')
    parser.add_argument('--reconcile', type=str, default=None,
                        help='限定调和方式 (逗号分隔)')
    args = parser.parse_args()

    # 加载配置
    # 注意：config.yaml 包含中文字符，必须用 utf-8 打开
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 解析实验名称
    exp_names = [e.strip() for e in args.exp.split(',')]
    reconcile_filter = (
        [r.strip() for r in args.reconcile.split(',')]
        if args.reconcile else None
    )

    # 运行
    for name in exp_names:
        if name not in cfg['experiments']:
            print(f"Warning: {name} not in config, skipping")
            continue
        run_experiment(name, cfg, reconcile_filter)
