"""
GA-MinT 对比实验: 加载已有基预测, 对比 mint_shrink / ga_mint_bd / ga_mint_gas

用法:
  python run_ga_mint.py E5_20260621_121531 E7_20260621_133716 E8_20260621_132744
"""
import sys, json, time, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

from src.reconciliation.mint import MinTShrink, GAMinT_BD, GAMinT_GAS
from src.evaluation.metrics import point_metrics, reconciliation_gain

OUT = Path('output')

def load_experiment(exp_dir: str):
    """从已有实验结果加载基预测和残差"""
    d = OUT / exp_dir
    preds = d / 'predictions'

    n_bottom = 275  # 深圳交通小区数 (常量)

    # 根据实验名确定 S 矩阵和分组标签
    exp_name = exp_dir.split('_')[0]

    if exp_name == 'E5':
        # behavior mid + behavior group: 都用 S_behavior
        S = np.load(str(OUT / 'S_behavior.npy'))
        n_middle = S.shape[0] - n_bottom - 1
        group_labels = S[n_bottom:n_bottom + n_middle, :].argmax(axis=0)

    elif exp_name == 'E7':
        # behavior mid + admin group: S_behavior 做中层, admin 做分组
        S = np.load(str(OUT / 'S_behavior.npy'))
        n_middle = S.shape[0] - n_bottom - 1
        from src.data.hierarchy import HierarchyBuilder
        import yaml
        with open('config.yaml', 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        hb = HierarchyBuilder(cfg['paths'])
        S_admin = np.load(str(OUT / 'S_admin.npy'))
        n_mid_admin = S_admin.shape[0] - n_bottom - 1
        group_labels = hb._extract_group_labels(S_admin, n_bottom, n_mid_admin)

    elif exp_name == 'E8':
        # admin mid + behavior group: S_admin 做中层, S_behavior 做分组
        S = np.load(str(OUT / 'S_admin.npy'))
        n_middle = S.shape[0] - n_bottom - 1
        S_behav = np.load(str(OUT / 'S_behavior.npy'))
        n_mid_behav = S_behav.shape[0] - n_bottom - 1
        group_labels = S_behav[n_bottom:n_bottom + n_mid_behav, :].argmax(axis=0)

    else:
        S = np.load(str(OUT / 'S_behavior.npy'))
        n_middle = S.shape[0] - n_bottom - 1
        group_labels = S[n_bottom:n_bottom + n_middle, :].argmax(axis=0)
    p_bot = np.load(str(preds / 'pred_bottom.npy'))
    p_mid = np.load(str(preds / 'pred_middle.npy'))
    p_top = np.load(str(preds / 'pred_top.npy'))
    y_true = np.load(str(preds / 'y_true.npy'))

    p_bot_val = np.load(str(preds / 'pred_bottom_val.npy'))
    p_mid_val = np.load(str(preds / 'pred_middle_val.npy'))
    p_top_val = np.load(str(preds / 'pred_top_val.npy'))
    y_val_true = np.load(str(preds / 'y_val_true.npy'))

    n_middle = S.shape[0] - n_bottom - 1
    n_taus = p_bot.shape[2]
    tau_med = n_taus // 2

    # 中位数基预测
    y_hat_val = np.column_stack([
        p_bot_val[:, :, tau_med],
        p_mid_val[:, :, tau_med] if n_middle > 0 else np.zeros((p_bot_val.shape[0], 0)),
        p_top_val[:, tau_med],
    ])
    y_hat_test = np.column_stack([
        p_bot[:, :, tau_med],
        p_mid[:, :, tau_med] if n_middle > 0 else np.zeros((p_bot.shape[0], 0)),
        p_top[:, tau_med],
    ])
    residuals = y_hat_val - y_val_true
    y_true_top = y_true[:, -1]

    return {
        'S': S, 'group_labels': group_labels,
        'y_hat_test': y_hat_test, 'y_true_top': y_true_top,
        'residuals': residuals, 'n_bottom': n_bottom,
    }

def main(exp_dirs):
    methods = {
        'mint_shrink': MinTShrink,
        'ga_mint_bd': GAMinT_BD,
        'ga_mint_gas': GAMinT_GAS,
    }

    print(f"\n{'='*70}")
    print(f"  GA-MinT Comparison: {len(exp_dirs)} experiments x {len(methods)} methods")
    print(f"{'='*70}")

    for exp_dir in exp_dirs:
        exp_name = exp_dir.split('_')[0]
        print(f"\n--- {exp_name} ({exp_dir}) ---")
        data = load_experiment(exp_dir)

        # Bottom-Up baseline
        bu_sum = data['y_hat_test'][:, :data['n_bottom']].sum(axis=1)
        m_bu = point_metrics(data['y_true_top'], bu_sum)

        print(f"  Bottom-Up RMSE: {m_bu['rmse']:.1f}")
        print(f"  {'Method':<16} {'TopRMSE':>8} {'Delta%':>8}")
        print(f"  {'-'*34}")

        for method_name, ReconcilerCls in methods.items():
            t0 = time.time()
            if 'ga_mint' in method_name:
                rec = ReconcilerCls(data['S'], data['group_labels'])
            else:
                rec = ReconcilerCls(data['S'])
            rec.fit(data['residuals'])
            y_rec = rec.reconcile(data['y_hat_test'])
            city_rec = y_rec[:, :data['n_bottom']].sum(axis=1)
            m = point_metrics(data['y_true_top'], city_rec)
            gain = reconciliation_gain(m_bu['rmse'], m['rmse'])
            elapsed = time.time() - t0
            print(f"  {method_name:<16} {m['rmse']:>8.1f} {gain['delta_rmse_pct']:>+7.1f}%  ({elapsed:.2f}s)")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        # 查找最新的 E5/E7/E8 目录
        dirs = sorted(OUT.glob('E5_*')) + sorted(OUT.glob('E7_*')) + sorted(OUT.glob('E8_*'))
        # 去重取最新
        latest = {}
        for d in dirs:
            exp = d.name.split('_')[0]
            latest[exp] = d.name
        main(list(latest.values()))
    else:
        main(sys.argv[1:])
