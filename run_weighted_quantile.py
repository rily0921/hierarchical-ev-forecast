"""
方向2: 加权分位数损失对比实验
─────────────────────────────
在同一个分组上对比 等权 vs 尾部加权 的校准效果
"""
import sys, numpy as np, pandas as pd, warnings, os, time
warnings.filterwarnings('ignore')
os.environ['LGB_VERBOSITY'] = '-1'
sys.path.insert(0, '.')

from src.data.loader import DataLoader
from src.data.features import FeatureBuilder
from src.data.hierarchy import HierarchyBuilder
from src.models.lgbm_quantile import IndependentQuantileTrainer
from src.evaluation.calibration import pit_histogram, interval_coverage, reliability_diagram

RAW = 'E:/Desktop/毕业论文/data/UrbanEV-main/UrbanEV-main/data'
S_BEHAV = 'output/S_behavior.npy'
TAUS = np.arange(0.05, 1.0, 0.05)
LGB_P = dict(n_estimators=300, learning_rate=0.08, num_leaves=31,
             min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
             reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1, n_jobs=4)

print("Loading data...")
loader = DataLoader(RAW)
data = loader.load()
fb = FeatureBuilder(data['global_feat'])

hb = HierarchyBuilder({
    'S_behavior': S_BEHAV, 'S_admin': 'output/S_admin.npy',
    'S_spatial': 'output/S_spatial.npy',
    'admin_map': 'chapter3/data/processed/district_mapping.json',
})
h = hb.build('behavior', data['occ_bottom'])
members_list = hb.get_group_members(h)

n_train = 2880 - 168; n_val = 720

# ── 选最大的组, 只取前 5 个 TAZ 做快速测试 ──
group_idx = max(range(len(members_list)), key=lambda g: len(members_list[g]))
members = members_list[group_idx][:5]
print(f'\nTesting on Group {group_idx}: {len(members)} TAZs')

for alpha, label in [(0.0, 'Equal Weight'), (1.0, 'Tail Weighted (α=1)'), (2.0, 'Tail Weighted (α=2)')]:
    print(f'\n--- {label} ---')
    trainer = IndependentQuantileTrainer(TAUS, LGB_P, tail_weight_alpha=alpha)

    pv_all = np.zeros((n_val, len(members), len(TAUS)), dtype=np.float32)
    pt_all = np.zeros((data['T_test'], len(members), len(TAUS)), dtype=np.float32)
    t0 = time.time()
    for j, col_idx in enumerate(members):
        pv, pt = trainer.train(data['occ_bottom'][:, col_idx], fb, n_train, n_val)
        pv_all[:, j, :] = pv
        pt_all[:, j, :] = pt

    # 校准评估
    y_true_test = data['occ_bottom'][2880+720:, members]
    pit = pit_histogram(y_true_test, pt_all, TAUS)
    cov = interval_coverage(y_true_test, pt_all, TAUS)
    pit_dev = np.mean(np.abs(pit['densities'] - 1.0))

    print(f'  Training: {time.time()-t0:.0f}s')
    print(f'  PIT deviation: {pit_dev:.4f} (lower=better)')
    for level in ['50%', '80%', '90%']:
        c = cov[level]
        print(f'  {level}: nominal={c["nominal_coverage"]}, actual={c["actual_coverage"]:.4f}, '
              f'width={c["avg_interval_width"]:.3f}')

    # RMSE (median)
    y_med = pt_all[:, :, 9]
    rmse = np.sqrt(np.mean((y_true_test - y_med)**2))
    print(f'  Median RMSE: {rmse:.2f}')

print('\nDone.')
