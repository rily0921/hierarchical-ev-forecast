"""
实验追踪: 结果保存 + 配置快照 + 时间记录

用法:
    tracker = ExperimentTracker('E5', '/path/to/output')
    tracker.save('mint_shrink', reconciled, metrics, config)
    tracker.finalize(elapsed_seconds=1234.5)
"""

import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime


class ExperimentTracker:
    """管理单个实验的运行记录和结果保存"""

    def __init__(self, exp_name: str, output_dir: str):
        self.exp_name = exp_name
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = Path(output_dir) / f'{exp_name}_{self.run_id}'
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._start_time = time.time()
        self._results = {}

    def save(self, rec_name: str,
             reconciled: np.ndarray = None,
             metrics: dict = None,
             config: dict = None):
        """
        保存一种调和方式的结果

        参数:
            rec_name:    'mint_shrink' | 'ga_mint_bd' | ...
            reconciled:  (T, N_total) 调和后预测 (可选)
            metrics:     评估指标字典
            config:      完整配置 (首次调用时保存快照)
        """
        rec_dir = self.run_dir / rec_name
        rec_dir.mkdir(exist_ok=True)

        # 保存配置快照 (仅首次)
        if config is not None and not (self.run_dir / 'config_snapshot.json').exists():
            with open(self.run_dir / 'config_snapshot.json', 'w', encoding='utf-8') as f:
                json.dump(self._make_serializable(config), f, indent=2,
                          ensure_ascii=False)

        # 保存调和结果
        if reconciled is not None:
            np.save(rec_dir / 'reconciled.npy', reconciled.astype(np.float32))

        # 保存指标
        if metrics is not None:
            with open(rec_dir / 'metrics.json', 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

        self._results[rec_name] = metrics

    def save_predictions(self, data: dict):
        """保存基预测 (与调和无关的原始预测)"""
        pred_dir = self.run_dir / 'predictions'
        pred_dir.mkdir(exist_ok=True)
        for key, arr in data.items():
            if isinstance(arr, np.ndarray):
                np.save(pred_dir / f'{key}.npy', arr.astype(np.float32))

    def finalize(self, elapsed_seconds: float = None):
        """保存运行摘要"""
        if elapsed_seconds is None:
            elapsed_seconds = time.time() - self._start_time

        # 生成调和效果对比摘要
        summary = {
            'experiment': self.exp_name,
            'run_id': self.run_id,
            'completed_at': datetime.now().isoformat(),
            'elapsed_seconds': round(elapsed_seconds, 1),
            'methods': {},
        }

        for rec_name, metrics in self._results.items():
            if metrics:
                summary['methods'][rec_name] = self._extract_key_metrics(metrics)

        with open(self.run_dir / 'summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f'\n[Tracker] Results saved to {self.run_dir}')
        return summary

    def _extract_key_metrics(self, metrics: dict) -> dict:
        """从完整指标中提取关键数字"""
        key = {}
        if 'city' in metrics:
            key['top_rmse_before'] = metrics['city'].get('rmse_before')
            key['top_rmse_after'] = metrics['city'].get('rmse_after')
            key['improvement_pct'] = metrics['city'].get('improvement_pct')
        if 'bottom' in metrics:
            key['bottom_rmse'] = metrics['bottom'].get('rmse')
            key['bottom_mape'] = metrics['bottom'].get('mape')
        if 'calibration' in metrics:
            key['pit_deviation'] = metrics['calibration'].get('pit_deviation')
        return key

    def _make_serializable(self, obj):
        """将配置对象转为 JSON 可序列化"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
