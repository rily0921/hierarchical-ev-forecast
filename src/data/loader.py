"""
数据加载模块：从 UrbanEV 原始文件 → 标准化的训练/验证/测试格式

用法:
    loader = DataLoader("E:/Desktop/毕业论文/data/UrbanEV-main/UrbanEV-main/data")
    data = loader.load()
    # data['occ_bottom']  (4344, 275)
    # data['global_feat'] (4344, 10)
    # data['zone_cols']   ['1', '2', ..., '275']
"""

import pandas as pd
import numpy as np
from pathlib import Path


class DataLoader:
    """加载 UrbanEV 数据集，输出标准化的 ndarray 格式"""

    def __init__(self, raw_dir: str, train_hours: int = 2880, val_hours: int = 720):
        self.raw_dir = Path(raw_dir)
        self.T_train = train_hours
        self.T_val = val_hours

    # ── 主入口 ──────────────────────────────────────
    def load(self) -> dict:
        """加载全部数据，返回标准字典"""
        occ = self._load_csv('occupancy.csv')
        self.zone_cols = [c for c in occ.columns]
        self.n_bottom = len(self.zone_cols)

        time_feat = self._build_time_features(occ.index)
        weather_feat = self._load_weather(occ.index)

        # 全局特征: 时间(6) + 天气(4) = 10 维
        global_feat = np.column_stack([time_feat, weather_feat]).astype(np.float32)

        T_total = len(occ)
        T_test = T_total - self.T_train - self.T_val

        return {
            'occ_bottom':  occ[self.zone_cols].values.astype(np.float32),
            'global_feat': global_feat,
            'time_index':  occ.index,
            'zone_cols':   self.zone_cols,
            'n_bottom':    self.n_bottom,
            'T_total':     T_total,
            'T_train':     self.T_train,
            'T_val':       self.T_val,
            'T_test':      T_test,
        }

    # ── 内部方法 ────────────────────────────────────
    def _load_csv(self, filename: str) -> pd.DataFrame:
        """加载 CSV 并解析时间索引"""
        df = pd.read_csv(self.raw_dir / filename)
        df['time'] = pd.to_datetime(df['time'])
        return df.set_index('time')

    def _build_time_features(self, idx: pd.DatetimeIndex) -> np.ndarray:
        """构建 6 维时间特征"""
        hour = idx.hour.values.astype(np.float32)
        wday = idx.dayofweek.values.astype(np.float32)
        month = idx.month.values.astype(np.float32)

        return np.column_stack([
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * wday / 7),
            np.cos(2 * np.pi * wday / 7),
            (wday >= 5).astype(np.float32),   # is_weekend
            month,
        ])

    def _load_weather(self, idx: pd.DatetimeIndex) -> np.ndarray:
        """加载并合并中心站+机场天气 → 4 维特征"""
        wc = self._load_csv('weather_central.csv')
        wa = self._load_csv('weather_airport.csv')

        # 与 G 脚本一致: T, U, P 取两站均值, nRAIN 用中心站
        return np.column_stack([
            ((wc['T'].values + wa['T'].values) / 2).astype(np.float32),
            ((wc['U'].values + wa['U'].values) / 2).astype(np.float32),
            ((wc['P'].values + wa['P'].values) / 2).astype(np.float32),
            wc['nRAIN'].values.astype(np.float32),
        ])
