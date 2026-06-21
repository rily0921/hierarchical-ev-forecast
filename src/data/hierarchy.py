"""
层次结构构建：S 矩阵加载 + 中层/顶层聚合 + 分组标签提取

支持的层次类型:
  - admin:    8 个行政区 (S_admin.npy)
  - spatial:  10 个空间聚类组 (S_spatial.npy)
  - behavior: 10 个行为聚类组 (S_behavior.npy)
  - two_level: 无中层 (直接底层→顶层)

用法:
    hb = HierarchyBuilder(paths_dict)
    h = hb.build('behavior', occ_bottom)
    # h.S           (N_total, N_bottom)
    # h.n_middle     int
    # h.occ_middle   (T, n_middle)
    # h.occ_top      (T,)
    # h.group_labels (275,) int  ← 用于分组建模和 GA-MinT
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Hierarchy:
    """层次结构的完整描述"""
    name: str               # 'admin' | 'spatial' | 'behavior' | 'two_level'
    S: np.ndarray           # (N_total, N_bottom) 汇总矩阵
    n_bottom: int           # 275
    n_middle: int           # 8 或 10 或 0
    n_total: int            # 284/286/276
    group_labels: np.ndarray  # (n_bottom,) int  底层节点的分组标签
    group_sizes: list       # 各组节点数


class HierarchyBuilder:
    """加载 S 矩阵、构建层聚合、提取分组标签"""

    def __init__(self, paths: dict):
        """
        paths: config.yaml 中的 paths 字典
        """
        self.paths = paths

    def build(self, hierarchy_name: str, occ_bottom: np.ndarray,
              zone_cols: list = None) -> Hierarchy:
        """
        构建层次结构

        参数:
            hierarchy_name: 'admin' | 'spatial' | 'behavior' | 'two_level'
            occ_bottom:     (T, 275) 底层 occupancy 矩阵
            zone_cols:      TAZ 列名列表 (仅 admin 需要，用于匹配行政区映射)

        返回:
            Hierarchy 对象
        """
        if hierarchy_name == 'two_level':
            return self._build_two_level(occ_bottom)
        elif hierarchy_name == 'admin':
            return self._build_from_S('admin', occ_bottom, zone_cols)
        elif hierarchy_name == 'spatial':
            return self._build_from_S('spatial', occ_bottom, zone_cols)
        elif hierarchy_name == 'behavior':
            return self._build_from_S('behavior', occ_bottom, zone_cols)
        else:
            raise ValueError(f"Unknown hierarchy: {hierarchy_name}")

    # ── 内部方法 ────────────────────────────────────

    def _build_two_level(self, occ_bottom: np.ndarray) -> Hierarchy:
        """两层结构：底层(275) → 顶层(1)，无中层"""
        n_bottom = occ_bottom.shape[1]
        S = np.zeros((n_bottom + 1, n_bottom))
        S[:n_bottom, :] = np.eye(n_bottom)
        S[-1, :] = 1.0

        return Hierarchy(
            name='two_level',
            S=S,
            n_bottom=n_bottom,
            n_middle=0,
            n_total=n_bottom + 1,
            group_labels=np.zeros(n_bottom, dtype=int),  # 无分组
            group_sizes=[n_bottom],
        )

    def _build_from_S(self, name: str, occ_bottom: np.ndarray,
                      zone_cols: list = None) -> Hierarchy:
        """从 .npy 文件加载 S 矩阵并构建 Hierarchy"""
        # 加载 S 矩阵
        key = f'S_{name}'
        if key not in self.paths:
            raise KeyError(f"Path not found: {key}")
        S = np.load(self.paths[key])
        n_bottom = S.shape[1]
        n_middle = S.shape[0] - n_bottom - 1

        # 提取分组标签: 每个底层节点属于哪个中层组
        group_labels = self._extract_group_labels(S, n_bottom, n_middle)
        group_sizes = [int((group_labels == g).sum()) for g in range(n_middle)]

        return Hierarchy(
            name=name,
            S=S,
            n_bottom=n_bottom,
            n_middle=n_middle,
            n_total=S.shape[0],
            group_labels=group_labels,
            group_sizes=group_sizes,
        )

    def _extract_group_labels(self, S: np.ndarray, n_bottom: int,
                              n_middle: int) -> np.ndarray:
        """从 S 矩阵的中层行提取每个底层节点的分组标签"""
        labels = np.zeros(n_bottom, dtype=int)
        for d in range(n_middle):
            idx = S[n_bottom + d, :] == 1
            labels[idx] = d
        return labels

    def get_group_members(self, hierarchy: Hierarchy):
        """提取每个分组的 TAZ 列索引列表"""
        members = []
        for g in range(hierarchy.n_middle):
            members.append(
                np.where(hierarchy.group_labels == g)[0].tolist()
            )
        return members

    def build_middle_top(self, occ_bottom: np.ndarray,
                         hierarchy: Hierarchy) -> tuple:
        """
        从底层数据聚合中层和顶层

        参数:
            occ_bottom:  (T, n_bottom)
            hierarchy:   Hierarchy 对象

        返回:
            occ_middle:  (T, n_middle)  或 None (two_level 时)
            occ_top:     (T,)
        """
        if hierarchy.n_middle == 0:
            return None, occ_bottom.sum(axis=1)

        T = occ_bottom.shape[0]
        n_bottom = hierarchy.n_bottom
        occ_middle = np.zeros((T, hierarchy.n_middle), dtype=np.float32)
        for d in range(hierarchy.n_middle):
            idx = hierarchy.S[n_bottom + d, :] == 1
            occ_middle[:, d] = occ_bottom[:, idx].sum(axis=1)

        occ_top = occ_bottom.sum(axis=1)
        return occ_middle, occ_top

    def load_group_labels_from_source(self, source: str,
                                      zone_cols: list,
                                      n_middle: int) -> np.ndarray:
        """
        从中层定义之外的其他来源加载分组标签

        用于 E7 (中层=behavior, 分组=admin) 和 E8 (中层=admin, 分组=behavior)

        参数:
            source:    'admin' | 'behavior'
            zone_cols: TAZ 列名列表
            n_middle:  中层节点数 (用于验证)

        返回:
            group_labels: (275,) int
        """
        if source == 'admin':
            return self._get_admin_labels(zone_cols)
        elif source == 'behavior':
            return self._get_behavior_labels(n_middle)
        else:
            raise ValueError(f"Unknown group source: {source}")

    def _get_admin_labels(self, zone_cols: list) -> np.ndarray:
        """从行政区映射 JSON 加载分组标签 (GBK编码)"""
        # 尝试多种中文编码
        for enc in ['utf-8', 'gbk', 'gb18030', 'gb2312']:
            try:
                with open(self.paths['admin_map'], 'r', encoding=enc) as f:
                    dmap = json.load(f)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        zone_to_district = dmap['zone_to_district']
        district_names = dmap['district_names']

        labels = np.zeros(len(zone_cols), dtype=int)
        for j, zid in enumerate(zone_cols):
            if str(zid) in zone_to_district:
                dname = zone_to_district[str(zid)]
                labels[j] = district_names.index(dname)
        return labels

    def _get_behavior_labels(self, n_middle: int) -> np.ndarray:
        """从 S_behavior.npy 加载行为聚类分组标签"""
        S = np.load(self.paths['S_behavior'])
        n_bottom = S.shape[1]
        return self._extract_group_labels(S, n_bottom, n_middle)
