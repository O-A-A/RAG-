#!/usr/bin/env python3
"""
===============================================================================
实验结果可视化 — 论文风格对比柱状图 (matplotlib only)
===============================================================================
读取 results/summary.csv，生成 4 张独立对比图。

输出规格:
    - 格式: PDF + PNG
    - 分辨率: 600 DPI
    - 字体: Times New Roman
    - 保存路径: figures/

Usage:
    python experiments/plot_results.py                          # 从 results/summary.csv 读取
    python experiments/plot_results.py -i results/summary.csv   # 指定输入
    python experiments/plot_results.py --mock                   # 使用模拟数据测试
===============================================================================
"""

import os
import sys
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── 全局 matplotlib 配置 ─────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# ── 配色方案 (色盲友好, 学术风格) ─────────────────────────────────────
COLORS: Dict[str, str] = {
    "bm25":       "#4472C4",   # 深蓝
    "dense_bge":  "#ED7D31",   # 橙色
    "hybrid_a50": "#70AD47",   # 绿色
}

RETRIEVER_LABELS: Dict[str, str] = {
    "bm25":       "BM25",
    "dense_bge":  "Dense (BGE)",
    "hybrid_a50": "Hybrid",
}

METRIC_LABELS: Dict[str, str] = {
    "recall_at_5":   "Recall@5",
    "mrr":           "MRR",
    "exact_match":   "Exact Match",
    "f1":            "F1 Score",
}

Y_LIMITS: Dict[str, Tuple[float, float]] = {
    "recall_at_5":   (0.0, 1.0),
    "mrr":           (0.0, 1.0),
    "exact_match":   (0.0, 1.0),
    "f1":            (0.0, 1.0),
}


# ============================================================================
# 数据加载
# ============================================================================

def load_summary(path: str) -> List[Dict]:
    """从 CSV 加载 summary 数据。"""
    import csv
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row["recall_at_5"] = float(row["recall_at_5"])
            row["mrr"] = float(row["mrr"])
            row["exact_match"] = float(row["exact_match"])
            row["f1"] = float(row["f1"])
            rows.append(row)
    return rows


def mock_data() -> List[Dict]:
    """返回模拟数据（当 summary.csv 不存在时使用）。"""
    return [
        {"retriever": "bm25",       "recall_at_5": 0.6275, "mrr": 0.6022, "exact_match": 0.2000, "f1": 0.5892},
        {"retriever": "dense_bge",  "recall_at_5": 0.7182, "mrr": 0.7215, "exact_match": 0.2600, "f1": 0.6401},
        {"retriever": "hybrid_a50", "recall_at_5": 0.7936, "mrr": 0.7548, "exact_match": 0.3100, "f1": 0.6825},
    ]


# ============================================================================
# 单图绘制
# ============================================================================

def plot_single_metric(
    data: List[Dict],
    metric_key: str,
    output_dir: str,
) -> str:
    """
    为单个指标绘制论文风格对比柱状图。

    Args:
        data:       summary 数据列表。
        metric_key: 指标名 (recall_at_5 | mrr | exact_match | f1)。
        output_dir: 输出目录。

    Returns:
        str: 输出文件路径前缀。
    """
    metric_label = METRIC_LABELS[metric_key]
    retrievers = [d["retriever"] for d in data]
    values = [d[metric_key] for d in data]
    colors = [COLORS[r] for r in retrievers]
    labels = [RETRIEVER_LABELS[r] for r in retrievers]

    # ── 创建图形 ──
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    x = np.arange(len(retrievers))
    bars = ax.bar(
        x, values,
        width=0.52,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    # ── 数值标注 ──
    y_min, y_max = Y_LIMITS[metric_key]
    offset = (y_max - y_min) * 0.025

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{val:.4f}",
            ha="center", va="bottom",
            fontsize=11,
            fontfamily="serif",
            fontweight="bold",
        )

    # ── 坐标轴 ──
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontfamily="serif")
    ax.set_ylabel(metric_label, fontfamily="serif")
    ax.set_ylim(y_min - 0.02, y_max * 1.18)

    # 纵轴格式化为百分比风格
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:.2f}"
    ))

    # 横轴下方留白
    ax.set_xlim(-0.55, len(retrievers) - 0.45)

    # ── 网格 ──
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    # ── 保存 ──
    prefix = os.path.join(output_dir, f"comparison_{metric_key}")
    fig.savefig(f"{prefix}.pdf", format="pdf")
    fig.savefig(f"{prefix}.png", format="png")
    plt.close(fig)

    return prefix


# ============================================================================
# 主入口
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="生成论文风格 RAG 对比柱状图 (matplotlib)",
    )
    parser.add_argument("-i", "--input", default="results/summary.csv")
    parser.add_argument("-o", "--output", default="figures")
    parser.add_argument("--mock", action="store_true",
                        help="使用模拟数据测试")
    args = parser.parse_args()

    # ── 加载数据 ──
    if args.mock or not os.path.exists(args.input):
        print(f"[INFO] 使用模拟数据 (--mock 或 {args.input} 不存在)")
        data = mock_data()
    else:
        print(f"[INFO] 加载: {args.input}")
        data = load_summary(args.input)

    # ── 创建输出目录 ──
    os.makedirs(args.output, exist_ok=True)

    # ── 指标列表 ──
    metrics = ["recall_at_5", "mrr", "exact_match", "f1"]

    print(f"[INFO] 生成 {len(metrics)} 张图片 (PDF + PNG, 600 DPI)...")

    for metric_key in metrics:
        prefix = plot_single_metric(data, metric_key, args.output)
        print(f"  [OK] {os.path.basename(prefix)}.pdf + .png")

    print(f"\n[DONE] 图片保存至: {os.path.abspath(args.output)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
