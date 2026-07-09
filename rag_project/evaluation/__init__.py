"""
评估模块 (Evaluation Layer)

自动计算 RAG 系统的检索质量和生成质量指标。

模块组成:
    - metrics.py:   纯函数指标计算 (Recall@K, MRR, Exact Match, F1)
    - evaluator.py: 评估器 (单样本/批量评估 + 聚合统计)
    - reporter.py:  结果报告器 (CSV 人工评价导出 + 摘要格式化)

CSV 导出格式:
    Question, Ground Truth, Prediction, Retriever, EM, F1, Comment

独立运行:
    python -m rag_project.evaluation.evaluator -i samples.json -c results.csv
"""

# 纯函数指标
from .metrics import (
    recall_at_k,
    mrr,
    exact_match,
    f1_score,
    aggregate_retrieval,
    aggregate_generation,
)

# 数据结构
from .evaluator import EvalSample, EvalResult, Evaluator

# 报告器
from .reporter import Reporter

__all__ = [
    # ── 指标函数 ──
    "recall_at_k",
    "mrr",
    "exact_match",
    "f1_score",
    "aggregate_retrieval",
    "aggregate_generation",
    # ── 数据结构 ──
    "EvalSample",
    "EvalResult",
    # ── 评估器 ──
    "Evaluator",
    # ── 报告器 ──
    "Reporter",
]
