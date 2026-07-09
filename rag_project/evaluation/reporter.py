"""
===============================================================================
评估结果报告器 (Reporter) — CSV 导出 + 摘要格式化
===============================================================================
将 EvalResult 列表导出为多种格式，方便实验分析和论文写作。

输出格式:
    1. CSV (人工评价):  Question, Ground Truth, Prediction, Retriever, EM, F1, Comment
    2. JSON (完整结果): 每条样本的四个指标 + 元数据
    3. Text (控制台打印): 聚合指标摘要表格

CSV 列说明:
    ┌────────────┬──────────────┬────────────┬───────────┬─────┬─────┬─────────┐
    │  Question  │ Ground Truth │ Prediction │ Retriever │ EM  │ F1  │ Comment │
    ├────────────┼──────────────┼────────────┼───────────┼─────┼─────┼─────────┤
    │ 问题文本    │ 标准答案      │ 模型预测    │ 检索器名    │ 自动 │ 自动 │ 人工填写 │
    └────────────┴──────────────┴────────────┴───────────┴─────┴─────┴─────────┘

    其中 EM 和 F1 列由 Evaluator 自动计算并填入；
    Comment 列为空白，供人工评价时填写。
===============================================================================
"""

import csv
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from rag_project.evaluation.evaluator import EvalResult, EvalSample

import logging

logger = logging.getLogger(__name__)


class Reporter:
    """
    评估结果报告器。

    负责:
        - 导出人工评价 CSV（标准 7 列格式）
        - 导出完整评估结果 JSON
        - 生成控制台可读的聚合摘要

    Usage:
        >>> reporter = Reporter()
        >>> reporter.to_csv(results, "eval_output.csv")
        >>> reporter.print_summary(results)
    """

    # CSV 列定义（固定顺序）
    CSV_COLUMNS = [
        "Question",
        "Ground Truth",
        "Prediction",
        "Retriever",
        "EM",
        "F1",
        "Comment",
    ]

    # ==================================================================
    # CSV 导出
    # ==================================================================

    def to_csv(
        self,
        results: List[EvalResult],
        output_path: str,
        encoding: str = "utf-8-sig",
    ) -> str:
        """
        导出人工评价 CSV。

        生成 Excel 可直接打开的 CSV 文件（BOM + UTF-8）。
        前 6 列由系统自动填充，Comment 列留空供人工填写。

        Args:
            results:     评估结果列表。
            output_path: 输出 CSV 文件路径。
            encoding:    文件编码。默认 "utf-8-sig"（含 BOM，Excel 友好）。

        Returns:
            str: 输出文件的绝对路径。

        Raises:
            ValueError: results 为空。

        Example 输出:
            Question,Ground Truth,Prediction,Retriever,EM,F1,Comment
            什么是RAG？,检索增强生成,RAG是检索增强生成技术,bm25,0.0000,0.5000,
            深度学习的应用？,...,...,dense_bge,0.0000,0.7500,
        """
        if not results:
            raise ValueError("results 不能为空")

        # 确保输出目录存在
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        abs_path = os.path.abspath(output_path)

        with open(abs_path, "w", newline="", encoding=encoding) as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
            writer.writeheader()

            for r in results:
                writer.writerow(r.to_csv_row())

        logger.info(
            f"人工评价 CSV 导出完成: {abs_path}\n"
            f"  共 {len(results)} 行 (含表头), "
            f"  编码: {encoding}"
        )

        return abs_path

    # ==================================================================
    # 摘要打印
    # ==================================================================

    def print_summary(
        self,
        results: List[EvalResult],
        title: Optional[str] = None,
    ) -> str:
        """
        打印评估指标聚合摘要（控制台友好格式）。

        Args:
            results: 评估结果列表。
            title:   摘要标题（可选）。

        Returns:
            str: 格式化的摘要文本。
        """
        if not results:
            return "(无评估结果)"

        n = len(results)

        # 计算均值
        em_avg = sum(r.exact_match for r in results) / n
        f1_avg = sum(r.f1 for r in results) / n
        r5_avg = sum(r.recall_at_5 for r in results) / n
        mrr_avg = sum(r.mrr for r in results) / n

        # 统计检索器分布
        retriever_counts: Dict[str, int] = {}
        for r in results:
            name = r.sample.retriever_name or "unknown"
            retriever_counts[name] = retriever_counts.get(name, 0) + 1

        # 构建摘要
        lines = []
        width = 50
        lines.append("=" * width)
        lines.append(f"  {title or 'RAG Evaluation Summary'}")
        lines.append("=" * width)
        lines.append(f"  Total samples:     {n}")
        lines.append("-" * width)
        lines.append(f"  {'Recall@5':<20s} {r5_avg:>8.4f}")
        lines.append(f"  {'MRR':<20s} {mrr_avg:>8.4f}")
        lines.append(f"  {'Exact Match':<20s} {em_avg:>8.4f}")
        lines.append(f"  {'F1':<20s} {f1_avg:>8.4f}")
        lines.append("-" * width)
        lines.append(f"  Retriever distribution:")
        for name, count in retriever_counts.items():
            lines.append(f"    {name}: {count}")
        lines.append("=" * width)

        text = "\n".join(lines)
        print(text)
        return text

    # ==================================================================
    # 分组摘要（按 Retriever 分组对比）
    # ==================================================================

    def print_grouped_summary(
        self,
        results: List[EvalResult],
    ) -> str:
        """
        按 Retriever 分组打印对比摘要。

        用于实验中对比不同检索策略的性能。

        Args:
            results: 评估结果列表（可包含多个 Retriever 的结果）。

        Returns:
            str: 分组对比摘要文本。
        """
        # 按 retriever_name 分组
        groups: Dict[str, List[EvalResult]] = {}
        for r in results:
            name = r.sample.retriever_name or "unknown"
            groups.setdefault(name, []).append(r)

        if not groups:
            return "(无评估结果)"

        lines = []
        width = 70
        lines.append("=" * width)
        lines.append("  RAG Retrieval Strategy Comparison")
        lines.append("=" * width)
        header = (
            f"  {'Retriever':<20s} {'N':>5s}  "
            f"{'Recall@5':>8s}  {'MRR':>8s}  "
            f"{'EM':>8s}  {'F1':>8s}"
        )
        lines.append(header)
        lines.append("-" * width)

        for name in sorted(groups.keys()):
            group = groups[name]
            n = len(group)
            r5 = sum(r.recall_at_5 for r in group) / n
            mr = sum(r.mrr for r in group) / n
            em = sum(r.exact_match for r in group) / n
            f1 = sum(r.f1 for r in group) / n

            lines.append(
                f"  {name:<20s} {n:>5d}  "
                f"{r5:>8.4f}  {mr:>8.4f}  "
                f"{em:>8.4f}  {f1:>8.4f}"
            )

        lines.append("=" * width)
        text = "\n".join(lines)
        print(text)
        return text

    # ==================================================================
    # JSON 导出（完整结果）
    # ==================================================================

    def to_json(
        self,
        results: List[EvalResult],
        output_path: str,
    ) -> str:
        """
        导出完整评估结果为 JSON（含每条样本的四个指标 + 元数据）。

        Args:
            results:     评估结果列表。
            output_path: 输出 JSON 文件路径。

        Returns:
            str: 输出文件路径。
        """
        from rag_project.evaluation.evaluator import Evaluator

        evaluator = Evaluator()
        data = evaluator.results_to_dicts(results)
        summary = evaluator.aggregate(results)

        payload = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "n_samples": len(results),
            },
            "summary": summary,
            "results": data,
        }

        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info(f"完整评估结果 JSON 导出完成: {output_path}")
        return os.path.abspath(output_path)


# ============================================================================
# 模块自测
# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("Reporter 模块自测")
    print("=" * 60)

    from rag_project.evaluation.evaluator import Evaluator

    # 构建模拟评估结果
    samples = [
        EvalSample(
            question_id=f"q_{i:04d}",
            question=f"测试问题 {i}: 深度学习是什么？",
            ground_truth="深度学习是机器学习的分支",
            prediction=f"深度学习是AI技术{'的一个分支' if i % 2 == 0 else ''}",
            retriever_name=["bm25", "dense_bge", "hybrid_a50"][i % 3],
            retrieved_doc_ids=[f"doc_{j}" for j in range(5)],
            relevant_doc_ids={f"doc_{j}" for j in range(3)},
        )
        for i in range(9)
    ]

    evaluator = Evaluator(recall_k=5)
    results = evaluator.evaluate_batch(samples)

    reporter = Reporter()

    # 测试 CSV 导出
    csv_path = reporter.to_csv(results, "test_eval_output.csv")
    print(f"\nCSV 导出: {csv_path}")

    # 测试摘要
    print()
    reporter.print_summary(results, title="Test Summary")

    # 测试分组摘要
    print()
    reporter.print_grouped_summary(results)

    # 验证 CSV 内容
    print("\nCSV 内容预览:")
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            if i < 4:
                print(f"  {line.rstrip()}")
    print(f"  ... (共 {len(results) + 1} 行)")

    # 清理
    if os.path.exists("test_eval_output.csv"):
        os.remove("test_eval_output.csv")

    print("\n[PASS] Reporter 自测通过")
