"""
===============================================================================
评估器 (Evaluator) — 自动指标计算引擎
===============================================================================
接收问题、检索结果、生成答案和标准答案，自动计算所有评估指标。

核心能力:
    1. 单样本评估: evaluate_one() → 计算单条样本的 Recall@5, MRR, EM, F1
    2. 批量评估:   evaluate_batch() → 对多个样本计算 + 聚合统计
    3. 独立运行:   支持 CLI 和编程两种调用方式

输入 (EvalSample):
    - question_id:       问题唯一标识
    - question:          问题文本
    - ground_truth:      标准答案文本
    - prediction:        模型生成的答案
    - retriever_name:    使用的检索器名称
    - retrieved_doc_ids: 检索返回的文档 ID 列表（用于 Recall@5 / MRR）
    - relevant_doc_ids:  标注的相关文档 ID 集合（用于 Recall@5 / MRR）

输出 (EvalResult):
    继承 EvalSample + 自动计算的四个指标值 + comment 占位

设计原则:
    - 评估器不关心 Retriever 的具体实现，只消费其输出
    - 所有指标计算委托给 metrics.py 纯函数
    - 通过 dataclass 确保数据结构一目了然
===============================================================================
"""

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set

from rag_project.evaluation.metrics import (
    recall_at_k,
    mrr,
    exact_match,
    f1_score,
    aggregate_retrieval,
    aggregate_generation,
)

import logging

logger = logging.getLogger(__name__)


# ========================================================================
# 数据结构
# ========================================================================

@dataclass
class EvalSample:
    """
    评估输入样本。

    Attributes:
        question_id:       问题唯一标识。
        question:          问题文本。
        ground_truth:      标准答案（人工标注的正确回答）。
        prediction:        模型生成的预测答案。
        retriever_name:    检索器标识（如 "bm25", "dense_bge", "hybrid_a50"）。
        retrieved_doc_ids: 检索器返回的文档 ID 列表（有序，含排名）。
        relevant_doc_ids:  标注的相关文档 ID 集合（用于检索评估）。
    """
    question_id: str
    question: str
    ground_truth: str
    prediction: str
    retriever_name: str = ""
    retrieved_doc_ids: List[str] = field(default_factory=list)
    relevant_doc_ids: Set[str] = field(default_factory=set)

    # ── 工厂方法 ─────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalSample":
        """从字典构建 EvalSample（relevent_doc_ids 自动转为 set）。"""
        data = dict(data)
        if "relevant_doc_ids" in data and not isinstance(data["relevant_doc_ids"], set):
            data["relevant_doc_ids"] = set(data["relevant_doc_ids"])
        return cls(**{
            k: v for k, v in data.items()
            if k in cls.__dataclass_fields__
        })


@dataclass
class EvalResult:
    """
    评估输出结果（单条样本）。

    包含原始样本信息 + 自动计算的四个指标 + 人工评价注释。

    Attributes:
        sample:         原始评估样本。
        recall_at_5:    Recall@5 (检索质量)。
        mrr:            MRR (检索排序质量)。
        exact_match:    EM (生成精确匹配, 0/1)。
        f1:             F1 (生成 Token 级匹配)。
        comment:        人工评价注释（CSV 导出后由人类填写）。
    """
    sample: EvalSample
    recall_at_5: float = 0.0
    mrr: float = 0.0
    exact_match: float = 0.0
    f1: float = 0.0
    comment: str = ""

    # ── CSV 导出时的列顺序 ───────────────────────────────────────────

    def to_csv_row(self) -> Dict[str, str]:
        """
        转为 CSV 行字典。

        列顺序严格按照: Question, Ground Truth, Prediction,
                        Retriever, EM, F1, Comment

        Returns:
            Dict[str, str]: CSV 列名 → 值 的映射。
        """
        return {
            "Question":     self.sample.question,
            "Ground Truth": self.sample.ground_truth,
            "Prediction":   self.sample.prediction,
            "Retriever":    self.sample.retriever_name,
            "EM":           f"{self.exact_match:.4f}",
            "F1":           f"{self.f1:.4f}",
            "Comment":      self.comment,
        }


# ========================================================================
# 评估器
# ========================================================================

class Evaluator:
    """
    RAG 系统评估器。

    自动计算:
        - 检索质量: Recall@5, MRR
        - 生成质量: Exact Match, F1

    Usage:
        >>> evaluator = Evaluator()
        >>> sample = EvalSample(
        ...     question_id="q1",
        ...     question="什么是RAG？",
        ...     ground_truth="检索增强生成",
        ...     prediction="RAG是检索增强生成技术",
        ...     retriever_name="bm25",
        ...     retrieved_doc_ids=["d1","d3","d2"],
        ...     relevant_doc_ids={"d1","d2","d4"},
        ... )
        >>> result = evaluator.evaluate_one(sample)
        >>> print(result.recall_at_5, result.mrr, result.exact_match, result.f1)
    """

    def __init__(self, recall_k: int = 5):
        """
        初始化评估器。

        Args:
            recall_k: Recall@K 中的 K 值，默认 5。
        """
        self.recall_k = recall_k

    # ==================================================================
    # 单样本评估
    # ==================================================================

    def evaluate_one(self, sample: EvalSample) -> EvalResult:
        """
        对单个样本计算全部四个指标。

        Args:
            sample: 包含问题、答案、检索结果的 EvalSample。

        Returns:
            EvalResult: 自动计算 EM, F1, Recall@5, MRR 后的完整结果。
        """
        # ── 生成质量 ──
        em = exact_match(sample.prediction, sample.ground_truth)
        f1 = f1_score(sample.prediction, sample.ground_truth)

        # ── 检索质量 ──
        r5 = recall_at_k(
            sample.retrieved_doc_ids,
            sample.relevant_doc_ids,
            k=self.recall_k,
        )
        mr = mrr(sample.retrieved_doc_ids, sample.relevant_doc_ids)

        return EvalResult(
            sample=sample,
            recall_at_5=r5,
            mrr=mr,
            exact_match=em,
            f1=f1,
        )

    # ==================================================================
    # 批量评估
    # ==================================================================

    def evaluate_batch(self, samples: List[EvalSample]) -> List[EvalResult]:
        """
        批量评估多个样本。

        Args:
            samples: EvalSample 列表。

        Returns:
            List[EvalResult]: 每个样本的完整评估结果。
        """
        if not samples:
            logger.warning("samples 为空，返回空结果")
            return []

        logger.info(f"开始批量评估: {len(samples)} 个样本")

        results = [self.evaluate_one(s) for s in samples]

        logger.info(f"批量评估完成")
        return results

    # ==================================================================
    # 聚合统计
    # ==================================================================

    def aggregate(self, results: List[EvalResult]) -> Dict[str, float]:
        """
        对评估结果列表做聚合统计（均值）。

        将检索指标和生成指标分开聚合，返回简洁的统计字典。

        Args:
            results: evaluate_batch() 返回的结果列表。

        Returns:
            Dict: 聚合后的平均指标。
                {
                    "recall_at_5": 0.XX,
                    "mrr": 0.XX,
                    "exact_match": 0.XX,
                    "f1": 0.XX,
                    "n_samples": N,
                }
        """
        if not results:
            return {}

        n = len(results)

        # 分离检索和生成数据
        all_retrieved = [r.sample.retrieved_doc_ids for r in results]
        all_relevant = [r.sample.relevant_doc_ids for r in results]
        all_preds = [r.sample.prediction for r in results]
        all_gts = [r.sample.ground_truth for r in results]

        # 聚合
        retrieval_stats = aggregate_retrieval(
            all_retrieved, all_relevant,
            k_values=[self.recall_k],
        )
        gen_stats = aggregate_generation(all_preds, all_gts)

        return {
            f"recall_at_{self.recall_k}": retrieval_stats.get(
                f"recall@{self.recall_k}", 0.0
            ),
            "mrr": retrieval_stats.get("mrr", 0.0),
            "exact_match": gen_stats.get("exact_match", 0.0),
            "f1": gen_stats.get("f1", 0.0),
            "n_samples": n,
        }

    # ==================================================================
    # 便捷方法：从原始数据直接评估
    # ==================================================================

    def evaluate_from_raw(
        self,
        questions: List[str],
        ground_truths: List[str],
        predictions: List[str],
        retriever_name: str = "",
        retrieved_doc_ids_list: Optional[List[List[str]]] = None,
        relevant_doc_ids_list: Optional[List[Set[str]]] = None,
    ) -> List[EvalResult]:
        """
        从原始列表数据直接构建样本并评估。

        Args:
            questions:             问题文本列表。
            ground_truths:         标准答案列表。
            predictions:           预测答案列表。
            retriever_name:        检索器名称。
            retrieved_doc_ids_list: 检索结果 ID 列表的列表（可选）。
            relevant_doc_ids_list:  相关文档 ID 集合的列表（可选）。

        Returns:
            List[EvalResult]: 评估结果列表。
        """
        n = len(questions)
        if not (n == len(ground_truths) == len(predictions)):
            raise ValueError(
                f"列表长度不一致: questions={len(questions)}, "
                f"ground_truths={len(ground_truths)}, "
                f"predictions={len(predictions)}"
            )

        # 填充可选的检索数据
        if retrieved_doc_ids_list is None:
            retrieved_doc_ids_list = [[] for _ in range(n)]
        if relevant_doc_ids_list is None:
            relevant_doc_ids_list = [set() for _ in range(n)]

        samples = []
        for i in range(n):
            samples.append(EvalSample(
                question_id=f"q_{i:04d}",
                question=questions[i],
                ground_truth=ground_truths[i],
                prediction=predictions[i],
                retriever_name=retriever_name,
                retrieved_doc_ids=retrieved_doc_ids_list[i],
                relevant_doc_ids=relevant_doc_ids_list[i],
            ))

        return self.evaluate_batch(samples)

    # ==================================================================
    # 序列化
    # ==================================================================

    def results_to_dicts(self, results: List[EvalResult]) -> List[Dict[str, Any]]:
        """
        将评估结果列表转为字典列表（用于 JSON 序列化）。

        Args:
            results: 评估结果列表。

        Returns:
            List[Dict]: 可 JSON 序列化的字典列表。
        """
        output = []
        for r in results:
            d = asdict(r)
            # set → list for JSON
            d["sample"]["relevant_doc_ids"] = list(
                r.sample.relevant_doc_ids
            )
            output.append(d)
        return output

    def save_results_json(
        self,
        results: List[EvalResult],
        file_path: str,
    ) -> None:
        """
        将评估结果保存为 JSON 文件。

        Args:
            results:   评估结果列表。
            file_path: 输出 JSON 文件路径。
        """
        data = self.results_to_dicts(results)
        # 附加聚合统计
        summary = self.aggregate(results)
        payload = {
            "summary": summary,
            "results": data,
        }

        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info(f"评估结果已保存: {file_path} ({len(results)} 条)")


# ============================================================================
# CLI 独立运行
# ============================================================================

def _parse_args():
    """解析命令行参数。"""
    import argparse
    parser = argparse.ArgumentParser(
        description="RAG Evaluation — 自动计算 Recall@5, MRR, EM, F1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
输入格式 (JSON):
    [
        {
            "question_id": "q_0001",
            "question": "什么是RAG？",
            "ground_truth": "检索增强生成技术",
            "prediction": "RAG是...",
            "retriever_name": "bm25",
            "retrieved_doc_ids": ["d1", "d3"],
            "relevant_doc_ids": ["d1", "d2"]
        },
        ...
    ]

输出 CSV 格式:
    Question, Ground Truth, Prediction, Retriever, EM, F1, Comment

示例:
    python -m rag_project.evaluation.evaluator -i samples.json -o results.csv
    python -m rag_project.evaluation.evaluator -i samples.json --json-out metrics.json
        """,
    )
    parser.add_argument("--input", "-i", required=True,
                        help="输入 JSON 文件（EvalSample 数组）")
    parser.add_argument("--csv-out", "-c", default="eval_results.csv",
                        help="人工评价 CSV 输出路径 (默认: eval_results.csv)")
    parser.add_argument("--json-out", "-j", default=None,
                        help="完整评估结果 JSON 输出路径")
    parser.add_argument("--recall-k", "-k", type=int, default=5,
                        help="Recall@K 的 K 值 (默认: 5)")
    return parser.parse_args()


def main():
    """CLI 入口。"""
    args = _parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("  RAG Evaluation — 自动指标计算")
    print(f"  Recall@{args.recall_k} | MRR | Exact Match | F1")
    print("=" * 60)

    # 加载输入
    if not os.path.exists(args.input):
        print(f"[ERROR] 输入文件不存在: {args.input}")
        return 1

    logger.info(f"加载评估样本: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        raw = raw.get("results", raw.get("samples", []))
    if not isinstance(raw, list):
        print("[ERROR] 输入 JSON 应为 EvalSample 数组")
        return 1

    # 构建样本
    samples = [EvalSample.from_dict(r) for r in raw]
    logger.info(f"加载了 {len(samples)} 个样本")

    # 评估
    evaluator = Evaluator(recall_k=args.recall_k)
    results = evaluator.evaluate_batch(samples)

    # 聚合统计
    summary = evaluator.aggregate(results)
    print("\n" + "=" * 40)
    print("  聚合指标 Summary")
    print("=" * 40)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print()

    # 导出 CSV
    from rag_project.evaluation.reporter import Reporter
    reporter = Reporter()
    reporter.to_csv(results, args.csv_out)
    print(f"[OK] 人工评价 CSV 已导出: {args.csv_out}")

    # 导出 JSON (可选)
    if args.json_out:
        evaluator.save_results_json(results, args.json_out)
        print(f"[OK] 完整结果 JSON 已导出: {args.json_out}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
