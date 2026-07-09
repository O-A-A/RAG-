#!/usr/bin/env python3
"""
===============================================================================
RAG 检索策略对比实验 — 主运行脚本
===============================================================================
自动运行 BM25 / Dense / Hybrid 三组对比实验，输出标准化评估结果。

实验设计 (论文复现风格):
    - 固定变量: Embedding (BGE-small-zh), LLM (Qwen2.5-7B), chunk_size=512,
                chunk_overlap=50, top_k=5, Prompt 模板, 随机种子=42
    - 自变量:   Retriever 类型 (BM25 / Dense / Hybrid)
    - 因变量:   Recall@5, MRR, Exact Match, F1
    - 测试集:   100 条中文 NLP 领域 QA 对

运行模式:
    1. 完整模式 (默认):
       python -m rag_project.experiments.run_experiment
       → 加载 LLM, 运行检索 + 生成, 输出全部 4 个指标

    2. 仅检索模式 (无需 GPU):
       python -m rag_project.experiments.run_experiment --skip-llm
       → 仅运行检索评估, 输出 Recall@5 + MRR

输出文件 (自动生成于 results/ 目录):
    - experiment_result.csv: 单条样本详细结果 (100 条 × 3 组 = 300 行)
    - summary.csv:           三组实验聚合对比表
    - experiment_log.txt:    实验运行日志

可复现性保证:
    1. 所有随机种子固定为 42
    2. LLM temperature=0 (贪婪解码)
    3. 语料库和 QA 对硬编码 (每次实验输入完全相同)
    4. 实验参数写入输出文件头部
===============================================================================
"""

import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── 在 import 任何其他模块之前设置随机种子 ──
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# ── 项目路径 ──
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import logging


def setup_logging(log_dir: str) -> logging.Logger:
    """配置双路日志（控制台 + 文件）。"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    return logging.getLogger(__name__)


# ============================================================================
# 实验核心逻辑
# ============================================================================

class ExperimentRunner:
    """
    RAG 检索对比实验运行器。

    编排完整的实验流程:
        1. 加载语料 + 文本切分
        2. 构建 FAISS 索引 (Dense/Hybrid 共用)
        3. 初始化三种 Retriever
        4. 对 100 个问题逐个检索 + 评估
        5. 导出 experiment_result.csv + summary.csv

    Attributes:
        skip_llm (bool): 是否跳过 LLM (仅评估检索质量)。
        results_dir (str): 结果输出目录。
        logger: 日志记录器。
    """

    def __init__(
        self,
        skip_llm: bool = False,
        results_dir: str = "results",
    ):
        self.skip_llm = skip_llm
        self.results_dir = os.path.abspath(results_dir)
        os.makedirs(self.results_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # 用于持有加载的模块（延迟初始化）
        self.corpus: List[Dict] = []
        self.qa_pairs: List[Dict] = []
        self.chunks: List = []
        self.vector_store = None
        self.generator = None
        self.prompt_template = None

    # ==================================================================
    # 阶段 1: 数据准备
    # ==================================================================

    def _load_data(self) -> None:
        """加载语料库和 100 条 QA 对。"""
        from rag_project.experiments.sample_data import get_corpus, get_qa_pairs

        self.logger.info("=" * 55)
        self.logger.info("阶段 1/5: 加载实验数据")

        self.corpus = get_corpus()
        self.qa_pairs = get_qa_pairs()

        self.logger.info(f"  语料库: {len(self.corpus)} 篇文档")
        self.logger.info(f"  测试集: {len(self.qa_pairs)} 条 QA 对")

    # ==================================================================
    # 阶段 2: 文本切分 & FAISS 索引
    # ==================================================================

    def _build_index(self) -> None:
        """
        文本切分 + Embedding + FAISS 索引构建。

        三种 Retriever 中 Dense 和 Hybrid 共享同一个 FAISS 索引，
        BM25 使用相同的 chunks 但独立构建倒排索引。
        """
        from rag_project.data.splitter import TextSplitter
        from rag_project.embeddings.encoder import EmbeddingEncoder
        from rag_project.embeddings.vector_store import FAISSVectorStore
        from langchain_core.documents import Document

        self.logger.info("=" * 55)
        self.logger.info("阶段 2/5: 文本切分 & FAISS 索引构建")

        # ── 将语料转为 LangChain Document ──
        documents = []
        for item in self.corpus:
            documents.append(Document(
                page_content=item["text"],
                metadata={
                    "doc_id": item["id"],
                    "chunk_id": item["id"],
                    "title": item["title"],
                }
            ))

        # ── 文本切分 ──
        splitter = TextSplitter()
        self.chunks = splitter.split_documents(documents)
        self.logger.info(f"  切分完成: {len(documents)} 篇 → {len(self.chunks)} 个块")

        # ── BGE Embedding ──
        self.logger.info("  加载 BGE Embedding 模型...")
        encoder = EmbeddingEncoder()

        # ── FAISS 索引 ──
        self.logger.info("  构建 FAISS 索引...")
        self.vector_store = FAISSVectorStore(encoder)
        self.vector_store.build_from_documents(self.chunks)
        self.logger.info(f"  FAISS 索引: {self.vector_store.num_vectors} 个向量")

    # ==================================================================
    # 阶段 3: 初始化三种 Retriever
    # ==================================================================

    def _create_retrievers(self) -> Dict[str, Any]:
        """
        创建三种 Retriever 实例。

        Returns:
            Dict[str, BaseRetriever]: {"bm25": ..., "dense_bge": ..., "hybrid_rrf": ...}
        """
        from rag_project.retrievers.bm25_retriever import BM25Retriever
        from rag_project.retrievers.dense_retriever import DenseRetriever
        from rag_project.retrievers.hybrid_retriever import HybridRetriever

        self.logger.info("=" * 55)
        self.logger.info("阶段 3/5: 初始化三种 Retriever")

        # BM25 (所有 Retriever 共用相同的 chunks)
        bm25 = BM25Retriever(self.chunks, top_k=5)
        self.logger.info(f"  [1/3] {bm25}")

        # Dense (共用 FAISS 索引)
        dense = DenseRetriever(self.vector_store, top_k=5)
        self.logger.info(f"  [2/3] {dense}")

        # Hybrid (组合 BM25 + Dense, alpha=0.5)
        hybrid = HybridRetriever(bm25, dense, top_k=5, fusion_method="rrf")
        self.logger.info(f"  [3/3] {hybrid}")

        return {
            "bm25": bm25,
            "dense_bge": dense,
            "hybrid_rrf": hybrid,
        }

    # ==================================================================
    # 阶段 4: LLM 初始化 (可选)
    # ==================================================================

    def _init_llm(self) -> None:
        """初始化 LLM 生成器和 Prompt 模板。"""
        from rag_project.llm.prompt import RAGPromptTemplate
        from rag_project.llm.generator import LLMGenerator

        self.logger.info("=" * 55)
        self.logger.info("阶段 4/5: 初始化 LLM (Qwen2.5-7B-Instruct)")

        self.prompt_template = RAGPromptTemplate()
        self.generator = LLMGenerator()
        self.logger.info(f"  LLM 就绪: {self.generator}")

    # ==================================================================
    # 阶段 5: 运行实验
    # ==================================================================

    def _run_single_question(
        self,
        qa: Dict,
        retriever,
        retriever_name: str,
    ) -> Dict[str, Any]:
        """
        对单个问题执行检索 + 评估。

        Args:
            qa:              QA 对字典 (id, question, answer, relevant_doc_ids)。
            retriever:       Retriever 实例。
            retriever_name:  检索器名称。

        Returns:
            Dict: 包含所有评估指标的结果记录。
        """
        import time as time_module
        from rag_project.evaluation.metrics import (
            recall_at_k, mrr, exact_match, f1_score,
        )

        question = qa["question"]
        ground_truth = qa["answer"]
        relevant_ids = set(qa["relevant_doc_ids"])

        # ── 检索 ──
        t0 = time_module.perf_counter()
        retrieved = retriever.retrieve(question, top_k=5)
        retrieval_ms = (time_module.perf_counter() - t0) * 1000

        # 使用父文档 ID (metadata["doc_id"]) 而非 chunk_id 来匹配 relevant_doc_ids
        retrieved_doc_ids = [
            r.metadata.get("doc_id", r.doc_id) for r in retrieved
        ]

        # ── 检索指标 ──
        r5 = recall_at_k(retrieved_doc_ids, relevant_ids, k=5)
        mr = mrr(retrieved_doc_ids, relevant_ids)

        # ── 生成 (如果有 LLM) ──
        if not self.skip_llm and self.generator and self.prompt_template:
            context = "\n\n".join(
                f"[来源: {r.metadata.get('title', r.doc_id)}]\n{r.content}"
                for r in retrieved
            )
            prompt = self.prompt_template.build_prompt(
                question=question,
                context=context,
            )
            gen_start = time_module.perf_counter()
            prediction = self.generator.generate(prompt)
            gen_ms = (time_module.perf_counter() - gen_start) * 1000
        else:
            # 无 LLM: 使用占位符，EM/F1 不可计算
            prediction = "[LLM_NOT_AVAILABLE]"
            gen_ms = 0.0

        # ── 生成指标 ──
        em = exact_match(prediction, ground_truth)
        f1 = f1_score(prediction, ground_truth)

        return {
            "question_id": qa["id"],
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "retriever": retriever_name,
            "recall_at_5": round(r5, 4),
            "mrr": round(mr, 4),
            "exact_match": round(em, 4),
            "f1": round(f1, 4),
            "retrieval_time_ms": round(retrieval_ms, 2),
            "generation_time_ms": round(gen_ms, 2),
            "comment": "",
        }

    def _run_experiment_group(
        self,
        retriever,
        retriever_name: str,
    ) -> List[Dict[str, Any]]:
        """
        用指定 Retriever 运行全部 100 个问题。

        Args:
            retriever:       Retriever 实例。
            retriever_name:  检索器名称。

        Returns:
            List[Dict]: 100 条评估结果记录。
        """
        self.logger.info(f"\n{'─' * 50}")
        self.logger.info(f"  运行实验组: {retriever_name}")
        self.logger.info(f"{'─' * 50}")

        results: List[Dict] = []
        n = len(self.qa_pairs)

        for i, qa in enumerate(self.qa_pairs, 1):
            record = self._run_single_question(qa, retriever, retriever_name)
            results.append(record)

            # 每 20 条打印进度
            if i % 20 == 0 or i == n:
                recent = results[-20:]
                avg_r5 = np.mean([r["recall_at_5"] for r in recent])
                avg_mr = np.mean([r["mrr"] for r in recent])
                self.logger.info(
                    f"    [{i:3d}/{n}] "
                    f"Recall@5={avg_r5:.4f}  MRR={avg_mr:.4f}"
                )

        return results

    # ==================================================================
    # 主运行入口
    # ==================================================================

    def run(self) -> str:
        """
        运行完整实验。

        流程:
            1. 加载数据
            2. 构建索引
            3. 初始化检索器
            4. (可选) 初始化 LLM
            5. 依次运行三组实验
            6. 导出结果 CSV

        Returns:
            str: 结果输出目录的路径。
        """
        start_time = time.time()
        self.logger.info("=" * 55)
        self.logger.info("  RAG 检索策略对比实验 — 开始运行")
        self.logger.info(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"  模式: {'仅检索 (--skip-llm)' if self.skip_llm else '完整 RAG'}")
        self.logger.info(f"  随机种子: {SEED}")
        self.logger.info("=" * 55)

        # ── 阶段 1-3: 数据 + 索引 + 检索器 ──
        self._load_data()

        # 自动检测设备：CUDA 不可用时回退 CPU
        from rag_project.config import get_config
        cfg = get_config()
        try:
            import torch
            if not torch.cuda.is_available():
                self.logger.info("CUDA 不可用，Embedding 将使用 CPU")
                cfg.embedding_device = "cpu"
                cfg.llm_device = "cpu"
        except ImportError:
            cfg.embedding_device = "cpu"
            cfg.llm_device = "cpu"

        self._build_index()
        retrievers = self._create_retrievers()

        # ── 阶段 4: (可选) LLM ──
        if not self.skip_llm:
            try:
                import torch
                if torch.cuda.is_available():
                    self._init_llm()
                else:
                    self.logger.warning(
                        "CUDA 不可用，自动切换为仅检索模式。"
                        "如需生成指标 (EM/F1)，请在 GPU 环境下运行。"
                    )
                    self.skip_llm = True
            except ImportError:
                self.logger.warning("torch 未安装，自动切换为仅检索模式。")
                self.skip_llm = True
        else:
            self.logger.info("阶段 4/5: 跳过 LLM (--skip-llm)")

        # ── 阶段 5: 三组实验 ──
        self.logger.info("=" * 55)
        self.logger.info("阶段 5/5: 运行三组对比实验")
        self.logger.info(f"  每组 {len(self.qa_pairs)} 个问题")

        all_results: List[Dict[str, Any]] = []

        for retriever_name in ["bm25", "dense_bge", "hybrid_rrf"]:
            group_results = self._run_experiment_group(
                retrievers[retriever_name],
                retriever_name,
            )
            all_results.extend(group_results)

        # ── 导出结果 ──
        self._export_results(all_results)

        elapsed = time.time() - start_time
        self.logger.info(f"\n实验完成! 总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
        self.logger.info(f"结果目录: {self.results_dir}")

        return self.results_dir

    # ==================================================================
    # 结果导出
    # ==================================================================

    def _export_results(self, all_results: List[Dict[str, Any]]) -> None:
        """
        导出实验结果。

        生成两个文件:
            1. experiment_result.csv — 每条样本的详细结果
            2. summary.csv — 三组实验的聚合对比

        Args:
            all_results: 所有评估结果记录列表。
        """
        self.logger.info(f"\n{'=' * 55}")
        self.logger.info("导出实验结果")

        # ── 文件 1: experiment_result.csv ──
        result_path = os.path.join(self.results_dir, "experiment_result.csv")
        self._write_result_csv(all_results, result_path)

        # ── 文件 2: summary.csv ──
        summary_path = os.path.join(self.results_dir, "summary.csv")
        self._write_summary_csv(all_results, summary_path)

        # ── 终端打印摘要 ──
        self._print_summary_table(all_results)

    def _write_result_csv(
        self,
        results: List[Dict[str, Any]],
        path: str,
    ) -> None:
        """
        写入单条样本结果 CSV。

        CSV 列:
            question_id, question, ground_truth, prediction,
            retriever, recall_at_5, mrr, exact_match, f1,
            retrieval_time_ms, generation_time_ms, comment
        """
        fieldnames = [
            "question_id", "question", "ground_truth", "prediction",
            "retriever", "recall_at_5", "mrr", "exact_match", "f1",
            "retrieval_time_ms", "generation_time_ms", "comment",
        ]

        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        self.logger.info(f"  [OK] experiment_result.csv ({len(results)} 行)")

    def _write_summary_csv(
        self,
        results: List[Dict[str, Any]],
        path: str,
    ) -> None:
        """
        写入聚合摘要 CSV。

        按 retriever 分组计算均值，生成对比表。
        """
        # 分组聚合
        groups: Dict[str, List[Dict]] = {}
        for r in results:
            groups.setdefault(r["retriever"], []).append(r)

        summary_rows = []
        for name in ["bm25", "dense_bge", "hybrid_rrf"]:
            group = groups.get(name, [])
            n = len(group)
            if n == 0:
                continue

            summary_rows.append({
                "retriever": name,
                "n_questions": n,
                "recall_at_5": round(np.mean([r["recall_at_5"] for r in group]), 4),
                "mrr": round(np.mean([r["mrr"] for r in group]), 4),
                "exact_match": round(np.mean([r["exact_match"] for r in group]), 4),
                "f1": round(np.mean([r["f1"] for r in group]), 4),
                "avg_retrieval_ms": round(np.mean([r["retrieval_time_ms"] for r in group]), 2),
                "avg_generation_ms": round(np.mean([r["generation_time_ms"] for r in group]), 2),
            })

        fieldnames = [
            "retriever", "n_questions", "recall_at_5", "mrr",
            "exact_match", "f1", "avg_retrieval_ms", "avg_generation_ms",
        ]

        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

        self.logger.info(f"  [OK] summary.csv ({len(summary_rows)} 行)")

    def _print_summary_table(self, results: List[Dict[str, Any]]) -> None:
        """终端打印三组实验对比表。"""
        groups: Dict[str, List[Dict]] = {}
        for r in results:
            groups.setdefault(r["retriever"], []).append(r)

        print()
        print("=" * 75)
        print("  RAG 检索策略对比实验结果")
        print("=" * 75)
        header = (
            f"  {'Retriever':<16s} {'N':>5s}  "
            f"{'Recall@5':>9s}  {'MRR':>9s}  "
            f"{'EM':>9s}  {'F1':>9s}"
        )
        print(header)
        print("-" * 75)

        for name in ["bm25", "dense_bge", "hybrid_rrf"]:
            group = groups.get(name, [])
            n = len(group)
            if n == 0:
                continue
            r5 = np.mean([r["recall_at_5"] for r in group])
            mr = np.mean([r["mrr"] for r in group])
            em = np.mean([r["exact_match"] for r in group])
            f1 = np.mean([r["f1"] for r in group])
            print(
                f"  {name:<16s} {n:>5d}  "
                f"{r5:>9.4f}  {mr:>9.4f}  "
                f"{em:>9.4f}  {f1:>9.4f}"
            )

        print("=" * 75)
        if self.skip_llm:
            print("  注意: EM 和 F1 为占位值 (LLM 未加载)")
        print()


# ============================================================================
# CLI 入口
# ============================================================================

def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="RAG 检索策略对比实验 — BM25 vs Dense vs Hybrid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python -m rag_project.experiments.run_experiment              # 完整实验 (含 LLM)
    python -m rag_project.experiments.run_experiment --skip-llm   # 仅检索评估
    python -m rag_project.experiments.run_experiment -o my_results # 指定输出目录
        """,
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="跳过 LLM，仅评估检索质量 (Recall@5 + MRR)。无需 GPU。"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="results",
        help="结果输出目录 (默认: results)"
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    # 日志
    logger = setup_logging(args.output)

    # 设置 torch 种子 (如果可用)
    try:
        import torch
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info(f"PyTorch 随机种子已设置: {SEED} (CUDA deterministic)")
        else:
            logger.info(f"PyTorch 随机种子已设置: {SEED} (CPU)")
    except ImportError:
        logger.info("PyTorch 未安装 (仅检索模式可用)")

    # 运行实验
    runner = ExperimentRunner(
        skip_llm=args.skip_llm,
        results_dir=args.output,
    )
    runner.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
