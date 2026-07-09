"""
===============================================================================
混合检索器 (Hybrid Retriever) — 分数级融合
===============================================================================
融合 BM25 稀疏检索和 Dense 稠密检索的得分，获得优于单一方法的检索质量。

融合公式 (Score-Level Fusion):
    Hybrid_Score(d) = α × Dense_norm(d) + (1−α) × BM25_norm(d)

    其中:
        - α ∈ [0, 1]:              控制 Dense 语义匹配的权重
        - Dense_norm(d):            Dense 得分的 min-max 归一化值 [0, 1]
        - BM25_norm(d):             BM25 得分的 min-max 归一化值 [0, 1]
        - 若文档 d 未出现在某检索器结果中，该项 = 0

参数语义:
    - α = 1.0  →  纯 Dense 检索（完全依赖语义匹配）
    - α = 0.5  →  等权重融合（默认值，平衡关键词和语义）
    - α = 0.0  →  纯 BM25 检索（完全依赖关键词匹配）

为什么需要归一化:
    - BM25 得分:  无界正数 (如 0 ~ 50+)，量纲取决于词频和文档长度
    - Dense 得分:  余弦相似度 [-1, 1]
    - 直接加权求和会导致 BM25 主导 → 必须先各自 min-max 归一化到 [0, 1]

融合流程:
    1. 放大候选池:  BM25 和 Dense 各检索 top_k × MULTIPLIER 个候选
    2. 分别归一化:  对两组的得分独立做 min-max 归一化
    3. 候选合并:    取两组候选 doc_id 的并集
    4. 得分融合:    Hybrid = α × norm_dense + (1−α) × norm_bm25
    5. 排序截断:    按 Hybrid 得分降序 → 取 top-k

设计模式:
    组合模式 (Composition):  Hybrid 持有 BM25 和 Dense 的引用，
    而非继承它们。这避免了代码重复，且符合"组合优于继承"原则。
===============================================================================
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from langchain_core.documents import Document

from rag_project.retrievers.base import BaseRetriever, RetrievalResult
from rag_project.retrievers.bm25_retriever import BM25Retriever
from rag_project.retrievers.dense_retriever import DenseRetriever

import logging

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """
    混合检索器: BM25 + Dense 分数级融合。

    组合 BM25 和 Dense 两个检索器，通过 min-max 归一化 + 加权求和
    融合两者的得分，实现关键词匹配和语义匹配的优势互补。

    关键设计: 所有归一化逻辑复用 BaseRetriever.normalize_scores()，
    所有结果格式化复用 BaseRetriever._format_results()，
    Hybrid 自身仅包含融合编排逻辑。

    Attributes:
        bm25 (BM25Retriever):          BM25 稀疏检索器。
        dense (DenseRetriever):        Dense 稠密检索器。
        alpha (float):                 Dense 语义权重 (α ∈ [0, 1])。
        candidate_multiplier (int):    候选池放大系数。

    Usage:
        >>> bm25 = BM25Retriever(chunks, top_k=5)
        >>> dense = DenseRetriever(vector_store, top_k=5)
        >>> hybrid = HybridRetriever(bm25, dense, alpha=0.5, top_k=5)
        >>> results = hybrid.retrieve("什么是深度学习？")
        >>> for r in results:
        ...     print(f"[hybrid={r.score:.4f}] {r.content[:50]}...")
    """

    # 候选池放大系数
    # 每个子检索器先检索 top_k × MULTIPLIER 个候选，
    # 融合排序后再截断到 top_k。更大的候选池 → 更全面的融合结果。
    CANDIDATE_MULTIPLIER: int = 4

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        alpha: float = 0.5,
        top_k: int = 5,
        fusion_method: str = "rrf",
    ):
        """
        初始化混合检索器。

        Args:
            bm25_retriever:  已初始化的 BM25 检索器。
            dense_retriever: 已初始化的 Dense 检索器。
            alpha:           Dense 语义匹配的权重 (α ∈ [0, 1])。
                             仅当 fusion_method="linear" 时使用。
            top_k:           默认检索返回数量。
            fusion_method:   融合策略: "rrf" (倒数秩融合) 或 "linear" (线性加权)。
        """
        super().__init__(top_k=top_k)

        if fusion_method not in ("rrf", "linear"):
            raise ValueError(f"fusion_method 必须是 'rrf' 或 'linear'，实际值: {fusion_method}")
        if fusion_method == "linear" and not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha 必须在 [0, 1] 范围内，实际值: {alpha}")

        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.alpha = alpha
        self.fusion_method = fusion_method

        logger.info(
            f"HybridRetriever 初始化完成:\n"
            f"  BM25: {self.bm25}\n"
            f"  Dense: {self.dense}\n"
            f"  Fusion: {self.fusion_method}"
            + (f", α={self.alpha}" if fusion_method == "linear" else "")
            + f"\n  top_k={self.top_k}, candidate_multiplier={self.CANDIDATE_MULTIPLIER}"
        )

    @property
    def name(self) -> str:
        """检索器唯一标识名。"""
        if self.fusion_method == "rrf":
            return "hybrid_rrf"
        return f"hybrid_a{int(self.alpha * 100):02d}"

    # ========================================================================
    # 核心检索逻辑
    # ========================================================================

    def _retrieve_raw(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """执行混合检索并进行分数融合。根据 fusion_method 选择 RRF 或线性融合。"""
        candidate_k = self.top_k * self.CANDIDATE_MULTIPLIER

        # ---- 阶段 1: 放大检索 ----
        bm25_results = self.bm25.retrieve(query, top_k=candidate_k)
        dense_results = self.dense.retrieve(query, top_k=candidate_k)

        # ---- 阶段 2-5: 融合 ----
        if self.fusion_method == "rrf":
            return self._rrf_fuse(bm25_results, dense_results)
        else:
            return self._linear_fuse(bm25_results, dense_results)

    # ==================================================================
    # RRF 融合 (默认，基于排名)
    # ==================================================================

    def _rrf_fuse(
        self,
        bm25_results: List[RetrievalResult],
        dense_results: List[RetrievalResult],
    ) -> List[Tuple[Document, float]]:
        """
        RRF (Reciprocal Rank Fusion) 融合。

        RRF_score(d) = 1/(k_rrf + rank_bm25(d)) + 1/(k_rrf + rank_dense(d))
        若 d 不在某检索器结果中，该项 = 0。

        优势: 不受得分量纲差异影响，无需归一化。
        """
        k_rrf = 60

        # rank 映射 (1-based)
        bm25_rank = {r.doc_id: i + 1 for i, r in enumerate(bm25_results)}
        dense_rank = {r.doc_id: i + 1 for i, r in enumerate(dense_results)}

        # 内容映射
        info: Dict[str, RetrievalResult] = {}
        for r in bm25_results:
            info[r.doc_id] = r
        for r in dense_results:
            if r.doc_id not in info:
                info[r.doc_id] = r

        fused: List[Tuple[Document, float]] = []
        all_ids = set(bm25_rank.keys()) | set(dense_rank.keys())

        for doc_id in all_ids:
            rrf = 0.0
            if doc_id in bm25_rank:
                rrf += 1.0 / (k_rrf + bm25_rank[doc_id])
            if doc_id in dense_rank:
                rrf += 1.0 / (k_rrf + dense_rank[doc_id])

            obj = info.get(doc_id)
            if obj is None:
                continue

            doc = Document(
                page_content=obj.content,
                metadata={
                    **obj.metadata,
                    "_bm25_rank": bm25_rank.get(doc_id, -1),
                    "_dense_rank": dense_rank.get(doc_id, -1),
                }
            )
            fused.append((doc, rrf))

        fused.sort(key=lambda x: x[1], reverse=True)

        logger.info(
            f"[Hybrid RRF] 融合完成: "
            f"BM25={len(bm25_results)}, Dense={len(dense_results)}, "
            f"并集={len(all_ids)}"
        )
        return fused

    # ==================================================================
    # 线性融合 (备选，基于得分)
    # ==================================================================

    def _linear_fuse(
        self,
        bm25_results: List[RetrievalResult],
        dense_results: List[RetrievalResult],
    ) -> List[Tuple[Document, float]]:
        """
        线性加权融合。

        Hybrid = α × Dense_norm + (1-α) × BM25_norm
        需要先 min-max 归一化得分到 [0,1]。
        """
        norm_bm25 = self.normalize_scores(bm25_results)
        norm_dense = self.normalize_scores(dense_results)

        all_ids: Set[str] = set(norm_bm25.keys()) | set(norm_dense.keys())
        bm25_map = self._build_result_map(bm25_results)
        dense_map = self._build_result_map(dense_results)

        fused: List[Tuple[Document, float]] = []
        for doc_id in all_ids:
            bm25_norm = norm_bm25.get(doc_id, 0.0)
            dense_norm = norm_dense.get(doc_id, 0.0)
            hybrid_score = self.alpha * dense_norm + (1.0 - self.alpha) * bm25_norm

            result_obj = dense_map.get(doc_id) or bm25_map.get(doc_id)
            if result_obj is None:
                continue

            doc = Document(
                page_content=result_obj.content,
                metadata={
                    **result_obj.metadata,
                    "_bm25_norm": bm25_norm,
                    "_dense_norm": dense_norm,
                }
            )
            fused.append((doc, hybrid_score))

        fused.sort(key=lambda x: x[1], reverse=True)

        logger.info(
            f"[Hybrid Linear α={self.alpha}] 融合完成: "
            f"BM25={len(bm25_results)}, Dense={len(dense_results)}, "
            f"并集={len(all_ids)}"
        )
        return fused

    # ========================================================================
    # 工具方法
    # ========================================================================

    def set_alpha(self, alpha: float) -> None:
        """
        动态修改融合权重。

        Args:
            alpha: 新的 Dense 权重 (0~1)。
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha 必须在 [0, 1] 范围内，实际值: {alpha}")
        self.alpha = alpha
        logger.info(f"Hybrid α 已更新: {alpha}")

    def get_fusion_params(self) -> Dict[str, float]:
        """获取融合参数（用于实验记录）。"""
        return {
            "alpha": self.alpha,
            "bm25_weight": 1.0 - self.alpha,
            "dense_weight": self.alpha,
            "candidate_multiplier": self.CANDIDATE_MULTIPLIER,
        }

    def __repr__(self) -> str:
        return (
            f"HybridRetriever("
            f"name='{self.name}', "
            f"α={self.alpha}, "
            f"top_k={self.top_k})"
        )


# ============================================================================
# 模块自测
# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    print("=" * 60)
    print("HybridRetriever 模块自测 (Score-Level Fusion)")
    print("=" * 60)

    from langchain_core.documents import Document as LCDoc
    from rag_project.embeddings.encoder import EmbeddingEncoder
    from rag_project.embeddings.vector_store import FAISSVectorStore

    # 构建测试文档
    test_chunks = []
    texts = [
        "BM25是一种基于词频的检索算法，通过倒排索引实现高效的关键词匹配。",
        "深度学习使用神经网络从数据中学习特征表示，在图像识别领域表现优异。",
        "检索增强生成（RAG）将信息检索与大语言模型结合，提高生成的准确性。",
        "jieba分词支持精确模式、全模式和搜索引擎模式三种分词方式。",
        "Transformer架构通过自注意力机制实现了对序列数据的并行处理。",
        "BM25算法是TF-IDF的改进版本，考虑了词频饱和度和文档长度归一化。",
        "自然语言处理结合深度学习可以更好地理解文本语义和上下文。",
        "RAG系统中的检索器可以选择BM25或Dense等不同策略进行文档匹配。",
    ]
    for i, text in enumerate(texts):
        test_chunks.append(LCDoc(
            page_content=text,
            metadata={"chunk_id": f"chunk_{i}", "doc_id": "test", "title": f"文档{i}"}
        ))

    # 构建两个子检索器
    bm25 = BM25Retriever(test_chunks, top_k=5)

    encoder = EmbeddingEncoder()
    store = FAISSVectorStore(encoder)
    store.build_from_documents(test_chunks)
    dense = DenseRetriever(store, top_k=5)

    # 测试不同 α 值
    for alpha in [0.0, 0.5, 1.0]:
        hybrid = HybridRetriever(bm25, dense, alpha=alpha, top_k=3)
        query = "BM25检索算法"
        print(f"\n{'='*40}")
        print(f"α = {alpha} | 查询: '{query}'")
        print(f"{'='*40}")
        results = hybrid.retrieve(query)
        for r in results:
            print(f"  [hybrid={r.score:.4f}] "
                  f"bm25={r.metadata.get('_bm25_norm', '?'):.3f} "
                  f"dense={r.metadata.get('_dense_norm', '?'):.3f} "
                  f"| {r.content[:60]}...")

    print("\n[PASS] HybridRetriever 自测通过")
