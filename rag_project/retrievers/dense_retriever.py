"""
===============================================================================
Dense 稠密检索器 (Dense Retriever)
===============================================================================
基于 BGE Embedding + FAISS 向量相似度搜索的语义检索器。

核心思想:
    将查询和文档映射到同一 512 维语义向量空间，
    通过向量内积（≈ 余弦相似度）衡量查询与文档的语义相关性。

技术栈:
    - Embedding:  BAAI/bge-small-zh-v1.5 → 512 维归一化向量
    - 索引:       FAISS IndexFlatIP → 内积搜索（等价余弦相似度）
    - 得分范围:   [-1, 1]（归一化后的内积值）

与 BM25 的对比:
    ┌──────────┬─────────────────────┬──────────────────────┐
    │          │   BM25              │   Dense (BGE+FAISS)  │
    ├──────────┼─────────────────────┼──────────────────────┤
    │ 匹配方式 │   精确词匹配         │   语义向量相似度      │
    │ 同义词   │   ✗ 无法匹配        │   ✓ 自动匹配          │
    │ 计算资源 │   CPU               │   GPU (推荐)          │
    │ 速度     │   极快 (< 1ms)      │   快 (~10ms)          │
    │ 适用场景 │   关键词/实体检索    │   语义/自然语言查询    │
    └──────────┴─────────────────────┴──────────────────────┘

工作流程:
    1. 构建阶段 (FAISSVectorStore):
       所有文档块 → BGE 编码 → FAISS IndexFlatIP
    2. 检索阶段 (DenseRetriever):
       查询文本 → BGE 编码 → FAISS 内积搜索 → top-k 文档
===============================================================================
"""

from typing import List, Optional, Tuple

from langchain_core.documents import Document

from rag_project.retrievers.base import BaseRetriever, RetrievalResult
from rag_project.embeddings.vector_store import FAISSVectorStore

import logging

logger = logging.getLogger(__name__)


class DenseRetriever(BaseRetriever):
    """
    基于 BGE Embedding + FAISS 的稠密语义检索器。

    将查询编码为语义向量，在 FAISS 索引中搜索最相似的文档块。
    得分 = 查询向量 ⋅ 文档向量（已 L2 归一化，等价余弦相似度）。

    Attributes:
        vector_store (FAISSVectorStore): 已构建/加载的 FAISS 向量存储。

    Usage:
        >>> encoder = EmbeddingEncoder()
        >>> store = FAISSVectorStore(encoder)
        >>> store.build_from_documents(chunks)   # 或 store.load(path)
        >>> retriever = DenseRetriever(store, top_k=5)
        >>> results = retriever.retrieve("什么是深度学习？")
        >>> for r in results:
        ...     print(f"[cos={r.score:.4f}] {r.content[:50]}...")
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        top_k: int = 5,
    ):
        """
        初始化 Dense 检索器。

        Args:
            vector_store: 已构建索引的 FAISSVectorStore 实例。
                          必须先调用 build_from_documents() 或 load()。
            top_k:        默认检索返回数量。

        Raises:
            RuntimeError: vector_store 索引尚未构建。
        """
        super().__init__(top_k=top_k)

        if not vector_store.is_built:
            raise RuntimeError(
                "FAISSVectorStore 索引尚未构建。"
                "请先调用 vector_store.build_from_documents(chunks) "
                "或 vector_store.load(path)。"
            )

        self.vector_store = vector_store

        logger.info(
            f"DenseRetriever 初始化完成: "
            f"n_vectors={self.vector_store.num_vectors}, "
            f"top_k={self.top_k}"
        )

    @property
    def name(self) -> str:
        """检索器唯一标识名。"""
        return "dense_bge"

    # ========================================================================
    # 核心检索逻辑
    # ========================================================================

    def _retrieve_raw(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """
        执行 Dense 语义检索。

        委托 FAISSVectorStore.similarity_search_with_score() 完成:
            1. EmbeddingEncoder.encode_query(query) → 512 维向量
            2. FAISS IndexFlatIP.search() → top-k 最近邻
            3. 返回 (Document, inner_product_score) 列表

        Args:
            query: 用户查询文本（自然语言句子）。
            k:     请求的候选数量（传给 FAISS 的 k 参数）。

        Returns:
            List[Tuple[Document, float]]: 按内积得分降序排列的结果。
            得分范围 [-1, 1]，越大表示语义越相关。
        """
        return self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
        )

    # ========================================================================
    # 工具方法
    # ========================================================================

    def get_vector_store(self) -> FAISSVectorStore:
        """返回关联的 FAISS 向量存储（供 Hybrid 或外部使用）。"""
        return self.vector_store


# ============================================================================
# 模块自测
# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    print("=" * 60)
    print("DenseRetriever 模块自测 (BGE + FAISS)")
    print("=" * 60)

    from langchain_core.documents import Document as LCDoc
    from rag_project.embeddings.encoder import EmbeddingEncoder
    from rag_project.embeddings.vector_store import FAISSVectorStore

    # 构建测试文档
    test_chunks = [
        LCDoc(page_content=f"这是第 {i} 个测试文档，包含中文语义内容。" * 5,
              metadata={"chunk_id": f"chunk_{i}", "doc_id": "test"})
        for i in range(30)
    ]

    print(f"构建 {len(test_chunks)} 个测试文档...")
    encoder = EmbeddingEncoder()
    store = FAISSVectorStore(encoder)
    store.build_from_documents(test_chunks)

    retriever = DenseRetriever(store, top_k=3)

    query = "中文语义测试"
    print(f"\n查询: '{query}'")
    results = retriever.retrieve(query)
    for r in results:
        print(f"  [cos={r.score:.4f}] {r.content[:60]}...")

    print(f"\n{retriever}")
    print("[PASS] DenseRetriever 自测通过")
