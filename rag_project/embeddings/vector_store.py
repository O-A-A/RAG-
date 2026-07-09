"""
===============================================================================
FAISS 向量存储 (FAISS Vector Store)
===============================================================================
基于 Facebook AI Similarity Search (FAISS) 的高效向量相似度搜索引擎。

核心功能:
    1. 构建索引:    将文档块的 Embedding 向量存入 FAISS 索引
    2. 相似度搜索:   查询向量与索引中的文档向量计算内积（等价余弦相似度）
    3. 持久化:       将索引保存到磁盘，下次运行直接加载，避免重复编码

索引类型选择:
    - IndexFlatIP (内积索引): 暴力搜索，精确但速度随数据量线性增长
    - 适用数据规模: < 100K 文档（暴力搜索在 100K 以内完全可接受）
    - 为什么用 IP 而不是 L2:
      因为 EmbeddingEncoder 对向量做了 L2 归一化，
      归一化后有: 内积 = 余弦相似度，且内积搜索比余弦距离快

工作流程:
    1. EmbeddingEncoder.encode_documents() → 编码文本为向量
    2. FAISS.from_documents()             → 构建索引
    3. save_local()                       → 持久化保存
    4. load_local()                       → 后续直接加载
    5. similarity_search_with_score()      → 查询时检索

数据流:
    文本块 (List[Document])
        ↓ EmbeddingEncoder.encode_documents()
    向量矩阵 (np.ndarray [N, 512])
        ↓ FAISS.from_documents()
    FAISS 索引 (内存)
        ↓ save_local()
    磁盘文件 (index.faiss + index.pkl)
===============================================================================
"""

import os
from typing import List, Optional, Tuple

import numpy as np

# LangChain FAISS 封装
from langchain_community.vectorstores import FAISS

# LangChain Document
from langchain_core.documents import Document

# 项目内模块
from rag_project.config import get_config
from rag_project.embeddings.encoder import EmbeddingEncoder

import logging

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    FAISS 向量数据库封装。

    封装 LangChain 的 FAISS 向量存储，提供统一的构建、保存、加载和
    检索接口。与 EmbeddingEncoder 紧密协作。

    Attributes:
        config (Config):                    全局配置对象。
        encoder (EmbeddingEncoder):         Embedding 编码器实例。
        vector_store (Optional[FAISS]):     LangChain FAISS 实例（未构建时为 None）。
        _is_built (bool):                  标记索引是否已构建/加载。

    Usage:
        >>> encoder = EmbeddingEncoder()
        >>> store = FAISSVectorStore(encoder)
        >>> store.build_from_documents(chunks)     # 构建索引
        >>> store.save("faiss_index/")             # 保存
        >>> results = store.similarity_search("查询", k=5)  # 检索
    """

    def __init__(self, encoder: EmbeddingEncoder):
        """
        初始化 FAISS 向量存储。

        Args:
            encoder: EmbeddingEncoder 实例（用于编码查询文本）。

        注意: 初始化后 vector_store 为 None，需要调用 build_from_documents()
              或 load() 来填充实际的 FAISS 索引。
        """
        self.config = get_config()
        self.encoder = encoder

        # FAISS 索引实例（延迟构建）
        self.vector_store: Optional[FAISS] = None

        # 索引状态标记
        self._is_built: bool = False

        logger.info("FAISSVectorStore 初始化完成（索引尚未构建）")

    # ========================================================================
    # 索引构建
    # ========================================================================

    def build_from_documents(
        self,
        documents: List[Document],
    ) -> None:
        """
        从文档块列表构建 FAISS 索引。

        这是整个 RAG 系统中计算量最大的步骤，需要:
        1. 对每个文档块调用 EmbeddingEncoder 编码为向量
        2. 将所有向量存入 FAISS 内积索引

        构建完成后会记录文档数量、索引维度等元信息。

        Args:
            documents: 切分后的文本块列表 (LangChain Document)。

        Raises:
            ValueError: documents 列表为空时抛出。

        Note:
            - 构建时间 ≈ O(N * encode_time)，N 为文档数
            - 内存占用 ≈ N * 512 * 4 bytes (float32)
            - 10万文档约需 200MB 显存/内存

        Example:
            >>> store = FAISSVectorStore(encoder)
            >>> store.build_from_documents(chunks)
            >>> print(store.vector_store.index.ntotal)  # 索引中的向量数
        """
        if not documents:
            raise ValueError("documents 列表为空，无法构建 FAISS 索引。")

        n_docs = len(documents)
        logger.info(f"开始构建 FAISS 索引: {n_docs} 个文档块")

        # FAISS.from_documents() 内部流程:
        # 1. 调用 encoder.embedding_model.embed_documents() 编码所有文档
        # 2. 构建 faiss.IndexFlatIP (内积索引，维度=embedding_dim)
        # 3. 将所有向量通过 IndexFlatIP.add() 添加到索引
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.encoder.embedding_model,

            # 距离策略: 使用内积 (Inner Product)
            # 配合 L2 归一化后的向量，内积 = 余弦相似度
            distance_strategy="INNER_PRODUCT",

            # 相关参数:
            # - normalize_L2: 如需额外确保归一化可设为 True
            #   但我们的 encoder 已经做了归一化，无需重复
        )

        self._is_built = True

        # 获取索引统计信息
        n_total = self.vector_store.index.ntotal
        dim = self.vector_store.index.d

        logger.info(
            f"FAISS 索引构建完成:\n"
            f"  向量数量: {n_total}\n"
            f"  向量维度: {dim}\n"
            f"  索引类型: IndexFlatIP (内积搜索)\n"
            f"  搜索方式: 暴力搜索 (精确, O(N) 时间复杂度)"
        )

    # ========================================================================
    # 索引持久化
    # ========================================================================

    def save(self, path: Optional[str] = None) -> str:
        """
        将 FAISS 索引保存到磁盘。

        保存后生成两个文件:
            - index.faiss: FAISS 索引二进制文件
            - index.pkl:   文档元数据 (page_content + metadata) pickle 文件

        Args:
            path: 保存目录路径。
                  为 None 时使用 config.faiss_index_path。

        Returns:
            str: 索引保存的目录绝对路径。

        Raises:
            RuntimeError: 索引尚未构建，无法保存。
        """
        if not self._is_built or self.vector_store is None:
            raise RuntimeError(
                "索引尚未构建，请先调用 build_from_documents()。"
            )

        if path is None:
            path = self.config.faiss_index_path

        # 创建目录
        os.makedirs(path, exist_ok=True)
        abs_path = os.path.abspath(path)

        logger.info(f"保存 FAISS 索引到: {abs_path}")

        # FAISS.save_local() 会生成两个文件:
        #   1. index.faiss — 向量索引 (二进制)
        #   2. index.pkl   — 文档内容 (pickle)
        self.vector_store.save_local(abs_path)

        # 验证文件是否成功生成
        faiss_file = os.path.join(abs_path, "index.faiss")
        pkl_file = os.path.join(abs_path, "index.pkl")

        if os.path.exists(faiss_file) and os.path.exists(pkl_file):
            faiss_size = os.path.getsize(faiss_file) / (1024 * 1024)
            pkl_size = os.path.getsize(pkl_file) / (1024 * 1024)
            logger.info(
                f"索引保存成功:\n"
                f"  index.faiss: {faiss_size:.1f} MB\n"
                f"  index.pkl:   {pkl_size:.1f} MB"
            )
        else:
            logger.warning("索引保存可能不完整: 缺少部分文件")

        return abs_path

    def load(self, path: Optional[str] = None) -> None:
        """
        从磁盘加载已有的 FAISS 索引。

        加载后可以跳过 Embedding 编码和索引构建步骤，大幅节省启动时间。
        但需要确保加载的索引与当前 Embedding 模型兼容（同一模型生成的向量）。

        Args:
            path: 索引文件目录路径。
                  为 None 时使用 config.faiss_index_path。

        Raises:
            FileNotFoundError: 索引文件不存在。
            RuntimeError: FAISS 索引文件损坏或不兼容。

        Example:
            >>> store = FAISSVectorStore(encoder)
            >>> store.load("faiss_index/")  # 加载已有索引，跳过早先步骤
        """
        if path is None:
            path = self.config.faiss_index_path

        abs_path = os.path.abspath(path)

        if not os.path.exists(abs_path):
            raise FileNotFoundError(
                f"FAISS 索引目录不存在: {abs_path}\n"
                f"请先调用 build_from_documents() 并 save() 创建索引。"
            )

        logger.info(f"加载 FAISS 索引: {abs_path}")

        try:
            # FAISS.load_local() 内部:
            # 1. 读取 index.faiss 反序列化 FAISS 索引
            # 2. 读取 index.pkl 恢复文档内容
            # 3. 允许传入与构建时不同但兼容的 embedding 模型
            self.vector_store = FAISS.load_local(
                folder_path=abs_path,
                embeddings=self.encoder.embedding_model,
                # allow_dangerous_deserialization 需要显式授权
                # 因为我们信任自己保存的索引文件
                allow_dangerous_deserialization=True,
                # 距离策略需与构建时一致
                distance_strategy="INNER_PRODUCT",
            )

            self._is_built = True

        except Exception as e:
            raise RuntimeError(
                f"FAISS 索引加载失败: {e}\n"
                f"可能原因: 文件损坏、版本不兼容、"
                f"或 Embedding 模型不匹配。"
            )

        n_total = self.vector_store.index.ntotal
        logger.info(f"FAISS 索引加载成功: {n_total} 条向量")

    # ========================================================================
    # 相似度检索
    # ========================================================================

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Document]:
        """
        基于文本查询的相似度搜索（不返回分数）。

        对 query 编码 → 在 FAISS 中搜索 → 返回 top-k 文档块。

        Args:
            query: 查询文本。
            k:     返回的文档块数量。为 None 时使用 config.top_k。

        Returns:
            List[Document]: 按相似度降序排列的文档块列表。

        Raises:
            RuntimeError: 索引尚未构建或加载。
        """
        if not self._is_built or self.vector_store is None:
            raise RuntimeError("索引尚未构建，请先调用 build_from_documents() 或 load()。")

        if k is None:
            k = self.config.top_k

        results = self.vector_store.similarity_search(
            query=query,
            k=k,
        )

        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        基于文本查询的相似度搜索（带分数）。

        这是检索器的核心方法，返回每个文档块及其与查询的相关性得分。
        得分含义: 查询向量与文档向量的内积（≈ 余弦相似度），
                  取值范围 [-1, 1]，越大越相似。

        Args:
            query: 查询文本。
            k:     返回的文档块数量。为 None 时使用 config.top_k。

        Returns:
            List[Tuple[Document, float]]:
                按相似度降序排列的 (文档块, 得分) 元组列表。
                得分范围: [-1, 1]（归一化后的内积）

        Raises:
            RuntimeError: 索引尚未构建或加载。

        Example:
            >>> store = FAISSVectorStore(encoder)
            >>> store.load()
            >>> results = store.similarity_search_with_score("什么是RAG？", k=5)
            >>> for doc, score in results:
            ...     print(f"[{score:.4f}] {doc.page_content[:50]}...")
        """
        if not self._is_built or self.vector_store is None:
            raise RuntimeError("索引尚未构建，请先调用 build_from_documents() 或 load()。")

        if k is None:
            k = self.config.top_k

        # FAISS.similarity_search_with_score() 返回:
        # List[Tuple[Document, float]]
        # 得分 = 查询向量与文档向量的内积 (dot product)
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
        )

        return results

    def similarity_search_by_vector(
        self,
        query_embedding: np.ndarray,
        k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        基于向量的相似度搜索（跳过 query 编码）。

        当查询向量已经预计算好时（如批量评估场景），直接传入向量搜索。
        避免每次重复编码 query，节省计算资源。

        Args:
            query_embedding: 查询向量，形状为 (512,) 的 numpy 数组。
            k:               返回的文档块数量。为 None 时使用 config.top_k。

        Returns:
            List[Tuple[Document, float]]: (文档块, 得分) 列表。

        Raises:
            RuntimeError: 索引尚未构建或加载。
            ValueError:  查询向量维度不匹配。
        """
        if not self._is_built or self.vector_store is None:
            raise RuntimeError("索引尚未构建，请先调用 build_from_documents() 或 load()。")

        if k is None:
            k = self.config.top_k

        # 验证向量维度
        expected_dim = self.vector_store.index.d
        if query_embedding.shape[-1] != expected_dim:
            raise ValueError(
                f"查询向量维度不匹配: 期望 {expected_dim}, "
                f"实际 {query_embedding.shape[-1]}"
            )

        # 确保是 float32 类型
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        # 确保是 1D 向量
        if query_embedding.ndim == 2:
            query_embedding = query_embedding.squeeze(0)

        results = self.vector_store.similarity_search_with_score_by_vector(
            embedding=query_embedding.tolist(),
            k=k,
        )

        return results

    # ========================================================================
    # 属性 & 工具方法
    # ========================================================================

    @property
    def is_built(self) -> bool:
        """返回索引是否已构建或加载。"""
        return self._is_built

    @property
    def num_vectors(self) -> int:
        """
        返回索引中存储的向量数量。

        Returns:
            int: 向量总数，若索引未构建则返回 0。
        """
        if not self._is_built or self.vector_store is None:
            return 0
        return self.vector_store.index.ntotal

    def __len__(self) -> int:
        """返回索引中的向量数量（与 num_vectors 相同）。"""
        return self.num_vectors

    def __repr__(self) -> str:
        """返回向量存储的字符串表示。"""
        status = "built" if self._is_built else "not built"
        return (
            f"FAISSVectorStore("
            f"status='{status}', "
            f"n_vectors={self.num_vectors}, "
            f"index_path='{self.config.faiss_index_path}')"
        )


# ============================================================================
# 模块自测 (Module Self-Test)
# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("FAISSVectorStore 模块自测")
    print("=" * 60)

    # 需要先创建 Encoder 和模拟文档
    from langchain_core.documents import Document as LCDoc

    # 创建测试文档块
    test_chunks = [
        LCDoc(
            page_content=f"这是第 {i} 个测试文档块，包含一些中文文本内容。",
            metadata={"chunk_id": f"chunk_{i}", "doc_id": "test"},
        )
        for i in range(20)
    ]

    print(f"\n创建了 {len(test_chunks)} 个测试文档块")

    # 初始化编码器和向量存储
    encoder = EmbeddingEncoder()
    store = FAISSVectorStore(encoder)

    # 构建索引
    print("\n构建 FAISS 索引...")
    store.build_from_documents(test_chunks)
    print(f"索引状态: {store}")

    # 搜索测试
    print("\n相似度搜索测试:")
    query = "测试文档"
    print(f"  查询: '{query}'")
    results = store.similarity_search_with_score(query, k=3)
    for i, (doc, score) in enumerate(results):
        print(f"  #{i+1} [得分: {score:.4f}] {doc.page_content[:50]}...")

    # 保存和加载测试
    print("\n保存索引...")
    saved_path = store.save("./test_faiss_index")
    print(f"  已保存到: {saved_path}")

    # 创建新的 store 并从磁盘加载
    print("\n从磁盘加载索引...")
    store2 = FAISSVectorStore(encoder)
    store2.load("./test_faiss_index")
    print(f"  加载后: {store2}")

    # 验证搜索一致性
    results2 = store2.similarity_search_with_score(query, k=3)
    assert results[0][1] == results2[0][1], "保存/加载后得分不一致！"
    print("  验证通过: 保存/加载后搜索得分一致")

    # 清理测试文件
    import shutil
    if os.path.exists("./test_faiss_index"):
        shutil.rmtree("./test_faiss_index")

    print("\n[PASS] FAISSVectorStore 自测通过")
