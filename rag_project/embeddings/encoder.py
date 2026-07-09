"""
===============================================================================
Embedding 编码器 (Embedding Encoder)
===============================================================================
基于 BAAI/bge-small-zh-v1.5 模型的文本向量化编码器。
通过 LangChain 的 HuggingFaceEmbeddings 封装，支持 GPU 推理和批量编码。

模型信息:
    - 模型名称: BAAI/bge-small-zh-v1.5
    - 模型架构: BERT-small (约 102M 参数)
    - 输出维度: 512
    - 最大长度: 512 tokens
    - 语言: 中文（简体）为主，支持多语言
    - 特点: 针对检索任务优化的句子级嵌入

关键设计决策:
    1. 使用 normalize_embeddings=True 对输出向量做 L2 归一化
       → 使得 FAISS 内积 (Inner Product) 等价于余弦相似度 (Cosine Similarity)
       → FAISS IndexFlatIP 搜索速度远快于 IndexFlatL2 (余弦模式)
    2. 使用 LangChain HuggingFaceEmbeddings 封装
       → 兼容 LangChain 生态（可直接传入 FAISS.from_documents）
       → 支持多进程批量编码 (encode_kwargs 配置)
===============================================================================
"""

from typing import List, Optional

import numpy as np

# LangChain HuggingFace Embeddings 封装
from langchain_huggingface import HuggingFaceEmbeddings

# LangChain Document
from langchain_core.documents import Document

# 全局配置
from rag_project.config import get_config

import logging

logger = logging.getLogger(__name__)


class EmbeddingEncoder:
    """
    BGE 中文 Embedding 编码器。

    封装 langchain_huggingface.HuggingFaceEmbeddings，提供:
        - 单文本编码 (embed_query)
        - 批量文本编码 (embed_documents)
        - 直接返回 numpy 数组 (方便与 FAISS 交互)

    Attributes:
        config (Config):              全局配置对象。
        model_name (str):              Embedding 模型名称。
        embedding_model (HuggingFaceEmbeddings): LangChain Embedding 实例。

    Usage:
        >>> encoder = EmbeddingEncoder()
        >>> query_vec = encoder.encode_query("什么是RAG？")
        >>> doc_vecs = encoder.encode_documents(["文档1", "文档2"])
        >>> print(query_vec.shape)    # (512,)
        >>> print(doc_vecs.shape)     # (2, 512)
    """

    def __init__(self):
        """
        初始化 Embedding 编码器。

        从全局配置读取模型名称、设备和批次大小，加载 BGE 模型。
        首次加载时模型权重会自动下载到 HuggingFace 缓存目录。

        HuggingFaceEmbeddings 参数说明:
            - model_name:           HuggingFace 模型 ID 或本地路径
            - model_kwargs:         传递给 SentenceTransformer 的参数
                                     {'device': 'cuda'} 使用 GPU
            - encode_kwargs:        编码时的参数
                                     {'normalize_embeddings': True}  L2 归一化
                                     {'batch_size': 32}             批处理大小
            - multi_process:        是否使用多进程编码（数据量大时可开启）
            - show_progress:        是否显示编码进度条

        Raises:
            ImportError: 缺少 sentence-transformers 依赖时抛出。
            OSError: 模型下载失败或路径不存在时抛出。
        """
        self.config = get_config()
        self.model_name = self.config.embedding_model_name

        logger.info(f"正在加载 Embedding 模型: {self.model_name}")

        # 构建 model_kwargs：传递给底层 SentenceTransformer 的参数
        model_kwargs = {
            "device": self.config.embedding_device,
            # trust_remote_code 对于某些自定义模型可能需要，
            # BGE 是标准模型，不需要
        }

        # 构建 encode_kwargs：编码时的参数
        encode_kwargs = {
            # L2 归一化: 确保嵌入向量模长为 1
            # 归一化后，两个向量的内积 = 余弦相似度
            "normalize_embeddings": self.config.embedding_normalize,
            # 批量大小: 越大编码越快，但显存占用越高
            "batch_size": self.config.embedding_batch_size,
        }

        # 创建 LangChain HuggingFaceEmbeddings 实例
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                # HuggingFaceEmbeddings 内部使用 sentence-transformers
                # multi_process=True 可能导致 Windows 下问题，仅在 Linux 使用
                multi_process=False,
            )
        except ImportError as e:
            raise ImportError(
                f"缺少 sentence-transformers 依赖。\n"
                f"请运行: pip install sentence-transformers\n"
                f"原始错误: {e}"
            )

        logger.info(
            f"Embedding 模型加载完成: {self.model_name}\n"
            f"  设备: {self.config.embedding_device}\n"
            f"  归一化: {self.config.embedding_normalize}\n"
            f"  批次大小: {self.config.embedding_batch_size}"
        )

    def encode_query(self, query: str) -> np.ndarray:
        """
        编码单个查询文本。

        用于检索阶段：将用户问题编码为向量，然后与 FAISS 索引中的
        文档向量做相似度搜索。

        Args:
            query: 用户查询文本（单个字符串）。

        Returns:
            np.ndarray: 形状为 (512,) 的归一化向量。

        Example:
            >>> encoder = EmbeddingEncoder()
            >>> vec = encoder.encode_query("深度学习的应用有哪些？")
            >>> vec.shape
            (512,)
            >>> np.linalg.norm(vec)   # L2 归一化后模长 ≈ 1.0
            1.0000001
        """
        if not query or not query.strip():
            raise ValueError("查询文本为空，无法编码。")

        # embed_query 返回 List[float]，转为 numpy 数组
        embedding = np.array(
            self.embedding_model.embed_query(query),
            dtype=np.float32,
        )

        return embedding

    def encode_documents(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        批量编码文档文本。

        用于索引构建阶段：将所有文档块编码为向量矩阵。

        Args:
            texts:          文档文本列表。
            show_progress:  是否显示编码进度条。

        Returns:
            np.ndarray: 形状为 (N, 512) 的向量矩阵，每行一个文档块。

        Example:
            >>> encoder = EmbeddingEncoder()
            >>> vecs = encoder.encode_documents(["文本1", "文本2", "文本3"])
            >>> vecs.shape
            (3, 512)
        """
        if not texts:
            raise ValueError("texts 列表为空，无法编码。")

        logger.info(f"开始批量编码: {len(texts)} 条文本")

        # embed_documents 返回 List[List[float]]，转为 2D numpy 数组
        embeddings_list = self.embedding_model.embed_documents(texts)

        # 转为连续内存的 float32 数组（FAISS 要求 float32）
        embeddings_matrix = np.array(embeddings_list, dtype=np.float32)

        logger.info(
            f"批量编码完成: 输出形状 {embeddings_matrix.shape}, "
            f"dtype={embeddings_matrix.dtype}"
        )

        return embeddings_matrix

    def encode_documents_from_chunks(
        self,
        chunks: List[Document],
    ) -> np.ndarray:
        """
        从 LangChain Document 列表中提取文本并批量编码。

        这是一个便捷方法：自动提取每个 Document 的 page_content，
        然后调用 encode_documents 进行批量编码。

        Args:
            chunks: 切分后的文本块 (LangChain Document 列表)。

        Returns:
            np.ndarray: 形状为 (N, 512) 的向量矩阵。

        Example:
            >>> encoder = EmbeddingEncoder()
            >>> vecs = encoder.encode_documents_from_chunks(chunks)
            >>> len(chunks) == vecs.shape[0]
            True
        """
        # 提取每个 chunk 的文本内容
        texts = [chunk.page_content for chunk in chunks]

        logger.info(f"从 {len(chunks)} 个 chunk 中提取文本并编码")
        return self.encode_documents(texts)

    @property
    def embedding_dim(self) -> int:
        """
        获取 Embedding 向量的维度。

        BAAI/bge-small-zh-v1.5 固定输出 512 维。

        Returns:
            int: 向量维度 (512)。
        """
        return 512

    def __repr__(self) -> str:
        """返回编码器的字符串表示。"""
        return (
            f"EmbeddingEncoder("
            f"model='{self.model_name}', "
            f"dim={self.embedding_dim}, "
            f"device='{self.config.embedding_device}')"
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
    print("EmbeddingEncoder 模块自测")
    print("=" * 60)

    # 初始化编码器 (首次运行会下载模型)
    encoder = EmbeddingEncoder()

    # 测试查询编码
    query = "什么是检索增强生成？"
    print(f"\n查询: {query}")
    q_vec = encoder.encode_query(query)
    print(f"向量形状: {q_vec.shape}")
    print(f"向量范数: {np.linalg.norm(q_vec):.6f} (归一化后应 ≈ 1.0)")
    print(f"前5个值: {q_vec[:5]}")

    # 测试批量文档编码
    texts = [
        "自然语言处理是人工智能的重要分支。",
        "深度学习极大地推动了NLP的发展。",
        "BERT模型基于Transformer架构。",
    ]
    print(f"\n批量编码 {len(texts)} 条文本...")
    doc_vecs = encoder.encode_documents(texts)
    print(f"输出形状: {doc_vecs.shape}")

    # 测试相似度（简单验证）
    sim = np.dot(q_vec, doc_vecs[0])
    print(f"\n查询与文档0的余弦相似度: {sim:.4f}")

    print(f"\n{encoder}")
    print("\n[PASS] EmbeddingEncoder 自测通过")
