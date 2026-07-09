"""
===============================================================================
BM25 稀疏检索器 (BM25 Sparse Retriever)
===============================================================================
基于 rank_bm25 库 + jieba 中文分词实现的 BM25 检索器。

BM25 算法原理:
    Score(D, Q) = Σ IDF(q_i) × [f(q_i, D) × (k1 + 1)]
                  / [f(q_i, D) + k1 × (1 - b + b × |D| / avgdl)]

    其中:
        - IDF(q_i):   逆文档频率
        - f(q_i, D):  词频
        - |D|:        文档长度
        - avgdl:      平均文档长度
        - k1:         词频饱和度参数
        - b:          文档长度归一化参数

为什么用 rank_bm25 而非 LangChain BM25Retriever:
    1. rank_bm25 是 PyPI 上最流行的 BM25 Python 实现，学术引用广泛
    2. 直接暴露 BM25Okapi.get_scores() 获得原始分数，便于 Hybrid 融合
    3. 更轻量，无需 LangChain 依赖即可使用
    4. 支持自定义 tokenizer（通过传入已分词的 token 列表）

特点:
    - 基于精确词匹配，擅长关键词/专有名词检索
    - 纯 CPU 计算，速度极快 (< 1ms per query on 10K docs)
    - 无需 GPU，内存占用小
    - jieba 分词确保中文文本被正确切分为有意义的词语
===============================================================================
"""

from typing import List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import jieba

from rag_project.retrievers.base import BaseRetriever, RetrievalResult

import logging

logger = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):
    """
    BM25 稀疏检索器（rank_bm25 + jieba 中文分词）。

    工作流程:
        1. 初始化时对所有文档做 jieba 分词 → 构建 BM25Okapi 索引
        2. 检索时对 query 做 jieba 分词 → get_scores() → 排序取 top-k

    Attributes:
        documents (List[Document]): 所有文档块（含 page_content 和 metadata）。
        corpus (List[str]):         文档正文列表（仅文本）。
        tokenized_corpus (List[List[str]]): 分词后的语料库。
        bm25 (BM25Okapi):           rank_bm25 的 BM25Okapi 实例。

    Usage:
        >>> retriever = BM25Retriever(chunks, top_k=5)
        >>> results = retriever.retrieve("什么是深度学习？")
        >>> for r in results:
        ...     print(f"[{r.score:.4f}] {r.content[:50]}...")
    """

    def __init__(
        self,
        documents: List[Document],
        top_k: int = 5,
    ):
        """
        初始化 BM25 检索器并构建索引。

        Args:
            documents: LangChain Document 列表（经过切分后的文档块）。
            top_k:     默认检索返回数量。

        Raises:
            ValueError: documents 为空时抛出。

        索引构建过程:
            1. 提取所有文档的 page_content
            2. 用 jieba.lcut() 对每篇文档分词
            3. 将分词后的语料传入 BM25Okapi 构建倒排索引
        """
        super().__init__(top_k=top_k)

        if not documents:
            raise ValueError("documents 不能为空")

        self.documents: List[Document] = documents

        # 提取纯文本
        self.corpus: List[str] = [doc.page_content for doc in documents]

        # jieba 分词构建语料
        logger.info(f"BM25 分词中: {len(self.corpus)} 篇文档...")
        self.tokenized_corpus: List[List[str]] = [
            jieba.lcut(text) for text in self.corpus
        ]

        # 构建 BM25Okapi 索引
        # BM25Okapi 使用默认的 k1=1.5, b=0.75（学术界标准参数）
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # 计算平均文档长度（用于诊断）
        avg_len = np.mean([len(tokens) for tokens in self.tokenized_corpus])
        logger.info(
            f"BM25Retriever 初始化完成: "
            f"n_docs={len(self.documents)}, "
            f"avg_doc_len={avg_len:.1f} tokens, "
            f"top_k={self.top_k}"
        )

    @property
    def name(self) -> str:
        """检索器唯一标识名。"""
        return "bm25"

    # ========================================================================
    # 核心检索逻辑（子类只需实现此方法）
    # ========================================================================

    def _retrieve_raw(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """
        执行 BM25 检索（返回全部文档的排序结果，k 参数在此忽略
        因为 BM25 需要全量排序才能正确获得 top-k）。

        Args:
            query: 用户查询文本。
            k:     候选数量（BM25 忽略此参数，始终返回全量排序）。

        Returns:
            List[Tuple[Document, float]]: 按 BM25 得分降序排列的全部文档。
        """
        # 分词查询
        tokenized_query = jieba.lcut(query)

        # 计算所有文档的 BM25 得分（O(N) 复杂度）
        scores = self.bm25.get_scores(tokenized_query)

        # argsort 降序 → 获取所有文档索引按得分从高到低排列
        ranked_indices = np.argsort(scores)[::-1]

        # 构建 (Document, score) 列表
        raw_results: List[Tuple[Document, float]] = []
        for idx in ranked_indices:
            raw_results.append((self.documents[int(idx)], float(scores[int(idx)])))

        return raw_results

    # ========================================================================
    # 工具方法
    # ========================================================================

    def get_documents(self) -> List[Document]:
        """返回索引中的所有文档块。"""
        return self.documents


# ============================================================================
# 模块自测
# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    print("=" * 60)
    print("BM25Retriever 模块自测 (rank_bm25 + jieba)")
    print("=" * 60)

    # 构建测试文档
    test_docs = []
    for i, text in enumerate([
        "自然语言处理是人工智能的重要分支，研究人与计算机之间的自然语言通信。",
        "深度学习技术极大地推动了自然语言处理的发展，特别是Transformer架构。",
        "BM25是一种基于概率检索模型的排序函数，用于估计文档与查询的相关性。",
        "检索增强生成（RAG）结合了信息检索和文本生成技术，减少幻觉问题。",
        "BERT模型通过预训练和微调范式在多个NLP任务上取得了突破性成果。",
        "机器学习是人工智能的核心，包括监督学习、无监督学习和强化学习。",
        "jieba分词是Python中最为流行的中文分词工具，支持多种分词模式。",
        "FAISS是Facebook开发的高效向量相似度搜索库，支持多种索引类型。",
        "大语言模型如GPT和Qwen在自然语言理解和生成方面表现出色。",
        "词嵌入技术如Word2Vec和BGE将词语映射到高维语义向量空间。",
    ]):
        test_docs.append(Document(
            page_content=text,
            metadata={"chunk_id": f"chunk_{i}", "doc_id": "test", "title": f"文档{i}"}
        ))

    retriever = BM25Retriever(test_docs, top_k=3)

    # 测试查询
    for query in ["什么是BM25算法", "深度学习和自然语言处理的关系"]:
        print(f"\n查询: {query}")
        results = retriever.retrieve(query)
        for r in results:
            print(f"  [{r.score:.4f}] {r.content[:60]}...")

    print(f"\n{retriever}")
    print("[PASS] BM25Retriever 自测通过")
