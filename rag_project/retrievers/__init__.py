"""
检索器模块 (Retriever Layer)

实现三种检索策略（本实验的核心对比维度）：
    - BaseRetriever:   检索器抽象基类 + RetrievalResult 统一返回类型
    - BM25Retriever:   BM25 稀疏检索（rank_bm25 + jieba 分词）
    - DenseRetriever:  Dense 稠密检索（BGE Embedding + FAISS）
    - HybridRetriever: 混合检索（分数级加权融合: α×Dense + (1-α)×BM25）

统一接口:
    retrieve(query: str) -> List[RetrievalResult]
"""

from .base import BaseRetriever, RetrievalResult
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever
from .hybrid_retriever import HybridRetriever

__all__ = [
    # 基类 + 数据类型
    "BaseRetriever",
    "RetrievalResult",
    # 三种检索器
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
]
