"""
嵌入层模块 (Embedding Layer)

负责文本向量化和向量索引管理。
- encoder.py:      BGE Embedding 模型封装（BAAI/bge-small-zh-v1.5）
- vector_store.py: FAISS 向量数据库的构建、保存与加载
"""

from .encoder import EmbeddingEncoder
from .vector_store import FAISSVectorStore

__all__ = ["EmbeddingEncoder", "FAISSVectorStore"]
