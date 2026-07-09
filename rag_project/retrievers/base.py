"""
===============================================================================
检索器抽象基类 (Base Retriever + RetrievalResult)
===============================================================================
定义所有检索器的统一接口和共享逻辑。

设计原则:
    1. 统一返回类型:  所有 retriever.retrieve(query) 返回 List[RetrievalResult]
    2. 代码高度复用:   格式化、归一化、top-k 截断等通用逻辑集中在基类
    3. 里氏替换原则:   任意 Retriever 子类可在 Pipeline 中无缝互换

统一接口:
    retrieve(query: str) -> List[RetrievalResult]

RetrievalResult 字段:
    - doc_id:   文档块唯一标识
    - content:  文档块文本内容
    - score:    相关性得分（越高越相关，各 Retriever 内部自行保证单调性）
    - metadata: 原始元数据 (source, title, chunk_idx 等)
===============================================================================
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from langchain_core.documents import Document


# ============================================================================
# 统一检索结果数据结构
# ============================================================================

@dataclass
class RetrievalResult:
    """
    统一检索结果数据结构。

    所有 Retriever 子类必须返回此类型的列表，确保 Pipeline 和
    评估模块无需感知具体是哪种 Retriever。

    Attributes:
        doc_id:   文档块唯一标识（来自 metadata["chunk_id"]）。
        content:  文档块正文文本。
        score:    检索相关性得分（越高越相关）。
                  注意: 不同 Retriever 的得分量纲可能不同，
                  仅在 Hybrid 融合时需要归一化处理。
        metadata: 文档元数据字典，至少包含:
                      - chunk_id: 块编号
                      - doc_id:   原始文档 ID
                      - title:    文档标题

    Example:
        >>> r = RetrievalResult(
        ...     doc_id="doc_001_chunk_3",
        ...     content="深度学习是人工智能的重要分支...",
        ...     score=0.852,
        ...     metadata={"doc_id": "doc_001", "title": "深度学习简介"}
        ... )
    """
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 抽象基类
# ============================================================================

class BaseRetriever(ABC):
    """
    检索器抽象基类。

    所有具体检索器（BM25 / Dense / Hybrid）必须实现:
        - name 属性: 返回检索器唯一标识名
        - _retrieve_raw(): 返回原始检索结果 List[Tuple[Document, float]]

    retrieve() 是唯一对外的公开接口，它内部调用 _retrieve_raw() 然后
    统一格式化。子类只需关心"如何检索"，不需要关心"如何格式化"。

    Attributes:
        top_k (int): 默认检索返回的文档数量。

    Usage:
        >>> class MyRetriever(BaseRetriever):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_retriever"
        ...     def _retrieve_raw(self, query: str) -> List[Tuple[Document, float]]:
        ...         ...  # 实现具体检索逻辑
        ...         return [(doc, score), ...]
    """

    def __init__(self, top_k: int = 5):
        """
        初始化检索器基类。

        Args:
            top_k: 默认返回的文档数量。
        """
        self.top_k = top_k

    # ========================================================================
    # 子类必须实现的抽象成员
    # ========================================================================

    @property
    @abstractmethod
    def name(self) -> str:
        """
        检索器唯一标识名。

        用于日志记录、结果保存和实验对比。
        命名规范: 小写英文 + 下划线，如 "bm25"、"dense_bge"、"hybrid_alpha05"。

        Returns:
            str: 检索器名称。
        """
        raise NotImplementedError

    @abstractmethod
    def _retrieve_raw(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """
        执行原始检索，返回 (Document, score) 元组列表。

        子类必须实现此方法。不需要关心 top-k 截断和格式化，
        这些由基类的 retrieve() 统一处理。

        Args:
            query: 用户查询文本。
            k:     请求的候选数量（子类应返回至少 k 个结果）。

        Returns:
            List[Tuple[Document, float]]: 按得分降序排列的 (文档, 分数) 列表。
        """
        raise NotImplementedError

    # ========================================================================
    # 对外公开的统一接口
    # ========================================================================

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        统一检索接口。

        流程:
            1. 调用子类的 _retrieve_raw(query, k) 获取原始结果
            2. 截断到 top_k
            3. 统一格式化为 List[RetrievalResult]

        Args:
            query: 用户查询文本。
            top_k: 返回数量，为 None 时使用 self.top_k。

        Returns:
            List[RetrievalResult]: 格式化的检索结果列表。
        """
        k = top_k if top_k is not None else self.top_k

        # 调用子类实现获取原始结果
        raw = self._retrieve_raw(query, k=k)

        # 截断 + 格式化
        return self._format_results(raw[:k])

    # ========================================================================
    # 共享工具方法（所有子类复用，消除重复代码）
    # ========================================================================

    @staticmethod
    def _format_results(raw_results: List[Tuple[Document, float]]) -> List[RetrievalResult]:
        """
        将 (Document, score) 列表统一格式化为 List[RetrievalResult]。

        提取 Document 中的 page_content 和 metadata，
        封装为标准的 RetrievalResult 对象。

        Args:
            raw_results: (Document, score) 元组列表，已按得分降序排列。

        Returns:
            List[RetrievalResult]: 格式化后的结果列表。
        """
        formatted: List[RetrievalResult] = []

        for rank, (doc, score) in enumerate(raw_results, start=1):
            metadata = dict(doc.metadata)
            metadata["rank"] = rank

            formatted.append(RetrievalResult(
                doc_id=metadata.get("chunk_id", f"chunk_{rank}"),
                content=doc.page_content,
                score=float(score),
                metadata=metadata,
            ))

        return formatted

    @staticmethod
    def normalize_scores(
        results: List[RetrievalResult],
    ) -> Dict[str, float]:
        """
        对结果列表的得分做 min-max 归一化到 [0, 1]。

        用于 Hybrid 融合前的得分标准化：
        BM25 得分量纲（无界正数）和 Dense 得分（[-1, 1] 余弦相似度）
        无法直接加权求和，必须先各自归一化。

        归一化公式:
            norm(x) = (x - min) / (max - min)    if max > min
            norm(x) = 1.0                         if max == min

        Args:
            results: 检索结果列表。

        Returns:
            Dict[str, float]: doc_id → 归一化得分 的映射。
        """
        if not results:
            return {}

        scores = np.array([r.score for r in results])
        s_min, s_max = scores.min(), scores.max()

        if s_max == s_min:
            # 所有得分相同 → 全部归一化为 1.0
            return {r.doc_id: 1.0 for r in results}

        normalized: Dict[str, float] = {}
        for r in results:
            normalized[r.doc_id] = float((r.score - s_min) / (s_max - s_min))

        return normalized

    @staticmethod
    def _build_result_map(
        results: List[RetrievalResult],
    ) -> Dict[str, RetrievalResult]:
        """
        将结果列表转为 doc_id → RetrievalResult 的索引映射。

        Hybrid 融合时用于快速查找文档内容。

        Args:
            results: 检索结果列表。

        Returns:
            Dict[str, RetrievalResult]: doc_id 到结果对象的映射。
        """
        return {r.doc_id: r for r in results}

    # ========================================================================
    # 魔术方法
    # ========================================================================

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', top_k={self.top_k})"

    def __str__(self) -> str:
        return f"Retriever[{self.name}]"
