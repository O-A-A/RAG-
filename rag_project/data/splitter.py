"""
===============================================================================
文本切分器 (Text Splitter)
===============================================================================
将长文档切分为固定大小的文本块 (Chunks)，是 RAG 系统的关键预处理步骤。

核心参数 (来自全局配置, 单一变量原则):
    - chunk_size:    512 tokens  — 每个文本块的目标大小
    - chunk_overlap: 50 tokens   — 相邻块之间的重叠区域
    - separators:    中文友好的分隔符优先级列表

切分策略:
    使用 LangChain 的 RecursiveCharacterTextSplitter，按分隔符优先级
    递归切分。优先在自然边界（段落、句子）处断句，保持语义完整性。

为什么需要重叠 (Overlap):
    1. 防止重要信息被切分边界截断
    2. 增加相邻 chunk 的上下文连续性
    3. 提高检索召回率（信息可能跨越 chunk 边界）
===============================================================================
"""

from typing import List, Optional

# LangChain 文本切分器
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain 文档数据结构
from langchain_core.documents import Document

# 全局配置
from rag_project.config import get_config

import logging

logger = logging.getLogger(__name__)


class TextSplitter:
    """
    固定窗口大小的文本切分器。

    封装 LangChain 的 RecursiveCharacterTextSplitter，使用 config 中
    定义的 chunk_size (512) 和 chunk_overlap (50)，确保实验中所有
    Retriever 使用完全相同的切分结果。

    切分流程:
        1. 按 separators 中第一个分隔符尝试切分
        2. 若切分后的片段仍大于 chunk_size，则用下一个分隔符
        3. 递归执行，直到所有片段 ≤ chunk_size 或分隔符耗尽
        4. 相邻片段保留 chunk_overlap 的重叠

    Attributes:
        splitter (RecursiveCharacterTextSplitter): LangChain 切分器实例。
        config (Config): 全局配置对象。

    Usage:
        >>> splitter = TextSplitter()
        >>> chunks = splitter.split_documents(documents)
        >>> print(f"切分后共 {len(chunks)} 个文本块")
        >>> print(f"第一个块: {chunks[0].page_content[:100]}")
    """

    def __init__(self):
        """
        初始化文本切分器。

        从全局配置读取 chunk_size、chunk_overlap 和 separators，
        创建 RecursiveCharacterTextSplitter 实例。

        RecursiveCharacterTextSplitter 参数说明:
            - chunk_size:          每个块的目标最大字符数
            - chunk_overlap:       相邻块之间的重叠字符数
            - separators:          按优先级排列的分隔符列表
            - length_function:     计算文本长度的函数 (默认 len)
            - is_separator_regex:  是否将分隔符视为正则表达式
            - keep_separator:      是否在切分结果中保留分隔符
        """
        self.config = get_config()

        # 创建 LangChain 的递归字符切分器
        self.splitter = RecursiveCharacterTextSplitter(
            # 核心切割参数 — 来自全局配置
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,

            # 中文友好的分隔符优先队列
            # 从大到小尝试: 先按段落切，再按句子，最后按字符
            separators=self.config.separators,

            # 使用 Python 内置 len() 计算字符串长度
            # 注意: 这是字符级长度，实际 token 数可能不同
            # 对于精确的 token 级切分，可替换为 tokenizer-based 方法
            length_function=len,

            # 分隔符被视为普通字符串，非正则表达式
            is_separator_regex=False,

            # 在切分结果开头保留分隔符（确保上下文完整）
            keep_separator=True,

            # 若某片段无法继续切分（无分隔符可用），
            # 则保留原始字符串（即使超出 chunk_size）
            # 避免破坏未分隔的长词或代码片段
            add_start_index=False,
        )

        logger.info(
            f"TextSplitter 初始化完成: "
            f"chunk_size={self.config.chunk_size}, "
            f"chunk_overlap={self.config.chunk_overlap}, "
            f"separators={self.config.separators[:4]}..."
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        对文档列表执行文本切分。

        每个输入文档可能被切分为多个文本块。切分后每个块继承原文档的
        metadata，并额外添加 chunk 相关的元信息。

        Args:
            documents: 待切分的 LangChain Document 列表。

        Returns:
            List[Document]: 切分后的文本块列表，每个块包含:
                - page_content: 块文本内容
                - metadata:
                    - doc_id:    原始文档 ID
                    - title:     原始文档标题
                    - source:    原始文档来源
                    - chunk_id:  块编号 (格式: "{doc_id}_chunk_{idx}")
                    - chunk_idx: 块在文档中的索引 (从 0 开始)

        Example:
            输入: 1 篇 2000 字符的文档
            输出: ~4 个 512 字符的文本块 (含 50 重叠)

        Raises:
            ValueError: 输入的 documents 列表为空。
        """
        if not documents:
            raise ValueError("documents 列表为空，无法执行切分。请先加载文档。")

        total_chars = sum(len(doc.page_content) for doc in documents)
        logger.info(
            f"开始文本切分: {len(documents)} 篇文档, "
            f"共 {total_chars} 字符"
        )

        # 执行 LangChain RecursiveCharacterTextSplitter 切分
        all_chunks: List[Document] = self.splitter.split_documents(documents)

        # 为每个块添加唯一标识和索引
        for i, chunk in enumerate(all_chunks):
            doc_id = chunk.metadata.get("doc_id", "unknown")

            # 统计同一文档下的块序号
            # 通过遍历找到该块在所属文档中的索引
            chunk_idx = sum(
                1 for c in all_chunks[:i]
                if c.metadata.get("doc_id") == doc_id
            )

            # 更新 metadata: 添加 chunk 级别的标识信息
            chunk.metadata.update({
                "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
                "chunk_idx": chunk_idx,
            })

        # 统计切分结果
        unique_docs = set(c.metadata.get("doc_id") for c in all_chunks)
        avg_chunk_len = (
            sum(len(c.page_content) for c in all_chunks) / len(all_chunks)
            if all_chunks else 0
        )

        logger.info(
            f"文本切分完成: "
            f"{len(documents)} 篇文档 → {len(all_chunks)} 个文本块, "
            f"平均块长度: {avg_chunk_len:.0f} 字符, "
            f"涉及 {len(unique_docs)} 个原始文档"
        )

        return all_chunks

    def split_text(self, text: str, metadata: Optional[dict] = None) -> List[Document]:
        """
        对单段文本执行切分（不依赖 Document 对象）。

        适用于临时文本或流式输入的切分场景。

        Args:
            text:     待切分的原始文本。
            metadata: 附加元数据字典（可选）。

        Returns:
            List[Document]: 切分后的文本块列表。

        Example:
            >>> splitter = TextSplitter()
            >>> chunks = splitter.split_text("这是一段很长的文本...")
            >>> len(chunks)
        """
        if metadata is None:
            metadata = {}

        doc = Document(page_content=text, metadata=metadata)
        return self.split_documents([doc])

    def get_chunk_statistics(self, chunks: List[Document]) -> dict:
        """
        计算文本块集合的统计信息。

        用于数据分析和实验报告（如统计块的长度分布、方差等）。

        Args:
            chunks: 切分后的文本块列表。

        Returns:
            dict: 包含以下统计量的字典:
                - total_chunks:   总块数
                - total_chars:    总字符数
                - mean_length:    平均块长度
                - min_length:     最短块长度
                - max_length:     最长块长度
                - std_length:     块长度标准差
        """
        import statistics

        if not chunks:
            return {
                "total_chunks": 0,
                "total_chars": 0,
                "mean_length": 0,
                "min_length": 0,
                "max_length": 0,
                "std_length": 0,
            }

        lengths = [len(chunk.page_content) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "total_chars": sum(lengths),
            "mean_length": statistics.mean(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "std_length": statistics.stdev(lengths) if len(lengths) > 1 else 0,
        }


# ============================================================================
# 模块自测 (Module Self-Test)
# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("TextSplitter 模块自测")
    print("=" * 60)

    # 创建测试文档
    test_text = (
        "自然语言处理（Natural Language Processing，NLP）"
        "是人工智能和语言学领域的分支学科。"
        "它研究能实现人与计算机之间用自然语言进行有效通信的"
        "各种理论和方法。"
        "自然语言处理包括自然语言理解（NLU）和自然语言生成（NLG）两大方向。\n\n"
        "检索增强生成（Retrieval-Augmented Generation，RAG）"
        "是一种结合了信息检索和文本生成的 AI 技术架构。"
        "RAG 的核心思想是：在生成回答之前，先从外部知识库中检索相关信息，"
        "然后将检索结果作为上下文提供给大语言模型，从而提高生成质量。\n\n"
        "BM25（Best Matching 25）是一种基于概率检索模型的排序函数，"
        "被广泛应用于信息检索领域。它考虑了词频饱和度和文档长度归一化，"
        "是 TF-IDF 的改进版本。"
    )

    test_doc = Document(
        page_content=test_text,
        metadata={"doc_id": "test_001", "title": "测试文档"}
    )

    # 测试切分
    splitter = TextSplitter()
    # 临时调小 chunk_size 用于测试
    splitter.splitter._chunk_size = 150
    chunks = splitter.split_documents([test_doc])

    print(f"\n切分结果: {len(chunks)} 个文本块\n")
    for chunk in chunks:
        cid = chunk.metadata.get("chunk_id", "?")
        print(f"[{cid}] ({len(chunk.page_content)} 字符)")
        print(f"  {chunk.page_content[:100]}...")
        print()

    # 统计信息
    stats = splitter.get_chunk_statistics(chunks)
    print("统计信息:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n[PASS] TextSplitter 自测通过")
