"""
===============================================================================
文档加载器 (Document Loader)
===============================================================================
负责从多种格式（JSONL、TXT、目录批量）加载文档语料，统一转换为
LangChain Document 格式，供后续文本切分和索引使用。

支持的格式:
    - JSONL: 每行一个 JSON 对象，格式为 {"id", "title", "text"}
    - TXT:  纯文本文件
    - 目录: 递归加载目录下所有支持的文件

所有加载方法返回 List[langchain.schema.Document] 统一格式。
===============================================================================
"""

import json
import os
from pathlib import Path
from typing import List, Optional

# LangChain 核心数据结构
from langchain_core.documents import Document

# 导入全局配置
from rag_project.config import get_config

# 日志工具
import logging

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    统一的文档加载器。

    将不同来源的文档语料转换为 LangChain 的标准 Document 格式。
    LangChain Document 包含两个核心字段:
        - page_content (str): 文档正文内容
        - metadata (dict):   文档元数据 (id, title, source 等)

    Usage:
        >>> loader = DocumentLoader()
        >>> docs = loader.load_from_jsonl("data/corpus.jsonl")
        >>> print(len(docs))
        >>> print(docs[0].page_content[:100])
    """

    def __init__(self):
        """初始化文档加载器，从全局配置读取数据目录路径。"""
        self.config = get_config()
        self.data_dir = self.config.data_dir

        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)

    def load_from_jsonl(
        self,
        file_path: Optional[str] = None,
        text_key: str = "text",
        id_key: str = "id",
        title_key: str = "title",
        encoding: str = "utf-8",
    ) -> List[Document]:
        """
        从 JSONL 文件加载文档语料。

        JSONL (JSON Lines) 格式：每行是一个完整的 JSON 对象。
        这是 RAG 语料库最常用的格式，每行代表一个独立文档。

        Args:
            file_path:  JSONL 文件路径。
                       为 None 时使用 config.data_dir / config.corpus_file。
            text_key:   JSON 对象中存放正文的字段名。默认 "text"。
            id_key:     JSON 对象中存放文档 ID 的字段名。默认 "id"。
            title_key:  JSON 对象中存放标题的字段名。默认 "title"。
            encoding:   文件编码。默认 "utf-8"。

        Returns:
            List[Document]: LangChain Document 列表，每个 Document 包含:
                - page_content: 文档正文 (text_key 对应的值)
                - metadata:     {"doc_id", "title", "source"}

        Raises:
            FileNotFoundError: 指定的文件不存在。
            json.JSONDecodeError: JSONL 行解析失败。

        Example:
            输入 JSONL 行:
            {"id": "001", "title": "深度学习简介", "text": "深度学习是..."}

            输出 Document:
            Document(
                page_content="深度学习是...",
                metadata={"doc_id": "001", "title": "深度学习简介", "source": "data/corpus.jsonl"}
            )
        """
        # 解析文件路径：优先使用传入路径，否则使用配置默认路径
        if file_path is None:
            file_path = os.path.join(self.data_dir, self.config.corpus_file)

        file_path = os.path.abspath(file_path)

        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"语料文件不存在: {file_path}\n"
                f"请将 JSONL 格式的语料放入 {self.data_dir} 目录，"
                f"或在 config 中修改 corpus_file 配置。"
            )

        logger.info(f"开始加载语料: {file_path}")

        documents: List[Document] = []
        parse_errors: int = 0

        with open(file_path, "r", encoding=encoding) as f:
            for line_num, line in enumerate(f, start=1):
                # 跳过空行
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"第 {line_num} 行 JSON 解析失败: {e}")
                    parse_errors += 1
                    continue

                # 提取核心字段
                doc_text = record.get(text_key, "")
                doc_id = record.get(id_key, f"doc_{line_num}")
                doc_title = record.get(title_key, "")

                # 跳过空文档
                if not doc_text.strip():
                    logger.debug(f"跳过空文档: {doc_id}")
                    continue

                # 构建 LangChain Document
                doc = Document(
                    page_content=doc_text,
                    metadata={
                        "doc_id": str(doc_id),
                        "title": doc_title,
                        "source": file_path,
                    }
                )
                documents.append(doc)

        logger.info(
            f"语料加载完成: 共 {len(documents)} 篇文档"
            + (f", {parse_errors} 行解析失败" if parse_errors else "")
        )

        return documents

    def load_from_txt(
        self,
        file_path: str,
        doc_title: Optional[str] = None,
        encoding: str = "utf-8",
    ) -> List[Document]:
        """
        从单个纯文本文件加载文档。

        适用于单个 TXT 格式的语料文件。

        Args:
            file_path: TXT 文件路径。
            doc_title: 文档标题（可选）。
                       为 None 时使用文件名作为标题。
            encoding:  文件编码。默认 "utf-8"。

        Returns:
            List[Document]: 包含一个 Document 的列表。
        """
        file_path = os.path.abspath(file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 使用文件名作为默认标题
        if doc_title is None:
            doc_title = os.path.splitext(os.path.basename(file_path))[0]

        logger.info(f"加载纯文本文档: {file_path}")

        with open(file_path, "r", encoding=encoding) as f:
            text = f.read()

        doc = Document(
            page_content=text,
            metadata={
                "doc_id": doc_title,
                "title": doc_title,
                "source": file_path,
            }
        )

        logger.info(f"纯文本文档加载完成: {len(text)} 字符")
        return [doc]

    def load_from_directory(
        self,
        dir_path: Optional[str] = None,
        file_extensions: Optional[List[str]] = None,
        encoding: str = "utf-8",
    ) -> List[Document]:
        """
        批量加载目录下的所有文档文件。

        递归遍历目录，加载所有匹配扩展名的文件。
        每个文件作为一个独立文档。

        Args:
            dir_path:        目标目录路径。为 None 时使用 config.data_dir。
            file_extensions: 要加载的文件扩展名列表。
                             为 None 时默认加载 [".txt", ".jsonl", ".md"]。
            encoding:        文件编码。默认 "utf-8"。

        Returns:
            List[Document]: 所有加载的文档列表。
        """
        if dir_path is None:
            dir_path = self.data_dir

        dir_path = os.path.abspath(dir_path)

        if file_extensions is None:
            file_extensions = [".txt", ".jsonl", ".md"]

        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"目录不存在: {dir_path}")

        logger.info(f"批量加载目录: {dir_path} (扩展名: {file_extensions})")

        documents: List[Document] = []

        # 递归遍历目录
        for root, _, files in os.walk(dir_path):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in file_extensions:
                    continue

                file_path = os.path.join(root, filename)
                doc_title = os.path.splitext(filename)[0]

                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        text = f.read()
                except Exception as e:
                    logger.warning(f"读取文件失败 [{file_path}]: {e}")
                    continue

                if not text.strip():
                    continue

                doc = Document(
                    page_content=text,
                    metadata={
                        "doc_id": doc_title,
                        "title": doc_title,
                        "source": file_path,
                    }
                )
                documents.append(doc)

        logger.info(f"目录加载完成: 共 {len(documents)} 篇文档")
        return documents

    def load_documents(
        self,
        source: Optional[str] = None,
        source_type: str = "jsonl",
    ) -> List[Document]:
        """
        统一的文档加载入口。

        根据 source_type 自动选择对应的加载方法。

        Args:
            source:      数据来源路径。为 None 时使用 config 默认路径。
            source_type: 数据格式类型。
                         可选: "jsonl", "txt", "directory"。

        Returns:
            List[Document]: 加载的文档列表。

        Example:
            >>> loader = DocumentLoader()
            >>> # 加载默认 JSONL 语料
            >>> docs = loader.load_documents()
            >>> # 加载指定 TXT 文件
            >>> docs = loader.load_documents("article.txt", "txt")
            >>> # 批量加载目录
            >>> docs = loader.load_documents("./my_corpus/", "directory")
        """
        if source_type == "jsonl":
            return self.load_from_jsonl(source)
        elif source_type == "txt":
            if source is None:
                raise ValueError("txt 模式需要指定文件路径 (source 参数)")
            return self.load_from_txt(source)
        elif source_type == "directory":
            return self.load_from_directory(source)
        else:
            raise ValueError(
                f"不支持的 source_type: '{source_type}'。"
                f"可选值: 'jsonl', 'txt', 'directory'"
            )


# ============================================================================
# 模块自测 (Module Self-Test)
# ============================================================================
if __name__ == "__main__":
    # 配置日志, 便于调试
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("DocumentLoader 模块自测")
    print("=" * 60)

    loader = DocumentLoader()

    # 创建示例数据用于测试
    sample_path = os.path.join(loader.data_dir, loader.config.corpus_file)
    if not os.path.exists(sample_path):
        # 生成示例 JSONL 文件
        os.makedirs(loader.data_dir, exist_ok=True)
        sample_data = [
            {
                "id": "doc_001",
                "title": "自然语言处理简介",
                "text": (
                    "自然语言处理（Natural Language Processing，NLP）"
                    "是人工智能和语言学领域的分支学科。"
                    "它研究能实现人与计算机之间用自然语言进行有效通信的"
                    "各种理论和方法。"
                ),
            },
            {
                "id": "doc_002",
                "title": "检索增强生成",
                "text": (
                    "检索增强生成（Retrieval-Augmented Generation，RAG）"
                    "是一种结合了信息检索和文本生成的 AI 技术架构。"
                    "它通过从外部知识库检索相关信息来增强大语言模型的"
                    "生成质量，减少幻觉问题。"
                ),
            },
        ]
        with open(sample_path, "w", encoding="utf-8") as f:
            for record in sample_data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[INFO] 已创建示例语料: {sample_path}")

    # 测试加载
    docs = loader.load_from_jsonl()
    print(f"\n加载文档数: {len(docs)}")
    for doc in docs:
        print(f"\n--- {doc.metadata['doc_id']}: {doc.metadata['title']} ---")
        print(f"  内容预览: {doc.page_content[:80]}...")
        print(f"  来源: {doc.metadata['source']}")

    print("\n[PASS] DocumentLoader 自测通过")
