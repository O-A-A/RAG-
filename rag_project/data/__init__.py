"""
数据层模块 (Data Layer)

负责文档语料的加载和预处理。
- loader.py:   从多种格式（JSONL、TXT）加载文档
- splitter.py: 固定窗口大小的文本切分器
"""

from .loader import DocumentLoader
from .splitter import TextSplitter

__all__ = ["DocumentLoader", "TextSplitter"]
