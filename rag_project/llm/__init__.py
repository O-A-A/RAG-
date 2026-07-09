"""
LLM 层模块 (Language Model Layer)

负责提示词管理和文本生成。
- prompt.py:    固定的 Prompt 模板（系统提示 + 上下文 + 问题）
- generator.py: Qwen2.5-7B-Instruct 模型封装与推理
"""

from .prompt import RAGPromptTemplate
from .generator import LLMGenerator

__all__ = ["RAGPromptTemplate", "LLMGenerator"]
