"""
LLM 层模块 (Language Model Layer)

负责提示词管理和文本生成。
- prompt.py:        固定的 Prompt 模板（系统提示 + 上下文 + 问题）
- generator.py:     Qwen2.5-7B-Instruct 本地模型推理
- api_generator.py: DeepSeek / OpenAI 兼容 API 调用
"""

from .prompt import RAGPromptTemplate
from .generator import LLMGenerator
from .api_generator import APIGenerator

__all__ = ["RAGPromptTemplate", "LLMGenerator", "APIGenerator"]
