"""
===============================================================================
RAG Prompt 模板 (Prompt Template)
===============================================================================
定义 RAG 系统使用的固定提示词模板，所有 Retriever 使用相同的 Prompt，
确保实验中仅检索策略发生变化。

Prompt 设计原则:
    1. 角色定义:   明确 LLM 作为"基于文档的助手"角色
    2. 行为约束:   要求仅基于给定上下文回答，不得编造
    3. 边界声明:   无法回答时明确告知，防止幻觉
    4. 上下文注入:  将检索到的文档片段作为 {context} 变量填充
    5. 问题注入:    将用户问题作为 {question} 变量填充

Prompt 结构 (Chat Format):
    ┌─────────────────────────────────────────┐
    │ System Message (系统提示)                │
    │  - 角色、约束、行为规范                   │
    ├─────────────────────────────────────────┤
    │ User Message (用户消息)                  │
    │  - 上下文信息（检索结果拼接）              │
    │  - 用户问题                              │
    │  - 回答指令                              │
    └─────────────────────────────────────────┘

为什么使用 Chat Format:
    - Qwen2.5-7B-Instruct 是 Chat 模型，期望 ChatML 格式输入
    - System/User/Assistant 角色分离有助于模型遵循指令
    - 与 Transformers tokenizer.apply_chat_template() 兼容
===============================================================================
"""

from typing import List, Dict, Any, Optional

# LangChain Prompt 模板
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompt_values import PromptValue

# 全局配置
from rag_project.config import get_config

import logging

logger = logging.getLogger(__name__)


class RAGPromptTemplate:
    """
    RAG 系统的固定提示词模板。

    封装 LangChain ChatPromptTemplate，提供统一的 prompt 构建接口。
    所有检索策略使用完全相同的 prompt 模板格式。

    Attributes:
        config (Config):              全局配置对象。
        template (ChatPromptTemplate): LangChain 聊天提示词模板。
        system_prompt (str):            系统提示词文本。
        user_prompt_template (str):     用户提示词模板（含 {context} 和 {question} 占位符）。

    Usage:
        >>> prompt_template = RAGPromptTemplate()
        >>> prompt = prompt_template.build_prompt(
        ...     question="什么是RAG？",
        ...     context="检索增强生成（RAG）是..."
        ... )
        >>> print(prompt)
    """

    def __init__(self):
        """
        初始化 RAG Prompt 模板。

        从全局配置读取 system_prompt 和 user_prompt_template，
        构建 LangChain ChatPromptTemplate。

        模板变量:
            - {context}:  检索到的文档片段内容（自动填充）
            - {question}: 用户原始问题
        """
        self.config = get_config()

        # 从配置读取提示词文本
        self.system_prompt = self.config.system_prompt
        self.user_prompt_template = self.config.user_prompt_template

        # 构建 LangChain ChatPromptTemplate
        # 使用 from_messages 方法构建多角色对话模板
        self.template = ChatPromptTemplate.from_messages([
            # 系统消息: 定义 LLM 的角色、约束和行为规范
            ("system", self.system_prompt),

            # 用户消息: 包含上下文和问题的模板
            ("human", self.user_prompt_template),
        ])

        logger.info(
            f"RAGPromptTemplate 初始化完成:\n"
            f"  系统提示词长度: {len(self.system_prompt)} 字符\n"
            f"  用户模板长度: {len(self.user_prompt_template)} 字符"
        )

    # ========================================================================
    # Prompt 构建
    # ========================================================================

    def build_prompt(
        self,
        question: str,
        context: str,
    ) -> str:
        """
        构建完整的 RAG Prompt 字符串。

        将检索到的文档上下文和用户问题填充到模板中，
        生成可直接送入 LLM 的完整提示词。

        Args:
            question: 用户原始问题。
            context:  检索到的文档片段拼接而成的上下文字符串。

        Returns:
            str: 完整的 Prompt 字符串，可直接作为 LLM 输入。

        Example:
            >>> prompt_template = RAGPromptTemplate()
            >>> prompt = prompt_template.build_prompt(
            ...     question="什么是深度学习？",
            ...     context="[文档1] 深度学习是...\\n[文档2] 深度神经网络..."
            ... )
            >>> # prompt 可以直接传入 generator.generate(prompt)
        """
        if not question:
            raise ValueError("question 不能为空")
        if not context:
            logger.warning("context 为空，LLM 将无法基于文档回答")

        # 使用 LangChain 模板格式化
        formatted_messages = self.template.format_messages(
            context=context,
            question=question,
        )

        # 合并所有消息为单个字符串
        # ChatPromptTemplate.format_messages() 返回 List[BaseMessage]
        # 需要转为字符串供 transformers pipeline 使用
        prompt_str = self._messages_to_string(formatted_messages)

        return prompt_str

    def build_prompt_from_chunks(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        chunk_separator: str = "\n\n---\n\n",
    ) -> str:
        """
        从检索结果列表构建 Prompt。

        自动将检索到的文档块拼接为上下文字符串，然后构建完整 Prompt。
        每个文档块标注来源信息，便于 LLM 引用和用户追溯。

        Args:
            question:          用户问题。
            retrieved_chunks:  检索结果列表（来自 Retriever.retrieve()）。
                               每个元素包含 "content", "chunk_id", "metadata" 等字段。
            chunk_separator:   文档块之间的分隔符。
                               默认使用换行 + 分隔线，视觉上清晰分块。

        Returns:
            str: 完整的 Prompt 字符串。

        Example:
            >>> results = retriever.retrieve("什么是RAG？", top_k=5)
            >>> prompt = prompt_template.build_prompt_from_chunks(
            ...     question="什么是RAG？",
            ...     retrieved_chunks=results,
            ... )
        """
        if not retrieved_chunks:
            logger.warning("retrieved_chunks 为空，context 将为空")
            return self.build_prompt(question=question, context="（未检索到相关文档）")

        # 拼接所有文档块为上下文字符串
        context_parts: List[str] = []

        for i, chunk in enumerate(retrieved_chunks, start=1):
            # 兼容 RetrievalResult dataclass 和 dict 两种格式
            if hasattr(chunk, 'doc_id'):
                chunk_id = chunk.doc_id
                content = chunk.content
                title = chunk.metadata.get("title", "") if chunk.metadata else ""
            else:
                chunk_id = chunk.get("chunk_id", f"chunk_{i}")
                content = chunk.get("content", "")
                title = chunk.get("metadata", {}).get("title", "")

            # 格式化每个文档块
            source_info = f"[来源: {title}]" if title else f"[片段 {chunk_id}]"
            chunk_text = f"{source_info}\n{content}"

            context_parts.append(chunk_text)

        # 拼接所有块
        context = chunk_separator.join(context_parts)

        return self.build_prompt(question=question, context=context)

    # ========================================================================
    # 内部工具方法
    # ========================================================================

    def _messages_to_string(self, messages: List) -> str:
        """
        将 LangChain 消息列表转换为单个字符串。

        使用 Qwen2.5 的 ChatML 格式（或兼容格式）拼接消息。
        如果当前 tokenizer 可用，优先使用其 apply_chat_template() 方法。

        ChatML 格式示例:
            <|im_start|>system
            你是...<|im_end|>
            <|im_start|>user
            上下文:...\n问题:...<|im_end|>
            <|im_start|>assistant
            （待生成）

        Args:
            messages: LangChain BaseMessage 列表。

        Returns:
            str: 拼接后的完整提示词字符串。
        """
        parts: List[str] = []

        for msg in messages:
            role = msg.type  # "system", "human", "ai"
            content = msg.content

            if role == "system":
                # 系统消息
                parts.append(
                    f"<|im_start|>system\n{content}<|im_end|>"
                )
            elif role == "human":
                # 用户消息
                parts.append(
                    f"<|im_start|>user\n{content}<|im_end|>"
                )
            elif role == "ai":
                # 助手消息（用于 few-shot 示例）
                parts.append(
                    f"<|im_start|>assistant\n{content}<|im_end|>"
                )

        # 添加 assistant 标记，指示模型开始生成
        parts.append("<|im_start|>assistant\n")

        return "\n".join(parts)

    # ========================================================================
    # 工具方法
    # ========================================================================

    def get_system_prompt(self) -> str:
        """
        获取系统提示词文本。

        Returns:
            str: 系统提示词。
        """
        return self.system_prompt

    def get_template_variables(self) -> List[str]:
        """
        获取模板中使用的变量名。

        Returns:
            List[str]: 变量名列表，例如 ["context", "question"]。
        """
        return self.template.input_variables

    def __repr__(self) -> str:
        """返回 Prompt 模板的字符串表示。"""
        return (
            f"RAGPromptTemplate("
            f"variables={self.template.input_variables}, "
            f"system_len={len(self.system_prompt)})"
        )
