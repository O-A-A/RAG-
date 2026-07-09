"""
===============================================================================
API LLM 生成器 — DeepSeek / OpenAI 兼容 API
===============================================================================
通过云端 API 调用大语言模型，无需本地 GPU。

支持的 API:
    - DeepSeek:  https://api.deepseek.com
    - OpenAI:    https://api.openai.com
    - 其他 OpenAI 兼容 API

用法:
    export DEEPSEEK_API_KEY="sk-your-key"
    python -m rag_project.scripts.run_rag --interactive --api
===============================================================================
"""

import os
import logging

from rag_project.config import get_config

logger = logging.getLogger(__name__)


class APIGenerator:
    """
    API 云端 LLM 生成器。

    通过 LangChain ChatOpenAI 调用 OpenAI 兼容 API。
    支持 DeepSeek、OpenAI 及其他兼容服务。

    Attributes:
        llm:   LangChain ChatOpenAI 实例。
        model: 模型名称。
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        from langchain_openai import ChatOpenAI

        config = get_config()

        # API Key: 参数 > 环境变量 > config
        api_key = (
            api_key
            or os.environ.get("DEEPSEEK_API_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        if not api_key:
            raise ValueError(
                "未找到 API Key。请设置环境变量:\n"
                "  export DEEPSEEK_API_KEY='sk-your-key'\n"
                "或在初始化时传入: APIGenerator(api_key='sk-...')"
            )

        self.model = model
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature or config.llm_temperature,
            max_tokens=max_tokens or config.llm_max_new_tokens,
        )
        logger.info(f"API Generator 就绪: {model} @ {base_url}")

    def generate(self, prompt: str) -> str:
        """调用 API 生成回答。"""
        from langchain_core.messages import HumanMessage, SystemMessage

        # 解析 ChatML 格式，若非 ChatML 则作为单条 user 消息
        messages = self._parse_prompt(prompt)
        response = self.llm.invoke(messages)
        return response.content.strip()

    def _parse_prompt(self, prompt: str):
        """将 ChatML 格式 prompt 解析为 LangChain 消息，解析失败则作为纯 user 消息。"""
        from langchain_core.messages import HumanMessage, SystemMessage

        if "<|im_start|>" not in prompt:
            return [HumanMessage(content=prompt)]

        msgs = []
        parts = prompt.split("<|im_start|>")
        for p in parts:
            p = p.strip()
            if not p:
                continue
            p = p.replace("<|im_end|>", "")
            if p.startswith("system"):
                msgs.append(SystemMessage(content=p[6:].strip()))
            elif p.startswith("user"):
                msgs.append(HumanMessage(content=p[4:].strip()))
        return msgs if msgs else [HumanMessage(content=prompt)]

    def is_ready(self) -> bool:
        return self.llm is not None
