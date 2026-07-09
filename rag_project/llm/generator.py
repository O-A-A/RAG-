"""
===============================================================================
LLM 生成器 (LLM Generator)
===============================================================================
封装 Qwen2.5-7B-Instruct 模型的加载和推理，提供统一的文本生成接口。

模型信息:
    - 模型名称: Qwen/Qwen2.5-7B-Instruct
    - 模型架构: Transformer Decoder-only (类似 LLaMA)
    - 参数量:   ~7B (70亿)
    - 上下文窗口: 32K tokens
    - 语言:    中文/英文/多语言
    - 特点:    指令微调 (Instruction-tuned)，支持 ChatML 格式

加载策略:
    1. 4-bit 量化 (默认): 通过 bitsandbytes 加载 NF4 量化权重
       → 显存占用约 4-5 GB，适合消费级 GPU (RTX 3060/4060 等)
    2. 全精度: 设置 load_in_4bit=False
       → 显存占用约 14-16 GB，需要 RTX 3090/4090 或 A100
    3. 8-bit 量化: 可通过修改配置实现
       → 显存占用约 7-8 GB

关键设计决策:
    1. temperature=0 + do_sample=False → 贪婪解码，确保每次输出完全相同
       （这是论文复现的关键：消除 LLM 端的随机性）
    2. max_new_tokens=512 → 限制生成长度，避免超出上下文窗口
    3. 使用 transformers.pipeline → 简化推理代码，自动处理 tokenization

使用方法:
    方式 1 — LangChain 集成:
        >>> generator = LLMGenerator()
        >>> response = generator.generate(prompt)

    方式 2 — 原始 transformers pipeline:
        >>> generator = LLMGenerator()
        >>> response = generator.generate_raw(messages)
===============================================================================
"""

import logging
from typing import List, Optional, Dict, Any

# 全局配置
from rag_project.config import get_config

logger = logging.getLogger(__name__)

# 延迟导入（避免在不使用 LLM 时加载 PyTorch）
_HAS_TORCH = False
_HAS_TRANSFORMERS = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    pass

try:
    import transformers
    _HAS_TRANSFORMERS = True
except ImportError:
    pass


class LLMGenerator:
    """
    Qwen2.5-7B-Instruct 大语言模型生成器。

    封装模型加载和推理逻辑，提供简洁的 generate() 接口。
    支持 4-bit 量化加载以降低显存需求。

    Attributes:
        config (Config):               全局配置对象。
        model:                         HuggingFace 模型实例。
        tokenizer:                     HuggingFace Tokenizer 实例。
        pipeline:                      Transformers text-generation pipeline。
        langchain_llm:                 LangChain HuggingFacePipeline 封装（可选）。

    Usage:
        >>> generator = LLMGenerator()
        >>> prompt = "上下文：...\\n问题：什么是RAG？\\n请基于以上上下文回答："
        >>> answer = generator.generate(prompt)
        >>> print(answer)
    """

    def __init__(self):
        """
        初始化 LLM 生成器，加载 Qwen2.5-7B-Instruct 模型。

        加载流程:
            1. 检查依赖 (torch, transformers)
            2. 加载 Tokenizer（配置 ChatML 格式和 padding）
            3. 加载模型（使用 4-bit 量化，根据配置决定设备）
            4. 创建 text-generation pipeline
            5. 创建 LangChain 封装 (可选)

        Raises:
            ImportError: 缺少 torch 或 transformers 依赖。
            OSError: 模型下载失败。
            RuntimeError: GPU 不可用而配置要求使用 CUDA。
        """
        if not _HAS_TORCH or not _HAS_TRANSFORMERS:
            raise ImportError(
                "LLMGenerator 需要 PyTorch 和 Transformers 库。\n"
                "请运行: pip install torch transformers accelerate bitsandbytes"
            )

        self.config = get_config()
        self.model_name = self.config.llm_model_name

        # ---- 设备检查 ----
        if self.config.llm_device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA 不可用，回退到 CPU。"
                "注意: 在 CPU 上运行 7B 模型速度极慢，建议使用 GPU。"
            )
            self.device = "cpu"
        else:
            self.device = self.config.llm_device

        logger.info(f"开始加载 LLM: {self.model_name} (device={self.device})")

        # ---- 加载 Tokenizer ----
        self._load_tokenizer()

        # ---- 加载模型 ----
        self._load_model()

        # ---- 创建 Pipeline ----
        self._create_pipeline()

        # ---- 创建 LangChain 封装 ----
        self._create_langchain_llm()

        logger.info(
            f"LLM 加载完成: {self.model_name}\n"
            f"  设备: {self.device}\n"
            f"  温度: {self.config.llm_temperature}\n"
            f"  最大输出 tokens: {self.config.llm_max_new_tokens}"
        )

    # ========================================================================
    # 内部加载方法
    # ========================================================================

    def _load_tokenizer(self) -> None:
        """
        加载 Qwen2.5 Tokenizer。

        配置:
            - padding_side="left": 左侧 padding（生成任务标准做法）
            - trust_remote_code=True: Qwen 系列需要此参数
            - use_fast=False: 使用慢速 tokenizer（更稳定）
        """
        logger.info("加载 Tokenizer...")

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",        # 生成任务：左侧 padding
            trust_remote_code=True,     # Qwen 模型需要
            use_fast=False,             # 慢速 tokenizer 更稳定
        )

        # 设置 pad_token（Qwen tokenizer 可能没有默认 pad_token）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 设置 padding token ID
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        logger.info(
            f"Tokenizer 加载完成: "
            f"vocab_size={self.tokenizer.vocab_size}, "
            f"pad_token={self.tokenizer.pad_token}"
        )

    def _load_model(self) -> None:
        """
        加载 Qwen2.5-7B-Instruct 模型。

        支持三种加载模式:
            1. 4-bit 量化 (load_in_4bit=True): 使用 bitsandbytes NF4
            2. 正常加载 (load_in_4bit=False, device="cuda"): FP16
            3. CPU 加载 (device="cpu"): FP32

        4-bit 量化参数说明:
            - bnb_4bit_compute_dtype=torch.float16:
              计算时使用 FP16（速度快，精度略降）
            - bnb_4bit_quant_type="nf4":
              NF4 (NormalFloat4) 量化类型，信息保留优于 FP4
            - bnb_4bit_use_double_quant=True:
              嵌套量化（double quantization），进一步节省 ~0.4 bit/参数
        """
        logger.info(
            f"加载模型权重..."
            + (" (4-bit 量化)" if self.config.llm_load_in_4bit else "")
        )

        # 构建加载参数
        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,       # Qwen 系列需要
            "torch_dtype": torch.float16,    # 默认使用 FP16
            "device_map": "auto" if self.device == "cuda" else None,
        }

        # 4-bit 量化配置
        if self.config.llm_load_in_4bit and self.device == "cuda":
            try:
                import bitsandbytes as bnb  # noqa: F401

                load_kwargs.update({
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch.float16,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True,
                })
                logger.info("  使用 bitsandbytes NF4 4-bit 量化")
            except ImportError:
                logger.warning(
                    "bitsandbytes 未安装，回退到正常加载。\n"
                    "如需 4-bit 量化，请运行: pip install bitsandbytes"
                )
        elif self.device == "cpu":
            # CPU 模式: 使用 FP32
            load_kwargs["torch_dtype"] = torch.float32
            # 使用低内存模式加载
            load_kwargs["low_cpu_mem_usage"] = True
            logger.info("  CPU 模式: 使用 FP32 + 低内存加载")

        # 加载模型
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs,
        )

        # 模型统计信息
        if hasattr(self.model, "parameters"):
            try:
                # 获取参数量
                total_params = sum(p.numel() for p in self.model.parameters())
                logger.info(
                    f"  模型参数量: {total_params / 1e9:.2f}B"
                )

                # 显示显存占用（仅 CUDA）
                if self.device == "cuda":
                    mem_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                    logger.info(f"  GPU 显存占用: {mem_gb:.2f} GB")
            except Exception:
                pass

    def _create_pipeline(self) -> None:
        """
        创建 HuggingFace text-generation pipeline。

        Pipeline 自动处理:
            - Text → Token IDs → Model Forward → Token IDs → Text
            - 使用 model.generate() 进行自回归生成
            - 支持批处理（batch_size）
        """
        logger.info("创建 text-generation pipeline...")

        # 构建生成参数
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.config.llm_max_new_tokens,
            "temperature": self.config.llm_temperature,
            "do_sample": self.config.llm_do_sample,
            # 当 temperature=0 且 do_sample=False 时使用贪婪解码
            # 确保每次对同一输入生成完全相同的输出
        }

        # 仅在采样模式下添加额外参数
        if self.config.llm_do_sample:
            generation_kwargs.update({
                "top_p": 0.9,           # nucleus sampling
                "top_k": 50,            # top-k sampling
                "repetition_penalty": 1.1,  # 惩罚重复
            })

        self.pipeline = transformers.pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            # 使用模型默认的设备映射（已在加载时设置 device_map="auto"）
            device_map="auto" if self.device == "cuda" else None,
            # 生成参数
            **generation_kwargs,
        )

        logger.info("Pipeline 创建完成")

    def _create_langchain_llm(self) -> None:
        """
        创建 LangChain HuggingFacePipeline 封装。

        使得 Generator 可以同时作为独立模块和 LangChain Chain 的一部分使用。

        注意: 此方法可能在某些 LangChain 版本中签名不同。
        失败时将 langchain_llm 设为 None，不影响核心功能。
        """
        try:
            from langchain_community.llms import HuggingFacePipeline

            self.langchain_llm = HuggingFacePipeline(
                pipeline=self.pipeline,
                model_kwargs={
                    "temperature": self.config.llm_temperature,
                    "max_new_tokens": self.config.llm_max_new_tokens,
                }
            )
            logger.info("LangChain HuggingFacePipeline 封装创建成功")
        except Exception as e:
            logger.warning(
                f"创建 LangChain 封装失败: {e}\n"
                f"  这通常不影响核心功能，可直接使用 generate() 方法。"
            )
            self.langchain_llm = None

    # ========================================================================
    # 生成方法
    # ========================================================================

    def generate(self, prompt: str) -> str:
        """
        基于提示词生成回答。

        这是主要的生成接口。接收构建好的 Prompt 字符串，
        返回 LLM 生成的回答文本。

        生成流程:
            1. 将 prompt 传入 pipeline
            2. Pipeline 自动 tokenize → generate → decode
            3. 从生成结果中提取纯回答文本（移除输入 prompt 部分）

        Args:
            prompt: 完整的 Prompt 字符串（由 RAGPromptTemplate 构建）。

        Returns:
            str: LLM 生成的回答文本（纯回答，不包含输入 prompt）。

        Example:
            >>> generator = LLMGenerator()
            >>> prompt = "<|im_start|>system\\n你是...<|im_end|>\\n..."
            >>> answer = generator.generate(prompt)
            >>> print(answer)
            "根据上下文，检索增强生成（RAG）是..."
        """
        if not prompt:
            raise ValueError("prompt 不能为空")

        # 使用 text-generation pipeline 生成
        # pipeline 返回 List[Dict[str, str]]: [{"generated_text": "..."}]
        outputs = self.pipeline(
            prompt,
            # 以下参数可通过 pipeline 的 call_kwargs 传入
            # 这些在 _create_pipeline 中已经设为默认值
            return_full_text=False,  # 只返回新生成的部分（不包含输入）
            # pad_token_id 确保 batch 处理正确
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # 提取生成的文本
        # pipeline 返回格式: [{"generated_text": "回答内容"}]
        if isinstance(outputs, list) and len(outputs) > 0:
            generated_text = outputs[0].get("generated_text", "")
        else:
            generated_text = str(outputs)

        # 清理生成结果
        generated_text = self._clean_output(generated_text)

        return generated_text

    def generate_with_chat_template(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """
        使用 ChatML 格式的消息列表生成回答。

        适用于直接使用 Chat 格式（而非拼接好的 prompt 字符串）的场景。
        使用 tokenizer.apply_chat_template() 自动构建 ChatML 格式。

        Args:
            messages: 消息列表，格式为:
                [
                    {"role": "system", "content": "系统提示词"},
                    {"role": "user", "content": "用户问题"},
                ]

        Returns:
            str: LLM 生成的回答文本。

        Example:
            >>> messages = [
            ...     {"role": "system", "content": "你是一个助手"},
            ...     {"role": "user", "content": "解释什么是RAG"},
            ... ]
            >>> answer = generator.generate_with_chat_template(messages)
        """
        if not messages:
            raise ValueError("messages 不能为空")

        # 使用 tokenizer 的 chat_template 构建 ChatML 格式的 prompt
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,          # 返回字符串而非 token IDs
            add_generation_prompt=True,  # 添加 "<|im_start|>assistant\n"
        )

        return self.generate(prompt)

    def generate_raw(self, prompt: str) -> Dict[str, Any]:
        """
        生成回答并返回完整的生成结果（包含元数据）。

        与 generate() 的区别:
            - generate(): 只返回纯文本回答
            - generate_raw(): 返回完整字典（含 token 数、生成耗时等）

        Args:
            prompt: 完整的 Prompt 字符串。

        Returns:
            Dict: 包含以下字段:
                - "answer":       生成的回答文本
                - "full_output":  Pipeline 原始输出
                - "usage":        可选的使用统计（若可用）
        """
        outputs = self.pipeline(
            prompt,
            return_full_text=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        generated_text = outputs[0].get("generated_text", "") \
            if isinstance(outputs, list) else str(outputs)

        return {
            "answer": self._clean_output(generated_text),
            "full_output": outputs,
        }

    # ========================================================================
    # 输出处理
    # ========================================================================

    def _clean_output(self, text: str) -> str:
        """
        清理生成文本中的格式标记。

        移除 ChatML 格式标记（<|im_start|>, <|im_end|>等），
        以及多余的空行和白空格。

        Args:
            text: 原始生成文本。

        Returns:
            str: 清理后的纯回答文本。
        """
        if not text:
            return text

        # 移除 ChatML 标记
        # <|im_start|>role\n...<|im_end|>
        markers_to_remove = [
            "<|im_start|>", "<|im_end|>",
            "system", "user", "assistant",
            "<|endoftext|>",
        ]
        cleaned = text
        for marker in markers_to_remove:
            cleaned = cleaned.replace(marker, "")

        # 移除多余空行（连续 3 个以上换行合并为 2 个）
        while "\n\n\n" in cleaned:
            cleaned = cleaned.replace("\n\n\n", "\n\n")

        # 去除首尾空白
        cleaned = cleaned.strip()

        return cleaned

    # ========================================================================
    # 工具方法
    # ========================================================================

    def get_langchain_llm(self):
        """
        获取 LangChain 兼容的 LLM 实例。

        可用于构建 LangChain Chain（如 RetrievalQA、ConversationalRetrievalChain）。

        Returns:
            HuggingFacePipeline: LangChain LLM 封装。

        Raises:
            RuntimeError: LangChain 封装创建失败时抛出。
        """
        if self.langchain_llm is None:
            raise RuntimeError(
                "LangChain 封装不可用。请检查 langchain-community 是否正确安装。"
            )
        return self.langchain_llm

    def is_ready(self) -> bool:
        """
        检查模型是否已加载完成。

        Returns:
            bool: True 表示模型已加载并可进行推理。
        """
        return (
            self.model is not None
            and self.tokenizer is not None
            and self.pipeline is not None
        )

    def __repr__(self) -> str:
        """返回生成器的字符串表示。"""
        return (
            f"LLMGenerator("
            f"model='{self.model_name}', "
            f"device='{self.device}', "
            f"temperature={self.config.llm_temperature})"
        )
