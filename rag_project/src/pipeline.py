"""
===============================================================================
RAG Pipeline 编排模块 (RAG Pipeline Orchestrator)
===============================================================================
串联 Retriever、Prompt Template 和 LLM Generator，构成完整的
RAG (Retrieval-Augmented Generation) 问答流水线。

Pipeline 流程:
    ┌──────────┐    ┌──────────────┐    ┌───────────┐    ┌──────────┐
    │ Question │ →  │   Retriever  │ →  │  Prompt   │ →  │   LLM    │ → Answer
    │  (输入)   │    │ (retrieve()) │    │ (build()) │    │(generate)│   (输出)
    └──────────┘    └──────────────┘    └───────────┘    └──────────┘
                           │                                    │
                    检索结果列表                          生成回答文本
                    (top-k chunks)                     + 检索来源追溯

设计原则:
    1. 模块解耦: Pipeline 只依赖抽象接口 (BaseRetriever)，不关心具体实现
    2. 配置驱动: 所有参数通过 Config 控制，Pipeline 内部不硬编码
    3. 可追溯:   每次 run() 返回检索来源，便于验证回答的事实性
===============================================================================
"""

import time
from typing import Any, Dict, List, Optional

# 项目内模块
from rag_project.config import get_config
from rag_project.retrievers.base import BaseRetriever
from rag_project.llm.prompt import RAGPromptTemplate
from rag_project.llm.generator import LLMGenerator

import logging

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG 问答流水线编排器。

    将检索、Prompt 构建、生成三个步骤串联为统一的问答接口。
    Pipeline 本身不感知 Retriever 的具体实现（BM25/Dense/Hybrid），
    只需要满足 BaseRetriever 接口的任意检索器即可。

    Attributes:
        config (Config):                 全局配置对象。
        retriever (BaseRetriever):       检索器实例（BM25 / Dense / Hybrid）。
        prompt_template (RAGPromptTemplate): Prompt 模板。
        generator (LLMGenerator):        LLM 生成器。

    Usage:
        >>> # 1. 构建组件
        >>> retriever = BM25Retriever(chunks)
        >>> prompt_template = RAGPromptTemplate()
        >>> generator = LLMGenerator()
        >>>
        >>> # 2. 创建 Pipeline
        >>> pipeline = RAGPipeline(retriever, prompt_template, generator)
        >>>
        >>> # 3. 运行问答
        >>> result = pipeline.run("什么是检索增强生成？")
        >>> print(result["answer"])
        >>> print(result["retrieved_chunks"][0]["content"])
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        prompt_template: RAGPromptTemplate,
        generator: LLMGenerator,
    ):
        """
        初始化 RAG Pipeline。

        Args:
            retriever:       检索器实例（已初始化）。
            prompt_template: Prompt 模板实例。
            generator:       LLM 生成器实例（可为 None，仅做检索）。

        Raises:
            ValueError: generator 非 None 但模型未就绪时抛出。
        """
        self.config = get_config()
        self.retriever = retriever
        self.prompt_template = prompt_template
        self.generator = generator

        # 验证 Generator（如果提供）
        if generator is not None and not generator.is_ready():
            raise ValueError(
                "LLM 模型未加载完成。请检查模型路径和 GPU 可用性。"
            )

        logger.info(
            f"RAGPipeline 初始化完成:\n"
            f"  Retriever: {self.retriever}\n"
            f"  Generator: {self.generator or '(none, retrieval-only)'}\n"
            f"  Top-k: {self.config.top_k}"
        )

    # ========================================================================
    # 核心方法: 运行问答
    # ========================================================================

    def run(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_context: bool = True,
    ) -> Dict[str, Any]:
        """
        执行一次完整的 RAG 问答。

        流程:
            1. 检索:    调用 retriever.retrieve(question) 获取相关文档块
            2. 构建 Prompt: 将检索结果和问题填入 prompt 模板
            3. 生成:    调用 generator.generate(prompt) 生成回答
            4. 组装:    将回答、检索来源、耗时等整合为结果字典

        Args:
            question:       用户问题。
            top_k:          检索返回的文档数量。为 None 时使用 config.top_k。
            return_context: 是否在结果中包含检索到的文档片段（默认 True）。

        Returns:
            Dict[str, Any]: 包含以下字段的完整结果:
                - "question":          str   — 原始问题
                - "answer":            str   — LLM 生成的回答
                - "retrieved_chunks":  List  — 检索到的文档片段列表
                - "prompt":            str   — 送入 LLM 的完整 Prompt
                - "retrieval_time_ms": float — 检索耗时 (毫秒)
                - "generation_time_ms": float — 生成耗时 (毫秒)
                - "total_time_ms":     float — 总耗时 (毫秒)
                - "retriever_name":    str   — 检索器名称
                - "top_k":             int   — 实际使用的 top_k

        Example:
            >>> result = pipeline.run("什么是深度学习？")
            >>> print(result["answer"])
            >>> for chunk in result["retrieved_chunks"]:
            ...     print(f"  [{chunk['rank']}] {chunk['content'][:50]}...")
            >>> print(f"检索耗时: {result['retrieval_time_ms']:.1f} ms")
            >>> print(f"生成耗时: {result['generation_time_ms']:.1f} ms")
        """
        k = top_k if top_k is not None else self.config.top_k
        start_time = time.perf_counter()

        logger.info(f"[RAG Pipeline] 开始处理: '{question[:80]}...'")

        # ---- 步骤 1: 检索 ----
        retrieval_start = time.perf_counter()
        retrieved_chunks = self.retriever.retrieve(query=question, top_k=k)
        retrieval_time = (time.perf_counter() - retrieval_start) * 1000

        logger.info(
            f"[RAG Pipeline] 检索完成: {len(retrieved_chunks)} 条结果, "
            f"耗时 {retrieval_time:.1f} ms"
        )

        # ---- 步骤 2: 构建 Prompt ----
        prompt = self.prompt_template.build_prompt_from_chunks(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        logger.info(
            f"[RAG Pipeline] Prompt 构建完成: {len(prompt)} 字符"
        )

        # ---- 步骤 3: 生成 ----
        if self.generator is not None:
            generation_start = time.perf_counter()
            answer = self.generator.generate(prompt)
            generation_time = (time.perf_counter() - generation_start) * 1000
            logger.info(
                f"[RAG Pipeline] 生成完成: {len(answer)} 字符, "
                f"耗时 {generation_time:.1f} ms"
            )
        else:
            # 无 LLM：返回检索结果拼接
            generation_time = 0.0
            answer = "(LLM 未加载) 检索到以下相关内容:\n\n" + "\n\n".join(
                f"[{getattr(c, 'rank', i+1)}] ({getattr(c, 'score', 0):.4f}) {c.content[:200]}"
                for i, c in enumerate(retrieved_chunks)
            )

        # ---- 步骤 4: 组装结果 ----
        total_time = (time.perf_counter() - start_time) * 1000

        result = {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks if return_context else [],
            "prompt": prompt,
            "retrieval_time_ms": round(retrieval_time, 2),
            "generation_time_ms": round(generation_time, 2),
            "total_time_ms": round(total_time, 2),
            "retriever_name": self.retriever.name,
            "top_k": k,
        }

        logger.info(
            f"[RAG Pipeline] 完成: 总耗时 {total_time:.1f} ms, "
            f"回答长度 {len(answer)} 字符"
        )

        return result

    # ========================================================================
    # 批量处理
    # ========================================================================

    def run_batch(
        self,
        questions: List[str],
        top_k: Optional[int] = None,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        批量处理多个问题。

        逐个处理问题列表，返回每个问题的完整结果。
        注意: 目前是串行处理（确保每个问题独立且结果可复现）。

        Args:
            questions: 问题列表。
            top_k:     检索返回的文档数量。为 None 时使用 config.top_k。
            verbose:   是否在控制台打印进度信息。

        Returns:
            List[Dict]: 每个问题的完整结果列表。

        Example:
            >>> questions = ["什么是NLP？", "什么是RAG？", "什么是BM25？"]
            >>> results = pipeline.run_batch(questions)
            >>> for r in results:
            ...     print(f"Q: {r['question']}")
            ...     print(f"A: {r['answer'][:100]}...")
            ...     print()
        """
        logger.info(f"[RAG Pipeline] 开始批量处理: {len(questions)} 个问题")

        results: List[Dict[str, Any]] = []

        for i, question in enumerate(questions, start=1):
            if verbose:
                logger.info(f"处理 [{i}/{len(questions)}]: {question[:60]}...")

            try:
                result = self.run(question=question, top_k=top_k)
                result["index"] = i  # 添加序号
                results.append(result)
            except Exception as e:
                logger.exception(f"处理第 {i} 个问题失败: {e}")
                # 记录失败并继续处理后续问题
                results.append({
                    "question": question,
                    "answer": f"[ERROR] {e}",
                    "retrieved_chunks": [],
                    "prompt": "",
                    "retrieval_time_ms": 0,
                    "generation_time_ms": 0,
                    "total_time_ms": 0,
                    "retriever_name": self.retriever.name,
                    "top_k": top_k or self.config.top_k,
                    "index": i,
                    "error": str(e),
                })

        logger.info(f"[RAG Pipeline] 批量处理完成: {len(results)} 个结果")
        return results

    # ========================================================================
    # 检索器切换
    # ========================================================================

    def set_retriever(self, retriever: BaseRetriever) -> None:
        """
        替换当前 Pipeline 的检索器。

        用于实验对比场景：同一 Pipeline 实例可以切换不同 Retriever，
        而保持 Prompt 和 Generator 不变，实现"单一变量原则"。

        Args:
            retriever: 新的检索器实例。

        Example:
            >>> pipeline = RAGPipeline(dense_retriever, prompt, generator)
            >>> result_dense = pipeline.run("测试问题")
            >>> pipeline.set_retriever(bm25_retriever)
            >>> result_bm25 = pipeline.run("测试问题")  # 仅检索器变了
        """
        old_name = self.retriever.name
        self.retriever = retriever
        logger.info(
            f"RAGPipeline 检索器已切换: {old_name} → {retriever.name}"
        )

    # ========================================================================
    # 信息方法
    # ========================================================================

    def get_info(self) -> Dict[str, str]:
        """
        获取 Pipeline 的配置信息。

        Returns:
            Dict: 包含各组件信息的字典。
        """
        return {
            "retriever": str(self.retriever),
            "retriever_name": self.retriever.name,
            "generator": str(self.generator),
            "top_k": str(self.config.top_k),
            "chunk_size": str(self.config.chunk_size),
            "embedding_model": self.config.embedding_model_name,
            "llm_model": self.config.llm_model_name,
        }

    def __repr__(self) -> str:
        """返回 Pipeline 的字符串表示。"""
        return (
            f"RAGPipeline("
            f"retriever={self.retriever.name}, "
            f"top_k={self.config.top_k})"
        )
