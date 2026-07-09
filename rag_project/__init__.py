"""
rag_project -- 基于不同检索策略的 RAG 问答系统

本项目实现了一个标准的 Naive RAG（检索增强生成）系统，
支持三种检索策略：BM25 稀疏检索、BGE 稠密检索、以及混合检索。

项目结构:
    config/     -- 全局配置（所有可调参数集中管理）
    data/       -- 数据层（文档加载 + 文本切分）
    embeddings/ -- 嵌入层（BGE 模型封装 + FAISS 向量存储）
    retrievers/ -- 检索器层（BM25 / Dense / Hybrid）
    llm/        -- LLM 层（Prompt + Generator）
    evaluation/ -- 评估指标
    scripts/    -- 运行脚本
    src/        -- RAG Pipeline 编排
"""

__version__ = "1.0.0"
__author__ = "NLP Lab"
