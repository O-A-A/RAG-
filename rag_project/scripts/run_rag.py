#!/usr/bin/env python3
"""
===============================================================================
RAG 系统运行脚本 (Main Entry Point)
===============================================================================
演示从数据加载到问答生成的完整 RAG 流程。

运行方式:
    # 方式 1: 使用默认配置运行
    cd rag_project
    python scripts/run_rag.py

    # 方式 2: 指定检索器类型
    python scripts/run_rag.py --retriever bm25
    python scripts/run_rag.py --retriever dense_bge
    python scripts/run_rag.py --retriever hybrid_rrf

    # 方式 3: 交互模式（逐个输入问题）
    python scripts/run_rag.py --interactive

    # 方式 4: 从文件读取问题批量处理
    python scripts/run_rag.py --questions questions.txt --output results.json

流程概览:
    1. 加载配置
    2. 加载文档语料 (JSONL/TXT)
    3. 文本切分 (chunk_size=512, overlap=50)
    4. 构建/加载 FAISS 索引
    5. 初始化所选检索器
    6. 初始化 LLM
    7. 创建 RAG Pipeline
    8. 运行问答
===============================================================================
"""

import argparse
import json
import logging
import os
import sys

# 将项目根目录加入 Python 路径
# 确保可以从任意目录运行此脚本
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 设置全局随机种子（必须在其他 import 之前）
from rag_project.config import get_config, Config
cfg = get_config()

import random
import numpy as np

try:
    import torch
    torch.manual_seed(cfg.seed)
except ImportError:
    torch = None

random.seed(cfg.seed)
np.random.seed(cfg.seed)


def setup_logging():
    """
    配置日志系统。

    日志同时输出到:
        - 控制台 (终端)
        - 文件 (logs/ 目录，带时间戳)
    """
    log_level = getattr(logging, get_config().log_level.upper(), logging.INFO)
    log_dir = get_config().log_dir
    os.makedirs(log_dir, exist_ok=True)

    # 日志格式
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # 根 logger 配置
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            # 控制台输出
            logging.StreamHandler(sys.stdout),
            # 文件输出（带时间戳）
            logging.FileHandler(
                os.path.join(
                    log_dir,
                    f"rag_{logging.Formatter().formatTime}"
                    f"(logging.LogRecord('', 0, '', 0, '', (), None))"
                ),
                encoding="utf-8",
            ),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"日志系统初始化完成: level={get_config().log_level}")
    return logger


def setup_logging_simple():
    """简化版日志配置（避免 FileHandler 时间戳复杂性）。"""
    log_level = getattr(logging, get_config().log_level.upper(), logging.INFO)
    log_dir = get_config().log_dir
    os.makedirs(log_dir, exist_ok=True)

    from datetime import datetime
    log_file = os.path.join(
        log_dir,
        f"rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

    return logging.getLogger(__name__)


def print_banner():
    """打印项目横幅。"""
    banner = r"""
    ╔══════════════════════════════════════════════════════╗
    ║            RAG 检索增强生成系统                        ║
    ║    基于不同检索策略的问答系统                            ║
    ║                                                      ║
    ║    Embedding: BAAI/bge-small-zh-v1.5                 ║
    ║    LLM:       Qwen2.5-7B-Instruct                    ║
    ║    Vector DB: FAISS                                  ║
    ║    Chunk:     512 / Overlap: 50                      ║
    ║    Top-k:     5                                      ║
    ╚══════════════════════════════════════════════════════╝
    """
    print(banner)


def build_rag_system(retriever_type: str = "hybrid_rrf"):
    """
    构建完整的 RAG 系统。

    按顺序执行:
        1. 加载文档
        2. 文本切分
        3. 构建/加载 Embedding 和 FAISS
        4. 初始化检索器
        5. 初始化 LLM

    Args:
        retriever_type: 检索器类型。
            可选: "bm25", "dense_bge", "hybrid_rrf", "hybrid_linear"

    Returns:
        tuple: (pipeline, chunks) — RAG Pipeline 实例和文档块列表。

    Raises:
        FileNotFoundError: 语料文件不存在。
        RuntimeError: 模型加载或索引构建失败。
    """
    logger = logging.getLogger(__name__)

    # ---- 延迟导入（避免启动时加载所有重型依赖） ----
    from rag_project.data.loader import DocumentLoader
    from rag_project.data.splitter import TextSplitter
    from rag_project.embeddings.encoder import EmbeddingEncoder
    from rag_project.embeddings.vector_store import FAISSVectorStore
    from rag_project.retrievers.bm25_retriever import BM25Retriever
    from rag_project.retrievers.dense_retriever import DenseRetriever
    from rag_project.retrievers.hybrid_retriever import HybridRetriever
    from rag_project.llm.prompt import RAGPromptTemplate
    from rag_project.llm.generator import LLMGenerator
    from rag_project.src.pipeline import RAGPipeline

    # ====================================================================
    # 阶段 1: 加载文档
    # ====================================================================
    logger.info("=" * 50)
    logger.info("阶段 1/5: 加载文档")
    logger.info("=" * 50)

    from langchain_core.documents import Document

    loader = DocumentLoader()
    try:
        documents = loader.load_documents()
    except FileNotFoundError:
        logger.info("语料文件不存在，使用内置示例数据 (sample_data)")
        from rag_project.experiments.sample_data import get_corpus
        corpus = get_corpus()
        documents = [
            Document(
                page_content=d["text"],
                metadata={"doc_id": d["id"], "chunk_id": d["id"], "title": d["title"]}
            )
            for d in corpus
        ]

    if not documents:
        raise FileNotFoundError(
            f"未找到任何文档。"
            f"请将 JSONL 格式的语料放入 '{get_config().data_dir}' 目录，"
            f"或在 config 中修改 corpus_file 配置。"
        )

    # ====================================================================
    # 阶段 2: 文本切分
    # ====================================================================
    logger.info("\n" + "=" * 50)
    logger.info("阶段 2/5: 文本切分")
    logger.info("=" * 50)

    splitter = TextSplitter()
    chunks = splitter.split_documents(documents)

    # 打印切分统计
    stats = splitter.get_chunk_statistics(chunks)
    logger.info(
        f"切分统计: 总块数={stats['total_chunks']}, "
        f"平均长度={stats['mean_length']:.0f} 字符, "
        f"最短={stats['min_length']}, 最长={stats['max_length']}"
    )

    # ====================================================================
    # 阶段 3: Embedding & FAISS 索引
    # ====================================================================
    logger.info("\n" + "=" * 50)
    logger.info("阶段 3/5: 构建 Embedding & FAISS 索引")
    logger.info("=" * 50)

    encoder = EmbeddingEncoder()
    vector_store = FAISSVectorStore(encoder)

    # 检查是否已有保存的索引（跳过，交互模式仅需内存索引）
    logger.info("构建新的 FAISS 索引...")
    vector_store.build_from_documents(chunks)

    # ====================================================================
    # 阶段 4: 初始化检索器
    # ====================================================================
    logger.info("\n" + "=" * 50)
    logger.info("阶段 4/5: 初始化检索器")
    logger.info("=" * 50)

    logger.info(f"检索器类型: {retriever_type}")

    if retriever_type == "bm25":
        retriever = BM25Retriever(chunks)

    elif retriever_type == "dense_bge":
        retriever = DenseRetriever(vector_store)

    elif retriever_type in ("hybrid_rrf", "hybrid_linear"):
        bm25_ret = BM25Retriever(chunks)
        dense_ret = DenseRetriever(vector_store)
        if retriever_type == "hybrid_linear":
            retriever = HybridRetriever(bm25_ret, dense_ret, fusion_method="linear")
        else:
            retriever = HybridRetriever(bm25_ret, dense_ret, fusion_method="rrf")

    else:
        raise ValueError(
            f"不支持的检索器类型: '{retriever_type}'\n"
            f"可选: bm25, dense_bge, hybrid_rrf, hybrid_linear"
        )

    logger.info(f"检索器初始化完成: {retriever}")

    # ====================================================================
    # 阶段 5: 初始化 LLM & Pipeline
    # ====================================================================
    logger.info("\n" + "=" * 50)
    logger.info("阶段 5/5: 初始化 LLM & Pipeline")
    logger.info("=" * 50)

    prompt_template = RAGPromptTemplate()

    # 自动检测 CUDA：不可用时跳过 LLM
    try:
        import torch as _torch
        has_cuda = _torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    if has_cuda:
        generator = LLMGenerator()
    else:
        logger.warning("CUDA 不可用，跳过 LLM 加载。回答将来自检索结果拼接。")
        generator = None

    pipeline = RAGPipeline(retriever, prompt_template, generator)
    logger.info("RAG 系统构建完成！")

    return pipeline, chunks


def interactive_mode(pipeline):
    """
    交互式问答模式。

    用户可以在终端中逐个输入问题，实时查看 RAG 系统的回答。
    输入 'quit' 或 'exit' 退出。

    Args:
        pipeline: RAGPipeline 实例。
    """
    logger = logging.getLogger(__name__)

    print("\n" + "=" * 60)
    print("  交互式 RAG 问答模式")
    print("  输入 'quit' 或 'exit' 退出")
    print("  输入 'info' 查看 Pipeline 信息")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("🧑 您的问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n再见!")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("再见!")
            break

        if question.lower() == "info":
            info = pipeline.get_info()
            print("\n📋 Pipeline 信息:")
            for k, v in info.items():
                print(f"   {k}: {v}")
            print()
            continue

        # 运行 RAG Pipeline
        print("\n⏳ 正在检索和生成...\n")

        try:
            result = pipeline.run(question)

            # 打印回答
            print("🤖 回答:")
            print(f"   {result['answer']}\n")

            # 打印检索来源
            print("📚 检索来源:")
            for chunk in result["retrieved_chunks"]:
                source = chunk.metadata.get("title", "未知来源") if hasattr(chunk, 'metadata') else "未知来源"
                print(
                    f"   [{chunk.metadata.get('rank', '?')}] (得分: {chunk.score:.4f}) "
                    f"{chunk.content[:80]}..."
                )
            print()

            # 打印耗时信息
            print(
                f"⏱️  检索: {result['retrieval_time_ms']:.0f} ms | "
                f"生成: {result['generation_time_ms']:.0f} ms | "
                f"总计: {result['total_time_ms']:.0f} ms\n"
            )

        except Exception as e:
            print(f"❌ 处理失败: {e}\n")
            logger.exception("交互问答异常")


def main():
    """主函数。"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="RAG 检索增强生成系统 - 运行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    python scripts/run_rag.py                          # 默认 Hybrid RRF 检索
    python scripts/run_rag.py --retriever bm25         # BM25 检索
    python scripts/run_rag.py --retriever dense_bge    # Dense 检索
    python scripts/run_rag.py --interactive            # 交互模式
    python scripts/run_rag.py --questions q.txt        # 从文件读取问题
        """,
    )
    parser.add_argument(
        "--retriever", "-r",
        type=str,
        default="hybrid_rrf",
        choices=["bm25", "dense_bge", "hybrid_rrf", "hybrid_linear"],
        help="检索器类型 (默认: hybrid_rrf)",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="进入交互式问答模式",
    )
    parser.add_argument(
        "--questions", "-q",
        type=str,
        default=None,
        help="包含问题列表的文件路径（每行一个问题）",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="结果输出文件路径 (JSON 格式)",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=None,
        help="检索返回的文档数量（覆盖 config 中的默认值）",
    )

    args = parser.parse_args()

    # 设置日志
    logger = setup_logging_simple()

    # 打印横幅
    print_banner()

    # 打印配置信息
    cfg = get_config()
    logger.info(f"配置: chunk_size={cfg.chunk_size}, top_k={cfg.top_k}")
    logger.info(f"Embedding: {cfg.embedding_model_name}")
    logger.info(f"LLM: {cfg.llm_model_name}")

    # 自动检测设备：CUDA 不可用时回退 CPU
    try:
        import torch
        if not torch.cuda.is_available():
            logger.info("CUDA 不可用，使用 CPU")
            cfg.embedding_device = "cpu"
            cfg.llm_device = "cpu"
    except ImportError:
        cfg.embedding_device = "cpu"
        cfg.llm_device = "cpu"

    # 构建 RAG 系统
    try:
        pipeline, chunks = build_rag_system(retriever_type=args.retriever)
    except Exception as e:
        logger.error(f"RAG 系统构建失败: {e}")
        logger.exception("详细错误信息:")
        return 1

    # ---- 运行模式选择 ----
    if args.questions:
        # 模式 1: 从文件读取问题批量处理
        logger.info(f"从文件读取问题: {args.questions}")

        if not os.path.exists(args.questions):
            logger.error(f"问题文件不存在: {args.questions}")
            return 1

        with open(args.questions, "r", encoding="utf-8") as f:
            questions = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]

        logger.info(f"共 {len(questions)} 个问题")
        results = pipeline.run_batch(questions, top_k=args.top_k)

        # 保存结果
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"结果已保存到: {args.output}")

        # 打印摘要
        print("\n" + "=" * 60)
        print("  批量处理结果摘要")
        print("=" * 60)
        for r in results:
            print(f"\nQ: {r['question']}")
            print(f"A: {r['answer'][:150]}...")
            print(f"   耗时: {r['total_time_ms']:.0f} ms")

    elif args.interactive or not args.questions:
        # 模式 2: 交互式问答
        interactive_mode(pipeline)

    return 0


if __name__ == "__main__":
    sys.exit(main())
