"""
===============================================================================
全局配置模块 (Global Configuration)
===============================================================================
所有可调参数集中管理于此文件，确保实验过程中遵循"单一变量原则"。
其他模块仅读取配置，不自行硬编码任何参数。

使用方法:
    from rag_project.config import get_config
    cfg = get_config()          # 获取默认配置
    cfg.chunk_size = 256        # 修改特定参数
    cfg = get_config()          # 同一会话中返回同一实例
===============================================================================
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

# 全局单例实例
_config: Optional["Config"] = None


@dataclass
class Config:
    """
    全局配置数据类。

    Attributes:
        embedding_model_name: HuggingFace Embedding 模型名称
        embedding_device: Embedding 模型推理设备
        chunk_size: 文本切片大小 (tokens)
        chunk_overlap: 相邻切片重叠大小 (tokens)
        top_k: 检索返回的文档片段数量
        llm_model_name: 大语言模型名称
        llm_temperature: LLM 采样温度 (0=贪婪解码)
        seed: 全局随机种子，确保可复现性
    """

    # ========================================================================
    # Embedding 模型配置
    # ========================================================================
    # BGE (BAAI General Embedding) 中文小型模型
    # 模型卡片: https://huggingface.co/BAAI/bge-small-zh-v1.5
    # 输出维度: 512，适用于中文语义相似度任务
    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"

    # 推理设备: "cuda" 使用 GPU，"cpu" 使用 CPU
    embedding_device: str = "cuda"

    # 是否对嵌入向量进行 L2 归一化
    # BGE 系列模型推荐开启归一化，使得内积等价于余弦相似度
    embedding_normalize: bool = True

    # 编码时的批次大小，根据 GPU 显存调整
    embedding_batch_size: int = 32

    # ========================================================================
    # 文本切分配置 (Chunking)
    # ========================================================================
    # 每个文本切片的目标 Token 数
    # 512 是 RAG 领域常用的平衡值：足够承载信息，又不会超出 LLM 上下文窗口
    chunk_size: int = 512

    # 相邻切片之间的重叠 Token 数
    # 50 个 Token 的重叠可减少边界信息丢失
    chunk_overlap: int = 50

    # 递归切分时使用的分隔符列表（按优先级从高到低排列）
    # 中文文本优先按段落、换行、标点切分，最后才按字符切分
    separators: List[str] = field(default_factory=lambda: [
        "\n\n",     # 双换行（段落边界）
        "\n",       # 单换行
        "。",       # 中文句号
        "！",       # 中文感叹号
        "？",       # 中文问号
        "；",       # 中文分号
        "，",       # 中文逗号
        " ",        # 空格
        ""          # 逐字符切分（最后手段）
    ])

    # ========================================================================
    # FAISS 向量数据库配置
    # ========================================================================
    # FAISS 索引文件持久化路径
    faiss_index_path: str = "./faiss_index"

    # 距离度量策略
    # "cosine": 余弦相似度 → 使用内积 + 归一化向量实现
    # "euclidean": 欧氏距离 → 使用 L2 距离
    faiss_distance_strategy: str = "cosine"

    # ========================================================================
    # Retriever (检索器) 配置
    # ========================================================================
    # 默认返回的文档片段数量
    top_k: int = 5

    # ---------- BM25 稀疏检索参数 ----------
    # k1: 词频饱和度参数 (通常取值 [1.2, 2.0])
    # 值越大，词频对得分的影响越大
    bm25_k1: float = 1.5

    # b: 文档长度归一化参数 (通常取值 [0, 1])
    # 值越大，长文档的惩罚越重；0 表示不做长度归一化
    bm25_b: float = 0.75

    # ---------- Hybrid 混合检索参数 ----------
    # 融合策略: "rrf" = Reciprocal Rank Fusion; "linear" = 加权线性组合
    hybrid_fusion_method: str = "rrf"

    # RRF 中的 k 参数（控制排名对最终得分的影响程度）
    # RRF 公式: score(d) = Σ 1 / (k + rank_r(d))
    # k=60 是常用值，来自论文结果
    hybrid_rrf_k: int = 60

    # 线性融合时，BM25 得分的权重 (λ)
    # Dense 权重 = 1 - alpha
    # alpha=0.5 表示两者等权重
    hybrid_linear_alpha: float = 0.5

    # ========================================================================
    # LLM 大语言模型配置
    # ========================================================================
    # Qwen2.5-7B-Instruct: 通义千问 2.5 系列指令微调模型
    # 模型卡片: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
    llm_model_name: str = "Qwen/Qwen2.5-7B-Instruct"

    # 模型推理设备
    llm_device: str = "cuda"

    # 采样温度: 0.0 = 贪婪解码 (每次输出确定，确保可复现)
    llm_temperature: float = 0.0

    # 最大生成 Token 数
    llm_max_new_tokens: int = 512

    # 是否进行随机采样 (False = 贪婪解码)
    llm_do_sample: bool = False

    # 是否使用 4-bit 量化加载模型 (节省显存)
    # Qwen2.5-7B 使用 4-bit 量化后约需 4-5GB 显存
    llm_load_in_4bit: bool = True

    # ========================================================================
    # Prompt 提示词配置
    # ========================================================================
    # 系统提示词：定义 LLM 的角色和行为约束
    # 重要：强调 LLM 只能基于给定上下文回答，防止幻觉
    system_prompt: str = (
        "你是一个基于给定文档回答问题的助手。"
        "请严格基于以下提供的上下文信息回答问题。"
        "如果无法从上下文中找到答案，请明确说明'根据给定的上下文无法回答此问题'。"
        "不要编造任何信息，不要使用你的先验知识。"
        "回答时请尽量引用上下文中的具体内容。"
    )

    # 用户提示词模板
    # {context}: 检索到的文档片段拼接
    # {question}: 用户问题
    user_prompt_template: str = (
        "上下文信息：\n"
        "{context}\n\n"
        "问题：{question}\n\n"
        "请基于以上上下文回答问题："
    )

    # ========================================================================
    # 数据路径配置
    # ========================================================================
    # 原始数据存放目录
    data_dir: str = "./data"

    # 语料文件名 (JSONL 格式，每行一个 JSON 文档)
    # 每行格式: {"id": "doc_001", "title": "文档标题", "text": "文档正文"}
    corpus_file: str = "corpus.jsonl"

    # ========================================================================
    # 全局随机种子
    # ========================================================================
    seed: int = 42

    # ========================================================================
    # 日志配置
    # ========================================================================
    log_level: str = "INFO"
    log_dir: str = "./logs"


def get_config() -> Config:
    """
    获取全局配置单例。

    首次调用时创建默认 Config 实例，后续调用返回同一实例。
    该设计确保所有模块共享完全相同的配置对象，修改一处即全局生效。

    Returns:
        Config: 全局配置对象。

    Example:
        >>> from rag_project.config import get_config
        >>> cfg = get_config()
        >>> print(cfg.chunk_size)  # 512
        >>> cfg.chunk_size = 256   # 修改配置
        >>> print(get_config().chunk_size)  # 256（同一实例）
    """
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """
    设置全局配置对象（用于测试或加载外部配置时替换默认实例）。

    Args:
        config: 新的 Config 实例。
    """
    global _config
    _config = config
