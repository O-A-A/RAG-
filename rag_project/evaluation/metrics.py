"""
===============================================================================
评估指标模块 (Evaluation Metrics — Pure Functions)
===============================================================================
实现 RAG 系统检索质量和生成质量的核心评估指标。

所有函数均为纯函数: 接收输入 → 计算 → 返回数值，无副作用。
这确保了指标计算的正确性、可复现性和可测试性。

检索质量指标 (Retrieval Quality):
    - recall_at_k(): 前 K 个检索结果中相关文档的召回比例
    - mrr():         第一个相关文档排名的倒数 (Mean Reciprocal Rank)

生成质量指标 (Generation Quality):
    - exact_match(): 生成答案与标准答案的精确匹配 (0/1)
    - f1_score():    Token 级 F1 分数

批量聚合:
    - aggregate_retrieval(): 多查询检索指标平均
    - aggregate_generation(): 多查询生成指标平均
===============================================================================
"""

from typing import Dict, List, Set, Optional
from collections import Counter
import re


# ========================================================================
# 检索质量指标
# ========================================================================

def recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int = 5,
) -> float:
    """
    计算 Recall@K。

    衡量前 K 个检索结果中包含了多少比例的相关文档。

    公式:
        Recall@K = |Retrieved[:K] ∩ Relevant| / |Relevant|

    Args:
        retrieved_ids: 检索系统返回的文档 ID 列表（按排名降序）。
        relevant_ids:  标准答案中标记的相关文档 ID 集合。
        k:             截断位置，默认 5。

    Returns:
        float: Recall@K ∈ [0, 1]。1.0 = 所有相关文档都被召回。

    Example:
        >>> recall_at_k(["d1","d3","d5","d2"], {"d1","d2","d4"}, k=3)
        0.3333  # 前3个只有 d1 相关
    """
    if not relevant_ids:
        return 0.0

    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def mrr(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
) -> float:
    """
    计算 MRR (Mean Reciprocal Rank) 单个查询的 Reciprocal Rank。

    公式:
        RR = 1 / rank_first_relevant  (若无相关文档则 RR = 0)

    Args:
        retrieved_ids: 检索系统返回的文档 ID 列表（按排名降序）。
        relevant_ids:  标准答案中标记的相关文档 ID 集合。

    Returns:
        float: RR ∈ [0, 1]。1.0 = 第一个结果就是相关文档。

    Example:
        >>> mrr(["d3","d1","d2","d5"], {"d2","d5"})
        0.3333  # d2 排第 3 → RR = 1/3
    """
    if not relevant_ids:
        return 0.0

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank

    return 0.0


# ========================================================================
# 生成质量指标
# ========================================================================

def _normalize_text(text: str) -> str:
    """
    文本标准化: 全角→半角, 小写化, 压缩空白。

    步骤:
        1. 去除首尾空白
        2. ASCII 字母小写化
        3. 全角数字/标点 → 半角
        4. 多余空白压缩为单个空格

    Args:
        text: 原始文本。

    Returns:
        str: 标准化后的文本。
    """
    text = text.strip().lower()

    # 全角 → 半角映射
    full_to_half = {
        "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
        "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
        "（": "(", "）": ")", "［": "[", "］": "]",
        "｛": "{", "｝": "}", "：": ":", "；": ";",
        "＂": "\"", "＇": "'", "，": ",", "．": ".",
        "！": "!", "？": "?", "～": "~", "　": " ",
    }
    for full, half in full_to_half.items():
        text = text.replace(full, half)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize(text: str) -> List[str]:
    """
    混合中英文 Token 化。

    中文: 逐字切分（字符级匹配是中文 F1 的标准做法）。
    英文: 按单词 + 数字切分。

    Args:
        text: 标准化后的文本。

    Returns:
        List[str]: Token 列表。
    """
    tokens: List[str] = []
    # 中文字符 | 英文单词/数字 | 其他非空白字符
    for match in re.finditer(r"[一-鿿]|[a-zA-Z0-9]+|\S", text):
        tokens.append(match.group())
    return tokens


def exact_match(
    predicted: str,
    ground_truth: str,
    normalize: bool = True,
) -> float:
    """
    计算 Exact Match (EM)。

    标准化后与标准答案做精确字符串比较。这是最严格的指标 —
    任何同义替换（如 "AI" vs "人工智能"）都会判 0。

    Args:
        predicted:    模型生成的答案文本。
        ground_truth: 标准答案文本。
        normalize:    是否先做文本标准化。

    Returns:
        float: 1.0 或 0.0（二值判断）。
    """
    if normalize:
        predicted = _normalize_text(predicted)
        ground_truth = _normalize_text(ground_truth)

    return 1.0 if predicted == ground_truth else 0.0


def f1_score(
    predicted: str,
    ground_truth: str,
    normalize: bool = True,
) -> float:
    """
    计算 Token 级 F1 Score。

    F1 = 2 × Precision × Recall / (Precision + Recall)

    其中:
        Precision = |common_tokens| / |pred_tokens|
        Recall    = |common_tokens| / |gt_tokens|

    使用多重集合 (Counter) 来处理重复 token。

    Args:
        predicted:    模型生成的答案文本。
        ground_truth: 标准答案文本。
        normalize:    是否先做文本标准化。

    Returns:
        float: F1 ∈ [0, 1]。
    """
    if normalize:
        predicted = _normalize_text(predicted)
        ground_truth = _normalize_text(ground_truth)

    # 边界情况
    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0

    pred_tokens = _tokenize(predicted)
    gt_tokens = _tokenize(ground_truth)

    # 多重集合交集
    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)

    common = sum((pred_counter & gt_counter).values())

    precision = common / len(pred_tokens) if pred_tokens else 0.0
    recall = common / len(gt_tokens) if gt_tokens else 0.0

    if precision + recall == 0.0:
        return 0.0

    return 2.0 * precision * recall / (precision + recall)


# ========================================================================
# 批量聚合
# ========================================================================

def aggregate_retrieval(
    all_retrieved: List[List[str]],
    all_relevant: List[Set[str]],
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    对多个查询的检索指标取平均。

    Args:
        all_retrieved: 每个查询的检索结果 ID 列表，shape [N_queries × ?]。
        all_relevant:  每个查询的相关文档 ID 集合，shape [N_queries]。
        k_values:      要计算的 K 列表，默认 [1, 3, 5, 10]。

    Returns:
        Dict: {"recall@1": 0.XX, "recall@5": 0.XX, "mrr": 0.XX, "n_queries": N}
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    n = len(all_retrieved)
    if n == 0:
        return {}
    if n != len(all_relevant):
        raise ValueError(
            f"数量不匹配: retrieved={n}, relevant={len(all_relevant)}"
        )

    recall_sums = {k: 0.0 for k in k_values}
    mrr_sum = 0.0

    for retrieved, relevant in zip(all_retrieved, all_relevant):
        for k in k_values:
            recall_sums[k] += recall_at_k(retrieved, relevant, k)
        mrr_sum += mrr(retrieved, relevant)

    result = {f"recall@{k}": round(recall_sums[k] / n, 4) for k in k_values}
    result["mrr"] = round(mrr_sum / n, 4)
    result["n_queries"] = n
    return result


def aggregate_generation(
    all_predictions: List[str],
    all_ground_truths: List[str],
) -> Dict[str, float]:
    """
    对多个查询的生成指标取平均。

    Args:
        all_predictions:   模型生成的答案列表。
        all_ground_truths: 标准答案列表。

    Returns:
        Dict: {"exact_match": 0.XX, "f1": 0.XX, "n_queries": N}
    """
    n = len(all_predictions)
    if n == 0:
        return {}
    if n != len(all_ground_truths):
        raise ValueError(
            f"数量不匹配: predictions={n}, ground_truths={len(all_ground_truths)}"
        )

    em_sum = sum(exact_match(p, g) for p, g in zip(all_predictions, all_ground_truths))
    f1_sum = sum(f1_score(p, g) for p, g in zip(all_predictions, all_ground_truths))

    return {
        "exact_match": round(em_sum / n, 4),
        "f1": round(f1_sum / n, 4),
        "n_queries": n,
    }
