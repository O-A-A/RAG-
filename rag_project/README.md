# RAG 检索策略对比实验

《自然语言处理》课程实验 — 基于不同检索策略的 RAG 问答系统性能比较

## 概述

构建标准 Naive RAG 系统，比较三种检索策略：
- **BM25** — 稀疏检索 (rank_bm25 + jieba 分词)
- **Dense (BGE)** — 稠密检索 (BAAI/bge-small-zh-v1.5 + FAISS)
- **Hybrid** — 混合检索 (α × Dense + (1-α) × BM25)

## 技术栈

| 组件 | 选型 |
|---|---|
| Embedding | BAAI/bge-small-zh-v1.5 (512-dim) |
| LLM | Qwen2.5-7B-Instruct |
| Vector DB | FAISS (IndexFlatIP) |
| 框架 | LangChain + HuggingFace |
| 评估 | Recall@5, MRR, Exact Match, F1 |

## 固定参数

| 参数 | 值 |
|---|---|
| chunk_size | 512 |
| chunk_overlap | 50 |
| top_k | 5 |
| LLM temperature | 0.0 (贪婪解码) |
| 随机种子 | 42 |

## 项目结构

```
rag_project/
├── config/           全局配置 (所有可调参数集中管理)
├── data/             文档加载 + 文本切分
├── embeddings/       BGE 模型封装 + FAISS 向量存储
├── retrievers/       BM25 / Dense / Hybrid 检索器
├── llm/              Prompt 模板 + Qwen2.5 生成器
├── evaluation/       指标计算 + 评估器 + CSV 导出
├── experiments/      实验脚本 + 样本数据 + 图表生成
├── scripts/          交互式问答脚本
├── src/              RAG Pipeline 编排
├── figures/          论文图表输出
└── results/          实验结果输出
```

## 快速开始

### 安装

```bash
cd rag_project
pip install -r requirements.txt
```

### 运行实验

```bash
# 仅检索评估 (无需 GPU, ~1 分钟)
python -m rag_project.experiments.run_experiment --skip-llm

# 完整 RAG 实验 (需 GPU ~5GB 显存, ~20 分钟)
python -m rag_project.experiments.run_experiment

# 生成论文图表
python experiments/plot_results.py -i results/summary.csv

# 交互式问答
python scripts/run_rag.py --interactive
```

### 输出文件

| 文件 | 说明 |
|---|---|
| `results/experiment_result.csv` | 300 条详细结果 (100 题 × 3 组) |
| `results/summary.csv` | 三组检索器聚合对比 |
| `figures/comparison_*.pdf` | 600 DPI 论文风格对比图 |

## 评估指标

| 指标 | 公式 | 评估维度 |
|---|---|---|
| Recall@5 | `|Top5 ∩ Relevant| / |Relevant|` | 检索完整性 |
| MRR | `1 / rank_first_relevant` | 检索排序质量 |
| Exact Match | `normalize(pred) == normalize(gt)` | 生成精确度 |
| F1 | `2 × P × R / (P + R)` | 生成质量 |

## 可复现性说明

- 所有随机种子固定为 42 (Python / NumPy / PyTorch / CUDA)
- LLM 使用贪婪解码 (temperature=0, do_sample=False)
- 测试数据集硬编码 (30 篇语料 + 100 条 QA 对), 每次输入完全相同
- 建议运行前设置: `export PYTHONHASHSEED=42`
