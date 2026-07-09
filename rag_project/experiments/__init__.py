"""
实验模块 (Experiments)

自动化运行 BM25 / Dense / Hybrid 三组对比实验。

模块:
    - sample_data.py:   100 QA 对 + 中文 NLP 语料生成（无需外部数据）
    - run_experiment.py: 实验主脚本（自动运行三组实验，输出 CSV 结果）

运行方式:
    python -m rag_project.experiments.run_experiment           # 完整实验
    python -m rag_project.experiments.run_experiment --skip-llm  # 仅检索评估
"""
