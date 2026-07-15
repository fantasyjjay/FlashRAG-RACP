# HotpotQA 实验结果与环境汇总

生成日期：2026-07-05

## 1. 统计口径

- 数据集：HotpotQA dev，共 7,405 条。
- 问题类型：5,918 条 bridge，1,487 条 comparison。
- 表中 EM、F1、Acc、Precision、Recall 和 Retrieval Recall 均转换为百分数。
- Retrieval Recall 后的 `@K` 来自对应实验的原始指标文件。不同 `K` 的结果不能直接视为同一指标。
- “全量主实验”与“全量 QD 派生实验”均使用 7,405 条数据；1000 条实验和 smoke test 单独列出。
- Reranker 标记综合原始 prepare 配置和实验命名判断。部分派生 generate 目录的配置会继承其他阶段字段。
- Prediction fusion 使用两次答案生成结果做离线融合，不属于单次生成设置。
- 软件版本、Git commit 和服务器状态记录自报告生成时的当前环境；历史实验跨越多个代码版本，具体实验参数应以各输出目录中的 `config.yaml` 和 `run.log` 为准。

## 2. 核心结论

- 全部单方法主实验最高分：IterRetGen + reranker，EM 38.74，F1 50.49。
- 包含离线双预测融合时，最佳 QD 结果为 EM 37.57，F1 48.65。
- 当前无 reranker 最高 EM/F1：IRCoT，EM 36.26，F1 47.50；但 IRCoT 最终最多
  使用 10 篇文档，EFC static-bridge 使用 6 篇，仍需补 EFC final-top10 做同预算比较。
- EFC 相比无 reranker IterRetGen：EM 提升 0.90，F1 提升 1.03，Acc 提升 1.27 个百分点。
- EFC 从初始全量版本到最佳版本：EM 从 26.12 提升至 35.57，F1 从 34.12 提升至 46.69。

## 3. 无 Reranker 专项整理

### 3.1 全量结果

| 排名 | 方法/实验 | Final K | EM | F1 | Acc | Precision | Recall | Retrieval Recall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | **IRCoT no-reranker** | 10 | **36.26** | **47.50** | 39.57 | **51.40** | 46.93 | 64.77 (@10) |
| 2 | EFC static-bridge | 6 | 35.57 | 46.69 | 42.40 | 48.20 | **49.27** | 68.35 (@6) |
| 3 | EFC 后续 router 修改版 | 6 | 35.54 | 46.68 | **42.42** | 48.18 | 49.26 | 68.39 (@6) |
| 4 | EFC IterRetGen 主干版 | 6 | 35.52 | 46.64 | 42.38 | 48.12 | 49.24 | **68.43 (@6)** |
| 5 | 原生 IterRetGen | 10 | 34.67 | 45.66 | 41.13 | 47.12 | 47.97 | 62.86 (@10) |
| 6 | Native RAG top10 | 10 | 33.54 | 44.80 | 39.93 | 46.34 | 47.03 | 57.61 (@5) |
| 7 | EFC router-v6 全量版 | 6 | 33.65 | 44.50 | 40.59 | 45.81 | 47.22 | 63.94 (@6) |
| 8 | Native RAG top15 | 15 | 33.26 | 44.46 | 40.85 | 45.90 | 47.77 | 57.61 (@5) |
| 9 | Native RAG top20 | 20 | 32.52 | 43.93 | 40.84 | 45.32 | 47.75 | 57.61 (@5) |
| 10 | Native RAG top5 | 5 | 31.74 | 42.36 | 38.15 | 43.71 | 44.71 | 57.61 (@5) |
| 11 | 初始 EFC | 5 | 26.12 | 34.12 | 29.86 | 35.61 | 35.03 | 55.14 (@5) |

对应实验目录：

| 简称 | 实验目录 |
|---|---|
| IRCoT no-reranker | `hotpotqa_2026_07_09_14_27_ircot-no-rerank-cache-full` |
| EFC static-bridge | `hotpotqa_2026_06_08_13_17_efc-static-bridge-v1-full` |
| EFC 后续 router 修改版 | `hotpotqa_2026_06_08_16_50_full` |
| EFC IterRetGen 主干版 | `hotpotqa_2026_06_07_12_41_efc-iterretgen-hotpotqa-full` |
| 原生 IterRetGen | `hotpotqa_2026_06_05_16_46_iterretgen-native-no-reranker-gpu3` |
| Native RAG top10 | `hotpotqa_2026_05_30_12_40_native-no-rerank-derived-top10-generate` |
| EFC router-v6 全量版 | `hotpotqa_2026_06_06_19_55_efc-hotpotqa-full-v2` |
| Native RAG top15 | `hotpotqa_2026_05_30_13_13_native-no-rerank-derived-top15-generate` |
| Native RAG top20 | `hotpotqa_2026_05_29_16_02_native-no-rerank-top20-generate` |
| Native RAG top5 | `hotpotqa_2026_05_30_12_27_native-no-rerank-derived-top5-generate` |
| 初始 EFC | `hotpotqa_2026_06_06_17_26_efc-hotpotqa-full` |

### 3.2 EFC 相对无 Reranker 基线的提升

| 对比 | EM | F1 | Acc | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| EFC static-bridge - 原生 IterRetGen | **+0.90** | **+1.03** | **+1.27** | **+1.08** | **+1.30** |
| EFC static-bridge - Native RAG top10 | **+2.03** | **+1.90** | **+2.47** | **+1.86** | **+2.24** |
| EFC static-bridge - Native RAG top5 | **+3.84** | **+4.34** | **+4.25** | **+4.49** | **+4.56** |
| EFC static-bridge - 初始 EFC | **+9.45** | **+12.57** | **+12.55** | **+12.59** | **+14.24** |

Retrieval Recall 未在此表计算差值，因为 Native、IterRetGen 和 EFC 使用的评测 K 不完全相同。

### 3.3 Native RAG 的文档数量影响

| Final K | EM | F1 | Acc | Precision | Recall |
|---:|---:|---:|---:|---:|---:|
| 5 | 31.74 | 42.36 | 38.15 | 43.71 | 44.71 |
| **10** | **33.54** | **44.80** | 39.93 | **46.34** | 47.03 |
| 15 | 33.26 | 44.46 | **40.85** | 45.90 | **47.77** |
| 20 | 32.52 | 43.93 | 40.84 | 45.32 | 47.75 |

无 reranker 的 Native RAG 在 top10 获得最佳 EM/F1。继续增加到 15 或 20 篇文档会增加上下文噪声，F1 反而下降。

### 3.4 当前无 Reranker 对比的缺口与补实验优先级

#### A. 主表必补：相同模型、检索器和语料

| 优先级 | 方法 | 对比价值 | 本地状态 |
|---:|---|---|---|
| 已完成 | IRCoT | 标准多跳“推理一步、检索一步”循环基线，直接检验 EFC router 是否优于固定交替检索 | no-reranker 全量 EM 36.26、F1 47.50；需补 EFC final-top10 同预算实验 |
| P0 | Adaptive-RAG | 同样根据问题复杂度路由到 no-RAG、单步 RAG 或多步 RAG，是与 EFC 最直接的 router 基线 | FlashRAG 已实现；缺少 Adaptive classifier checkpoint |
| P0 | Full-QD no-reranker | 检验自适应路由是否优于对所有问题统一拆分 | 当前 RACP 代码可运行；尚无全量结果 |
| P1 | FLARE | 根据生成 token 置信度决定何时检索，并用预测内容构造检索 query | FlashRAG 已实现；需要先验证当前 vLLM 的 token-score 接口 |
| P1 | Self-Ask + Search | 显式生成 follow-up question 并逐步检索，属于经典问题分解式多跳基线 | FlashRAG 已实现 pipeline；需要接入当前统一实验入口 |
| P1 | RACP no-reranker | 对比动态文档截断/压缩与 EFC 多轮检索 | 目前只有 1,000 条结果 |

建议主表统一设置：

```text
Dataset          = HotpotQA dev 7405
Retriever        = bge-large-en-v1.5
Corpus/Index     = wiki18_100w
Generator        = Llama-3.1-8B-Instruct
Reranker         = false
Initial top-k    = 20
Final context-k  = 6
Temperature      = 0
```

IRCoT、FLARE、Self-Ask 的最大检索轮数应统一为 2，或额外报告平均检索调用次数，避免以更多检索预算换取分数。

#### B. 扩展表：需要专门训练模型

| 方法 | 特点 | 为什么不放入同预算主表 |
|---|---|---|
| RQ-RAG | 训练模型执行 query rewrite、decomposition 和 disambiguation | 需要专用 RQ-RAG Llama2-7B checkpoint |
| Self-RAG | 使用专门训练的 retrieval、relevance、support 和 utility 控制 token | 不能直接替换为普通 Llama-3.1-8B |
| CoRAG | 训练模型逐步生成 retrieval chain，并动态改写 query | 使用多跳 QA 数据训练的 CoRAG checkpoint |
| Search-R1 | 通过强化学习学习推理过程中何时搜索和如何多轮搜索 | 使用 RL 训练的 Qwen checkpoint |
| R1-Searcher/AutoRefine/O2-Searcher | 训练型 search-and-reasoning 方法 | 模型、训练数据和生成预算均与 EFC 不同 |

这些方法可以作为“现代训练型检索推理模型”参考，但应单独报告 checkpoint、训练数据和最大生成 token。

#### C. 不同检索基础设施：单独系统表

| 方法 | 差异 |
|---|---|
| HippoRAG 2 | 使用图结构、phrase/passage node 和 Personalized PageRank，不再是同一个 Flat dense index |
| HopRAG | 使用逻辑感知的多跳检索结构，需要重新构建索引 |
| GraphRAG 类方法 | 索引、图构建成本和检索单元均与当前 passage RAG 不同 |

这些方法适合作为系统级扩展对比，不适合用来单独证明 EFC router 的收益。

#### D. 次要无 Reranker 对照

- SuRe：候选答案生成、摘要与排序，偏 branching RAG。
- REPLUG：按检索分数融合多文档生成概率。
- Selective-Context no-reranker：验证压缩方法在没有 cross-encoder 时的效果。
- LongLLMLingua no-reranker：验证 prompt compression 是否可以代替多轮检索。

推荐执行顺序：

1. EFC final-top10 同预算全量
2. Full-QD no-reranker
3. Adaptive-RAG no-reranker
4. Self-Ask no-reranker
5. FLARE no-reranker
6. RQ-RAG、CoRAG、Search-R1 扩展表

当前可以支持的严格结论是：

> IRCoT 在当前已完成的 HotpotQA 全量无 reranker 实验中取得最高 EM 和 F1；
> EFC static-bridge 使用更少的最终文档并取得更高 Acc 和 Recall，且超过原生 IterRetGen。
> 在完成 EFC final-top10 前，不能把 IRCoT 与 EFC 的差值完全归因于算法。

## 4. 全量主实验

| 实验 | 样本数 | Reranker | EM | F1 | Acc | Precision | Recall | Retrieval Recall |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|
| `hotpotqa_2026_06_03_15_33_iterretgen-rerank-top5` | 7405 | 是 | 38.74 | 50.49 | 45.55 | 52.14 | 52.67 | 68.48 (@10) |
| `hotpotqa_2026_07_09_14_27_ircot-no-rerank-cache-full` | 7405 | 否 | 36.26 | 47.50 | 39.57 | 51.40 | 46.93 | 64.77 (@10) |
| `hotpotqa_2026_06_02_13_41_full-qd-planner3b-parser-v2-title-dedup-top5` | 7405 | 是 | 36.22 | 47.78 | 43.11 | 49.33 | 50.10 | 65.10 (@5) |
| `hotpotqa_2026_06_02_10_10_full-qd-planner3b-reranker-large-top5` | 7405 | 是 | 36.03 | 47.52 | 42.80 | 49.17 | 49.77 | 64.36 (@10) |
| `hotpotqa_2026_05_30_15_28_native-rerank-derived-top5-generate` | 7405 | 是 | 35.94 | 47.38 | 42.65 | 49.04 | 49.63 | 64.21 (@10) |
| `hotpotqa_2026_06_01_15_41_mhr` | 7405 | 是 | 35.80 | 47.25 | 42.55 | 48.87 | 49.54 | 63.79 (@5) |
| `hotpotqa_2026_06_01_20_59_mhr-title-default-fill-cache` | 7405 | 是 | 35.76 | 47.12 | 42.44 | 48.74 | 49.49 | 63.42 (@5) |
| `hotpotqa_2026_05_30_15_54_native-rerank-derived-top10-generate` | 7405 | 是 | 35.50 | 46.91 | 42.50 | 48.54 | 49.32 | 68.20 (@10) |
| `hotpotqa_2026_06_01_19_06_mhr-more-planner` | 7405 | 是 | 35.58 | 46.84 | 42.24 | 48.49 | 49.19 | 62.94 (@5) |
| `hotpotqa_2026_06_08_13_17_efc-static-bridge-v1-full` | 7405 | 否 | 35.57 | 46.69 | 42.40 | 48.20 | 49.27 | 68.35 (@6) |
| `hotpotqa_2026_06_08_16_50_full` | 7405 | 否 | 35.54 | 46.68 | 42.42 | 48.18 | 49.26 | 68.39 (@6) |
| `hotpotqa_2026_06_02_18_45_full-qd-parser-v2-title-dedup-top6` | 7405 | 是 | 35.03 | 46.64 | 42.78 | 47.98 | 49.71 | 66.28 (@6) |
| `hotpotqa_2026_06_07_12_41_efc-iterretgen-hotpotqa-full` | 7405 | 否 | 35.52 | 46.64 | 42.38 | 48.12 | 49.24 | 68.43 (@6) |
| `hotpotqa_2026_06_03_18_52_ircot-rerank-top5` | 7405 | 是 | 35.31 | 46.57 | 50.63 | 48.27 | 59.19 | 64.15 (@10) |
| `hotpotqa_2026_05_30_16_27_native-rerank-derived-top15-generate` | 7405 | 是 | 34.34 | 45.80 | 42.11 | 47.20 | 49.22 | 68.20 (@10) |
| `hotpotqa_2026_06_05_16_46_iterretgen-native-no-reranker-gpu3` | 7405 | 否 | 34.67 | 45.66 | 41.13 | 47.12 | 47.97 | 62.86 (@10) |
| `hotpotqa_2026_06_03_14_56_llmlingua-rerank-top5/hotpotqa_2026_06_03_16_11_llmlingua-rerank-top5-generate` | 7405 | 是 | 33.71 | 45.24 | 38.33 | 47.71 | 46.00 | 64.21 (@5) |
| `hotpotqa_2026_05_30_17_15_native-rerank-top20-generate` | 7405 | 是 | 33.38 | 44.82 | 41.93 | 45.96 | 48.98 | 68.20 (@10) |
| `hotpotqa_2026_05_30_12_40_native-no-rerank-derived-top10-generate` | 7405 | 否 | 33.54 | 44.80 | 39.93 | 46.34 | 47.03 | 57.61 (@5) |
| `hotpotqa_2026_06_06_19_55_efc-hotpotqa-full-v2` | 7405 | 否 | 33.65 | 44.50 | 40.59 | 45.81 | 47.22 | 63.94 (@6) |
| `hotpotqa_2026_05_30_13_13_native-no-rerank-derived-top15-generate` | 7405 | 否 | 33.26 | 44.46 | 40.85 | 45.90 | 47.77 | 57.61 (@5) |
| `hotpotqa_2026_05_29_16_02_native-no-rerank-top20-generate` | 7405 | 否 | 32.52 | 43.93 | 40.84 | 45.32 | 47.75 | 57.61 (@5) |
| `hotpotqa_2026_05_30_12_27_native-no-rerank-derived-top5-generate` | 7405 | 否 | 31.74 | 42.36 | 38.15 | 43.71 | 44.71 | 57.61 (@5) |
| `hotpotqa_2026_06_03_14_46_selective-context-rerank-top5/hotpotqa_2026_06_03_15_21_selective-context-rerank-top5-generate` | 7405 | 是 | 30.87 | 41.64 | 35.25 | 44.14 | 42.54 | 64.21 (@5) |
| `hotpotqa_2026_06_06_17_26_efc-hotpotqa-full` | 7405 | 否 | 26.12 | 34.12 | 29.86 | 35.61 | 35.03 | 55.14 (@5) |
| `hotpotqa_2026_05_31_14_41_native-zero-shot-hotpotqa-dev` | 7405 | 是 | 17.65 | 26.54 | 23.24 | 27.20 | 29.64 | - |

## 5. 全量 QD 融合与答案模板实验

| 实验 | 样本数 | Reranker | EM | F1 | Acc | Precision | Recall | Retrieval Recall |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|
| `qd-fusion-w0p25-top5-pred-fusion` | 7405 | 是 | 37.57 | 48.65 | 42.27 | 50.97 | 49.16 | 64.89 (@5) |
| `qd-fusion-w0p25-top5` | 7405 | 是 | 36.42 | 48.02 | 43.25 | 49.58 | 50.33 | 64.89 (@5) |
| `qd-fusion-w0p1-top5` | 7405 | 是 | 36.38 | 47.91 | 43.23 | 49.48 | 50.22 | 65.20 (@5) |
| `qd-fusion-w0p0-top5-strict` | 7405 | 是 | 37.07 | 47.82 | 41.40 | 50.25 | 48.35 | 65.10 (@5) |
| `qd-fusion-w0p25-top5-strict` | 7405 | 是 | 37.07 | 47.81 | 41.50 | 50.20 | 48.33 | 64.89 (@5) |
| `qd-fusion-w0p0-top5` | 7405 | 是 | 36.22 | 47.78 | 43.11 | 49.33 | 50.10 | 65.10 (@5) |
| `qd-fusion-w0p1-top5-strict` | 7405 | 是 | 36.91 | 47.75 | 41.46 | 50.11 | 48.37 | 65.20 (@5) |
| `qd-fusion-w0p5-top5` | 7405 | 是 | 35.85 | 47.52 | 42.54 | 49.18 | 49.75 | 64.21 (@5) |
| `qd-fusion-w0p5-top5-strict` | 7405 | 是 | 36.37 | 47.30 | 40.86 | 49.69 | 47.84 | 64.21 (@5) |
| `qd-fusion-w0p75-top5-strict` | 7405 | 是 | 36.15 | 46.79 | 40.27 | 49.15 | 47.26 | 63.12 (@5) |
| `qd-fusion-w0p0-top5-answer-only` | 7405 | 是 | 33.34 | 46.72 | 43.89 | 47.45 | 51.00 | 65.10 (@5) |
| `qd-fusion-w0p75-top5` | 7405 | 是 | 35.25 | 46.65 | 41.80 | 48.24 | 48.90 | 63.12 (@5) |
| `qd-fusion-w0p25-top5-exact` | 7405 | 是 | 34.22 | 46.27 | 42.31 | 47.38 | 49.19 | 64.89 (@5) |
| `qd-fusion-w1p0-top5` | 7405 | 是 | 34.73 | 45.97 | 41.20 | 47.54 | 48.19 | 61.89 (@5) |
| `qd-fusion-w1p0-top5-strict` | 7405 | 是 | 35.41 | 45.95 | 39.59 | 48.26 | 46.47 | 61.89 (@5) |

这些实验的共同父目录为：

`racp/output/hotpotqa_2026_06_02_19_47_full-qd-qdrerank-title-dedup-top5/`

## 6. 1000 条实验与 Smoke Test

这些结果用于开发和消融，不应与 7,405 条全量结果直接比较。

| 实验 | 样本数 | Reranker | EM | F1 | Acc | Precision | Recall | Retrieval Recall |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|
| `hotpotqa_2026_06_01_14_10_rs-mhr-smoke` | 1 | 是 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 0.00 (@5) |
| `hotpotqa_2026_06_07_14_30_efc-full-vllm-smoke20-v2` | 20 | 否 | 55.00 | 63.51 | 60.00 | 65.83 | 65.50 | 60.00 (@6) |
| `hotpotqa_2026_06_08_13_08_efc-static-bridge-v1-smoke200` | 200 | 否 | 39.00 | 50.67 | 47.00 | 51.55 | 54.64 | 66.50 (@6) |
| `hotpotqa_2026_06_07_12_31_efc-iterretgen-seed4-v12-smoke200` | 200 | 否 | 38.50 | 50.17 | 46.50 | 51.06 | 54.14 | 67.00 (@6) |
| `hotpotqa_2026_06_08_15_06_efc-main-v2-router-only-smoke200` | 200 | 否 | 37.00 | 49.85 | 45.50 | 51.14 | 53.73 | 67.00 (@6) |
| `hotpotqa_2026_06_07_12_28_efc-iterretgen-answerfix-v11-smoke200` | 200 | 否 | 39.00 | 49.75 | 47.50 | 50.46 | 53.67 | 66.50 (@6) |
| `hotpotqa_2026_06_06_19_40_efc-router-v6-llm-query-smoke200` | 200 | 否 | 36.00 | 47.01 | 44.00 | 47.64 | 50.69 | 64.00 (@6) |
| `hotpotqa_2026_06_07_12_18_efc-iterretgen-final-v10-smoke200` | 200 | 否 | 35.50 | 46.83 | 41.50 | 48.08 | 48.69 | 66.50 (@6) |
| `hotpotqa_2026_06_07_12_02_efc-iterretgen-probe-query-v8-smoke200` | 200 | 否 | 34.50 | 46.57 | 40.50 | 48.27 | 48.07 | 68.50 (@6) |
| `hotpotqa_2026_06_07_11_44_efc-iterretgen-main-v7-smoke200` | 200 | 否 | 35.00 | 46.55 | 40.50 | 48.00 | 47.98 | 66.50 (@6) |
| `hotpotqa_2026_06_08_14_53_efc-main-v2-router-gen2-smoke200` | 200 | 否 | 35.00 | 46.20 | 43.50 | 46.85 | 50.16 | 68.00 (@6) |
| `hotpotqa_2026_06_06_19_34_efc-router-v5-baseline-final-smoke200` | 200 | 否 | 33.50 | 45.48 | 42.50 | 46.10 | 49.24 | 62.50 (@6) |
| `hotpotqa_2026_06_07_12_09_efc-iterretgen-probe-query-v9-smoke200` | 200 | 否 | 34.50 | 45.29 | 40.00 | 46.35 | 46.62 | 64.00 (@6) |
| `hotpotqa_2026_06_07_11_55_efc-iterretgen-raw-y1-v7-smoke200` | 200 | 否 | 32.50 | 44.31 | 38.50 | 46.10 | 45.86 | 65.50 (@6) |
| `hotpotqa_2026_05_27_14_50_racp-decomp-gapk-mmr-smoke` | 50 | 是 | 32.00 | 42.36 | 48.00 | 40.02 | 54.00 | 62.00 (@5) |
| `hotpotqa_2026_06_06_19_24_efc-router-v3-smoke200` | 200 | 否 | 29.50 | 40.50 | 35.00 | 42.22 | 41.82 | 62.50 (@6) |
| `hotpotqa_2026_06_06_19_30_efc-router-v4-probe-final-smoke200` | 200 | 否 | 29.00 | 39.00 | 34.50 | 39.96 | 40.60 | 62.50 (@6) |
| `hotpotqa_2026_05_27_15_12_racp-decomp-gapk-mmr-reranker-1000` | 1000 | 是 | 26.40 | 38.65 | 38.30 | 39.27 | 45.95 | 61.80 (@5) |
| `hotpotqa_2026_05_27_19_47_racp-decomp-reranker-top8` | 1000 | 是 | 26.00 | 38.07 | 38.50 | 38.49 | 46.30 | 61.50 (@5) |
| `hotpotqa_2026_05_27_12_50_racp-gapk+mmr+reranker` | 1000 | 是 | 26.30 | 37.87 | 37.60 | 38.68 | 44.69 | 60.50 (@5) |
| `hotpotqa_2026_06_06_19_14_efc-router-v2-smoke200` | 200 | 否 | 29.00 | 37.84 | 34.00 | 39.09 | 39.02 | 61.00 (@6) |
| `hotpotqa_2026_05_22_14_58_racp` | 1000 | 是 | 26.50 | 37.81 | 37.30 | 38.63 | 44.32 | 60.40 (@5) |
| `hotpotqa_2026_05_27_17_42_racp-decomp-gapk-mmr-no-reranker-1000` | 1000 | 否 | 24.60 | 36.18 | 36.80 | 36.62 | 43.62 | 51.60 (@5) |
| `hotpotqa_2026_05_14_15_39_racp` | 1000 | 是 | 25.10 | 35.82 | 32.70 | 37.59 | 39.79 | 60.40 (@5) |
| `hotpotqa_2026_05_14_20_23_racp` | 1000 | 是 | 25.10 | 35.82 | 32.70 | 37.59 | 39.79 | 60.40 (@5) |
| `hotpotqa_2026_05_27_13_09_racp` | 1000 | 否 | 23.10 | 34.11 | 33.80 | 34.70 | 40.55 | 52.90 (@5) |
| `hotpotqa_2026_06_06_13_00_efc-smoke-50-final2` | 50 | 否 | 26.00 | 33.70 | 30.00 | 33.83 | 34.67 | 52.00 (@5) |
| `hotpotqa_2026_06_06_12_40_efc-smoke-1d` | 1 | 否 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 (@5) |
| `hotpotqa_2026_06_06_12_40_efc-smoke-1d/final-prompt-v2` | 1 | 否 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 (@5) |

## 7. 最佳 EFC 详细统计

实验目录：

`racp/output/hotpotqa_2026_06_08_13_17_efc-static-bridge-v1-full`

### 7.1 Router 决策

| Router 决策 | 数量 | 比例 |
|---|---:|---:|
| direct | 949 | 12.82% |
| generation_guided | 5990 | 80.89% |
| static_qd | 466 | 6.29% |

### 7.2 实际执行路线

| 实际路线 | 数量 | 比例 | EM | F1 |
|---|---:|---:|---:|---:|
| direct | 955 | 12.90% | 47.12 | 59.30 |
| generation_guided | 5990 | 80.89% | 34.71 | 46.11 |
| static_qd | 460 | 6.21% | 22.83 | 28.09 |

Router 决策与实际路线的差异来自 planner 失败后的 fallback。

### 7.3 Missing-query 策略

| 策略 | 数量 | 比例 |
|---|---:|---:|
| none | 955 | 12.90% |
| llm_missing_hop | 5860 | 79.14% |
| static_qd_bridge_evidence | 375 | 5.06% |
| heuristic_repair | 130 | 1.76% |
| static_qd | 85 | 1.15% |

### 7.4 成本与证据质量

| 指标 | 数值 |
|---|---:|
| 平均 LLM 调用次数 | 2.8718 |
| 平均检索调用次数 | 1.9332 |
| 平均候选池文档数 | 25.5246 |
| 平均最终文档数 | 6.0000 |
| Planner 失败率 | 1.84% |
| Planner heuristic repair 比例 | 1.76% |
| Role coverage | 78.38% |
| 最终标题唯一率 | 99.74% |
| Support title recall@6 | 65.44% |
| 两个支持标题同时命中 | 47.43% |

Planner 失败原因：

| 原因 | 数量 |
|---|---:|
| missing-hop query 与原问题过于相似 | 104 |
| missing-hop 输出为空或无效 | 26 |
| static-QD bridge 输出为空或无效 | 6 |

## 8. 实验环境

### 8.1 硬件

| 项目 | 配置 |
|---|---|
| 服务器架构 | x86_64 |
| CPU | 2 × Intel Xeon Gold 6430 |
| CPU 核心/线程 | 64 物理核心，128 逻辑线程 |
| 系统内存 | 1.0 TiB |
| GPU | 5 × NVIDIA GeForce RTX 4090 |
| 单卡显存 | 24,564 MiB |
| NVIDIA Driver | 550.120 |
| 主要实验用卡 | 单张 RTX 4090，最佳 EFC 配置记录为 GPU 3 |

### 8.2 软件

| 项目 | 版本 |
|---|---|
| 操作系统 | Ubuntu 22.04.5 LTS |
| Linux Kernel | 6.8.0-57-generic |
| Python | 3.10.20 |
| PyTorch | 2.10.0+cu128 |
| PyTorch CUDA Runtime | 12.8 |
| cuDNN | 9.1.0.2 |
| Transformers | 4.57.6 |
| vLLM | 0.19.0 |
| FAISS | 1.8.0 |
| NumPy | 1.26.4 |
| PyYAML | 6.0.3 |
| tqdm | 4.67.3 |
| FlashRAG | 本地源码版本 |
| 报告生成时 Git commit | `58dcbe616eab02f7d35dbeb8cb3e14121fd9b03e` |
| Conda 环境 | `/home/guanjunjie/.conda/envs/flashrag` |

### 8.3 主要模型与数据配置

| 项目 | 配置 |
|---|---|
| 最终生成模型 | Llama-3.1-8B-Instruct |
| EFC Probe 模型 | Llama-3.1-8B-Instruct |
| EFC Planner 模型 | Llama-3.1-8B-Instruct |
| QD Planner | 部分实验为 Llama-3.2-3B-Instruct |
| Dense Retriever | bge-large-en-v1.5 |
| Cross-encoder Reranker | bge-reranker-large |
| 检索语料 | wiki18_100w |
| Dense Index | bge-large-en-v1.5 Flat FAISS index |
| FAISS 运行设备 | CPU，`faiss_gpu=false` |
| Retriever 精度 | FP16 |
| Retriever batch size | 1024 |
| Generator framework | vLLM |
| EFC Planner framework | Hugging Face Transformers |
| Generator batch size | 2 |
| 最大生成输入长度 | 4096 |
| 默认生成方式 | Greedy，`do_sample=false` |
| 随机种子 | 2024 |

最佳 EFC 的主要参数：

| 参数 | 数值 |
|---|---:|
| initial_topk | 20 |
| probe_topk | 5 |
| final_topk | 6 |
| qd_num | 2 |
| qd_topk | 5 |
| missing_query_num | 1 |
| generation-guided topk | 10 |
| RRF k | 60 |
| original_seed_count | 4 |
| planner batch chunk | 32 |
| planner HF inference batch | 8 |

## 9. 可比性说明

1. 主实验的最终文档数量存在 top5、top6、top10、top15 和 top20 差异。
2. Retrieval Recall 使用了不同的 `@K`，因此表中保留原始 K，不进行强行统一排序。
3. QD 部分实验使用 3B planner，EFC 使用 8B planner。
4. EFC 和无 reranker IterRetGen 不使用 cross-encoder；Native rerank、MHR、QD、IRCoT 和 rerank IterRetGen 使用了 `bge-reranker-large`。
5. Strict、answer-only、exact 和 prediction-fusion 改变了答案生成或后处理方式，不应只归因于检索质量。
6. 1000 条和 smoke 结果只用于开发决策，不可作为全量方法排名。
