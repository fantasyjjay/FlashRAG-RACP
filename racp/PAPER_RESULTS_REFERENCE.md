# RACP / EFC-RAG 已完成实验结果与论文写作参考

更新时间：2026-08-13

> 本文是论文写作参考摘要，不替代唯一正式实验台账
> [`PAPER_EXPERIMENT_PLAN.md`](PAPER_EXPERIMENT_PLAN.md)。本文只收录台账中状态为
> `FINAL` 的全量实验，不收录 smoke、失败运行、暂停的 TRACE、运行中的消融或暂停的
> Adaptive-RAG。Modified Adaptive-k 是独立的largest-gap对照，不是官方Adaptive-RAG。
> 当前论文数值以最新版 Table 5 为准；对应 FINAL 目录存在时再用 `metric_score.txt` 和
> `config.yaml` 完成溯源。NQ 新增三项及 EFC-RAG 新 F1 的产物映射仍待补齐。

## 1. 实验范围与完成情况

当前 Table 5 共有 32 项主结果；其中 29 项已有产物审计，新增的 3 项 NQ 结果仍待补 provenance：

| 方法 | HotpotQA | 2WikiMultiHopQA | MuSiQue | NQ |
|---|---|---|---|---|
| No-RAG | FINAL | FINAL | FINAL | FINAL |
| Standard RAG | FINAL | FINAL | FINAL | FINAL |
| IterRetGen | FINAL | FINAL | FINAL | FINAL |
| Full-QD | FINAL | FINAL | FINAL | FINAL（产物待补） |
| IRCoT | FINAL | FINAL | FINAL | FINAL（产物待补） |
| EFC-RAG | FINAL | FINAL | FINAL | FINAL |
| FLARE | FINAL | FINAL | FINAL | FINAL |
| Modified Adaptive-k | FINAL | FINAL | FINAL | FINAL（产物待补） |

暂不进入本文结果表：

- TRACE：HotpotQA、2Wiki 已暂停，MuSiQue 运行失败，三者均无 FINAL。
- Adaptive-RAG：缺少可审计的官方 classifier checkpoint，当前暂停。
- HotpotQA W11的title-dedup-only、controlled final top5、matched maximum retrieval
  budget和w/o redundancy已全部通过全量验收并登记FINAL。
- 最新 Table 5 已纳入 NQ Full-QD、IRCoT 和 Modified Adaptive-k；表格数值为正式结果，
  但实验 ID、FINAL 目录、commit、配置、逐样本预测和成本仍待补齐。

## 2. 数据集与评测范围

| 数据集 | FlashRAG 名称 | Split | 样本数 | 用途 |
|---|---|---|---:|---|
| HotpotQA | `hotpotqa` | dev | 7,405 | 主要多跳数据集、路由和消融分析 |
| 2WikiMultiHopQA | `2wikimultihopqa` | dev | 12,576 | 跨实体、跨文档多跳泛化 |
| MuSiQue | `musique` | dev | 2,417 | 组合性和依赖式多跳泛化 |
| Natural Questions | `nq` | test | 3,610 | 单跳泛化与过度检索检查 |

所有结果均使用完整目标 split，`test_sample_num: null`，没有抽样。历史上使用不同
retriever、reranker、生成模型或 1,000 条子集的结果不在本文中。

## 3. 统一测试条件

### 3.1 模型与检索环境

| 项目 | 设置 |
|---|---|
| Generator | `Llama-3.1-8B-Instruct` |
| 默认生成框架 | vLLM；TRACE 例外使用 HF，但 TRACE 尚未写入本文结果 |
| Generator 最大输入长度 | 4,096 tokens |
| 普通答案最大生成长度 | 32 tokens |
| Sampling | 关闭；`do_sample=false`，temperature=0 |
| Retriever | `bge-large-en-v1.5` |
| Corpus | `wiki18_100w` Wikipedia corpus |
| Index | BGE Flat index |
| Retriever query 最大长度 | 128 tokens |
| Retriever batch size | 1,024 |
| Retriever precision | FP16 |
| Reranker | 所有执行检索的方法均关闭 reranker |
| Refiner | No-RAG、Standard、IterRetGen、Full-QD、IRCoT、EFC、FLARE 均为 `null` |
| Seed | 2024 |
| 主要指标 | EM、F1 |
| 辅助指标 | Acc、Precision、Recall、Retrieval Recall |

No-RAG 使用 `NoOpRetriever`，不执行检索或 rerank。MuSiQue No-RAG 的基础配置中曾保留
`use_reranker: true`，但因根本没有调用 retriever/reranker，不影响闭卷结果。缓存只用于
避免重复计算相同 query；正式评测仍使用相同文档、dense score 和 top-k 口径。

### 3.2 硬件与软件

| 项目 | 环境 |
|---|---|
| GPU | 5 × NVIDIA GeForce RTX 4090，单卡 24,564 MiB |
| CPU | 2 × Intel Xeon Gold 6430，64 物理核 / 128 线程 |
| 内存 | 1.0 TiB |
| Python | Conda 环境 `flashrag` |
| PyTorch | 2.10.0+cu128 |
| CUDA runtime | 12.8 |
| Transformers | 4.57.6 |
| vLLM | 0.19.0 |
| NVIDIA driver | 550.120 |

大多数实验为单卡生成。2Wiki IRCoT 使用 4 卡 tensor parallel；这只改变执行方式，不改变
模型、prompt、top-k 或生成参数。

### 3.3 代码基线与复现说明

当前主要基线为 Git commit `3b4d780` 加未提交的兼容性 diff。少数较早但已审计的 2Wiki
运行分别来自 `e6b4456` 或 `3e62049`。兼容性修改主要修复 FLARE token logprob、动态
cache-miss 返回维度和评测字段，不应在论文中描述为 EFC 算法改进。

Modified Adaptive-k 三项运行单独锁定在 clean commit
`5f13649a4e5dcdf4ccbdcf19f24bd173db46a69b`；每个FINAL目录的
`adaptive_k_summary.json` 保存实际命令、Git状态、K分布、缓存来源和指标。

因此本文结果属于“统一模型、统一 retriever、无 reranker 条件下的受控比较”，不是对各篇
原论文 headline number 的逐项复刻。提交论文或发布代码前，应将当前工作区固定为一个明确
commit，并为所有 FINAL 目录保留对应配置与命令。

## 4. 方法与锁定参数

### 4.1 名称与代码入口

必须区分以下两个容易混淆的入口：

- `zero-shot` / `naive_run()`：No-RAG，闭卷生成，不检索。
- `naive` CLI / `SequentialPipeline.run()`：Standard RAG，执行一次标准检索再生成。

论文中统一使用 No-RAG 和 Standard RAG，不把代码中的 `naive` 写成 No-RAG。

### 4.2 各方法参数

| 方法 | 检索与最终上下文 | 轮次 / 生成 | 关键参数 | Retrieval Recall 口径 |
|---|---|---|---|---|
| No-RAG | 无检索、无 context | 1 次答案生成 | `max_tokens=32` | N/A |
| Standard RAG | 原问题 BGE top10；final K=10 | 1 次检索 + 1 次答案生成 | `retrieval_topk=10` | @10 |
| IterRetGen | 每轮 BGE top5 | 固定 3 轮检索—生成 | `iter_num=3`，每轮 `max_tokens=32` | 配置键为 @10，但实际只评估第三轮最终5篇 |
| Full-QD | 原问题 top20；2 条子问题各 top5；title-dedup 选 final top10 | 1 次 planner + 子问题检索 + 1 次答案生成 | `subquery_num=2`、`subquery_topk=5`、planner batch=16、planner max tokens=96 | @10 |
| IRCoT | 每轮 top5；两轮累积去重，最终最多10篇 | 2 轮 thought；未结束样本追加 final answer | `max_iter=2`、固定 demonstration、thought/final各32 tokens | 配置 @10；实际最终可少于10篇 |
| EFC-RAG | original top20、probe top5；扩展检索；final top10 | probe + 路由相关 planner + final answer | 见下节 | @10 |
| FLARE | 低置信度句子触发 BGE top5；未触发则最终列表为空 | 最多5轮 | threshold=0.2、look-ahead=64、总生成上限256、`max_iter=5` | @5；未触发样本按零召回计入 |
| Modified Adaptive-k | 原问题BGE top20；按raw dense score最大相邻gap选择final K=6--8 | 1次检索 + 1次答案生成，无planner | `selection_method=gap`、`search_ratio=0.9`、`buffer=5`、`max_k=8` | variable K；配置键虽名为@10，实际评估全部6--8篇，不可直接当固定Recall@10 |

### 4.3 EFC-RAG 完整参数

EFC-RAG 的正式流程为：

```text
question
  -> original-query BGE top20
  -> 使用 top5 生成 probe answer
  -> router: direct / generation_guided / static_qd
  -> missing-hop top10，或两条 QD query 各 top5
  -> RRF 与角色/来源加权融合
  -> soft title dedup，选择 final top10
  -> Llama-3.1-8B 生成最终短答案
```

| 参数 | 锁定值 |
|---|---:|
| `force_route` | `auto` |
| `initial_topk` | 20 |
| `probe_topk` | 5 |
| `final_topk` | 10 |
| `missing_query_num` | 1 |
| `missing_query_mode` | `llm` |
| `gen_topk` | 10 |
| `qd_num` | 2 |
| `qd_topk` | 5 |
| `rrf_k` | 60 |
| `rrf_weight` | 1.00 |
| `role_weight` | 0.30 |
| `title_weight` | 0.02 |
| `source_weight` | 0.05 |
| `redundancy_weight` | 0.01 |
| `title_dedup_soft` | true |
| `max_same_title` | 2 |
| `original_seed_count` | 4 |
| `static_bridge_evidence_topk` | 5 |
| `static_bridge_original_count` | 4 |
| `static_bridge_qd_count` | 2 |
| Probe max tokens | 128 |
| Planner max tokens | 96 |
| Planner batch / inference batch | 32 / 8 |
| Final answer max tokens | 32 |
| Planner GPU memory utilization | 0.75 |
| Final generator GPU memory utilization | 0.85 |

自动路由逻辑的核心规则：comparison 两侧实体均被 top5 覆盖时走 `direct`；probe 暴露新
桥接实体时走 `generation_guided`；probe 无效或不确定且没有桥接实体时走 `static_qd`；
单跳问题或完整 probe 没有新的检索桥接点时走 `direct`。

## 5. 已完成主结果

所有数值均为百分数。粗体仅标记当前已完成方法中的该列最优值，TRACE 尚未完成，因此最终
论文表的粗体可能变化。

### 5.1 HotpotQA dev（7,405）

| 方法 | EM | F1 | Acc | Precision | Recall | Retrieval Recall |
|---|---:|---:|---:|---:|---:|---:|
| No-RAG | 17.68 | 26.54 | 23.27 | 27.22 | 29.63 | N/A |
| Standard RAG | 33.57 | 44.82 | 39.93 | 46.36 | 47.04 | 65.12 (@10) |
| IterRetGen | 34.67 | 45.66 | 41.13 | 47.12 | 47.97 | 62.86（配置@10；实际最终5篇） |
| Full-QD | 33.67 | 44.59 | 40.23 | 46.16 | 46.90 | 65.04 (@10) |
| IRCoT | 36.26 | 47.50 | 39.57 | **51.40** | 46.93 | 64.77 (@10) |
| EFC-RAG | **37.85** | **49.21** | **42.35** | 48.89 | **49.43** | **71.51 (@10)** |
| FLARE | 16.04 | 23.26 | 21.27 | 23.88 | 26.54 | 1.93 (@5) |
| Modified Adaptive-k | 31.99 | 43.07 | 38.99 | 44.48 | 45.97 | 60.93 (variable K=6--8) |

EFC-RAG 相比 Standard RAG 高 4.28 EM、4.39 F1；相比 IterRetGen 高 3.18/3.55，
相比 IRCoT 高 1.59/1.71。Retrieval Recall 与其他辅助指标来自旧轨迹，不能与新版
EM/F1 拼成同一次运行。

### 5.2 2WikiMultiHopQA dev（12,576）

| 方法 | EM | F1 | Acc | Precision | Recall | Retrieval Recall |
|---|---:|---:|---:|---:|---:|---:|
| No-RAG | 17.22 | 26.51 | 30.93 | 25.01 | 35.28 | N/A |
| Standard RAG | 16.56 | 25.60 | 25.37 | 25.32 | 30.59 | 46.20 (@10) |
| IterRetGen | 16.90 | 26.59 | 30.16 | 25.58 | 34.69 | 42.48（配置@10；实际最终5篇） |
| Full-QD | 18.23 | 26.84 | 25.56 | 26.69 | 30.86 | 49.57 (@10) |
| IRCoT | 33.07 | **39.39** | **35.46** | **40.99** | **39.41** | 50.15 (@10) |
| EFC-RAG | **34.35** | 39.37 | 28.92 | 29.04 | 33.98 | **58.17 (@10)** |
| FLARE | 9.37 | 20.45 | 32.11 | 17.51 | 36.22 | 2.27 (@5) |
| Modified Adaptive-k | 15.18 | 25.14 | 27.87 | 24.25 | 32.74 | 42.22 (variable K=6--8) |

EFC-RAG 相比 Standard RAG 高 17.79/13.77，相比 IterRetGen 高 17.45/12.78，
相比 Full-QD 高 16.12/12.53（EM/F1）。相较 IRCoT，EM 高 1.28、F1 低 0.02。

2Wiki 中 No-RAG 略高于 Standard RAG，表明单次检索在该设置下可能引入噪声。论文不能
宣称 EFC 在 2Wiki 的两个指标都超过 IRCoT；可准确表述为 EM 较高、F1 低 0.02。

### 5.3 MuSiQue dev（2,417）

| 方法 | EM | F1 | Acc | Precision | Recall | Retrieval Recall |
|---|---:|---:|---:|---:|---:|---:|
| No-RAG | 3.56 | 9.94 | 6.37 | 10.46 | 11.73 | N/A |
| Standard RAG | 6.45 | 14.11 | 8.98 | 14.83 | 15.40 | 32.02 (@10) |
| IterRetGen | 8.07 | 15.22 | 10.47 | 15.92 | 16.63 | 29.58（配置@10；实际最终5篇） |
| Full-QD | 7.61 | 15.42 | 10.30 | 16.17 | 16.60 | 33.88 (@10) |
| IRCoT | 9.38 | 17.42 | 12.00 | **19.40** | 17.40 | 30.33 (@10) |
| EFC-RAG | **11.93** | **19.65** | **13.24** | 18.31 | **19.21** | **42.08 (@10)** |
| FLARE | 2.15 | 5.42 | 3.89 | 5.75 | 6.68 | 1.32 (@5) |
| Modified Adaptive-k | 6.16 | 13.03 | 8.69 | 13.62 | 14.70 | 28.13 (variable K=6--8) |

EFC-RAG 相比 Standard RAG 高 5.48/5.54，相比 IterRetGen 高 3.86/4.43，
相比 Full-QD 高 4.32/4.23，相比 IRCoT 高 2.55/2.23（EM/F1）。

### 5.3.1 Modified Adaptive-k 三数据集对照

三个FINAL使用相同设置：原问题BGE top20候选；在前90%排序候选内寻找最大相邻dense
score gap；`buffer=5`、`max_k=8`，因此final K为6--8；seed=2024；
Llama-3.1-8B-Instruct；`max_tokens=32`；无sampling、reranker、refiner或planner。每条样本
只有1次检索和1次答案生成。prepare复用只读top20 cache，cache中的文档和raw dense score
原样参与选择，只影响速度。运行均为单卡、TP=1、clean commit `5f13649`。
代码模板与落盘prompt核对确认，final-answer prompt和短答案要求与Standard RAG一致；
主要受控变量是final context从固定top10变为按score gap选择的6--8篇。

| 数据集 | 平均K | K=6 / 7 / 8 | 相对top10文档减少 | ΔEM / ΔF1 vs Standard | 活跃耗时 |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 6.7837 | 3,541 / 1,925 / 1,939 | 32.16% | -1.58 / -1.75 | 15:53 |
| 2Wiki | 6.7706 | 6,193 / 3,075 / 3,308 | 32.29% | -1.38 / -0.46 | 28:25 |
| MuSiQue | 6.8324 | 1,158 / 506 / 753 | 31.68% | -0.29 / -1.08 | 5:29 |

三个数据集都以约32%的最终上下文压缩换来小幅答案质量下降。表中的检索指标必须称为
variable-K Retrieval Recall：现有evaluator字段虽然名为`retrieval_recall_top10`，实际因
最大K=8而评估全部6--8篇，不能无注释地视为固定Recall@10。

### 5.4 Natural Questions test（3,610）

| 方法 | EM | F1 | Acc | Precision | Recall | Retrieval Recall |
|---|---:|---:|---:|---:|---:|---:|
| No-RAG | 21.63 | 32.42 | 35.54 | 31.17 | 42.33 | N/A |
| Standard RAG | 36.95 | 49.15 | **54.49** | 47.63 | **59.88** | **82.74 (@10)** |
| IterRetGen | 37.53 | 47.38 | 53.05 | **48.12** | 58.12 | 75.62（旧产物字段；实际最终5篇） |
| Full-QD | 34.88 | 47.07 | --- | --- | --- | --- |
| IRCoT | **39.31** | 49.53 | --- | --- | --- | --- |
| Modified Adaptive-k | 36.43 | 48.90 | --- | --- | --- | --- |
| EFC-RAG | 38.32 | **50.44** | 52.44 | 45.87 | 57.94 | 79.86（旧产物字段） |
| FLARE | 21.22 | 30.85 | 33.43 | 29.95 | 39.53 | 3.21 (@5) |

EFC-RAG 比 Standard RAG 高 1.37/1.29，比 IterRetGen 高 0.79/3.06，比 Full-QD
高 3.44/3.37（EM/F1）。相较 IRCoT，EFC 的 EM 低 0.99、F1 高 0.91。
旧 Retrieval Recall 与辅助指标不能与新版 EM/F1 拼成同一次运行。
FLARE 还略低于 No-RAG，说明该主动检索策略不适合当前单跳设置。

## 6. EFC-RAG 路由与效率

| 数据集 | Avg. LLM calls | Avg. retrieval calls | Avg. candidate docs | Final docs | Direct | Static-QD | Generation-guided |
|---|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 2.8718 | 1.9332 | 25.5246 | 10 | 955（12.90%） | 460（6.21%） | 5,990（80.89%） |
| 2Wiki | 2.9051 | 2.0161 | 26.9002 | 10 | 1,207（9.60%） | 1,410（11.21%） | 9,959（79.19%） |
| MuSiQue | 2.9876 | 2.0426 | 27.6719 | 10 | 35（1.45%） | 138（5.71%） | 2,244（92.84%） |
| NQ | 2.1008 | 1.1042 | 20.5006 | 10 | 3,246（89.92%） | 12（0.33%） | 352（9.75%） |

该分布说明 router 能明显区分数据集：三个多跳数据集主要进入 generation-guided，而 NQ
有 89.92% 进入 direct。需要谨慎的是，NQ direct 比例高并不等于零额外开销；EFC direct
仍包含 probe 和 final answer，平均 LLM 调用为 2.1008，高于 Standard RAG 的一次答案生成。

2Wiki 的额外诊断：EFC support-title recall 为 48.90%，IRCoT 为 45.27%；EFC 两个
support title 同时命中率为 28.96%，IRCoT 为 23.54%。然而 IRCoT 的 EM/F1 仍明显更高，
说明主要瓶颈在证据到答案的推理与答案格式，而不是单纯没有检到证据。

## 7. FLARE 触发率、口径与耗时

| 数据集 | 触发样本 / 总样本 | 动态检索次数 | 未触发样本 | 总耗时 | Retrieval Recall@5 |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 1,353 / 7,405（18.27%） | 1,406 | 6,052 | 4:49:24 | 1.93 |
| 2Wiki | 1,524 / 12,576（12.12%） | 1,610 | 11,052 | 6:12:41 | 2.27 |
| MuSiQue | 327 / 2,417（13.53%） | 340 | 2,090 | 1:23:27 | 1.32 |
| NQ | 511 / 3,610（14.16%） | 573 | 3,099 | 2:17:38 | 3.21 |

FLARE 未触发检索时，`retrieval_result` 是合法空列表，其 Retrieval Recall 按零计入平均值。
因此低 Recall 不是评测字段缺失。FLARE 的 Recall@5 也不能与其他方法的 Recall@10 直接
作等预算比较。

## 8. 已完成消融结果

| 数据集 | ID | 变体 | EM | F1 | Acc | Precision | Recall | Retrieval Recall（实际K） |
|---|---|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | `HP-AB-01` | 完整EFC | 35.85 | 47.21 | 42.35 | 48.89 | 49.43 | 71.51 (@10) |
| HotpotQA | `HP-AB-02` | direct-only | 33.59 | 44.58 | 39.91 | 46.14 | 46.83 | 65.63 |
| HotpotQA | `HP-AB-03` | generation-guided-only | 35.61 | 46.95 | 42.07 | 48.65 | 49.12 | 71.60 (@10) |
| HotpotQA | `HP-AB-04` | w/o static-QD | 35.76 | 47.09 | 42.15 | 48.80 | 49.22 | 71.49 |
| HotpotQA | `HP-AB-05` | Full-QD | 33.67 | 44.59 | 40.23 | 46.16 | 46.90 | 65.04 |
| HotpotQA | `HP-AB-06` | w/o role weight | 35.77 | 47.02 | 42.20 | 48.67 | 49.24 | 71.05 |
| HotpotQA | `HP-AB-07` | w/o original seed reservation | 35.80 | 47.09 | 42.36 | 48.76 | 49.37 | 71.41 |
| HotpotQA | `HP-AB-08` | w/o source weight | 35.85 | 47.21 | 42.35 | 48.89 | 49.43 | 71.51 (@10) |
| HotpotQA | `HP-AB-09` | RRF-only selector | 33.30 | 43.67 | 39.30 | 45.19 | 46.04 | 67.13 (@10) |
| HotpotQA | `HP-AB-10` | title-dedup-only selector | 35.18 | 46.57 | 41.78 | 48.20 | 48.90 | 70.90 (@10) |
| HotpotQA | `HP-AB-11` | controlled final top5 | 35.53 | 46.63 | 41.82 | 48.21 | 48.71 | 66.31 (@5) |
| HotpotQA | `HP-AB-12` | matched maximum retrieval budget | **36.66** | **48.17** | **43.58** | **49.71** | **50.60** | 69.82 (@≤10) |
| HotpotQA | `HP-AB-13` | w/o redundancy penalty | 36.04 | 47.46 | 42.59 | 49.16 | 49.65 | **72.21 (@10)** |
| MuSiQue | `MU-AB-01` | 完整EFC | **9.93** | **17.65** | **13.24** | **18.31** | **19.21** | **42.08** |
| MuSiQue | `MU-AB-02` | generation-guided-only | 9.64 | 17.45 | 12.95 | 18.12 | 19.04 | 41.54 |
| MuSiQue | `MU-AB-03` | Full-QD | 7.61 | 15.42 | 10.30 | 16.17 | 16.60 | 33.88 |

表中未单独标注的HotpotQA/MuSiQue消融均为@10；`HP-AB-11`是受控@5，
`HP-AB-12`的实际final context为5--10篇。这里的Retrieval Recall仍是现有evaluator
的gold-answer string containment hit rate，不是supporting-title recall。

HotpotQA direct-only覆盖7,405/7,405条，全部走direct；平均LLM/检索调用为2.0000/1.0000，
平均候选池20篇、最终10篇，总耗时42:46。完整EFC的EM/F1/Retrieval Recall分别高
2.26/2.63/5.88个百分点。direct-only仍包含probe与final两次生成，不等同Standard RAG。

MuSiQue generation-guided-only覆盖2,417/2,417条，实际2,407条generation-guided、
10条direct fallback；平均LLM/检索调用3.0000/1.9959，平均候选池27.7245篇、最终10篇，
总耗时35:48。完整EFC的EM/F1/Retrieval Recall分别高0.29/0.20/0.54个百分点，说明
自适应router在MuSiQue上的收益为正但较小。

HotpotQA新增四项均覆盖7,405/7,405条。w/o static-QD、w/o role weight、w/o original
seed reservation相对完整EFC的F1分别下降0.12、0.19、0.12个百分点；w/o source weight
的六项汇总指标与完整EFC完全相同。w/o static-QD使用auto router，static样本转为direct，
其余三项保持与完整EFC相同的route分布和调用预算。

RRF-only同样覆盖7,405条，route和调用预算与完整EFC相同，但F1下降3.54个百分点，
support-title recall和双支持标题命中率分别下降20.66和28.28个百分点。结合标题唯一率
99.53%→70.54%，说明selector组合约束的重要价值来自title diversity。

title-dedup-only把RRF-only的EM/F1/answer-hit分别恢复1.88/2.90/3.77个百分点，但仍比
完整EFC低0.68/0.64/0.61个百分点。这说明title diversity解释了组合selector收益的大部分，
但role/source/original seed/static packing的剩余组合贡献仍存在，不能把全部收益只归因于
标题去重。该变体的route、调用预算和候选池与完整EFC相同，7,405条全部使用隔离的
`efc_rrf_title_dedup_only`选择器。

controlled final top5只改变final与评测K，EM/F1相对完整EFC仅下降0.32/0.58个百分点；
它不减少Probe、Planner或检索调用，只减少最终context。其66.31是Recall@5，不能与
其他消融的Recall@10直接作等预算排序。

matched maximum retrieval budget把initial/gen top-k均设为5并关闭static，严格满足最多
2次逻辑检索和10篇raw文档；实际平均检索1.8089次、最终文档7.8357篇。它比正式IRCoT
高0.41 EM、0.67 F1和5.05个answer-hit百分点，但只匹配最大检索预算，不匹配prompt、
推理机制或逐样本LLM成本，也不是单变量消融。论文中不得简写成“完全同预算”。

w/o redundancy只将权重0.01改为0，F1与answer-hit反而高0.25/0.70个百分点。因此当前
HotpotQA没有观察到冗余惩罚的正收益。由于相同context下仍有少量生成文本漂移，这一小幅
差值应配合paired检验表述为“未观察到收益”，不宣称移除惩罚能稳定提升。

## 9. 可用于论文的主要结论

1. **EFC 稳定超过较简单的多跳基线。** 在 HotpotQA、2Wiki、MuSiQue 三个多跳数据集
   上，EFC-RAG 的 EM/F1 均高于 Standard RAG、IterRetGen 和 Full-QD。
2. **与 IRCoT 的比较取决于数据集和指标。** EFC 在 HotpotQA、MuSiQue 的 EM/F1 均较高；
   2Wiki 为 +1.28/-0.02，NQ 为 -0.99/+0.91。论文不应宣称全面 SOTA。
3. **EFC 的旧轨迹显示较高证据覆盖。** 旧 Retrieval Recall@10 为 71.51、58.17 和 42.08，
   但不得与新版 EM/F1 作逐样本因果连接。
4. **固定分解不是最优主干。** Full-QD 在四个数据集均低于 EFC，支持把 QD 保留为
   probe 失败时的辅助分支，而不是所有问题的固定步骤。
5. **路由具有明显的数据集适应性。** generation-guided 在多跳数据集占 79%–93%，而
   NQ 的 direct 占 89.92%。
6. **单跳增益伴随额外成本。** NQ 上 EFC 的 EM/F1 比 Standard RAG 高 1.37/1.29，
   但平均需要约 2.10 次 LLM 调用，不支持单跳效率优势。
7. **FLARE 在当前统一环境中不具竞争力。** 四个数据集上的 EM/F1 和检索召回均较低，
   主要原因之一是多数样本没有触发动态检索。
8. **消融支持自适应扩展。** HotpotQA direct-only明显低于完整EFC；MuSiQue固定走
   generation-guided也略低于完整EFC，说明router带来稳定但数据集相关的收益。
9. **Modified Adaptive-k提供上下文压缩，而非质量提升。** largest-gap设置在三个多跳
   数据集将平均K降至约6.8，但F1相对Standard RAG分别下降1.75、0.46和1.08个百分点。

## 10. 建议的论文表述

可将主要结果概括为：

> 在统一的 Llama-3.1-8B-Instruct、BGE-large-en-v1.5 和无 reranker 设置下，EFC-RAG
> 在 HotpotQA、2WikiMultiHopQA 和 MuSiQue 上均超过 Standard RAG、原生 IterRetGen
> 与固定 Full-QD。相较 IterRetGen，EFC 的 EM/F1 分别高 3.18/3.55、17.45/12.78
> 和 3.86/4.43 个百分点。相较 IRCoT，EFC 在 HotpotQA 和 MuSiQue 的 EM/F1 均较高；
> 2Wiki 为 +1.28/-0.02，NQ 为 -0.99/+0.91。NQ 相较 Standard RAG 为 +1.37/+1.29，
> 但仍支付 Probe 调用成本。

论文中不应使用以下过强表述：

- “EFC 超过所有 RAG 基线”——2Wiki F1 和 NQ EM 低于 IRCoT。
- “EFC 在所有数据集和指标都是最优”——Table 5 中有两个指标居第二。
- “EFC direct 等于单次 Standard RAG 成本”——direct 仍执行 probe 和 final generation。
- “IterRetGen Retrieval Recall@10”而不解释——当前最终列表实际只有第三轮5篇。
- 把 FLARE Recall@5 与其他方法 Recall@10 当作相同预算直接排序。
- 把 Modified Adaptive-k 的variable-K recall写成固定Recall@10，或把它与需要classifier
  的Adaptive-RAG视为同一个方法。

## 11. FINAL 结果来源

### HotpotQA

| ID | 方法 | FINAL 目录 |
|---|---|---|
| HP-NR-01 | No-RAG | [`output/hotpotqa_2026_07_12_14_26_hotpotqa-zero-shot-full`](output/hotpotqa_2026_07_12_14_26_hotpotqa-zero-shot-full) |
| HP-NR-02 | Standard RAG | [`output/hotpotqa_2026_07_12_14_26_hotpotqa-standard-rag-no-rerank-top10-full`](output/hotpotqa_2026_07_12_14_26_hotpotqa-standard-rag-no-rerank-top10-full) |
| HP-NR-03 | IterRetGen | [`output/hotpotqa_2026_06_05_16_46_iterretgen-native-no-reranker-gpu3`](output/hotpotqa_2026_06_05_16_46_iterretgen-native-no-reranker-gpu3) |
| HP-NR-04 | Full-QD | [`output/hotpotqa_2026_07_10_09_27_full-qd-no-rerank-top10-full`](output/hotpotqa_2026_07_10_09_27_full-qd-no-rerank-top10-full) |
| HP-NR-05 | IRCoT | [`output/hotpotqa_2026_07_09_14_27_ircot-no-rerank-cache-full`](output/hotpotqa_2026_07_09_14_27_ircot-no-rerank-cache-full) |
| HP-NR-07 | EFC-RAG | [`output/hotpotqa_2026_07_09_15_52_efc-static-bridge-final-top10-full`](output/hotpotqa_2026_07_09_15_52_efc-static-bridge-final-top10-full) |
| HP-NR-08 | FLARE | [`output/hotpotqa_2026_07_13_09_22_hotpotqa-flare-no-rerank-top5-full-v2`](output/hotpotqa_2026_07_13_09_22_hotpotqa-flare-no-rerank-top5-full-v2) |
| HP-NR-10 | Modified Adaptive-k | [`output/hotpotqa_2026_07_16_17_35_hotpotqa-modified-adaptive-k-gap-b5-max8-no-rerank-full`](output/hotpotqa_2026_07_16_17_35_hotpotqa-modified-adaptive-k-gap-b5-max8-no-rerank-full) |
| HP-AB-02 | direct-only | [`output/hotpotqa_2026_07_15_10_01_hotpotqa-efc-force-direct-top10-full`](output/hotpotqa_2026_07_15_10_01_hotpotqa-efc-force-direct-top10-full) |
| HP-AB-03 | generation-guided-only | [`output/hotpotqa_2026_07_14_13_49_hotpotqa-efc-force-generation-guided-no-static-top10-full`](output/hotpotqa_2026_07_14_13_49_hotpotqa-efc-force-generation-guided-no-static-top10-full) |
| HP-AB-04 | w/o static-QD | [`output/hotpotqa_2026_07_15_11_05_hotpotqa-efc-auto-no-static-top10-full-v3-online-fill`](output/hotpotqa_2026_07_15_11_05_hotpotqa-efc-auto-no-static-top10-full-v3-online-fill) |
| HP-AB-06 | w/o role weight | [`output/hotpotqa_2026_07_15_10_16_hotpotqa-efc-no-role-weight-top10-full`](output/hotpotqa_2026_07_15_10_16_hotpotqa-efc-no-role-weight-top10-full) |
| HP-AB-07 | w/o original seed reservation | [`output/hotpotqa_2026_07_15_11_05_hotpotqa-efc-no-original-seed-top10-full`](output/hotpotqa_2026_07_15_11_05_hotpotqa-efc-no-original-seed-top10-full) |
| HP-AB-08 | w/o source weight | [`output/hotpotqa_2026_07_15_11_05_hotpotqa-efc-no-source-weight-top10-full`](output/hotpotqa_2026_07_15_11_05_hotpotqa-efc-no-source-weight-top10-full) |
| HP-AB-09 | RRF-only selector | [`output/hotpotqa_2026_07_15_12_46_hotpotqa-efc-rrf-only-top10-full`](output/hotpotqa_2026_07_15_12_46_hotpotqa-efc-rrf-only-top10-full) |
| HP-AB-10 | title-dedup-only selector | [`output/hotpotqa_2026_07_15_13_54_hotpotqa-efc-title-dedup-only-top10-full`](output/hotpotqa_2026_07_15_13_54_hotpotqa-efc-title-dedup-only-top10-full) |
| HP-AB-11 | controlled final top5 | [`output/hotpotqa_2026_07_15_13_54_hotpotqa-efc-final-top5-controlled-full`](output/hotpotqa_2026_07_15_13_54_hotpotqa-efc-final-top5-controlled-full) |
| HP-AB-12 | matched maximum retrieval budget | [`output/hotpotqa_2026_07_15_13_54_hotpotqa-efc-ircot-matched-budget10-full`](output/hotpotqa_2026_07_15_13_54_hotpotqa-efc-ircot-matched-budget10-full) |
| HP-AB-13 | w/o redundancy penalty | [`output/hotpotqa_2026_07_15_13_54_hotpotqa-efc-no-redundancy-weight-top10-full`](output/hotpotqa_2026_07_15_13_54_hotpotqa-efc-no-redundancy-weight-top10-full) |

### 2WikiMultiHopQA

| ID | 方法 | FINAL 目录 |
|---|---|---|
| 2W-NR-01 | No-RAG | [`output/2wikimultihopqa_2026_07_12_14_26_2wiki-zero-shot-full`](output/2wikimultihopqa_2026_07_12_14_26_2wiki-zero-shot-full) |
| 2W-NR-02 | Standard RAG | [`output/2wikimultihopqa_2026_07_10_15_04_2wiki-naive-no-rerank-top10-full`](output/2wikimultihopqa_2026_07_10_15_04_2wiki-naive-no-rerank-top10-full) |
| 2W-NR-03 | IterRetGen | [`output/2wikimultihopqa_2026_07_10_15_52_2wiki-iterretgen-no-rerank-full`](output/2wikimultihopqa_2026_07_10_15_52_2wiki-iterretgen-no-rerank-full) |
| 2W-NR-04 | Full-QD | [`output/2wikimultihopqa_2026_07_12_14_31_2wiki-full-qd-no-rerank-top10-full`](output/2wikimultihopqa_2026_07_12_14_31_2wiki-full-qd-no-rerank-top10-full) |
| 2W-NR-05 | IRCoT | [`output/2wikimultihopqa_2026_07_12_17_09_2wiki-ircot-no-rerank-cache-full-v2`](output/2wikimultihopqa_2026_07_12_17_09_2wiki-ircot-no-rerank-cache-full-v2) |
| 2W-NR-07 | EFC-RAG | [`output/2wikimultihopqa_2026_07_12_11_11_2wiki-efc-no-rerank-top10-full`](output/2wikimultihopqa_2026_07_12_11_11_2wiki-efc-no-rerank-top10-full) |
| 2W-NR-08 | FLARE | [`output/2wikimultihopqa_2026_07_13_14_26_2wiki-flare-no-rerank-top5-full`](output/2wikimultihopqa_2026_07_13_14_26_2wiki-flare-no-rerank-top5-full) |
| 2W-NR-10 | Modified Adaptive-k | [`output/2wikimultihopqa_2026_07_16_17_35_2wiki-modified-adaptive-k-gap-b5-max8-no-rerank-full`](output/2wikimultihopqa_2026_07_16_17_35_2wiki-modified-adaptive-k-gap-b5-max8-no-rerank-full) |

### MuSiQue

| ID | 方法 | FINAL 目录 |
|---|---|---|
| MU-NR-01 | No-RAG | [`output/musique_2026_07_12_16_14_musique-zero-shot-full`](output/musique_2026_07_12_16_14_musique-zero-shot-full) |
| MU-NR-02 | Standard RAG | [`output/musique_2026_07_12_19_43_musique-naive-no-rerank-top10-full`](output/musique_2026_07_12_19_43_musique-naive-no-rerank-top10-full) |
| MU-NR-03 | IterRetGen | [`output/musique_2026_07_12_21_35_musique-iterretgen-no-rerank-full`](output/musique_2026_07_12_21_35_musique-iterretgen-no-rerank-full) |
| MU-NR-04 | Full-QD | [`output/musique_2026_07_12_19_43_musique-full-qd-no-rerank-top10-prepare`](output/musique_2026_07_12_19_43_musique-full-qd-no-rerank-top10-prepare) |
| MU-NR-05 | IRCoT | [`output/musique_2026_07_12_22_30_musique-ircot-no-rerank-cache-full`](output/musique_2026_07_12_22_30_musique-ircot-no-rerank-cache-full) |
| MU-NR-07 | EFC-RAG | [`output/musique_2026_07_12_20_15_musique-efc-no-rerank-top10-prepare`](output/musique_2026_07_12_20_15_musique-efc-no-rerank-top10-prepare) |
| MU-NR-08 | FLARE | [`output/musique_2026_07_14_09_25_musique-flare-no-rerank-top5-full`](output/musique_2026_07_14_09_25_musique-flare-no-rerank-top5-full) |
| MU-NR-10 | Modified Adaptive-k | [`output/musique_2026_07_16_17_35_musique-modified-adaptive-k-gap-b5-max8-no-rerank-full`](output/musique_2026_07_16_17_35_musique-modified-adaptive-k-gap-b5-max8-no-rerank-full) |
| MU-AB-02 | generation-guided-only | [`output/musique_2026_07_15_10_01_musique-efc-force-generation-guided-no-static-top10-full`](output/musique_2026_07_15_10_01_musique-efc-force-generation-guided-no-static-top10-full) |

### Natural Questions

| ID | 方法 | FINAL 目录 |
|---|---|---|
| NQ-NR-01 | No-RAG | [`output/nq_2026_07_12_19_21_nq-zero-shot-full`](output/nq_2026_07_12_19_21_nq-zero-shot-full) |
| NQ-NR-02 | Standard RAG | [`output/nq_2026_07_14_09_54_nq-standard-rag-no-rerank-top10-full`](output/nq_2026_07_14_09_54_nq-standard-rag-no-rerank-top10-full) |
| NQ-NR-03 | IterRetGen | [`output/nq_2026_07_14_10_36_nq-iterretgen-no-rerank-full`](output/nq_2026_07_14_10_36_nq-iterretgen-no-rerank-full) |
| 待补 | Full-QD | `UNKNOWN`（Table 5: 34.88/47.07） |
| 待补 | IRCoT | `UNKNOWN`（Table 5: 39.31/49.53） |
| 待补 | Modified Adaptive-k | `UNKNOWN`（Table 5: 36.43/48.90） |
| NQ-NR-05 | EFC-RAG | [`output/nq_2026_07_14_09_54_nq-efc-no-rerank-top10-full`](output/nq_2026_07_14_09_54_nq-efc-no-rerank-top10-full) |
| NQ-NR-06 | FLARE | [`output/nq_2026_07_14_10_36_nq-flare-no-rerank-top5-full`](output/nq_2026_07_14_10_36_nq-flare-no-rerank-top5-full) |

## 12. 尚待补齐后再更新的内容

- TRACE 三个多跳数据集的最终质量、Retrieval Recall@5 与总耗时；当前按用户指令暂停。
- Adaptive-RAG：只有取得并锁定 classifier 后才可加入。
- 当前预先计划的HotpotQA/MuSiQue核心与W11扩展消融均已FINAL；后续只按论文缺口新增实验。
- 所有方法统一的端到端耗时、峰值显存和调用成本表；当前只有 EFC、FLARE、IRCoT 的
  部分效率信息已经完成审计。
