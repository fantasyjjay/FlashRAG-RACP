# RACP / EFC-RAG 项目周报

报告周期：2026-06-29 至 2026-07-05
项目状态统计截止：2026-07-05
主要任务：HotpotQA 多跳问答、无 reranker 自适应检索、基线补全与缓存复用

## 1. 本周摘要

本周工作的重点不是继续微调 EFC 的局部权重，而是整理当前阶段结果、确认方法边界，
并开始补充严格可比的无 reranker 多跳基线。主要完成情况如下：

1. 汇总 HotpotQA 已有全量、1000 条和 smoke 实验，形成统一实验报告。
2. 明确当前 EFC-RAG 最佳全量结果及其相对基线的实际提升。
3. 完成 IRCoT no-reranker 独立运行入口，并统一到当前 BGE-large、Llama-3.1-8B
   和 wiki18_100w 实验环境。
4. 修复 FlashRAG IRCoT 中的文档排序、无效检索、最终答案缺失和答案解析问题。
5. 验证 IRCoT 对已有 top20 检索缓存的复用方式，以及新 thought query 的增量缓存机制。
6. 使用 2 张 RTX 4090 完成 20 条 IRCoT 真实模型 smoke，最终流程指标为
   EM 45.00、F1 47.86。

当前阶段最重要的结论是：

> EFC-RAG 已经稳定超过相同无 reranker 条件下的原生 IterRetGen，但领先幅度有限；
> 下一阶段需要通过完整基线实验和证据质量分析确认收益来源，而不是继续仅根据
> 200 条 smoke 调整 router 比例。

## 2. 项目目标与当前方法定位

项目目标是在 HotpotQA 多跳问答上构建不依赖 cross-encoder reranker 的自适应 RAG，
在控制检索和 LLM 调用成本的同时超过固定流程的 IterRetGen。

当前 EFC-RAG 以 IterRetGen 为主干：

```text
原问题检索 top20
-> 使用原始 top5 生成 probe answer
-> 根据问题、初始证据和 probe 进行路由
-> direct / generation-guided / static-QD
-> 对需要补证据的样本生成 missing-hop query
-> 增量检索并合并候选文档
-> RRF、原始证据保留、角色覆盖和标题去重
-> 选择最终 top6
-> Llama-3.1-8B-Instruct 生成短答案
```

QD 在当前方法中是辅助模块，不是主干。大多数样本应优先通过 probe 暴露桥接实体，
再生成 missing-hop query；只有 probe 无法形成有效桥接信息时才回退到 static-QD。

## 3. 当前各阶段使用的模型

| 阶段 | 模型或组件 | 作用 |
|---|---|---|
| 初始检索 | bge-large-en-v1.5 | 对原问题检索 wiki18_100w top20 |
| Probe 生成 | Llama-3.1-8B-Instruct | 基于初始 top5 暴露桥接实体或直接答案 |
| Router | 规则特征与 probe 信息 | 决定 direct、generation-guided 或 static-QD |
| Missing-hop planner | Llama-3.1-8B-Instruct | 把 probe 中的桥接信息压缩为下一跳检索 query |
| Static-QD planner | Llama-3.1-8B-Instruct | 在 probe 不充分时生成辅助分解 query |
| 增量检索 | bge-large-en-v1.5 | 检索 missing-hop 或 QD query |
| 文档融合 | Query-level RRF 与规则选择 | 合并原问题和增量检索结果，保留 top6 |
| 最终生成 | Llama-3.1-8B-Instruct + vLLM | 基于最终证据生成答案 |

最佳 EFC 全量实验平均每条样本调用：

| 成本项 | 平均值 |
|---|---:|
| LLM 调用次数 | 2.8718 |
| 检索调用次数 | 1.9332 |
| 候选池文档数 | 25.5246 |
| 最终上下文文档数 | 6.0000 |

## 4. 本周工程进展

### 4.1 HotpotQA 实验结果整理

已生成完整实验档案：

`racp/HOTPOTQA_EXPERIMENT_REPORT.md`

报告统一整理了以下内容：

- HotpotQA dev 7,405 条全量实验。
- Native RAG、IterRetGen、QD、EFC、IRCoT 等结果。
- reranker 与 no-reranker 分组。
- EM、F1、Acc、Precision、Recall 和 Retrieval Recall。
- EFC router 分流、成本、planner 失败和支持文档命中情况。
- 硬件、软件、模型、语料和索引环境。
- 当前无 reranker 基线缺口和补实验顺序。

### 4.2 IRCoT no-reranker 入口

在 `examples/methods/run_exp.py` 中新增 `ircot-no-rerank` 方法入口，参数集中支持：

- 数据集与 split。
- 测试样本数。
- GPU 列表，限制 1 至 4 张。
- 检索 top-k。
- 最大 IRCoT 迭代数。
- 最大生成 token。
- vLLM 显存利用率。
- 检索缓存路径。
- 实验保存名称。

该入口固定使用：

```text
Retriever  = bge-large-en-v1.5
Generator  = Llama-3.1-8B-Instruct
Corpus     = wiki18_100w
Reranker   = false
Refiner    = none
Framework  = vLLM
```

这样可以避免旧的 `my_config.yaml` 默认 reranker 或 Selective-Context 混入 IRCoT，
保证后续结果可以与 EFC 和原生 IterRetGen 进行无 reranker 对比。

### 4.3 IRCoT 实现修复

本周发现并修复了四类问题。

#### 问题一：文档排序方向错误

原实现使用升序排列 dense retrieval score，把低分文档放在高分文档前面。
现已改为高分优先排列。

#### 问题二：最后一轮存在无效检索

原实现会在最后一次 thought 生成后继续检索，但后面不再进行答案生成。
这次检索不会影响答案，却会增加耗时、检索调用数和 Retrieval Recall。

现改为只有在后续仍有生成轮次时才检索。

#### 问题三：到达最大轮数后没有最终答案

无 reranker 时，第二轮通常刚得到第二跳证据。原实现直接把两句 thought 当作预测，
没有要求模型输出 `So the answer is:`，导致首轮 smoke 的 EM 为 0。

现为未提前作答的样本增加批量 finalization：

```text
已有 thought + 最终文档
-> 不再检索
-> 强制生成最短 final answer
-> 追加 So the answer is:
```

#### 问题四：答案后附加说明导致 EM 为 0

Llama-3.1-8B 会在正确短答案后继续输出 Note、Next step 或解释。原 parser 将这些内容
全部保留，导致答案虽然包含正确字符串，但 Exact Match 为 0。

现已在生成端增加短答案停止条件，并在 parser 中只保留最终答案的第一段。

### 4.4 检索缓存验证

IRCoT 可以复用现有 query-level 检索缓存，但缓存命中单位是完整 query 字符串：

1. HotpotQA 原问题与缓存键完全一致时，直接读取已有 top20，并按本次 top5 截取。
2. IRCoT 生成的新 thought query 首次通常不在缓存中，需要在线 BGE 检索。
3. 新 query 的结果会写入本次输出目录的 `retrieval_cache.json`。
4. 再次使用该合并缓存运行相同配置时，原问题和 thought query 都可以命中。

本周实测：

| 缓存阶段 | 条目数 | 新增 |
|---|---:|---:|
| 原始缓存 | 15,044 | - |
| 首次 IRCoT smoke 后 | 15,064 | 20 |
| 使用合并缓存复跑 | 15,064 | 0 |

这说明 20 个原始问题命中已有缓存，20 个第一跳 thought query 在首次运行时回源检索，
第二次运行则全部复用。

需要注意：普通缓存模式允许 miss 后在线检索；不能对首次 IRCoT 全量实验使用
cache-only，因为大部分新 thought query 尚未存在。

## 5. 当前最佳全量结果

### 5.1 无 Reranker 主表

| 方法 | Final K | EM | F1 | Acc | Precision | Recall | Retrieval Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| **EFC static-bridge** | 6 | **35.57** | **46.69** | 42.40 | **48.20** | **49.27** | 68.35 (@6) |
| EFC router 修改版 | 6 | 35.54 | 46.68 | **42.42** | 48.18 | 49.26 | 68.39 (@6) |
| EFC IterRetGen 主干版 | 6 | 35.52 | 46.64 | 42.38 | 48.12 | 49.24 | **68.43 (@6)** |
| 原生 IterRetGen | 10 | 34.67 | 45.66 | 41.13 | 47.12 | 47.97 | 62.86 (@10) |
| Native RAG top10 | 10 | 33.54 | 44.80 | 39.93 | 46.34 | 47.03 | 57.61 (@5) |
| Native RAG top5 | 5 | 31.74 | 42.36 | 38.15 | 43.71 | 44.71 | 57.61 (@5) |

### 5.2 EFC 相对基线的提升

相对原生 IterRetGen：

| 指标 | 提升 |
|---|---:|
| EM | +0.90 个百分点 |
| F1 | +1.03 个百分点 |
| Acc | +1.27 个百分点 |
| Precision | +1.08 个百分点 |
| Recall | +1.30 个百分点 |

相对 Native RAG top10：

| 指标 | 提升 |
|---|---:|
| EM | +2.03 个百分点 |
| F1 | +1.90 个百分点 |
| Acc | +2.47 个百分点 |
| Precision | +1.86 个百分点 |
| Recall | +2.24 个百分点 |

当前可以严格支持的结论是：

> 在已经完成的 HotpotQA 全量无 reranker 实验中，EFC static-bridge 的 EM、F1、
> Precision 和 Recall 均为最高，并超过原生 IterRetGen。

当前还不能声称 EFC 超过所有流行多跳 RAG 方法，因为 IRCoT no-reranker、
Adaptive-RAG、Full-QD no-reranker、Self-Ask 和 FLARE 尚未完成统一条件下的全量实验。

### 5.3 Reranker 参考上限

当前所有单方法全量实验最高结果为：

| 方法 | Reranker | EM | F1 |
|---|:---:|---:|---:|
| IterRetGen + bge-reranker-large | 是 | 38.74 | 50.49 |
| EFC static-bridge | 否 | 35.57 | 46.69 |

这表明 cross-encoder reranker 仍能带来明显增益。当前 EFC 的研究价值主要是：

- 不加载 reranker。
- 使用多轮 query 补充第二跳证据。
- 在较低重排成本下超过 no-reranker IterRetGen。

## 6. Router 表现分析

最佳 EFC 全量实验：

`hotpotqa_2026_06_08_13_17_efc-static-bridge-v1-full`

### 6.1 分流比例

| 实际路线 | 样本数 | 比例 | EM | F1 |
|---|---:|---:|---:|---:|
| direct | 955 | 12.90% | 47.12 | 59.30 |
| generation-guided | 5,990 | 80.89% | 34.71 | 46.11 |
| static-QD | 460 | 6.21% | 22.83 | 28.09 |

### 6.2 当前判断

#### Direct 分支

Direct 是表现最好的分支，但占比只有约 13%。这不是坏现象：HotpotQA 是多跳数据集，
多数问题不应仅依靠初始证据直接作答。Direct 的高分说明 router 对高置信简单样本的
筛选基本有效。

#### Generation-guided 分支

Generation-guided 占 80.89%，是当前方法的主要收益来源，也是下一阶段最值得优化的
部分。其 F1 为 46.11，接近整体结果，因此整体性能主要由 missing-hop query 质量、
第二跳召回和最终文档选择决定。

#### Static-QD 分支

Static-QD 只占 6.21%，F1 为 28.09，显著低于其他分支。其问题包括：

- 分解 query 可能缺少初始证据中出现的真实桥接实体。
- 无 reranker 时，两个子问题的候选文档难以稳定进入最终 top6。
- 分支样本本身更难，存在选择偏差，不能只用 route-wise 分数判断 QD 完全无效。
- 继续投入大量时间微调 static-QD，对整体指标的上限贡献有限。

因此当前策略是将 QD 保持为 fallback，而不是扩大 QD 路由比例。

## 7. 证据质量与主要瓶颈

最佳 EFC 的证据诊断：

| 指标 | 数值 |
|---|---:|
| Role coverage | 78.38% |
| 最终标题唯一率 | 99.74% |
| Support title recall@6 | 65.44% |
| 两个支持标题同时命中 | 47.43% |
| Planner 失败率 | 1.84% |
| Planner heuristic repair 比例 | 1.76% |

当前标题去重已经接近饱和，继续加强 title dedup 的收益空间很小。更明显的瓶颈是：

1. 只有 47.43% 的样本能够同时命中两个支持标题。
2. 即使支持标题进入候选池，也不一定都进入最终 top6。
3. Missing-hop query 有时与原问题过于相似，无法补充第二跳信息。
4. Final top6 的容量限制可能丢弃一篇必要证据，但扩大上下文又可能增加噪声。
5. 最终生成仍会受到实体截断、答案格式和错误证据干扰。

后续优化应优先围绕“第二个支持文档是否进入最终上下文”展开，而不是继续大幅调整
三条 router 的总体占比。

## 8. IRCoT Smoke 结果

本周共进行三次 20 条真实模型验证：

| 版本 | EM | F1 | Acc | 主要问题 |
|---|---:|---:|---:|---|
| 原始 IRCoT no-reranker | 0.00 | 7.95 | 25.00 | 到达 max_iter 后没有最终答案 |
| 增加 finalization | 0.00 | 6.62 | 45.00 | 正确答案后附加长解释，parser 全部保留 |
| 修复停止条件与 parser | **45.00** | **47.86** | **45.00** | 流程正常 |

最终 smoke 完整指标：

| 指标 | 数值 |
|---|---:|
| EM | 45.00 |
| F1 | 47.86 |
| Acc | 45.00 |
| Precision | 50.00 |
| Recall | 47.00 |
| Retrieval Recall@10 | 55.00 |

结果目录：

`racp/output/hotpotqa_2026_07_05_12_52_ircot-no-rerank-cache-smoke20-v2`

该结果只能证明代码流程、vLLM、缓存和答案解析已经正常，不能用于判断 IRCoT
是否超过 EFC。20 条样本方差很大，必须完成 7,405 条全量实验。

## 9. 实验环境

### 9.1 硬件

| 项目 | 配置 |
|---|---|
| CPU | 2 × Intel Xeon Gold 6430 |
| CPU | 64 物理核心，128 逻辑线程 |
| 内存 | 1.0 TiB |
| GPU | 5 × NVIDIA GeForce RTX 4090 |
| 单卡显存 | 24,564 MiB |
| NVIDIA Driver | 550.120 |

### 9.2 软件

| 项目 | 版本 |
|---|---|
| Ubuntu | 22.04.5 LTS |
| Python | 3.10.20 |
| PyTorch | 2.10.0+cu128 |
| Transformers | 4.57.6 |
| vLLM | 0.19.0 |
| FAISS | 1.8.0 |
| Conda 环境 | `/home/guanjunjie/.conda/envs/flashrag` |

### 9.3 IRCoT GPU 设置

IRCoT smoke 使用 GPU 0、1，vLLM tensor parallel size 为 2，显存利用率为 0.65。
当前实现限制最多使用 4 张 GPU。

Llama-3.1-8B 在 2 张 4090 上已经有足够显存。当前服务器的 GPU P2P 不可用，
使用 4 卡 tensor parallel 会增加 NCCL 通信开销，不一定比 2 卡更快。因此全量
IRCoT 推荐先使用 2 卡，剩余 GPU 可用于其他实验。

## 10. 当前代码状态

当前分支：

`flashrag-experiments`

当前提交：

`58dcbe6 测试`

本周尚未提交的主要改动：

| 文件 | 改动 |
|---|---|
| `examples/methods/run_exp.py` | 新增 IRCoT no-reranker 统一入口和参数 |
| `flashrag/pipeline/active_pipeline.py` | 修复 IRCoT 排序、无效检索、提前停止和 finalization |
| `flashrag/utils/pred_parse.py` | 修复 IRCoT 最终答案解析 |
| `racp/HOTPOTQA_EXPERIMENT_REPORT.md` | 完整实验结果报告 |
| `racp/WEEKLY_REPORT_2026_07_05.md` | 本周报告 |

在启动全量 IRCoT 前，应先提交或保存当前代码版本，使最终实验可以准确关联 commit。

## 11. 当前问题与风险

### 11.1 EFC 提升已经进入小幅优化区间

最佳 EFC 与原生 IterRetGen 的 F1 差距只有 1.03 个百分点。继续修改 router 阈值可能
只会造成不同分支之间的样本迁移，无法提高第二跳证据质量。

### 11.2 不同实验的 K 尚未完全统一

EFC 使用 final top6，原生 IterRetGen 和部分基线使用 top10。Retrieval Recall 的
评测 K 也不同，因此目前不能只根据 Retrieval Recall 数值判断检索器优劣。

### 11.3 无 Reranker 基线仍不完整

目前缺少：

- IRCoT no-reranker 全量。
- Full-QD no-reranker 全量。
- Adaptive-RAG no-reranker。
- Self-Ask + Search no-reranker。
- FLARE no-reranker。

其中 Adaptive-RAG 还缺少 classifier checkpoint；FLARE 需要先验证 vLLM token-score
接口。IRCoT 已具备全量运行条件，应作为第一优先级。

### 11.4 Flat FAISS 首次加载较慢

wiki18_100w 的 Flat index 体积较大。即使原问题命中缓存，只要存在新的 thought query，
仍然需要加载语料和 FAISS index。首次冷启动较慢；操作系统 page cache 热起来后，
重复加载会明显加快。

### 11.5 Smoke 结果不可直接用于论文对比

20 条和 200 条结果适合检查方向，但不同版本之间可能存在较大抽样波动。算法结论必须
以同一 7,405 条 HotpotQA dev、相同模型、相同检索器和相同预算的全量结果为准。

## 12. 下周计划

### P0：完成 IRCoT no-reranker 全量实验

统一设置：

```text
Dataset       = HotpotQA dev, 7,405
Retriever     = bge-large-en-v1.5
Generator     = Llama-3.1-8B-Instruct
Reranker      = false
Top-k/round   = 5
Max iter      = 2
Max documents = 10 before duplicate removal
GPU           = 0,1
```

全量实验结束后需要分析：

- EM、F1、Acc、Precision、Recall。
- Retrieval Recall@10。
- 提前回答比例。
- 实际平均检索轮数。
- Finalization 使用比例。
- 缓存命中和新增 query 数。
- 与 EFC、IterRetGen 的同条件差值。

### P0：统一无 Reranker 主表预算

至少统一报告：

- 原始检索 top-k。
- 每轮检索 top-k。
- 最大检索轮数。
- 最终上下文文档数。
- 平均 LLM 调用数。
- 平均检索调用数。
- 是否使用额外 planner。

避免某方法通过更多检索或更长上下文获得不可直接比较的分数。

### P1：补 Full-QD no-reranker

目的不是继续优化 EFC 的 QD 分支，而是回答一个实验问题：

> 自适应路由是否优于对所有 HotpotQA 问题统一进行问题分解？

### P1：分析 generation-guided 失败样本

从全量 EFC 中按以下类型各抽样：

- missing-hop query 与原问题相似。
- 第二支持标题未进入候选池。
- 第二支持标题进入候选池但未进入 top6。
- 两篇支持文档齐全但最终答案错误。

这四类问题分别对应 planner、retriever、selector 和 generator，能够避免继续盲调
router 阈值。

### P2：评估 Self-Ask 和 FLARE 接入成本

Self-Ask 可以作为显式 follow-up question 基线；FLARE 可以作为生成中动态检索基线。
如果实现成本较高，应先完成 IRCoT 和 Full-QD，再决定是否加入主表。

## 13. 下周验收标准

1. 完成 IRCoT no-reranker 7,405 条全量实验。
2. 将 IRCoT 全量指标和运行环境写入 `HOTPOTQA_EXPERIMENT_REPORT.md`。
3. 输出 EFC、IterRetGen、IRCoT 三者的统一预算对比表。
4. 完成至少 100 条 EFC generation-guided 失败样本归因。
5. 保存并提交当前 IRCoT 修复代码，使实验结果与 Git commit 一一对应。

## 14. 阶段结论

当前 EFC-RAG 已从初始版本 EM 26.12、F1 34.12，提高到 EM 35.57、F1 46.69，
并在无 reranker 条件下超过原生 IterRetGen。方法的主干定位已经清晰：

```text
IterRetGen 式 probe 和补检索
+ 自适应路由
+ QD fallback
+ 无 cross-encoder 的证据融合
```

下一阶段的重点应从“继续改变 router 比例”转向两件事：

1. 补齐统一条件下的多跳基线，确认 EFC 的相对位置。
2. 提高 generation-guided 分支的第二支持文档命中与最终保留率。

这两项完成后，才能判断当前方法是已经接近无 reranker 配置下的性能上限，还是仍有
可通过 query 生成和证据选择获得的明确提升空间。
