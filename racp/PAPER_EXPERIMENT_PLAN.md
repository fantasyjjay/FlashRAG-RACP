# RACP/EFC-RAG 论文实验计划与结果台账

最后更新：2026-07-15
W11运行代码基线：`063e29dde59191c2c249ec1bde669834dc3e88ee` + title-dedup-only
隔离diff（关键四文件hash已归档，本轮文档/代码提交负责将该diff固化到Git历史）

## 0. 下一轮对话快速交接

### 0.1 项目目标

本项目在 FlashRAG 上实现并评测 EFC-RAG（Evidence-Feedback Controlled RAG）。方法以
IterRetGen 的“生成中间信息后继续检索”为主干，通过 probe answer 和首轮证据状态将
问题路由到三个分支：

1. `direct`：首轮证据足够，直接使用最终证据生成答案；
2. `generation_guided`：probe 暴露新的桥接实体，生成 missing-hop query 后继续检索；
3. `static_qd`：probe 无有效桥接实体或不确定，使用固定问题分解作为辅助回退。

论文不以 SOTA 为目标。主要论点是：在普通 8B 模型、同一 dense retriever、无
reranker 的条件下，自适应证据反馈路由比 Standard RAG、固定 Full-QD 和原生 IterRetGen
取得更好的质量/成本平衡，并在单跳 NQ 上避免不必要的多轮检索。

### 0.2 当前最重要结论

- HotpotQA 当前 EFC-RAG：EM 35.85、F1 47.21、Retrieval Recall@10 71.51。
- IRCoT：EM 36.26、F1 47.50，答案指标略高于 EFC，但检索召回和 Acc 低于 EFC。
- 原生 IterRetGen：EM 34.67、F1 45.66，EFC 分别提升 1.18 和 1.55 个百分点。
- Full-QD：EM 33.67、F1 44.59。固定拆分不适合作为主干，QD 只保留为辅助模块。
- EFC 在 HotpotQA 平均检索 1.9332 次、平均 LLM 调用 2.8718 次；80.89% 样本进入
  `generation_guided`，12.90% 进入 `direct`，6.21% 进入 `static_qd`。
- 2Wiki Standard RAG：EM 16.56、F1 25.60、Retrieval Recall@10 46.20；原生
  IterRetGen：EM 16.90、F1 26.59、Retrieval Recall@10 42.48；EFC-RAG：EM 20.35、
  F1 29.37、Retrieval Recall@10 58.17。
- 2Wiki EFC 相比 IterRetGen 的 EM/F1/检索召回分别提升 3.45/2.78/15.70 个百分点，
  相比 Standard RAG 的 EM/F1 分别提升 3.79/3.77 个百分点。
- EFC 已在 HotpotQA 和 2Wiki 两个多跳数据集上同时超过 IterRetGen 的 EM/F1，并持续
  提高 Retrieval Recall@10；已满足“至少两个多跳数据集超过 IterRetGen”的关键标准。
- No-RAG 闭卷下界已补齐：HotpotQA EM/F1 17.68/26.54，2Wiki 17.22/26.51；2Wiki
  No-RAG 略高于 Standard RAG，说明单轮检索在该设置下会引入一定噪声。
- HotpotQA Standard RAG 已补齐：EM 33.57、F1 44.82、Retrieval Recall@10 65.12。
- MuSiQue No-RAG 闭卷下界已补齐：EM 3.56、F1 9.94；完整 dev 2,417 条均有预测。
- 2Wiki Full-QD 已补齐：EM 18.23、F1 26.84、Retrieval Recall@10 49.57；高于
  Standard RAG，但仍低于 EFC-RAG 的 EM/F1 20.35/29.37。
- 2Wiki IRCoT 已补齐：EM 33.07、F1 39.39、Retrieval Recall@10 50.15。IRCoT 的答案
  指标显著高于 EFC-RAG；EFC 的检索召回 58.17 仍高于 IRCoT。论文不能宣称 EFC 在
  2Wiki 上超过所有多轮推理检索方法，只能主张超过 Standard RAG、IterRetGen 和 Full-QD。
- FLARE、TRACE 和 Adaptive-RAG 已纳入主表候选。FLARE/TRACE 必须先通过当前
  Llama-3.1/BGE/no-reranker 环境兼容性审计；Adaptive-RAG 在取得并锁定分类器前不得
  启动正式全量。
- W11四项消融已全量验收：title-dedup-only较RRF-only恢复2.90 F1，证明title
  diversity是selector的主要收益来源，但仍不能解释全部收益。
- controlled final top5较完整EFC仅降0.58 F1；matched maximum retrieval budget较IRCoT
  高0.67 F1，但只匹配最大检索预算；w/o redundancy未观察到penalty正收益。

### 0.3 下一步正式任务

当前优先级：

1. 对已完成主结果和消融统一重算answer-hit/support-title口径并做paired bootstrap；
2. 将W11的`063e29d + title-dedup diff`归档到Git commit，不重跑已验收结果；
3. 若篇幅需要更完整正交表，再补always-static或w/o title weight；
4. TRACE和Adaptive-RAG保持暂停，只在明确要求且解决现有阻塞后恢复。

任何新对话开始后，应先检查最新输出目录、GPU 进程和 Git 状态，确认没有上一轮已完成
但尚未登记的正式结果，再决定下一条命令。

### 0.4 W11 运行时代码边界

W11四项实验同秒启动于`063e29d`加同一份未提交diff。该diff只增加默认
关闭的title-dedup-only隔离选择器、CLI/配置入口和测试：

```text
racp/config.yaml
racp/efc.py
racp/run_racp.py
racp/test_efc.py
```

`HP-AB-10/run_command.txt`保存了这四个文件的SHA-256，并与验收时工作区完全匹配；
`HP-AB-11/12/13`的命令记录了同一code state，但没有各自独立的hash。本轮Git归档
固化该diff后即可关闭provenance缺口；不要回滚或重跑已完成FINAL。

## 1. 文档用途与记录规则

本文档只跟踪计划写入论文的正式实验。以下内容不进入本文档：

- smoke test、少量样本调试和性能预估；
- 启动失败、显存溢出或未完成评测的运行；
- router 调参过程中被后续版本替代的中间结果；
- 配置口径不一致且尚未完成审计的历史结果；
- 未被选作论文方法的 prompt、top-k 或答案融合派生结果。

状态定义：

| 状态 | 含义 |
|---|---|
| `FINAL` | 全量完成、配置和日志已审计，可以写入论文 |
| `TODO` | 尚未执行的正式论文实验 |
| `RERUN` | 有历史结果，但配置口径不一致，必须统一重跑 |
| `RUNNING` | 正在执行正式全量实验 |
| `PAUSED` | 已按实验排期主动停止，当前无进程且不得登记部分结果；恢复时先审计可复用工件并使用新目录 |
| `FAILED` | 正式全量因异常退出，未产生可登记指标；只记录故障状态和可复用工件，修复后必须用新目录重跑 |
| `REJECTED` | 全量完成但不再进入论文；只保留拒绝原因，不记录分数 |
| `N/A` | 该方法不适用于对应数据集或不计划报告 |

只有状态为 `FINAL` 的实验可以填写论文结果数值和最终输出目录。

## 2. 数据集与评测范围

| 数据集 | 本地名称 | Split | 样本数 | 作用 |
|---|---|---:|---:|---|
| HotpotQA | `hotpotqa` | dev | 7,405 | 主多跳数据集；完整 router 分析和消融 |
| 2WikiMultiHopQA | `2wikimultihopqa` | dev | 12,576 | 跨实体、多文档多跳泛化 |
| MuSiQue | `musique` | dev | 2,417 | 更强组合性和依赖式多跳泛化 |
| Natural Questions | `nq` | test | 3,610 | 单跳补充实验；检查是否过度检索 |

历史 NQ 结果使用过 `e5`、Llama-3、reranker 或 1,000 条子集，不进入新论文表。历史 2Wiki 结果存在 prepare/generate 配置继承歧义，也不直接进入新论文表。

### 2.1 本地路径

| 资源 | 路径 |
|---|---|
| 仓库根目录 | `/home/guanjunjie/jayj/FlashRAG` |
| RACP 目录 | `/home/guanjunjie/jayj/FlashRAG/racp` |
| Conda Python | `/home/guanjunjie/.conda/envs/flashrag/bin/python` |
| 数据集根目录 | `/home/guanjunjie/my_datasets/FlashRAG_datasets` |
| 模型根目录 | `/home/guanjunjie/my_models` |
| BGE index | `/home/guanjunjie/my_datasets/FlashRAG_datasets/indexes/wiki18_100w_bge-large-en-v1.5/bge-large-en-v1.5_Flat.index` |
| Wikipedia corpus | `/home/guanjunjie/my_datasets/FlashRAG_datasets/retrieval-corpus/wiki18_100w.jsonl` |
| 实验输出 | `/home/guanjunjie/jayj/FlashRAG/racp/output` |
| 公共缓存 | `/home/guanjunjie/jayj/FlashRAG/racp/output/cache` |

### 2.2 硬件与软件环境

| 项目 | 当前环境 |
|---|---|
| GPU | 5 x NVIDIA GeForce RTX 4090，单卡 24,564 MiB |
| 项目 GPU 上限 | 最多使用 4 张，不占用第 5 张作为默认计划 |
| CPU | 2 x Intel Xeon Gold 6430，64 物理核 / 128 线程 |
| 内存 | 1.0 TiB |
| Python | Conda 环境 `flashrag` |
| PyTorch | 2.10.0+cu128 |
| CUDA runtime | 12.8 |
| Transformers | 4.57.6 |
| vLLM | 0.19.0 |
| NVIDIA driver | 550.120 |

EFC/QD 单次实验默认使用一张 4090，并按 planner、retriever、final generator 顺序复用
显存。IRCoT no-reranker 入口支持 1-4 张 GPU。FAISS Flat 检索会大量占用 CPU 和内存；
不要同时启动多个首次全量检索任务，否则会产生严重 CPU/NUMA 竞争。

### 2.3 关键代码文件

| 文件 | 作用 |
|---|---|
| [`run_racp.py`](run_racp.py) | EFC-RAG、Full-QD、原始 RACP、RS-MHR 的统一入口；负责 prepare/generate/full、缓存和评测 |
| [`efc.py`](efc.py) | EFC router 特征、三分支决策、RRF、角色感知文档选择和标题去重 |
| [`config.yaml`](config.yaml) | 模型、数据集、index、generator、retriever 和 metric 基础配置 |
| [`run_exp.py`](run_exp.py) | Standard RAG、No-RAG、IterRetGen、FLARE、TRACE、Adaptive-RAG 等通用基线入口；`naive` CLI 实际对应 Standard RAG |
| [`../examples/methods/run_exp.py`](../examples/methods/run_exp.py) | 当前定制的 `ircot-no-rerank` 多 GPU/cache 入口 |
| [`../flashrag/pipeline/active_pipeline.py`](../flashrag/pipeline/active_pipeline.py) | IRCoT/主动检索 pipeline；当前存在未提交修复 |
| [`../flashrag/utils/pred_parse.py`](../flashrag/utils/pred_parse.py) | IRCoT 等方法的最终答案解析 |
| [`README.md`](README.md) | 历史使用说明；若与本文档冲突，以本文档和最终 `config.yaml` 为准 |

### 2.4 EFC-RAG 实际执行流程

```text
问题 q
  -> BGE 原问题检索 top20 (R0)
  -> Llama-3.1-8B 用 R0 top5 生成 probe answer
  -> router 提取问题类型、证据分数、标题覆盖、probe 不确定性和新实体
      -> direct: 不生成新 query
      -> generation_guided: Llama-3.1-8B 生成 1 条 missing-hop query，检索 top10
      -> static_qd: Llama-3.1-8B 生成 2 条辅助 query，每条检索 top5
  -> 原始与扩展候选按 RRF、来源和角色覆盖合并
  -> title-diverse / role-aware pack 选 final top10
  -> 干净 vLLM 子进程中的 Llama-3.1-8B 生成最终短答案
  -> EM/F1/Acc/Precision/Recall/Retrieval Recall@10 评测
```

模型调用分工：

| 阶段 | 模型 | 说明 |
|---|---|---|
| Dense retrieval | `bge-large-en-v1.5` | 原问题和新增 query 都使用同一 Flat index |
| Probe generation | `Llama-3.1-8B-Instruct` | 输入首轮 top5，最多生成 128 tokens |
| Missing-hop / QD planner | `Llama-3.1-8B-Instruct` | 输出 JSON query；planner max tokens 96 |
| Final answer | `Llama-3.1-8B-Instruct` | 输入 final top10，最多生成 32 tokens |
| Reranker | 无 | 论文主表统一关闭 cross-encoder reranker |

router 的关键逻辑位于 `efc.py::compute_router_features` 和 `efc.py::decide_route`：

- comparison 的两侧实体都被 top5 覆盖时走 `direct`；
- probe 暴露原问题中没有的新实体时优先走 `generation_guided`；
- probe 无效、不确定且没有桥接实体时走 `static_qd`；
- 非多跳问题或完整 probe 没有可用于下一跳的新实体时走 `direct`；
- planner 输出无效时使用 heuristic repair、static-QD 或 direct 回退，不能让样本丢失。

### 2.5 当前未提交代码修改边界

当前正式代码基线为 commit `3b4d780` 加工作区兼容性 diff。不得把这些修改笼统写成
“算法改进”；每项影响如下：

| 文件 | 修改 | 是否改变论文方法逻辑 | 影响范围 |
|---|---|---|---|
| `flashrag/generator/generator.py` | 修复 vLLM `return_scores` 中未初始化 `output_scores`，并读取实际生成 token 的 logprob | 恢复 FLARE 所需的 canonical token-confidence，旧实现会崩溃；不是阈值调参 | 仅调用 `return_scores=True` 的 FLARE；已完成的 No-RAG、Standard、IterRetGen、Full-QD、IRCoT、EFC 不受影响 |
| `flashrag/pipeline/active_pipeline.py` | FLARE 无检索时写入空 `retrieval_result`；每轮 query/docs 另存轨迹；动态单query通过单元素 `batch_search` 解包 | 否；不改变置信度判断、query、top-k、检索内容或答案，只修复 cache miss 返回维度、评测字段与可追溯性 | 2026-07-13 起的 FLARE 正式运行 |
| `racp/run_exp.py` | 暴露 FLARE/Adaptive 参数、统一 no-refiner/top-k/metric 配置并支持 `test_sample_num` | 不改变已锁定基线；使参数可显式记录 | FLARE、TRACE、Adaptive 入口及 smoke 控制 |
| `flashrag/pipeline/pipeline.py` | Adaptive 多跳分支 `max_iter` 改为可配置 | 是可控实验设置；论文计划值锁定为2，不再沿用旧默认5 | Adaptive 尚无正式结果且当前暂停，因此未影响论文成绩 |

所有使用上述 diff 的 FINAL 目录均记录为 `3b4d780 + 未提交兼容性 diff`。FLARE 修复前的
smoke、失败运行和缺少 Retrieval Recall 的输出全部排除，不得与修复后的正式结果混用。

## 3. 统一实验口径

所有可直接控制的配置固定如下：

| 项目 | 统一设置 |
|---|---|
| Generator | `Llama-3.1-8B-Instruct` |
| Generator framework | vLLM |
| Retriever | `bge-large-en-v1.5` |
| Corpus / index | `wiki18_100w` / BGE Flat index |
| Reranker | 关闭 |
| 原问题候选 top-k | 20（适用时） |
| 最终上下文 K | 10 |
| 最大新增检索 query | 2 |
| Sampling | 关闭，temperature = 0 |
| 数据抽样 | 关闭，使用完整 split |
| 主要指标 | EM、F1 |
| 辅助指标 | Acc、Precision、Recall、Retrieval Recall@10 |
| 效率指标 | 平均 LLM 调用、平均检索调用、候选池大小、最终文档数、耗时 |

不同算法的 canonical 每轮检索 top-k 可能不同。例如当前 IRCoT 和 IterRetGen 每轮取 5，EFC 从原问题 top20 构造候选池。论文中必须同时披露平均检索调用次数和候选池大小，不能只比较最终 K。

### 3.0.1 各方法锁定关键参数

下表是论文主表的集中参数台账。最终仍以各 FINAL 目录中的 `config.yaml` 为审计依据；
若 FINAL 配置与本表不一致，该结果必须重跑，不能静默修改本表迁就结果。

| 方法 | 检索与上下文 | 轮次/调用 | 锁定生成与方法参数 | 检索指标语义 |
|---|---|---|---|---|
| No-RAG (`zero-shot`) | 不检索；无 context | 1次答案生成 | Llama-3.1-8B；`max_tokens=32`；`do_sample=false`；`refiner=null` | N/A |
| Standard RAG (`naive`) | 原问题单次 BGE top10；final K=10 | 1次检索 + 1次生成 | `retrieval_topk=10`；no-reranker；`max_tokens=32` | Recall@10 |
| IterRetGen | 每轮 BGE top5 | 固定3轮检索-生成 | `iter_num=3`；no-reranker；每轮生成沿用 `max_tokens=32` | 当前实现最终 `retrieval_result` 仅为第三轮5篇；配置键虽名为@10，必须同时披露实际5篇 |
| Full-QD | 原问题top20；2条子问题各top5；title-dedup选final top10 | 1次planner + 子问题检索 + 1次答案生成 | `planner_subquery_num=2`；`subquery_topk=5`；`selection_method=title_dedup_topk`；`selection_topk=10`；planner batch=16、max tokens=96 | Recall@10 |
| IRCoT | 每轮top5；两轮累积去重，最多10篇 | `max_iter=2` thought；未结束样本追加1次final-answer调用 | 固定IRCoT demonstration；thought/final `max_tokens=32`；no-reranker；正式可用GPU0-3 TP | Recall@10；实际每条最终文档可少于10篇 |
| EFC-RAG | original top20、probe top5；missing-hop top10或2条QD各top5；final top10 | probe + route-dependent planner + final answer | missing query=1；QD=2；RRF k=60；权重 `1.0/0.30/0.02/0.05/0.01`；soft title dedup；`max_same_title=2`；probe/planner/final max tokens=`128/96/32`；planner batch=32 | Recall@10；同时报告平均LLM/检索调用和候选池 |
| FLARE | 仅低置信度句子触发BGE top5；final字段为最后一次触发的top5，无触发则空列表 | 最多5轮 | confidence threshold=0.2；look-ahead=64 tokens；总生成上限256；`max_iter=5`；vLLM token logprob；no-reranker | Recall@5；无检索样本计空列表，不得当作缺失数据 |
| TRACE | 原问题BGE top5；最终使用5条推理链 | 三元组抽取、链构造、最终生成 | HF framework；exemplars=3；max chain length=4；triple select top5；choices=20；min triple prob=1e-4；beams=5；candidate chains=20；`n_context=5`；context=`triples` | Recall@5；必须同时报告耗时 |
| Adaptive-RAG | classifier路由为No-RAG/单次RAG/IRCoT；检索分支top5 | multi-hop `max_iter=2` | classifier batch=16；必须显式提供并审计本地checkpoint；no-reranker | 当前暂停，无合法FINAL |

所有方法共用 seed=2024、完整 split、BGE-large-en-v1.5/wiki18_100w、Llama-3.1-8B、
temperature=0/no sampling。除 TRACE 必须使用 HF、IRCoT 可使用TP外，其余生成默认vLLM单卡。

### 3.1 运行入口与 stage

所有命令都从仓库根目录执行，推荐直接使用确定的 Conda Python，避免 shell 中的 Conda
配置权限问题：

```bash
cd /home/guanjunjie/jayj/FlashRAG
export PYTHONPATH=.
PY=/home/guanjunjie/.conda/envs/flashrag/bin/python
```

自定义方法入口：

```bash
$PY racp/run_racp.py --method <efc|qd|racp|rs_mhr> --stage <prepare|generate|full> ...
```

基线入口：

```bash
$PY racp/run_exp.py --method_name <zero-shot|naive|iterretgen|adaptive|...> ...
$PY examples/methods/run_exp.py --method_name ircot-no-rerank ...
```

`run_racp.py` 的 stage 含义：

| Stage | 行为 | 产物 |
|---|---|---|
| `prepare` | planner/probe、检索、候选融合、最终 prompt 构造 | `prompt_cache.json`、retrieval cache、`config.yaml` |
| `generate` | 读取 prompt cache，启动 generator，生成答案并评测 | `metric_score.txt`、`intermediate_data.json` |
| `full` | 在一次命令中完成 prepare 和 generate | 上述全部产物 |

EFC、QD、RS-MHR 的 `full + vLLM` 已修复为：prepare 完成后保存 prompt cache，再在干净
子进程启动 final generator。这样可避免 planner/retriever 已初始化 CUDA 后触发 vLLM
`spawn` 递归和显存耗尽。正式 EFC/QD 推荐使用 `--stage full`；中途失败时可从已保存的
`prompt_cache.json` 单独执行 `generate`。

### 3.2 必须显式指定的参数

不要依赖 `run_racp.py` 的无参数默认值。当前默认 stage 是 `prepare`，默认 EFC
`final_topk=6`，而论文锁定设置是 final top10。正式命令至少显式指定：

| 参数 | 论文设置/说明 |
|---|---|
| `--method` | EFC 用 `efc`；固定分解用 `qd` |
| `--stage` | 正式完整运行用 `full` |
| `--dataset_name` | `hotpotqa` / `2wikimultihopqa` / `musique` / `nq` |
| `--split` | 三个多跳数据集用 `dev`；NQ 用 `test` |
| `--gpu_id` | 单卡如 `2`；IRCoT 可用 `0,1,2,3` |
| `--save_note` | 必须包含数据集、方法、no-rerank、top10、full |
| `--no_reranker` | 主表必须开启 |
| `--test_sample_num` | 正式全量不要传；最终 config 应为 `null` |
| `--initial_topk` | EFC 为 20 |
| `--probe_topk` | EFC 为 5 |
| `--final_topk` | EFC 为 10，必须显式传入 |
| `--missing_query_num` | 1 |
| `--qd_num / --qd_topk` | 2 / 5 |
| `--gen_topk` | 10 |
| `--planner_model` | `Llama-3.1-8B-Instruct` |
| `--planner_batch_size` | 正式 EFC 记录为 32；QD 已验证设置为 16 |
| `--planner_gpu_memory_utilization` | 0.75 |
| `--generate_gpu_memory_utilization` | 0.85 |

### 3.3 缓存类型与现状

| 缓存类型 | 内容 | 可复用阶段 | 注意事项 |
|---|---|---|---|
| Query retrieval cache | `query -> docs + dense scores` | 原问题和完全相同的新 query 检索 | EFC/IRCoT 使用；新生成 query 不命中时仍需在线检索 |
| QD retrieval-topk bundle | planner 输出、原问题和子问题候选、分数 | Full-QD 离线重新选择/生成 | 通过 `--load_retrieval_topk_cache_path` 使用，不是普通 query cache |
| Prompt cache | 已选择文档和最终生成 prompt | 只用于 `--stage generate` | 不能传给 `--retrieval_cache_path` |

当前可用正式缓存：

| 数据集/用途 | 路径 | 状态 |
|---|---|---|
| HotpotQA 原问题 top20 query cache | `racp/output/cache/hotpotqa_dev_bge_large_no_rerank_top20_retrieval_cache.json` | 有效软链接；不要删除其目标目录 |
| HotpotQA EFC 扩展 query cache | `racp/output/hotpotqa_2026_06_08_13_17_efc-static-bridge-v1-full/retrieval_cache.json` | 包含部分历史生成 query；最终 EFC top10 使用过 |
| HotpotQA Full-QD bundle | `racp/output/cache/hotpotqa_full_qd_planner8b_no_rerank_top10_retrieval_topk.json` | 可复用 |
| 2Wiki 原问题 top20 query cache | `racp/output/cache/2wikimultihopqa_dev_bge_large_no_rerank_top20_retrieval_cache.json` | 可复用 |
| MuSiQue 原问题 top20 query cache | `racp/output/cache/musique_dev_bge_large_no_rerank_top20_retrieval_cache.json` | 已审计；2,417条样本全部命中、2,412个唯一问题键 |
| NQ 原问题 top20 query cache | `racp/output/cache/nq_test_bge_large_no_rerank_top20_retrieval_cache.json` | 已审计；完整test 3,610条全部命中、每键top20 |

缓存规则：

- `--retrieval_cache_only` 只适合 planner/probe/query 生成完全不变且扩展缓存完整的 EFC
  重跑。跨数据集第一次正式运行禁止使用它。
- 普通 `--use_retrieval_cache` 允许 cache miss；缺失的新 query 会在线检索，并在本次
  输出目录形成合并后的 `retrieval_cache.json`。
- Full-QD 的子问题由 LLM 生成。原问题 cache 可复用，但子问题首次出现时仍需检索；
  完成后应保存 retrieval-topk bundle。
- 每次正式运行都保留自己的 cache，不覆盖或删除历史 FINAL 目录。

### 3.3.1 并行执行与缓存优先规则

本项目有 5 张 4090，但正式实验最多使用 GPU 0-3，GPU 4 默认保留。并行调度不能只看
空闲显存，还必须区分任务是否会加载 81GB BGE Flat index：

| 通道 | 任务类型 | 并发上限 | 典型任务 |
|---|---|---:|---|
| `R`（online retrieval） | cache miss、动态 query、首次建 cache | 1 | EFC、IterRetGen、IRCoT、FLARE、首次 Standard/QD prepare |
| `G`（generation/cache-only） | No-RAG、prompt-cache generate、完全 cache-hit | 最多 3 | No-RAG、Standard/EFC/QD 的 generate、改造后的 TRACE |
| `TP`（多卡单任务） | 一个模型跨多张 GPU tensor parallel | 独占所用 GPU | 2Wiki IRCoT GPU 0-3 |

固定规则：

1. 每个数据集先串行建立一份 BGE/no-reranker/original-query top20 cache，再启动其他方法；
   top5/top10 方法由 `cache_manager` 对 top20 结果切片，不重复检索。
2. 同一时刻最多运行一个可能产生 cache miss 的检索型任务。多个任务同时读取 Flat index
   会重复占用约 81GB 内存并争抢 CPU/内存带宽，即使 GPU 空闲也禁止盲目并行。
3. 能拆成 `prepare/generate` 的方法优先串行执行 `prepare`，然后将多个 `generate` 分配到
   GPU 0-3 并行。prompt cache 只供 generate 使用，不作为 query retrieval cache。
4. IterRetGen、IRCoT、FLARE 和 Adaptive-RAG 的多跳分支会在生成过程中产生新 query，
   只能复用原问题和历史相同 query；cache miss 必须允许在线检索并保存到本次输出目录。
5. TRACE 只检索原问题，理论上可以完全复用 top20 query cache；但当前 `DenseRetriever`
   即使 cache 全命中也会加载 index。启动 TRACE 全量/分片前，先实现并审计 cache-only
   retriever，避免每个 GPU 进程重复加载 81GB index。
6. TRACE 全量优先按数据分片在 GPU 0-3 并行，分片必须互斥、合并后恢复原始顺序，并只对
   合并后的完整 split 评测。分片中间结果和单片指标不登记为 FINAL。
7. 所有公共 cache 视为只读输入；新 query 合并 cache 写到本次输出目录。确认完整后再建立
   新的 `racp/output/cache/` 软链接，禁止覆盖现有 FINAL cache 或 QD bundle。

新增方法的缓存语义：

| 方法 | 可复用缓存 | 仍需在线检索 | 并行策略 |
|---|---|---|---|
| FLARE | 数据集 original-query top20 cache | 低置信度句子生成的新 query | 每次一个数据集；不与其他 `R` 任务并行 |
| TRACE | original-query top20 cache（取前 5） | 无 | cache-only 改造后按 4 个数据分片并行 |
| Adaptive-RAG | original-query top20 cache | multi-hop 分支的 IRCoT query | 暂停；classifier 锁定后作为单个 `R` 任务运行 |

### 3.4 EFC-RAG 正式命令模板

下面是当前论文 EFC top10 的完整参数模板。替换数据集、split、GPU、缓存和 save note；
首次没有缓存时删除 `--use_retrieval_cache` 与 `--retrieval_cache_path`，保留
`--save_retrieval_cache`。

```bash
$PY racp/run_racp.py \
  --method efc \
  --stage full \
  --dataset_name <DATASET> \
  --split <SPLIT> \
  --gpu_id <GPU> \
  --save_note <DATASET>-efc-no-rerank-top10-full \
  --no_reranker \
  --initial_topk 20 \
  --probe_topk 5 \
  --final_topk 10 \
  --enable_generation_guided \
  --enable_static_qd_fallback \
  --missing_query_num 1 \
  --missing_query_mode llm \
  --qd_num 2 \
  --qd_topk 5 \
  --gen_topk 10 \
  --rrf_k 60 \
  --rrf_weight 1.0 \
  --role_weight 0.30 \
  --title_weight 0.02 \
  --source_weight 0.05 \
  --redundancy_weight 0.01 \
  --title_dedup_soft \
  --max_same_title 2 \
  --original_seed_count 4 \
  --static_bridge_evidence_topk 5 \
  --static_bridge_original_count 4 \
  --static_bridge_qd_count 2 \
  --probe_max_tokens 128 \
  --planner_model Llama-3.1-8B-Instruct \
  --planner_max_tokens 96 \
  --planner_batch_size 32 \
  --planner_inference_batch_size 8 \
  --planner_gpu_memory_utilization 0.75 \
  --generate_gpu_memory_utilization 0.85 \
  --use_retrieval_cache \
  --retrieval_cache_path <QUERY_CACHE.json> \
  --save_retrieval_cache
```

下一项 2Wiki EFC 的可直接执行版本：

```bash
$PY racp/run_racp.py \
  --method efc --stage full \
  --dataset_name 2wikimultihopqa --split dev --gpu_id 2 \
  --save_note 2wiki-efc-no-rerank-top10-full \
  --no_reranker \
  --initial_topk 20 --probe_topk 5 --final_topk 10 \
  --enable_generation_guided --enable_static_qd_fallback \
  --missing_query_num 1 --missing_query_mode llm \
  --qd_num 2 --qd_topk 5 --gen_topk 10 \
  --rrf_k 60 --rrf_weight 1.0 \
  --role_weight 0.30 --title_weight 0.02 --source_weight 0.05 \
  --redundancy_weight 0.01 --title_dedup_soft --max_same_title 2 \
  --original_seed_count 4 \
  --static_bridge_evidence_topk 5 \
  --static_bridge_original_count 4 --static_bridge_qd_count 2 \
  --probe_max_tokens 128 \
  --planner_model Llama-3.1-8B-Instruct --planner_max_tokens 96 \
  --planner_batch_size 32 --planner_inference_batch_size 8 \
  --planner_gpu_memory_utilization 0.75 \
  --generate_gpu_memory_utilization 0.85 \
  --use_retrieval_cache \
  --retrieval_cache_path racp/output/cache/2wikimultihopqa_dev_bge_large_no_rerank_top20_retrieval_cache.json \
  --save_retrieval_cache
```

### 3.5 Full-QD 正式命令模板

```bash
$PY racp/run_racp.py \
  --method qd --stage full \
  --dataset_name <DATASET> --split <SPLIT> --gpu_id <GPU> \
  --save_note <DATASET>-full-qd-no-rerank-top10-full \
  --no_reranker \
  --retrieval_topk 20 \
  --planner_subquery_num 2 --subquery_topk 5 \
  --selection_method title_dedup_topk --selection_topk 10 \
  --planner_model Llama-3.1-8B-Instruct \
  --planner_batch_size 16 --planner_gpu_memory_utilization 0.75 \
  --generate_gpu_memory_utilization 0.85 \
  --use_retrieval_cache --retrieval_cache_path <QUERY_CACHE.json> \
  --save_retrieval_topk_cache_path <DATASET_QD_BUNDLE.json>
```

Full-QD 当前将不同 query 的原始 cosine score 直接合并，无 reranker 时会引入较多噪声。
这是需要保留的固定分解基线，不再以提升它为主要开发目标。

### 3.6 基线正式命令

No-RAG：

```bash
$PY racp/run_exp.py \
  --method_name zero-shot \
  --dataset_name <DATASET> --split <SPLIT> --gpu_id <GPU> \
  --save_note <DATASET>-zero-shot-full
```

Standard RAG no-reranker，final context top10（代码入口名为 `naive`）：

```bash
$PY racp/run_exp.py \
  --method_name naive \
  --dataset_name <DATASET> --split <SPLIT> --gpu_id <GPU> \
  --save_note <DATASET>-naive-no-rerank-top10-full \
  --retrieval_topk 10 --no_reranker \
  --use_retrieval_cache --retrieval_cache_path <QUERY_CACHE.json> \
  --save_retrieval_cache
```

原生 IterRetGen：当前 `racp/run_exp.py` 中固定 `iter_num=3`。正式运行保持该设置，不要在
不同数据集间静默修改：

```bash
$PY racp/run_exp.py \
  --method_name iterretgen \
  --dataset_name <DATASET> --split <SPLIT> --gpu_id <GPU> \
  --save_note <DATASET>-iterretgen-no-rerank-full \
  --retrieval_topk 5 --no_reranker \
  --use_retrieval_cache --retrieval_cache_path <QUERY_CACHE.json> \
  --save_retrieval_cache
```

IRCoT no-reranker：使用定制入口，固定 `max_iter=2`、每轮 top5，因此 Retrieval
Recall 记为 @10。可使用 1-4 张 GPU：

```bash
$PY examples/methods/run_exp.py \
  --method_name ircot-no-rerank \
  --dataset_name <DATASET> --split <SPLIT> \
  --gpu_id 0,1,2,3 \
  --save_note <DATASET>-ircot-no-rerank-cache-full \
  --retrieval_cache_path <QUERY_CACHE.json> \
  --retrieval_topk 5 --retrieval_batch_size 1024 \
  --max_iter 2 --max_tokens 32 --gpu_memory_utilization 0.75
```

Adaptive-RAG：

```bash
$PY racp/run_exp.py \
  --method_name adaptive \
  --dataset_name <DATASET> --split <SPLIT> --gpu_id <GPU> \
  --save_note <DATASET>-adaptive-rag-no-rerank-full \
  --adaptive_model_path <AUDITED_LOCAL_CLASSIFIER> \
  --adaptive_batch_size 16 --adaptive_max_iter 2 \
  --retrieval_topk 5 --no_reranker \
  --use_retrieval_cache --retrieval_cache_path <QUERY_CACHE.json> \
  --save_retrieval_cache
```

Adaptive-RAG 没有官方可直接使用的 classifier checkpoint；复现手册引用的第三方
`illuminoplanet/combined_flan_t5_xl_classifier` 和旧代码中的
`illuminoplanet/adaptive-rag-classifier` 当前均不可获取。本地也没有该模型。入口现要求
显式传入已审计的本地 checkpoint，并把多跳 IRCoT 分支固定为 `max_iter=2`。正式全量前
仍必须检查 A/B/C 标签映射和三条分支的检索语义；未经审计不得标为 FINAL。

FLARE（原方法 canonical 每次主动检索 top5）：

```bash
$PY racp/run_exp.py \
  --method_name flare \
  --dataset_name <DATASET> --split <SPLIT> --gpu_id <GPU> \
  --save_note <DATASET>-flare-no-rerank-top5-full \
  --retrieval_topk 5 --no_reranker \
  --flare_threshold 0.2 --flare_look_ahead_steps 64 \
  --flare_max_generation_length 256 --flare_max_iter 5 \
  --use_retrieval_cache --retrieval_cache_path <QUERY_CACHE.json> \
  --save_retrieval_cache
```

FLARE 依赖逐 token 概率并按置信度生成新检索 query。原问题可命中公共缓存，运行时产生的
query 通常需要在线检索；每个样本最多五轮，不能按普通批量 RAG 的速度估计全量耗时。
其检索指标按 canonical 每次 top5 记录为 Retrieval Recall@5。

TRACE（原方法检索 top5，最终使用五条推理链）：

```bash
$PY racp/run_exp.py \
  --method_name trace \
  --dataset_name <DATASET> --split <SPLIT> --gpu_id <GPU> \
  --save_note <DATASET>-trace-no-rerank-top5-full \
  --retrieval_topk 5 --no_reranker \
  --use_retrieval_cache --retrieval_cache_path <QUERY_CACHE.json> \
  --save_retrieval_cache
```

TRACE 强制使用 Hugging Face generator，因为三元组抽取和推理链构造需要输出 logits；还会
额外加载 E5 encoder 选择 demonstrations。它与主表使用同一 Llama-3.1 生成模型和 BGE
retriever，但执行框架不同且成本显著高于 Standard RAG，正式结果必须同时报告耗时；检索
指标按输入 top5 记录为 Retrieval Recall@5。

### 3.7 运行前和运行后检查

运行前：

```bash
git status --short
nvidia-smi
pgrep -af 'run_racp.py|run_exp.py|VLLM::EngineCore' || true
```

运行后：

```bash
find racp/output -maxdepth 2 -name metric_score.txt -printf '%T@ %h\n' | sort -nr | head
cat racp/output/<FINAL_DIR>/metric_score.txt
rg -n 'Traceback|RuntimeError|CUDA out of memory| failed at ' racp/output/<FINAL_DIR>/run.log
```

确认全量结果有效后，更新本文档的四处内容：进度总表、正式实验编号、FINAL 结果表、
更新日志。不要登记 smoke、失败运行或未采用的调参结果。

## 4. 主结果实验矩阵

### 4.1 进度总表

| 方法 | HotpotQA | 2Wiki | MuSiQue | NQ |
|---|---|---|---|---|
| No-RAG | `FINAL` | `FINAL` | `FINAL` | `FINAL` |
| Standard RAG | `FINAL` | `FINAL` | `FINAL` | `FINAL` |
| IterRetGen | `FINAL` | `FINAL` | `FINAL` | `FINAL` |
| Full-QD | `FINAL` | `FINAL` | `FINAL` | `N/A` |
| IRCoT | `FINAL` | `FINAL` | `FINAL` | `N/A` |
| FLARE | `FINAL` | `FINAL` | `FINAL` | `FINAL` |
| TRACE | `PAUSED` | `PAUSED` | `FAILED` | `N/A` |
| Adaptive-RAG | `TODO` | `TODO` | `TODO` | `TODO` |
| EFC-RAG | `FINAL` | `FINAL` | `FINAL` | `FINAL` |

NQ 主表不运行 Full-QD；IRCoT 仅在需要展示固定多轮检索对单跳任务的额外成本时加入补充表。

### 4.2 正式实验编号

| ID | 数据集 | 方法 | 状态 | 目标/备注 |
|---|---|---|---|---|
| `HP-NR-01` | HotpotQA | No-RAG | `FINAL` | 闭卷下界；`zero-shot` / `naive_run()` |
| `HP-NR-02` | HotpotQA | Standard RAG | `FINAL` | `naive` CLI；统一 no-refiner、no-reranker、top10 全量结果 |
| `HP-NR-03` | HotpotQA | IterRetGen | `FINAL` | 主干基线 |
| `HP-NR-04` | HotpotQA | Full-QD | `FINAL` | 固定拆分基线 |
| `HP-NR-05` | HotpotQA | IRCoT | `FINAL` | 经典多轮推理检索基线 |
| `HP-NR-06` | HotpotQA | Adaptive-RAG | `TODO` | 直接 router 基线；先验证 classifier checkpoint |
| `HP-NR-07` | HotpotQA | EFC-RAG | `FINAL` | 完整方法，final context K=10 |
| `HP-NR-08` | HotpotQA | FLARE | `FINAL` | GPU 0；每次检索 top5，固定五轮；全量耗时4:49:24 |
| `HP-NR-09` | HotpotQA | TRACE | `PAUSED` | 2026-07-15按用户指令停止；无可登记指标；完整`save_triples.json`已保留，恢复时用新目录重跑reasoning chain |
| `2W-NR-01` | 2Wiki | No-RAG | `FINAL` | 闭卷下界；`zero-shot` / `naive_run()` |
| `2W-NR-02` | 2Wiki | Standard RAG | `FINAL` | `naive` CLI；统一 no-refiner、no-reranker、top10 全量结果 |
| `2W-NR-03` | 2Wiki | IterRetGen | `FINAL` | 原生三轮主干基线，每轮 top5 |
| `2W-NR-04` | 2Wiki | Full-QD | `FINAL` | 固定拆分基线；final top10，全量 12,576 条 |
| `2W-NR-05` | 2Wiki | IRCoT | `FINAL` | 四卡；`max_iter=2`、每轮 top5、无 reranker |
| `2W-NR-06` | 2Wiki | Adaptive-RAG | `TODO` | router 基线 |
| `2W-NR-07` | 2Wiki | EFC-RAG | `FINAL` | 完整方法，final context K=10 |
| `2W-NR-08` | 2Wiki | FLARE | `FINAL` | 主动检索基线；每次检索top5、固定五轮；全量耗时6:12:41 |
| `2W-NR-09` | 2Wiki | TRACE | `PAUSED` | 2026-07-15按用户指令停止；尚未生成`save_triples.json`且无可登记指标，恢复时需从triple extraction重跑 |
| `MU-NR-01` | MuSiQue | No-RAG | `FINAL` | 闭卷下界；不加载 retriever，与 2Wiki Full-QD 并行完成 |
| `MU-NR-02` | MuSiQue | Standard RAG | `FINAL` | no-refiner、no-reranker、top10；完整 dev 2,417 条 |
| `MU-NR-03` | MuSiQue | IterRetGen | `FINAL` | 原生三轮、每轮 top5；完整 dev 2,417 条 |
| `MU-NR-04` | MuSiQue | Full-QD | `FINAL` | final top10；完整 dev 2,417 条 |
| `MU-NR-05` | MuSiQue | IRCoT | `FINAL` | GPU 0-3 TP；两轮、每轮 top5；完整 dev 2,417 条 |
| `MU-NR-06` | MuSiQue | Adaptive-RAG | `TODO` | router 基线 |
| `MU-NR-07` | MuSiQue | EFC-RAG | `FINAL` | final top10；完整 dev 2,417 条 |
| `MU-NR-08` | MuSiQue | FLARE | `FINAL` | 完整dev 2,417条；top5、固定五轮；327条触发检索 |
| `MU-NR-09` | MuSiQue | TRACE | `FAILED` | 全量在796/2,417（`dev_796`）处因空path query触发`np.concatenate([])`异常；无metric/intermediate/prediction，不登记分数；完整triple缓存可复用，修复后新目录重跑 |
| `NQ-NR-01` | NQ | No-RAG | `FINAL` | 单跳闭卷下界；完整 test 3,610 条 |
| `NQ-NR-02` | NQ | Standard RAG | `FINAL` | cache-hit top10、no-reranker；完整test 3,610条 |
| `NQ-NR-03` | NQ | IterRetGen | `FINAL` | 完整test 3,610条；原生三轮、每轮top5、最终列表5篇 |
| `NQ-NR-04` | NQ | Adaptive-RAG | `TODO` | 单跳 router 基线 |
| `NQ-NR-05` | NQ | EFC-RAG | `FINAL` | final top10；完整EFC路由；direct占89.92% |
| `NQ-NR-06` | NQ | FLARE | `FINAL` | 完整test 3,610条；top5、固定五轮；511条触发检索 |

## 5. 已锁定的论文结果

所有数值按百分数记录。表中 `FINAL` 项均已通过正式全量、输出覆盖、配置和日志审计。

### 5.1 HotpotQA 主结果

| ID | 方法 | EM | F1 | Acc | Precision | Recall | Retrieval Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| `HP-NR-01` | No-RAG | 17.68 | 26.54 | 23.27 | 27.22 | 29.63 | N/A |
| `HP-NR-02` | Standard RAG | 33.57 | 44.82 | 39.93 | 46.36 | 47.04 | 65.12 (@10) |
| `HP-NR-03` | IterRetGen | 34.67 | 45.66 | 41.13 | 47.12 | 47.97 | 62.86 (@10) |
| `HP-NR-04` | Full-QD | 33.67 | 44.59 | 40.23 | 46.16 | 46.90 | 65.04 (@10) |
| `HP-NR-05` | IRCoT | **36.26** | **47.50** | 39.57 | **51.40** | 46.93 | 64.77 (@10) |
| `HP-NR-07` | EFC-RAG | 35.85 | 47.21 | **42.35** | 48.89 | **49.43** | **71.51 (@10)** |
| `HP-NR-08` | FLARE | 16.04 | 23.26 | 21.27 | 23.88 | 26.54 | 1.93 (@5) |

最终结果目录：

| ID | 输出目录 |
|---|---|
| `HP-NR-01` | [`output/hotpotqa_2026_07_12_14_26_hotpotqa-zero-shot-full`](output/hotpotqa_2026_07_12_14_26_hotpotqa-zero-shot-full) |
| `HP-NR-02` | [`output/hotpotqa_2026_07_12_14_26_hotpotqa-standard-rag-no-rerank-top10-full`](output/hotpotqa_2026_07_12_14_26_hotpotqa-standard-rag-no-rerank-top10-full) |
| `HP-NR-03` | [`output/hotpotqa_2026_06_05_16_46_iterretgen-native-no-reranker-gpu3`](output/hotpotqa_2026_06_05_16_46_iterretgen-native-no-reranker-gpu3) |
| `HP-NR-04` | [`output/hotpotqa_2026_07_10_09_27_full-qd-no-rerank-top10-full`](output/hotpotqa_2026_07_10_09_27_full-qd-no-rerank-top10-full) |
| `HP-NR-05` | [`output/hotpotqa_2026_07_09_14_27_ircot-no-rerank-cache-full`](output/hotpotqa_2026_07_09_14_27_ircot-no-rerank-cache-full) |
| `HP-NR-07` | [`output/hotpotqa_2026_07_09_15_52_efc-static-bridge-final-top10-full`](output/hotpotqa_2026_07_09_15_52_efc-static-bridge-final-top10-full) |
| `HP-NR-08` | [`output/hotpotqa_2026_07_13_09_22_hotpotqa-flare-no-rerank-top5-full-v2`](output/hotpotqa_2026_07_13_09_22_hotpotqa-flare-no-rerank-top5-full-v2) |

FLARE 使用完整 dev 7,405 条样本，`test_sample_num: null`、`refiner_name: null`、
`use_reranker: false`、`do_sample: false`。锁定参数为 threshold=0.2、look-ahead=64、
总生成上限256、`max_iter=5`、每次动态检索top5；每条样本均执行5轮。1,353条样本至少
触发一次检索，6,052条从未触发；共1,406次动态检索，所有样本均存在合法的
`retrieval_result`（未触发时为空列表）。因此 Recall@5 的1.93%包含未触发样本的零召回，
不是字段缺失。总耗时4:49:24；完整启动日志、原始命令和7405条中间结果均保存在FINAL
目录，日志无Traceback、RuntimeError或OOM。

### 5.2 HotpotQA EFC-RAG 效率与路由

| 指标 | 最终值 |
|---|---:|
| 平均 LLM 调用 | 2.8718 |
| 平均检索调用 | 1.9332 |
| 平均候选池文档数 | 25.5246 |
| 最终文档数 | 10.0000 |
| direct | 955 / 12.90% |
| static-QD | 460 / 6.21% |
| generation-guided | 5,990 / 80.89% |

其余方法的效率数据需要从正式最终日志统一提取后再写入效率表。

### 5.3 2Wiki 主结果

| ID | 方法 | EM | F1 | Acc | Precision | Recall | Retrieval Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| `2W-NR-01` | No-RAG | 17.22 | 26.51 | 30.93 | 25.01 | 35.28 | N/A |
| `2W-NR-02` | Standard RAG | 16.56 | 25.60 | 25.37 | 25.32 | 30.59 | 46.20 (@10) |
| `2W-NR-03` | IterRetGen | 16.90 | 26.59 | 30.16 | 25.58 | 34.69 | 42.48 (@10) |
| `2W-NR-04` | Full-QD | 18.23 | 26.84 | 25.56 | 26.69 | 30.86 | 49.57 (@10) |
| `2W-NR-05` | IRCoT | **33.07** | **39.39** | **35.46** | **40.99** | **39.41** | 50.15 (@10) |
| `2W-NR-07` | EFC-RAG | 20.35 | 29.37 | 28.92 | 29.04 | 33.98 | **58.17 (@10)** |
| `2W-NR-08` | FLARE | 9.37 | 20.45 | 32.11 | 17.51 | 36.22 | 2.27 (@5) |

最终结果目录：

| ID | 输出目录 | Git commit |
|---|---|---|
| `2W-NR-01` | [`output/2wikimultihopqa_2026_07_12_14_26_2wiki-zero-shot-full`](output/2wikimultihopqa_2026_07_12_14_26_2wiki-zero-shot-full) | `3b4d780` |
| `2W-NR-02` | [`output/2wikimultihopqa_2026_07_10_15_04_2wiki-naive-no-rerank-top10-full`](output/2wikimultihopqa_2026_07_10_15_04_2wiki-naive-no-rerank-top10-full) | `e6b4456` |
| `2W-NR-03` | [`output/2wikimultihopqa_2026_07_10_15_52_2wiki-iterretgen-no-rerank-full`](output/2wikimultihopqa_2026_07_10_15_52_2wiki-iterretgen-no-rerank-full) | `3e62049` |
| `2W-NR-04` | [`output/2wikimultihopqa_2026_07_12_14_31_2wiki-full-qd-no-rerank-top10-full`](output/2wikimultihopqa_2026_07_12_14_31_2wiki-full-qd-no-rerank-top10-full) | `3b4d780` + 未提交兼容性 diff |
| `2W-NR-05` | [`output/2wikimultihopqa_2026_07_12_17_09_2wiki-ircot-no-rerank-cache-full-v2`](output/2wikimultihopqa_2026_07_12_17_09_2wiki-ircot-no-rerank-cache-full-v2) | `3b4d780` + 未提交兼容性 diff |
| `2W-NR-07` | [`output/2wikimultihopqa_2026_07_12_11_11_2wiki-efc-no-rerank-top10-full`](output/2wikimultihopqa_2026_07_12_11_11_2wiki-efc-no-rerank-top10-full) | `3e62049` |
| `2W-NR-08` | [`output/2wikimultihopqa_2026_07_13_14_26_2wiki-flare-no-rerank-top5-full`](output/2wikimultihopqa_2026_07_13_14_26_2wiki-flare-no-rerank-top5-full) | `3b4d780` + 未提交兼容性 diff |

该运行使用完整 dev 12,576 条样本；每条最终使用 10 篇文档，`refiner_name: null`、
`use_reranker: false`、`do_sample: false`。`racp/run_exp.py` 未生成 `run.log`，但
`config.yaml`、`metric_score.txt`、`intermediate_data.json` 和完整 retrieval cache 均已审计。

IterRetGen 同样使用完整 dev 12,576 条样本，固定运行三轮、每轮检索 top5；三轮输出和
最终预测均覆盖全部样本。配置中的指标键为 `retrieval_recall_top10`，但当前

FLARE 使用完整 dev 12,576 条样本，`test_sample_num: null`、`refiner_name: null`、
`use_reranker: false`、`do_sample: false`。锁定参数为 threshold=0.2、look-ahead=64、
总生成上限256、`max_iter=5`、每次动态检索top5；每条样本均执行5轮。1,524条样本至少
触发一次检索，11,052条从未触发；共1,610次动态检索，所有样本均存在合法的
`retrieval_result`。Recall@5的2.27%包含未触发样本的零召回，不是字段缺失。总耗时
6:12:41；完整日志、原始命令及12,576条中间结果均保存在FINAL目录，日志无Traceback、
RuntimeError或OOM。
`IterativePipeline` 写入最终 `retrieval_result` 的是第三轮 5 篇文档，而不是三轮并集；
论文报告时必须同时披露“三轮 × top5”和该评测语义，不能把它解释为最终上下文含 10 篇。

Full-QD 使用原问题 top20、两条 Llama-3.1 子问题各 top5，经 title-dedup 选择 final
top10；完整输出和预测均覆盖 12,576 条，并保存 462MB retrieval-topk bundle。配置为
`refiner_name: null`、`use_reranker: false`、`do_sample: false`，日志无 Traceback 或 OOM。

IRCoT 使用四卡 tensor parallel，固定两轮、每轮 top5。首轮未结束的样本继续按 thought
检索，最终 retrieval list 为累积去重后的最多 10 篇文档；部分样本不足 10 篇，因此评测
发出 length warning，但没有丢样本。最终预测、检索结果均覆盖 12,576/12,576。配置为
`refiner_name: null`、`use_reranker: false`、`do_sample: false`，退出时只有 vLLM semaphore/
shared-memory 清理 warning，无 Traceback、RuntimeError 或 OOM。

### 5.4 2Wiki EFC-RAG 效率与路由

| 指标 | 最终值 |
|---|---:|
| 平均 LLM 调用 | 2.9051 |
| 平均检索调用 | 2.0161 |
| 平均候选池文档数 | 26.9002 |
| 最终文档数 | 10.0000 |
| direct | 1,207 / 9.60% |
| static-QD | 1,410 / 11.21% |
| generation-guided | 9,959 / 79.19% |
| planner failure | 0.68% |
| planner repair | 0.57% |
| support title recall@10 | 48.90% |
| both support title hit | 28.96% |

分路由答案表现：`direct` EM/F1 34.88/41.89，`static_qd` 15.89/23.21，
`generation_guided` 19.22/28.73。完整运行覆盖 12,576 条样本，final top10，日志无
Traceback、OOM 或失败结束标记。

### 5.4.1 2Wiki IRCoT 与 EFC-RAG 差距诊断

对两个 FINAL 的 12,576 条中间结果使用同一 supporting-title 口径重新统计：

| 指标 | IRCoT | EFC-RAG |
|---|---:|---:|
| EM / F1 | 33.07 / 39.39 | 20.35 / 29.37 |
| support title recall | 45.27% | **48.90%** |
| both support titles hit | 23.54% | **28.96%** |
| 平均最终文档数 | 8.1566 | 10.0000 |
| 平均 LLM 调用 | 2.9688 | 2.9051 |
| 平均检索调用 | 2.0000 | 2.0161 |
| 预测平均词数 | 2.29 | 4.41 |
| 超过 10 词的预测 | 0.64% | 14.46% |

按最终命中的 supporting title 数量分组后，IRCoT 仍持续更高：

| supporting-title hits | IRCoT EM / F1 | EFC-RAG EM / F1 |
|---:|---:|---:|
| 0 | 16.18 / 21.25 | 8.59 / 16.13 |
| 1 | 33.09 / 39.79 | 20.34 / 29.39 |
| 2 | 48.49 / 56.05 | 29.43 / 39.95 |

因此主要差距不是“EFC 没检到证据”，而是证据到答案的转换：IRCoT 在每轮 prompt 中保留
显式 thought，用一条固定多跳 demonstration 引导逐步推理，并对 12,183/12,576 条未在两轮
内自然结束的样本追加严格的 shortest-answer finalization。EFC 的 probe 主要用于路由和
生成 missing-hop query，最终答案阶段仍是普通 documents-to-short-answer prompt，没有把
probe/桥接关系作为显式推理轨迹传给 final generator。EFC 还固定输入 10 篇文档，噪声高于
IRCoT 平均 8.16 篇；其长答案和“没有相关信息”式拒答明显更多。

当前 IRCoT 的 `max_iter=2` 实际语义是“两轮 thought generation + 必要时一次 final answer
generation”，96.87% 样本共调用三次 LLM。论文比较效率时必须披露，不能把它写成严格两次
LLM 调用。EFC 的改进方向应优先放在 structured evidence reasoning、将 probe/bridge 显式
传入最终 prompt、动态减少 final context 噪声和更严格的答案格式，而不是继续单独提高
retrieval recall。任何修改必须先在 HotpotQA/MuSiQue 上确定并冻结，不能直接在 2Wiki dev
上按最终答案调参后汇报。

### 5.5 MuSiQue 主结果

| ID | 方法 | EM | F1 | Acc | Precision | Recall | Retrieval Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| `MU-NR-01` | No-RAG | 3.56 | 9.94 | 6.37 | 10.46 | 11.73 | N/A |
| `MU-NR-02` | Standard RAG | 6.45 | 14.11 | 8.98 | 14.83 | 15.40 | 32.02 (@10) |
| `MU-NR-03` | IterRetGen | 8.07 | 15.22 | 10.47 | 15.92 | 16.63 | 29.58 (@10 配置键；最终5篇) |
| `MU-NR-04` | Full-QD | 7.61 | 15.42 | 10.30 | 16.17 | 16.60 | 33.88 (@10) |
| `MU-NR-05` | IRCoT | **10.38** | 17.42 | 12.00 | **19.40** | 17.40 | 30.33 (@10) |
| `MU-NR-07` | EFC-RAG | 9.93 | **17.65** | **13.24** | 18.31 | **19.21** | **42.08 (@10)** |
| `MU-NR-08` | FLARE | 2.15 | 5.42 | 3.89 | 5.75 | 6.68 | 1.32 (@5) |

最终结果目录：

| ID | 输出目录 | Git commit |
|---|---|---|
| `MU-NR-01` | [`output/musique_2026_07_12_16_14_musique-zero-shot-full`](output/musique_2026_07_12_16_14_musique-zero-shot-full) | `3b4d780` + 未提交兼容性 diff |
| `MU-NR-02` | [`output/musique_2026_07_12_19_43_musique-naive-no-rerank-top10-full`](output/musique_2026_07_12_19_43_musique-naive-no-rerank-top10-full) | `3b4d780` + 未提交兼容性 diff |
| `MU-NR-03` | [`output/musique_2026_07_12_21_35_musique-iterretgen-no-rerank-full`](output/musique_2026_07_12_21_35_musique-iterretgen-no-rerank-full) | `3b4d780` + 未提交兼容性 diff |
| `MU-NR-04` | [`output/musique_2026_07_12_19_43_musique-full-qd-no-rerank-top10-prepare`](output/musique_2026_07_12_19_43_musique-full-qd-no-rerank-top10-prepare) | `3b4d780` + 未提交兼容性 diff |
| `MU-NR-05` | [`output/musique_2026_07_12_22_30_musique-ircot-no-rerank-cache-full`](output/musique_2026_07_12_22_30_musique-ircot-no-rerank-cache-full) | `3b4d780` + 未提交兼容性 diff |
| `MU-NR-07` | [`output/musique_2026_07_12_20_15_musique-efc-no-rerank-top10-prepare`](output/musique_2026_07_12_20_15_musique-efc-no-rerank-top10-prepare) | `3b4d780` + 未提交兼容性 diff |
| `MU-NR-08` | [`output/musique_2026_07_14_09_25_musique-flare-no-rerank-top5-full`](output/musique_2026_07_14_09_25_musique-flare-no-rerank-top5-full) | `3b4d780` + 未提交兼容性 diff |

该运行使用完整 dev 2,417 条样本，预测覆盖 2,417/2,417，`refiner_name: null`、
`do_sample: false`，且没有 `retrieval_result`。基础配置中的 `use_reranker: true` 对
No-RAG 的 `NoOpRetriever` 不生效；没有执行检索或 rerank。日志无 Traceback、OOM 或失败标记。

Standard RAG 使用完整 dev 2,417 条样本，每条 final context 恰为 10 篇文档；配置为
`test_sample_num: null`、`refiner_name: null`、`use_reranker: false`、`do_sample: false`。
完整启动日志已复制到 FINAL 目录，日志无 Traceback、RuntimeError 或 OOM。

Full-QD 同样覆盖完整 dev 2,417 条，每条 final context 恰为 10 篇文档；2,386 条完整接受
两条子问题、25 条部分接受、6 条 fallback，平均接受 1.9847 条子问题。配置和日志审计
通过，无 Traceback、RuntimeError 或 OOM。

IterRetGen 的三轮生成与检索均覆盖 2,417 条，每轮 top5；最终 `retrieval_result` 仅保存第三轮
5篇文档，因此评测发出 length warning。配置指标键仍名为 `retrieval_recall_top10`，表中保留
原始键值但明确标注实际最终列表为5篇，不能解释成三轮并集或真实 top10。

EFC-RAG 每条 final context 恰为10篇；generation-guided/static-QD/direct 分别为
2,244/138/35 条。平均 LLM 调用2.9876、检索调用2.0426、候选池27.6719篇，日志无
Traceback、RuntimeError 或 OOM。

IRCoT 两轮中间输出均覆盖2,417条，2,327条（96.28%）在两轮 thought 后追加 final-answer
调用，平均 LLM 调用2.9628。累积去重后的最终文档数为5–10篇、平均8.5937篇，因此评测
出现 length warning；没有丢样本，只有9条空预测。完整日志仅有 vLLM semaphore/shared-memory
退出清理 warning，无 Traceback、RuntimeError 或 OOM。

FLARE 覆盖完整dev 2,417条，threshold=0.2、look-ahead=64、总生成上限256、固定最多五轮、
动态检索top5。327条样本至少触发一次检索，2,090条从未触发，共340次动态检索；最终
`retrieval_result` 分别为5篇或合法空列表。总耗时1:23:27，完整日志和命令已归档，
无Traceback、RuntimeError或OOM。

### 5.6 NQ 主结果

| ID | 方法 | EM | F1 | Acc | Precision | Recall | Retrieval Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| `NQ-NR-01` | No-RAG | 21.63 | 32.42 | 35.54 | 31.17 | 42.33 | N/A |
| `NQ-NR-02` | Standard RAG | 36.95 | 49.15 | **54.49** | 47.63 | **59.88** | **82.74 (@10)** |
| `NQ-NR-03` | IterRetGen | **37.53** | **49.38** | 53.05 | **48.12** | 58.12 | 75.62 (@10配置键；最终5篇) |
| `NQ-NR-05` | EFC-RAG | 35.32 | 47.44 | 52.44 | 45.87 | 57.94 | 79.86 (@10) |
| `NQ-NR-06` | FLARE | 21.22 | 30.85 | 33.43 | 29.95 | 39.53 | 3.21 (@5) |

最终结果目录：

| ID | 输出目录 | Git commit |
|---|---|---|
| `NQ-NR-01` | [`output/nq_2026_07_12_19_21_nq-zero-shot-full`](output/nq_2026_07_12_19_21_nq-zero-shot-full) | `3b4d780` + 未提交兼容性 diff |
| `NQ-NR-02` | [`output/nq_2026_07_14_09_54_nq-standard-rag-no-rerank-top10-full`](output/nq_2026_07_14_09_54_nq-standard-rag-no-rerank-top10-full) | `3b4d780` + 未提交兼容性 diff |
| `NQ-NR-03` | [`output/nq_2026_07_14_10_36_nq-iterretgen-no-rerank-full`](output/nq_2026_07_14_10_36_nq-iterretgen-no-rerank-full) | `3b4d780` + 未提交兼容性 diff |
| `NQ-NR-05` | [`output/nq_2026_07_14_09_54_nq-efc-no-rerank-top10-full`](output/nq_2026_07_14_09_54_nq-efc-no-rerank-top10-full) | `3b4d780` + 未提交兼容性 diff |
| `NQ-NR-06` | [`output/nq_2026_07_14_10_36_nq-flare-no-rerank-top5-full`](output/nq_2026_07_14_10_36_nq-flare-no-rerank-top5-full) | `3b4d780` + 未提交兼容性 diff |

该运行使用完整 test 3,610 条样本，预测覆盖 3,610/3,610，`test_sample_num: null`、
`refiner_name: null`、`do_sample: false`。No-RAG 使用 `NoOpRetriever`，没有执行检索或
rerank；已将完整启动日志复制到 FINAL 目录，日志无 Traceback、RuntimeError 或 OOM。

Standard RAG 使用完整test 3,610条，每条final context恰为10篇；原问题全部命中统一top20
缓存并截取top10，`test_sample_num: null`、`refiner_name: null`、`use_reranker: false`、
`do_sample: false`。完整日志与命令已归档，无Traceback、RuntimeError或OOM。

EFC-RAG 同样覆盖3,610/3,610条且每条最终选择10篇。direct/generation-guided/static-QD
分别为3,246/352/12条，平均LLM调用2.1008、检索调用1.1042、候选池20.5006篇。
generation-guided分支EM/F1为40.34/53.01，但整体EM/F1仍比Standard低1.63/1.71；其
Retrieval Recall@10也低2.88，说明NQ上重路由和重选文档没有超过直接top10 RAG。
完整日志、命令、prompt cache及动态检索cache均已归档，无Traceback、RuntimeError或OOM。

IterRetGen 覆盖完整test 3,610条，固定三轮且每轮检索top5，三轮检索、生成和最终预测均
无缺失；最终`retrieval_result`仅保存第三轮5篇。总耗时约39:34，配置指标键仍为
`retrieval_recall_top10`，表中已明确其实际评测列表为5篇。其EM/F1比Standard RAG高
0.58/0.23，但Retrieval Recall低7.12。

FLARE 覆盖完整test 3,610条，锁定参数与其他数据集一致。511条样本至少触发一次检索，
3,099条从未触发，共573次动态检索；最终列表为top5或合法空列表。总耗时2:17:38，
EM/F1略低于No-RAG 0.41/1.57，说明单跳NQ上低置信度主动检索没有带来收益。两项完整
日志与命令均已归档，无Traceback、RuntimeError或OOM。

### 5.7 其余尚无可写入论文的最终结果

| 数据集 | 当前结论 |
|---|---|
| MuSiQue | No-RAG、Standard RAG、IterRetGen、Full-QD、IRCoT、EFC-RAG、FLARE已完成 |
| NQ | No-RAG、Standard RAG、IterRetGen、EFC-RAG、FLARE已完成；Adaptive暂停 |

## 6. 消融实验计划

完整消融以 HotpotQA 为主；MuSiQue 只补最关键的 router 对照，避免重复消耗。

| ID | 数据集 | 消融 | 状态 | 证明目标 |
|---|---|---|---|---|
| `HP-AB-01` | HotpotQA | EFC-RAG 完整方法 | `FINAL` | 完整模型，复用 `HP-NR-07` |
| `HP-AB-02` | HotpotQA | direct-only | `FINAL` | 完整dev 7,405条；EFC force-direct，仍保留probe+final两次调用和EFC final-top10选择，不等同Standard RAG |
| `HP-AB-03` | HotpotQA | generation-guided-only | `FINAL` | 完整dev 7,405条；强制generation-guided并关闭static fallback；7,371条generation-guided、34条direct fallback |
| `HP-AB-04` | HotpotQA | 去除 static-QD | `FINAL` | 完整dev 7,405条；auto router保持不变，只关闭static fallback；动态query在线补齐到本次独立cache |
| `HP-AB-05` | HotpotQA | Full-QD | `FINAL` | 固定分解对照，复用 `HP-NR-04` |
| `HP-AB-06` | HotpotQA | w/o role weight | `FINAL` | 完整dev 7,405条；仅将`role_weight`从0.30改为0，其余锁定参数不变；cache-only |
| `HP-AB-07` | HotpotQA | w/o original seed reservation | `FINAL` | 完整dev 7,405条；仅将`original_seed_count`从4改为0；cache-only |
| `HP-AB-08` | HotpotQA | w/o source weight | `FINAL` | 完整dev 7,405条；仅将`source_weight`从0.05改为0；cache-only；汇总指标与完整EFC完全相同 |
| `HP-AB-09` | HotpotQA | RRF-only selector | `FINAL` | 完整dev 7,405条；三个route统一只按RRF选final top10；完整EFC cache-only；默认EFC路径不变 |
| `HP-AB-10` | HotpotQA | title-dedup-only selector | `FINAL` | 完整dev 7,405条；三个route仅保留融合RRF与title-diverse first pass；cache-only |
| `HP-AB-11` | HotpotQA | controlled final top5 | `FINAL` | 完整dev 7,405条；仅将锁定EFC的final/评测K从10改为5；cache-only |
| `HP-AB-12` | HotpotQA | matched maximum retrieval budget | `FINAL` | 完整dev 7,405条；initial5、最多1条generation query×top5、关闭static；最多2次逻辑检索和10篇raw文档 |
| `HP-AB-13` | HotpotQA | w/o redundancy penalty | `FINAL` | 完整dev 7,405条；仅将`redundancy_weight=0.01`改为0；cache-only |
| `MU-AB-01` | MuSiQue | EFC-RAG 完整方法 | `FINAL` | 复用 `MU-NR-07`，无需重跑 |
| `MU-AB-02` | MuSiQue | generation-guided-only | `FINAL` | 完整dev 2,417条；强制generation-guided并关闭static fallback；2,407条generation-guided、10条direct fallback |
| `MU-AB-03` | MuSiQue | Full-QD | `FINAL` | 复用 `MU-NR-04`，无需重跑 |

`HP-AB-03` 的 FINAL 输出为
`output/hotpotqa_2026_07_14_13_49_hotpotqa-efc-force-generation-guided-no-static-top10-full`。
结果覆盖7,405/7,405条，运行时间13:49:04--15:17:04，总耗时1:28:00；EM/F1/Acc/
Precision/Recall/Retrieval Recall@10分别为35.61/46.95/42.07/48.65/49.12/71.60。
强制路由决策为100% generation-guided，但missing-hop planner失败或修复后有34条回退direct，
实际route为7,371条generation-guided、34条direct、0条static-QD；平均LLM调用3.0000、
检索调用1.9954、候选池26.2138篇、最终10篇。config、metric、intermediate、run.log、
run_command和独立retrieval cache均已归档，未发现Traceback、RuntimeError或OOM。

`HP-AB-02` 的 FINAL 输出为
`output/hotpotqa_2026_07_15_10_01_hotpotqa-efc-force-direct-top10-full`。
结果覆盖7,405/7,405条，运行时间10:01:08--10:43:54，总耗时42:46；EM/F1/Acc/
Precision/Recall/Retrieval Recall@10分别为33.59/44.58/39.91/46.14/46.83/65.63。
全部样本均为direct，平均LLM调用2.0000、检索调用1.0000、候选池20篇、最终10篇。
相对完整EFC，EM/F1/Retrieval Recall分别低2.26/2.63/5.88个百分点，证明probe后的
自适应扩展与选择带来实质收益。该变体仍包含probe和final两次生成，不能记为Standard RAG。

`MU-AB-02` 的 FINAL 输出为
`output/musique_2026_07_15_10_01_musique-efc-force-generation-guided-no-static-top10-full`。
结果覆盖2,417/2,417条，运行时间10:01:08--10:36:56，总耗时35:48；EM/F1/Acc/
Precision/Recall/Retrieval Recall@10分别为9.64/17.45/12.95/18.12/19.04/41.54。
实际route为2,407条generation-guided和10条direct fallback；平均LLM调用3.0000、检索
调用1.9959、候选池27.7245篇、最终10篇。完整EFC的EM/F1/Retrieval Recall分别高
0.29/0.20/0.54个百分点，说明MuSiQue上的自适应router收益为正但幅度较小。

`HP-AB-04` 的 FINAL 输出为
`output/hotpotqa_2026_07_15_11_05_hotpotqa-efc-auto-no-static-top10-full-v3-online-fill`。
结果覆盖7,405/7,405条，运行时间11:05:31--12:07:14，总耗时1:01:43；EM/F1/Acc/
Precision/Recall/Retrieval Recall@10分别为35.76/47.09/42.15/48.80/49.22/71.49。
实际route为1,415条direct、5,990条generation-guided、0条static-QD；平均LLM/检索
调用2.8089/1.8089，平均候选池25.1866篇、最终10篇。源cache只读，动态query miss
在线补齐后只保存到该FINAL目录；旧cache-only失败目录没有指标，仍不登记成绩。

`HP-AB-06` 的 FINAL 输出为
`output/hotpotqa_2026_07_15_10_16_hotpotqa-efc-no-role-weight-top10-full`。
结果覆盖7,405/7,405条，运行时间10:16:51--11:15:55，总耗时59:04；EM/F1/Acc/
Precision/Recall/Retrieval Recall@10分别为35.77/47.02/42.20/48.67/49.24/71.05。
只将`role_weight=0.30`改为0；平均LLM/检索调用2.8718/1.9332，平均候选池
25.5246篇、最终10篇，route为955/460/5,990条direct/static-QD/generation-guided。

`HP-AB-07` 的 FINAL 输出为
`output/hotpotqa_2026_07_15_11_05_hotpotqa-efc-no-original-seed-top10-full`。
结果覆盖7,405/7,405条，运行时间11:05:32--12:05:23，总耗时59:51；EM/F1/Acc/
Precision/Recall/Retrieval Recall@10分别为35.80/47.09/42.36/48.76/49.37/71.41。
只将`original_seed_count=4`改为0；调用预算、候选池和route分布与完整EFC相同。

`HP-AB-08` 的 FINAL 输出为
`output/hotpotqa_2026_07_15_11_05_hotpotqa-efc-no-source-weight-top10-full`。
结果覆盖7,405/7,405条，运行时间11:05:31--12:05:17，总耗时59:46；只将
`source_weight=0.05`改为0。EM/F1/Acc/Precision/Recall/Retrieval Recall@10分别为
35.85/47.21/42.35/48.89/49.43/71.51，与完整EFC六项汇总指标完全相同；调用预算、
候选池和route分布也相同。该结论只证明当前数据和设置下汇总指标无变化，不外推为该项
在所有数据集恒无作用。

`HP-AB-09` 的 FINAL 输出为
`output/hotpotqa_2026_07_15_12_46_hotpotqa-efc-rrf-only-top10-full`。
结果覆盖7,405/7,405条，运行时间12:46:34--13:41:43，总耗时55:09；EM/F1/Acc/
Precision/Recall/Retrieval Recall@10分别为33.30/43.67/39.30/45.19/46.04/67.13。
route和调用预算与完整EFC相同，但三个route均只按融合RRF选择final top10，绕过role/title/
source/redundancy、original seed reservation、title diversity和static配额。support-title recall
从完整EFC的68.11%降至47.45%，双支持标题命中率从51.24%降至22.96%，证明组合selector
整体有效；该结果不能单独归因于任一组件。

`HP-AB-10` 的 FINAL 输出为
`output/hotpotqa_2026_07_15_13_54_hotpotqa-efc-title-dedup-only-top10-full`。
结果覆盖7,405/7,405条，运行时间13:54:21--14:49:56，总耗时55:35；EM/F1/Acc/
Precision/Recall/Retrieval Recall@10分别为35.18/46.57/41.78/48.20/48.90/70.90。
7,405条均使用`efc_rrf_title_dedup_only`：先按融合RRF排序，每个小写title优先一篇，
不足10篇时再按RRF回填重复title；role/source/redundancy、original seed和static quota均
不参与选择。平均LLM/检索调用2.8718/1.9332、候选池25.5246篇、最终10篇，route与
完整EFC一致。相对RRF-only，EM/F1/answer-hit分别提高1.88/2.90/3.77个百分点；相对
完整EFC仍低0.68/0.64/0.61个百分点，说明title diversity恢复了selector增益的大部分，
但不能把全部增益只归因于title。

`HP-AB-11` 的 FINAL 输出为
`output/hotpotqa_2026_07_15_13_54_hotpotqa-efc-final-top5-controlled-full`。
结果覆盖7,405/7,405条，运行时间13:54:22--14:40:50，总耗时46:28；EM/F1/Acc/
Precision/Recall/Retrieval Recall@5分别为35.53/46.63/41.82/48.21/48.71/66.31。
除`final_topk=5`和评测K=5外，其余锁定参数、route、调用预算和候选池与完整EFC一致；
全部样本最终正好5篇。相对完整EFC的EM/F1仅低0.32/0.58个百分点，但Recall@5与
Recall@10预算不同，不能把5.20个百分点差值解释为同口径检索退化。该实验只缩短最终
context，不减少Probe、Planner或检索调用。

`HP-AB-12` 的 FINAL 输出为
`output/hotpotqa_2026_07_15_13_54_hotpotqa-efc-ircot-matched-budget10-full`。
结果覆盖7,405/7,405条，运行时间13:54:22--14:46:20，总耗时51:58；EM/F1/Acc/
Precision/Recall/Retrieval Recall@10分别为36.66/48.17/43.58/49.71/50.60/69.82。
该变体把initial/gen top-k均设为5、关闭static-QD，因而满足预先锁定的最多2次逻辑检索、
最多10篇raw文档和final K不超过10；实际平均LLM/检索调用2.8089/1.8089，候选池和最终
文档均值7.8357、范围5--10，route为1,415条direct和5,990条generation-guided。
它相对正式IRCoT高0.41 EM、0.67 F1和5.05个answer-hit百分点，但只匹配**最大检索预算**，
不匹配prompt、推理机制或逐样本LLM成本，且同时改变initial top-k、gen top-k和static路线，
不能作为单参数消融。运行从22,606键源cache读取，只在独立FINAL cache中新增4条动态query；
`run.log`把variable-K support-title诊断标作`@5`是显示标签问题，论文应写final-context
support-title recall，不把它解释为统一K=5。

`HP-AB-13` 的 FINAL 输出为
`output/hotpotqa_2026_07_15_13_54_hotpotqa-efc-no-redundancy-weight-top10-full`。
结果覆盖7,405/7,405条，运行时间13:54:21--14:53:41，总耗时59:20；EM/F1/Acc/
Precision/Recall/Retrieval Recall@10分别为36.04/47.46/42.59/49.16/49.65/72.21。
只将`redundancy_weight`从0.01改为0，route、调用预算、候选池和final10与完整EFC一致；
2,655条非direct样本的selected docs发生变化，证明开关实际生效。汇总F1比完整EFC高0.25，
因此当前HotpotQA没有观察到redundancy penalty的正收益。由于相同context下仍观察到少量
确定性生成文本漂移，这个小差值应经paired检验后再解释，不能声称移除惩罚可稳定提升。

W11四项均使用commit `063e29d`加运行时未提交的title-dedup隔离diff；AB10保存了四个
关键文件SHA-256，另外三项的`run_command.txt`记录相同code state。四项同秒启动，当前
关键文件hash仍与AB10归档一致；投稿复现包必须把该diff固化到commit，不需要重跑结果。

只有最终确实进入论文消融表的运行才会在本节填写结果。router 阈值搜索和调试版本不记录。

## 7. 正式执行顺序

### 7.0 当前运行

- W1 已完成：MuSiQue original-query BGE top20 cache 已覆盖完整 dev 2,417 条，
  并行的 NQ No-RAG 已登记 `FINAL`。缓存文件按 question 字符串存储，数据集包含 5 组
  重复问题，因此文件为 2,412 个唯一键；逐样本检查为 0 missing 且每个键均含 20 篇文档。
- W2/W3 已完成：MuSiQue Standard RAG、Full-QD、EFC-RAG、IterRetGen、IRCoT 均已
  登记 FINAL。W4 的 HotpotQA/2Wiki FLARE `HP-NR-08`、`2W-NR-08` 均已完成并登记；
  HotpotQA/2Wiki TRACE 已按用户指令暂停，不登记部分结果。
- MuSiQue FLARE `MU-NR-08` 已完成并登记FINAL。NQ original BGE top20、no-reranker
  缓存已完成并审计，稳定链接见缓存表。
- NQ Standard RAG `NQ-NR-02`、IterRetGen `NQ-NR-03`、EFC-RAG `NQ-NR-05`与
  FLARE `NQ-NR-06` 均已完成审计并登记FINAL；Adaptive-RAG继续暂停。
- HotpotQA/2Wiki TRACE `HP-NR-09`、`2W-NR-09` 已于2026-07-15按用户指令停止，相关
  Python进程和tmux均已退出，GPU0/2已释放。两项都没有metric或prediction，不登记部分
  结果；HotpotQA已保留完整triple缓存，2Wiki停止时尚无可复用`save_triples.json`。
- MuSiQue TRACE `MU-NR-09` 在796/2,417处异常退出，没有metric、intermediate或prediction，
  因此不登记成绩。已保留覆盖全部2,417×top5输入的`save_triples.json`（8,636篇唯一文档、
  96,785条triple）；推理链结果只在内存中，不能安全续跑，修复空path处理后必须用新目录重跑。
- HotpotQA generation-guided-only消融 `HP-AB-03` 已审计并登记FINAL：完整7,405条，
  EM/F1 35.61/46.95，Retrieval Recall@10 71.60，总耗时1:28:00。
- W8中HotpotQA direct-only `HP-AB-02`与MuSiQue generation-guided-only `MU-AB-02`
  已完成并登记FINAL。HotpotQA w/o static-QD `HP-AB-04`因动态query cache miss安全
  退出，无可登记成绩；错误消息只展示缺失列表前3项，实际缺失总数未落盘，不能据此认定
  只有3项。w/o role weight `HP-AB-06`仍在GPU0执行最终生成。
- W9四项均已完成并登记FINAL：w/o static-QD `HP-AB-04`使用独立合并cache，w/o role
  `HP-AB-06`、w/o original seed reservation `HP-AB-07`和w/o source weight `HP-AB-08`
  均cache-only；四项完整覆盖7,405条。W10 RRF-only `HP-AB-09`也已FINAL。
- W11四项`HP-AB-10/11/12/13`均已结束并通过FINAL验收。只有matched maximum budget
  运行发生4条online cache miss并写入独立cache，其余三项完整EFC cache-only；当前无
  W11 Python进程或tmux会话。
- `2W-NR-04` Full-QD、`2W-NR-05` IRCoT 和 `MU-NR-01` No-RAG 已完成并登记 FINAL。
- MuSiQue 公共缓存链接：
  `output/cache/musique_dev_bge_large_no_rerank_top20_retrieval_cache.json`。

### 7.1 并行波次

| 波次 | Retrieval 通道 `R` | 可并行的 Generation/cache-only 通道 `G` | 完成条件 |
|---|---|---|---|
| W0（完成） | 2Wiki IRCoT，GPU 0-3 TP | 无 | `2W-NR-05` 已 FINAL |
| W1（完成） | MuSiQue original top20 cache 已建立并审计 | NQ No-RAG 已 FINAL | 2,417/2,417 样本命中、每个唯一 query top20 |
| W2（完成） | MuSiQue Standard、Full-QD、EFC 均已完成 | prepare/generate 分通道并行 | `MU-NR-02/04/07` 已 FINAL |
| W3（完成） | MuSiQue IterRetGen、IRCoT 均已完成 | 无 | `MU-NR-03/05` 已 FINAL |
| W4（暂停） | HotpotQA/2Wiki TRACE已按用户指令停止；MuSiQue TRACE此前异常退出 | HP/Mu保留完整triple缓存；2Wiki无完整triple缓存；三者均无可登记指标 | 后续确认需要TRACE时再修复并用新目录重跑 |
| W5（完成） | NQ original top20 cache、Standard RAG、EFC-RAG、FLARE已完成 | IterRetGen已完成 | NQ非Adaptive主表完成 |
| W6（暂停） | Adaptive-RAG | 无 | 获得并锁定 classifier 后再排期 |
| W7（完成） | HotpotQA generation-guided-only消融已完成 | 独立合并cache已保存 | `HP-AB-03`已FINAL |
| W8（完成） | MuSiQue generation-guided-only已FINAL；HotpotQA w/o static-QD旧cache-only运行安全失败 | HotpotQA direct-only已FINAL | 失败目录不登记，转W9独立cache重跑 |
| W9（完成） | GPU1重跑HotpotQA w/o static-QD并在线补齐动态query | GPU0 w/o role、GPU3 w/o original seed、GPU4 w/o source均cache-only | `HP-AB-04/06/07/08`均FINAL |
| W10（完成） | 无新增检索；复用完整HotpotQA EFC cache | GPU0 HotpotQA RRF-only selector，单卡cache-only | `HP-AB-09`已FINAL |
| W11（完成） | GPU3 matched maximum retrieval budget，4条新query写独立cache | GPU0 title-dedup-only、GPU1 final top5、GPU4 w/o redundancy均cache-only | `HP-AB-10/11/12/13`均FINAL |

默认仍只安排一个会高频搜索Flat index的任务。W11已结束，当前没有这四项
消融对应的Python进程或tmux会话。增加任务前仍需检查GPU/CPU/NUMA资源，并继续
遵守本项目最多4张GPU的限制。
全部TRACE均无Python进程。增加任务前仍需检查显存、内存、CPU和独立缓存写入路径，并继续
遵守本项目最多4张GPU的限制。

### 阶段 A：跨数据集可行性

1. 2Wiki：`2W-NR-02`、`2W-NR-03`、`2W-NR-07` 已 `FINAL`
2. 按 W1-W3 完成 `MU-NR-02`、`MU-NR-03`、`MU-NR-04`、`MU-NR-05`、`MU-NR-07`
3. 判断 EFC-RAG 是否至少在两个多跳数据集上超过 IterRetGen。

### 阶段 B：补齐多跳主表

1. 2Wiki IRCoT 已完成；继续补齐 MuSiQue 的 Full-QD、IRCoT
2. 按 W4 在 HotpotQA、2Wiki 上运行 FLARE 和 TRACE
3. Adaptive-RAG 保持暂停；取得 classifier 后再恢复
4. 补齐其余 No-RAG 下界

### 阶段 C：单跳泛化

1. 在 NQ test 运行 No-RAG、Standard RAG、IterRetGen、FLARE、Adaptive-RAG、EFC-RAG
2. 检查 EFC-RAG 是否保持接近 Standard RAG 的 EM/F1
3. 报告 direct 路由比例和额外调用成本

### 阶段 D：消融与效率

1. 完成 HotpotQA router 消融
2. 完成 MuSiQue 三项核心消融
3. 从所有 FINAL 日志统一提取调用次数、候选池、耗时和路由比例

## 8. 结果进入论文前的验收清单

每个实验转为 `FINAL` 前必须全部满足：

- [ ] 使用完整目标 split，`test_sample_num: null`
- [ ] 生成模型为 `Llama-3.1-8B-Instruct`
- [ ] retriever 为 `bge-large-en-v1.5`
- [ ] reranker 关闭
- [ ] 最终 context K 和检索预算已记录
- [ ] `config.yaml`、`metric_score.txt`、`intermediate_data.json` 均存在
- [ ] runner 支持日志落盘时必须保留 `run.log`；旧 runner 至少保留完整终端结果或中间数据
- [ ] 可用日志中无 Traceback、OOM 或递归子进程异常
- [ ] 指标由最终输出直接读取，不从 smoke 或旧报告手工推算
- [ ] Retrieval Recall 的 K 与表头一致
- [ ] 输出目录不会被后续派生实验覆盖
- [ ] 记录对应 Git commit；若工作区非 clean，保存代码 diff

## 9. 更新日志

| 日期 | 更新内容 |
|---|---|
| 2026-07-15 | 验收并登记W11四项HotpotQA消融FINAL：`HP-AB-10` title-dedup-only为35.18/46.57/70.90@10，`HP-AB-11` controlled final top5为35.53/46.63/66.31@5，`HP-AB-12` matched maximum retrieval budget为36.66/48.17/69.82@≤10，`HP-AB-13` w/o redundancy为36.04/47.46/72.21@10（EM/F1/answer-hit）；四项均完整7,405条、无异常或缺失预测。AB12只匹配最大检索预算而非完整计算预算；AB13未观察到冗余惩罚正收益。运行代码为`063e29d`加已识别diff，提交后无需重跑。 |
| 2026-07-15 | 启动W11四项HotpotQA扩展消融：GPU0/tmux `racp_hp_ab10_0715`运行title-dedup-only，GPU1/tmux `racp_hp_ab11_0715`运行受控final top5，GPU3/tmux `racp_hp_ab12_0715`运行IRCoT-matched retrieval budget（预锁定最多2次逻辑查询、10篇raw文档、final K≤10），GPU4/tmux `racp_hp_ab13_0715`运行w/o redundancy；仅matched-budget允许在线补cache且只写独立输出，其余cache-only；title-dedup-only为默认关闭的新隔离开关，34/34直接回归测试通过 |
| 2026-07-15 | 审计并登记HotpotQA RRF-only selector `HP-AB-09` FINAL：完整7,405条，EM/F1/Retrieval Recall@10为33.30/43.67/67.13，总耗时55:09；相对完整EFC低2.55/3.54/4.38个百分点，support-title recall和双支持命中分别从68.11%/51.24%降至47.45%/22.96%，证明组合selector整体有效但不能把全部增益归因于单一权重 |
| 2026-07-15 | GPU0/tmux `racp_hp_ab09_0715`启动W10 HotpotQA RRF-only selector：新增默认关闭的`rrf_only_selection`隔离开关，三个route均按融合RRF分数选final top10，绕过role/title/source/redundancy、original seed reservation、title diversity与static配额；router、查询和生成设置不变，复用22,590键完整EFC cache并cache-only；代码hash、测试与完整命令已归档 |
| 2026-07-15 | 审计并登记W9四项HotpotQA消融FINAL：`HP-AB-04` w/o static-QD为35.76/47.09/71.49，`HP-AB-06` w/o role为35.77/47.02/71.05，`HP-AB-07` w/o original seed reservation为35.80/47.09/71.41，`HP-AB-08` w/o source为35.85/47.21/71.51（EM/F1/Retrieval Recall@10）；四项均覆盖7,405条且归档实际命令，后三项cache-only，AB04只向独立FINAL目录保存补齐cache |
| 2026-07-15 | 按最多4张本项目GPU的限制启动W9：GPU1/tmux `racp_hp_ab04_v3_0715`重跑w/o static-QD `HP-AB-04`，复用旧cache、允许动态query miss并将补齐后的合并cache只写新输出目录；GPU3/tmux `racp_hp_ab07_0715`启动w/o original seed reservation，GPU4/tmux `racp_hp_ab08_0715`启动w/o source weight，后二者cache-only；连同GPU0的w/o role共4卡，GPU2其他用户进程保持不动 |
| 2026-07-15 | 审计并登记HotpotQA direct-only `HP-AB-02` FINAL：完整dev 7,405条，EM/F1 33.59/44.58、Retrieval Recall@10 65.63、平均LLM/检索调用2.0000/1.0000、耗时42:46；相对完整EFC低2.26/2.63/5.88个百分点，且因保留probe不等同Standard RAG |
| 2026-07-15 | 审计并登记MuSiQue generation-guided-only `MU-AB-02` FINAL：完整dev 2,417条，EM/F1 9.64/17.45、Retrieval Recall@10 41.54、实际2,407条generation-guided与10条direct fallback、耗时35:48；完整EFC对应指标高0.29/0.20/0.54个百分点 |
| 2026-07-15 | HotpotQA w/o static-QD `HP-AB-04`在cache-only准备阶段因新动态query miss安全退出，错误只列出缺失列表前3项、实际总数UNKNOWN；无metric/intermediate/prediction且不登记成绩。后续须复用源cache但允许miss，并将合并cache写入新的独立输出目录；`HP-AB-06`继续在GPU0运行 |
| 2026-07-15 | GPU0启动HotpotQA w/o role weight消融 `HP-AB-06`（tmux `racp_hp_ab06_0715`）：仅将`role_weight=0.30`改为0，其余EFC锁定参数不变；基于正式完整EFC中间数据核对14,311个唯一原始/missing-hop/QD查询，对历史合并cache为0 miss且无短于top5条目，故采用cache-only并禁止写源缓存 |
| 2026-07-15 | 按用户指令暂停全部TRACE：终止HotpotQA `HP-NR-09`与2Wiki `2W-NR-09`进程并关闭相应tmux，清理MuSiQue失败运行遗留tmux；GPU0/2已释放，三项EFC消融继续运行。TRACE部分运行均无metric/prediction，不登记成绩；保留现有目录和日志，HotpotQA/MuSiQue完整triple缓存可供未来新目录重跑，2Wiki无完整triple缓存 |
| 2026-07-15 | 并行启动HotpotQA direct-only `HP-AB-02`（GPU4、tmux `racp_hp_ab02_0715`、原问题top20 cache-only）、HotpotQA w/o static-QD `HP-AB-04`（GPU1、tmux `racp_hp_ab04_0715`、活动`-v2`目录复用`HP-AB-03`完整合并cache）与MuSiQue generation-guided-only `MU-AB-02`（GPU3、tmux `racp_mu_ab02_0715`、本轮唯一online-retrieval通道）；三项命令与缓存语义已归档 |
| 2026-07-15 | 审计MuSiQue TRACE全量尝试：在796/2,417处因空path query异常退出，无metric、prediction或可续跑推理链，不登记成绩；保留覆盖完整输入的`save_triples.json`供修复后新目录重跑；HotpotQA/2Wiki TRACE继续运行 |
| 2026-07-15 | 审计并登记HotpotQA generation-guided-only消融 `HP-AB-03` FINAL：完整dev 7,405条，EM/F1 35.61/46.95、Retrieval Recall@10 71.60，实际7,371条generation-guided和34条direct fallback，总耗时1:28:00 |
| 2026-07-14 | 并行启动2Wiki TRACE `2W-NR-09`（GPU2、tmux `racp_2w_trace_0714`）与MuSiQue TRACE `MU-NR-09`（GPU3、tmux `racp_mu_trace_0714`），均复用原问题缓存并截取top5；GPU4启动HotpotQA generation-guided-only消融 `HP-AB-03`，强制generation-guided、关闭static fallback并使用独立缓存副本，启动审计无异常 |
| 2026-07-14 | 审计并登记MuSiQue FLARE `MU-NR-08`（完整dev 2,417条，EM/F1 2.15/5.42，327条触发、340次动态检索、耗时1:23:27）、NQ IterRetGen `NQ-NR-03`（完整test 3,610条，EM/F1 37.53/49.38，三轮各top5、耗时约39:34）和NQ FLARE `NQ-NR-06`（EM/F1 21.22/30.85，511条触发、573次动态检索、耗时2:17:38）；完整日志与命令归档 |
| 2026-07-14 | 审计并登记NQ Standard RAG `NQ-NR-02`（EM/F1 36.95/49.15，Retrieval Recall@10 82.74）与EFC-RAG `NQ-NR-05`（EM/F1 35.32/47.44，Retrieval Recall@10 79.86）完整test结果；启动下一轮NQ IterRetGen（GPU2、tmux `racp_nq_iterretgen_0714`）和FLARE（GPU3、tmux `racp_nq_flare_0714`），复用统一top20原问题缓存，GPU4保留 |
| 2026-07-14 | 启动NQ Standard RAG正式重跑 `NQ-NR-02`（GPU2、tmux `racp_nq_standard_0714`、cache-hit top10）和NQ EFC-RAG `NQ-NR-05`（GPU3、tmux `racp_nq_efc_0714`、完整锁定EFC top10参数）；GPU4保留，启动审计无异常 |
| 2026-07-14 | 完成并审计NQ test统一BGE top20、no-reranker原问题缓存：3,610/3,610条命中、每键20篇，建立稳定链接 `output/cache/nq_test_bge_large_no_rerank_top20_retrieval_cache.json`；日志与命令归档 |
| 2026-07-14 | 启动MuSiQue FLARE正式全量 `MU-NR-08`（GPU1、tmux `racp_mu_flare_0714`），并在GPU2/tmux `racp_nq_cache_0714`构建NQ test统一BGE top20、no-reranker缓存；与cache-hit HotpotQA TRACE并行，资源审计通过 |
| 2026-07-13 | 启动HotpotQA TRACE正式全量 `HP-NR-09`；GPU0、tmux `racp_hp_trace_0713`、原问题top5缓存、no-reranker、HF/KG-TRACE |
| 2026-07-13 | 审计并登记2Wiki FLARE全量结果 `2W-NR-08`；12,576条完整dev，EM/F1 9.37/20.45，1,524条触发检索、动态检索1,610次，总耗时6:12:41 |
| 2026-07-13 | 审计并登记 HotpotQA FLARE 全量结果 `HP-NR-08`；7,405条完整dev，EM/F1 16.04/23.26，1,353条触发检索、动态检索1,406次，总耗时4:49:24 |
| 2026-07-13 | 增加当前代码修改边界审计与九种方法的集中锁定参数表；明确 FLARE 兼容修复、Adaptive 两轮设置及各方法 Retrieval Recall 语义 |
| 2026-07-13 | 修复 FLARE 空 `retrieval_result` 与动态单query cache-miss返回维度问题；7项测试及含真实动态检索的6条 smoke 通过（不登记 smoke），以新目录重启 HotpotQA FLARE 正式全量 |
| 2026-07-12 | 审计并登记 MuSiQue IRCoT 全量结果 `MU-NR-05`；完整 dev 2,417 条，EM/F1 10.38/17.42；EFC 保持更高 F1、Recall 和 Retrieval Recall |
| 2026-07-12 | 审计并登记 MuSiQue EFC-RAG 与 IterRetGen：EFC EM/F1 9.93/17.65，IterRetGen 8.07/15.22；明确 IterRetGen 最终5篇文档的评测语义并启动四卡 IRCoT |
| 2026-07-12 | 审计并登记 MuSiQue Full-QD 全量结果 `MU-NR-04`；完整 dev 2,417 条、final top10，EM/F1 7.61/15.42；EFC 转生成并启动 IterRetGen |
| 2026-07-12 | 审计并登记 MuSiQue Standard RAG 全量结果 `MU-NR-02`；完整 dev 2,417 条、final top10，EM/F1 6.45/14.11；Full-QD 转生成、EFC 进入 prepare |
| 2026-07-12 | 完成 MuSiQue original-query BGE top20 公共缓存：完整 dev 2,417 条全部命中；因 5 组重复问题保存为 2,412 个唯一键，每键 top20；进入 W2 |
| 2026-07-12 | 审计并登记 NQ No-RAG 全量结果 `NQ-NR-01`；完整 test 3,610 条，EM/F1 21.63/32.42；W1 继续构建 MuSiQue top20 cache |
| 2026-07-12 | 审计并登记 2Wiki IRCoT 全量结果 `2W-NR-05`；EM/F1 33.07/39.39，显著高于 EFC，EFC 仅在 Retrieval Recall 上更高 |
| 2026-07-12 | 将 FLARE、TRACE、Adaptive-RAG 纳入缓存优先的并行波次计划；固定单 online-retrieval 通道、最多三条 generation/cache-only 通道，TRACE 全量前先完成 cache-only retriever 与四卡分片审计 |
| 2026-07-12 | 审计并登记 2Wiki Full-QD 全量结果 `2W-NR-04`；EM/F1 18.23/26.84，final top10，完整 12,576 条 |
| 2026-07-12 | 审计并登记 MuSiQue No-RAG 全量结果 `MU-NR-01`；完整 dev 2,417 条，EM/F1 3.56/9.94 |
| 2026-07-12 | 将 FLARE、TRACE 和 Adaptive-RAG 正式加入实验矩阵与命令；Adaptive 要求显式本地 classifier，FLARE/TRACE 先完成兼容性审计；登记 HotpotQA Standard RAG FINAL |
| 2026-07-12 | 完成并登记 HotpotQA、2Wiki No-RAG 闭卷下界；2Wiki No-RAG 略高于 Standard RAG，Standard RAG HotpotQA 仍在运行 |
| 2026-07-12 | 按代码语义统一命名：`naive` CLI / `SequentialPipeline.run()` 记为 Standard RAG，`zero-shot` / `naive_run()` 记为 No-RAG；并行启动 HotpotQA Standard RAG、HotpotQA No-RAG 和 2Wiki No-RAG |
| 2026-07-12 | 审计并登记 2Wiki EFC-RAG 全量结果 `2W-NR-07`；EFC 在 EM/F1 和 Retrieval Recall 上明显超过 IterRetGen，下一阶段推进到 MuSiQue |
| 2026-07-12 | 审计并登记 2Wiki IterRetGen 全量结果 `2W-NR-03`；记录三轮 top5 与最终检索列表的评测语义，下一项推进到 EFC-RAG |
| 2026-07-10 | 修复 baseline 意外继承 `selective-context` refiner，锁定正式代码基线 `e6b4456` |
| 2026-07-10 | 审计并登记 2Wiki Standard RAG 全量结果 `2W-NR-02`，下一项推进到 IterRetGen |
| 2026-07-10 | 建立正式论文实验计划；纳入 HotpotQA 的 IterRetGen、Full-QD、IRCoT 和 EFC-RAG 四项全量结果；排除全部 smoke 和配置不统一的历史结果 |
| 2026-07-10 | 补充项目目标、算法流程、环境、代码入口、参数、缓存规则、正式命令和跨对话交接信息 |

## 10. 已知问题与不可静默修改的约束

### 10.1 当前已知问题

1. Baseline 默认 refiner 已在 `e6b4456` 修复；Standard RAG（`naive` CLI）和 No-RAG（`zero-shot`）显式关闭 refiner，
   后续正式运行必须保持 `refiner_name: null`，只有显式 refiner 方法可以启用。
2. `run_racp.py` 默认 EFC final K 是 6，论文设置是 10。所有正式 EFC 命令必须显式传
   `--final_topk 10`。
3. `racp/run_exp.py` 中普通 `ircot` 默认最大 5 轮；论文 HotpotQA IRCoT 使用定制
   `ircot-no-rerank`、`max_iter=2`。后续必须使用定制入口。
4. Adaptive-RAG classifier 尚未完成本地 checkpoint 和 pipeline 配置审计。
5. Full-QD 在无 reranker 时直接合并不同 query 的 raw cosine score，分数未校准；其低分
   是当前固定分解基线的真实限制，不应通过加入 reranker 改变主表条件。
6. HotpotQA 的 EFC F1 略低于 IRCoT。论文应主张“相比 IterRetGen/Full-QD 的提升和更好
   的证据召回/质量成本平衡”，不能宣称所有答案指标最优。
7. 2Wiki 历史 Native 结果和 NQ 历史结果配置不统一，只能作为调试参考。
8. 当前 `IterativePipeline` 固定三轮各取 top5，但最终 `retrieval_result` 只保留第三轮
   top5；metric key 仍为 `retrieval_recall_top10`。论文中必须披露这一实现语义，后续若
   改为三轮候选并集评测属于口径变更，必须统一重跑，不能与现有 FINAL 混排。
9. FLARE 的 vLLM `return_scores=True` 路径在当前 FlashRAG 中存在兼容性修复；正式运行前
   必须通过 smoke，并确认概率取自实际生成 token，而不是 logprob 字典中的任意候选。
10. TRACE 会同时加载 HF 生成模型、BGE retriever 和 E5 demonstration encoder，单卡
    24GB 的显存余量与全量耗时必须先完成审计；不得未经耗时预估直接启动正式全量。

### 10.2 未经明确实验设计不得修改

- generator、retriever、corpus/index 和 no-reranker 条件；
- EFC final context K=10；
- IRCoT max_iter=2、retrieval_topk=5；
- IterRetGen 当前 canonical `iter_num=3`；
- 多跳数据使用 dev、NQ 使用 test；
- temperature=0、完整 split；
- FINAL 输出目录只读保留，不覆盖、不删除；
- 不能将 reranker 实验与 no-reranker 主表混排。

### 10.3 跨数据集判断标准

在继续补齐大规模基线前，先检查：

- EFC 是否在 HotpotQA、2Wiki、MuSiQue 中至少两个数据集超过 IterRetGen 的 EM/F1；
- EFC 与 IRCoT 的 F1 差距是否不超过约 0.5，或是否能用更低调用成本解释差距；
- EFC 是否持续提高 Retrieval Recall@10；
- NQ 上 EFC 是否接近 Standard RAG，并且 `direct` 比例显著高于多跳数据集；
- 如果 EFC 在 2Wiki/MuSiQue 明显退化，先分析 route 分布、support title recall 和 query
  cache，不要直接进行数据集专属阈值调参并在同一 dev 上汇报。

## 11. 下一轮对话提示词

可直接复制的提示词保存在：

[`NEXT_CONVERSATION_PROMPT.md`](NEXT_CONVERSATION_PROMPT.md)

下一轮对话必须先完整阅读本文档，再检查最新输出、进程和 Git 状态；本文档不是一次性
报告，而是后续所有论文正式实验的唯一进度台账。
