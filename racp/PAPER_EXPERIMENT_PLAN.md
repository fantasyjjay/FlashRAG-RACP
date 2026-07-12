# RACP/EFC-RAG 论文实验计划与结果台账

最后更新：2026-07-12
当前代码基线：`e6b4456`（2Wiki Naive RAG 正式运行基线）

## 0. 下一轮对话快速交接

### 0.1 项目目标

本项目在 FlashRAG 上实现并评测 EFC-RAG（Evidence-Feedback Controlled RAG）。方法以
IterRetGen 的“生成中间信息后继续检索”为主干，通过 probe answer 和首轮证据状态将
问题路由到三个分支：

1. `direct`：首轮证据足够，直接使用最终证据生成答案；
2. `generation_guided`：probe 暴露新的桥接实体，生成 missing-hop query 后继续检索；
3. `static_qd`：probe 无有效桥接实体或不确定，使用固定问题分解作为辅助回退。

论文不以 SOTA 为目标。主要论点是：在普通 8B 模型、同一 dense retriever、无
reranker 的条件下，自适应证据反馈路由比 Naive RAG、固定 Full-QD 和原生 IterRetGen
取得更好的质量/成本平衡，并在单跳 NQ 上避免不必要的多轮检索。

### 0.2 当前最重要结论

- HotpotQA 当前 EFC-RAG：EM 35.85、F1 47.21、Retrieval Recall@10 71.51。
- IRCoT：EM 36.26、F1 47.50，答案指标略高于 EFC，但检索召回和 Acc 低于 EFC。
- 原生 IterRetGen：EM 34.67、F1 45.66，EFC 分别提升 1.18 和 1.55 个百分点。
- Full-QD：EM 33.67、F1 44.59。固定拆分不适合作为主干，QD 只保留为辅助模块。
- EFC 在 HotpotQA 平均检索 1.9332 次、平均 LLM 调用 2.8718 次；80.89% 样本进入
  `generation_guided`，12.90% 进入 `direct`，6.21% 进入 `static_qd`。
- 2Wiki Naive RAG：EM 16.56、F1 25.60、Retrieval Recall@10 46.20；原生
  IterRetGen：EM 16.90、F1 26.59、Retrieval Recall@10 42.48；EFC-RAG：EM 20.35、
  F1 29.37、Retrieval Recall@10 58.17。
- 2Wiki EFC 相比 IterRetGen 的 EM/F1/检索召回分别提升 3.45/2.78/15.70 个百分点，
  相比 Naive 的 EM/F1 分别提升 3.79/3.77 个百分点。
- EFC 已在 HotpotQA 和 2Wiki 两个多跳数据集上同时超过 IterRetGen 的 EM/F1，并持续
  提高 Retrieval Recall@10；已满足“至少两个多跳数据集超过 IterRetGen”的关键标准。
- 下一阶段不是继续调 HotpotQA QD，而是优先验证 2Wiki 和 MuSiQue 的跨数据集收益。

### 0.3 下一步正式任务

按顺序完成以下全量论文实验：

1. 2Wiki：Naive RAG、IterRetGen、EFC-RAG；
2. MuSiQue：Naive RAG、IterRetGen、EFC-RAG；
3. 如果 EFC 至少在两个多跳数据集上超过 IterRetGen，再补 Full-QD、IRCoT 和
   Adaptive-RAG；
4. 最后在 NQ test 上运行 No-RAG、Naive RAG、IterRetGen、Adaptive-RAG 和 EFC-RAG。

任何新对话开始后，应先检查最新输出目录、GPU 进程和 Git 状态，确认没有上一轮已完成
但尚未登记的正式结果，再决定下一条命令。

### 0.4 当前未提交代码

当前 Git 工作区不是 clean，已知修改包括：

```text
examples/methods/run_exp.py
flashrag/pipeline/active_pipeline.py
flashrag/utils/pred_parse.py
racp/run_racp.py
```

这些修改包括 IRCoT no-reranker/cache 支持、pipeline 修复、答案解析和 EFC/QD 的干净
vLLM 子进程生成。不要回滚。下一批正式论文实验前，应先审计 diff 并提交一个明确的
实验基线 commit；提交后更新本文档顶部的 commit。

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
| [`run_exp.py`](run_exp.py) | Naive、No-RAG、IterRetGen、IRCoT、Adaptive-RAG 等通用基线入口 |
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
| MuSiQue query cache | 尚无 | 首次正式 EFC/检索运行需在线检索并保存 |
| NQ query cache | 尚无统一 BGE top20 正式缓存 | 首次正式运行需在线检索并保存 |

缓存规则：

- `--retrieval_cache_only` 只适合 planner/probe/query 生成完全不变且扩展缓存完整的 EFC
  重跑。跨数据集第一次正式运行禁止使用它。
- 普通 `--use_retrieval_cache` 允许 cache miss；缺失的新 query 会在线检索，并在本次
  输出目录形成合并后的 `retrieval_cache.json`。
- Full-QD 的子问题由 LLM 生成。原问题 cache 可复用，但子问题首次出现时仍需检索；
  完成后应保存 retrieval-topk bundle。
- 每次正式运行都保留自己的 cache，不覆盖或删除历史 FINAL 目录。

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

Naive RAG no-reranker，final context top10：

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
  --max_iter 2 --max_tokens 32 --gpu_memory_utilization 0.65
```

Adaptive-RAG：

```bash
$PY racp/run_exp.py \
  --method_name adaptive \
  --dataset_name <DATASET> --split <SPLIT> --gpu_id <GPU> \
  --save_note <DATASET>-adaptive-rag-no-rerank-full \
  --no_reranker
```

Adaptive-RAG 依赖 `illuminoplanet/adaptive-rag-classifier`。当前本地模型目录未确认存在该
checkpoint；在正式全量前必须检查 checkpoint、分类标签映射、三条子 pipeline 的
retrieval top-k 和最大迭代数。未经审计不得直接把 Adaptive-RAG 结果标为 FINAL。

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
| No-RAG | `TODO` | `TODO` | `TODO` | `TODO` |
| Naive RAG | `RERUN` | `FINAL` | `TODO` | `RERUN` |
| IterRetGen | `FINAL` | `FINAL` | `TODO` | `TODO` |
| Full-QD | `FINAL` | `TODO` | `TODO` | `N/A` |
| IRCoT | `FINAL` | `TODO` | `TODO` | `N/A` |
| Adaptive-RAG | `TODO` | `TODO` | `TODO` | `TODO` |
| EFC-RAG | `FINAL` | `FINAL` | `TODO` | `TODO` |

NQ 主表不运行 Full-QD；IRCoT 仅在需要展示固定多轮检索对单跳任务的额外成本时加入补充表。

### 4.2 正式实验编号

| ID | 数据集 | 方法 | 状态 | 目标/备注 |
|---|---|---|---|---|
| `HP-NR-01` | HotpotQA | No-RAG | `TODO` | 闭卷下界 |
| `HP-NR-02` | HotpotQA | Naive RAG | `RERUN` | 统一生成与 Retrieval Recall@10 口径 |
| `HP-NR-03` | HotpotQA | IterRetGen | `FINAL` | 主干基线 |
| `HP-NR-04` | HotpotQA | Full-QD | `FINAL` | 固定拆分基线 |
| `HP-NR-05` | HotpotQA | IRCoT | `FINAL` | 经典多轮推理检索基线 |
| `HP-NR-06` | HotpotQA | Adaptive-RAG | `TODO` | 直接 router 基线；先验证 classifier checkpoint |
| `HP-NR-07` | HotpotQA | EFC-RAG | `FINAL` | 完整方法，final context K=10 |
| `2W-NR-01` | 2Wiki | No-RAG | `TODO` | 闭卷下界 |
| `2W-NR-02` | 2Wiki | Naive RAG | `FINAL` | 统一 no-refiner、no-reranker、top10 全量结果 |
| `2W-NR-03` | 2Wiki | IterRetGen | `FINAL` | 原生三轮主干基线，每轮 top5 |
| `2W-NR-04` | 2Wiki | Full-QD | `TODO` | 固定拆分基线 |
| `2W-NR-05` | 2Wiki | IRCoT | `TODO` | 多轮推理检索基线 |
| `2W-NR-06` | 2Wiki | Adaptive-RAG | `TODO` | router 基线 |
| `2W-NR-07` | 2Wiki | EFC-RAG | `FINAL` | 完整方法，final context K=10 |
| `MU-NR-01` | MuSiQue | No-RAG | `TODO` | 闭卷下界 |
| `MU-NR-02` | MuSiQue | Naive RAG | `TODO` | 单次检索基线 |
| `MU-NR-03` | MuSiQue | IterRetGen | `TODO` | 主干基线 |
| `MU-NR-04` | MuSiQue | Full-QD | `TODO` | 固定拆分基线 |
| `MU-NR-05` | MuSiQue | IRCoT | `TODO` | 多轮推理检索基线 |
| `MU-NR-06` | MuSiQue | Adaptive-RAG | `TODO` | router 基线 |
| `MU-NR-07` | MuSiQue | EFC-RAG | `TODO` | 完整方法 |
| `NQ-NR-01` | NQ | No-RAG | `TODO` | 单跳闭卷下界 |
| `NQ-NR-02` | NQ | Naive RAG | `RERUN` | 历史结果配置不统一 |
| `NQ-NR-03` | NQ | IterRetGen | `TODO` | 检查固定迭代的噪声与成本 |
| `NQ-NR-04` | NQ | Adaptive-RAG | `TODO` | 单跳 router 基线 |
| `NQ-NR-05` | NQ | EFC-RAG | `TODO` | 检查 direct 路由占比和质量保持 |

## 5. 已锁定的论文结果

所有数值按百分数记录。当前 HotpotQA 四项和 2Wiki 的 Naive RAG、IterRetGen、EFC-RAG
通过正式全量与配置审计。

### 5.1 HotpotQA 主结果

| ID | 方法 | EM | F1 | Acc | Precision | Recall | Retrieval Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| `HP-NR-03` | IterRetGen | 34.67 | 45.66 | 41.13 | 47.12 | 47.97 | 62.86 (@10) |
| `HP-NR-04` | Full-QD | 33.67 | 44.59 | 40.23 | 46.16 | 46.90 | 65.04 (@10) |
| `HP-NR-05` | IRCoT | **36.26** | **47.50** | 39.57 | **51.40** | 46.93 | 64.77 (@10) |
| `HP-NR-07` | EFC-RAG | 35.85 | 47.21 | **42.35** | 48.89 | **49.43** | **71.51 (@10)** |

最终结果目录：

| ID | 输出目录 |
|---|---|
| `HP-NR-03` | [`output/hotpotqa_2026_06_05_16_46_iterretgen-native-no-reranker-gpu3`](output/hotpotqa_2026_06_05_16_46_iterretgen-native-no-reranker-gpu3) |
| `HP-NR-04` | [`output/hotpotqa_2026_07_10_09_27_full-qd-no-rerank-top10-full`](output/hotpotqa_2026_07_10_09_27_full-qd-no-rerank-top10-full) |
| `HP-NR-05` | [`output/hotpotqa_2026_07_09_14_27_ircot-no-rerank-cache-full`](output/hotpotqa_2026_07_09_14_27_ircot-no-rerank-cache-full) |
| `HP-NR-07` | [`output/hotpotqa_2026_07_09_15_52_efc-static-bridge-final-top10-full`](output/hotpotqa_2026_07_09_15_52_efc-static-bridge-final-top10-full) |

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
| `2W-NR-02` | Naive RAG | 16.56 | 25.60 | 25.37 | 25.32 | 30.59 | 46.20 (@10) |
| `2W-NR-03` | IterRetGen | 16.90 | 26.59 | **30.16** | 25.58 | **34.69** | 42.48 (@10) |
| `2W-NR-07` | EFC-RAG | **20.35** | **29.37** | 28.92 | **29.04** | 33.98 | **58.17 (@10)** |

最终结果目录：

| ID | 输出目录 | Git commit |
|---|---|---|
| `2W-NR-02` | [`output/2wikimultihopqa_2026_07_10_15_04_2wiki-naive-no-rerank-top10-full`](output/2wikimultihopqa_2026_07_10_15_04_2wiki-naive-no-rerank-top10-full) | `e6b4456` |
| `2W-NR-03` | [`output/2wikimultihopqa_2026_07_10_15_52_2wiki-iterretgen-no-rerank-full`](output/2wikimultihopqa_2026_07_10_15_52_2wiki-iterretgen-no-rerank-full) | `3e62049` |
| `2W-NR-07` | [`output/2wikimultihopqa_2026_07_12_11_11_2wiki-efc-no-rerank-top10-full`](output/2wikimultihopqa_2026_07_12_11_11_2wiki-efc-no-rerank-top10-full) | `3e62049` |

该运行使用完整 dev 12,576 条样本；每条最终使用 10 篇文档，`refiner_name: null`、
`use_reranker: false`、`do_sample: false`。`racp/run_exp.py` 未生成 `run.log`，但
`config.yaml`、`metric_score.txt`、`intermediate_data.json` 和完整 retrieval cache 均已审计。

IterRetGen 同样使用完整 dev 12,576 条样本，固定运行三轮、每轮检索 top5；三轮输出和
最终预测均覆盖全部样本。配置中的指标键为 `retrieval_recall_top10`，但当前
`IterativePipeline` 写入最终 `retrieval_result` 的是第三轮 5 篇文档，而不是三轮并集；
论文报告时必须同时披露“三轮 × top5”和该评测语义，不能把它解释为最终上下文含 10 篇。

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

### 5.5 其余尚无可写入论文的最终结果

| 数据集 | 当前结论 |
|---|---|
| MuSiQue | 尚未完成正式全量主表实验 |
| NQ | 历史结果使用不同模型、retriever、reranker 或样本子集；不记录数值 |

## 6. 消融实验计划

完整消融以 HotpotQA 为主；MuSiQue 只补最关键的 router 对照，避免重复消耗。

| ID | 数据集 | 消融 | 状态 | 证明目标 |
|---|---|---|---|---|
| `HP-AB-01` | HotpotQA | EFC-RAG 完整方法 | `FINAL` | 完整模型，复用 `HP-NR-07` |
| `HP-AB-02` | HotpotQA | direct-only | `RERUN` | 退化为单次 RAG；最终复用统一 Naive 结果 |
| `HP-AB-03` | HotpotQA | generation-guided-only | `TODO` | 检验 router 相比固定 IterRetGen 分支的收益 |
| `HP-AB-04` | HotpotQA | 去除 static-QD | `TODO` | 检验 QD 辅助模块的边际收益 |
| `HP-AB-05` | HotpotQA | Full-QD | `FINAL` | 固定分解对照，复用 `HP-NR-04` |
| `MU-AB-01` | MuSiQue | EFC-RAG 完整方法 | `TODO` | 更难依赖式多跳上的完整模型 |
| `MU-AB-02` | MuSiQue | generation-guided-only | `TODO` | 验证 router 是否优于固定主干 |
| `MU-AB-03` | MuSiQue | Full-QD | `TODO` | 验证固定分解在复杂多跳上的局限 |

只有最终确实进入论文消融表的运行才会在本节填写结果。router 阈值搜索和调试版本不记录。

## 7. 正式执行顺序

### 阶段 A：跨数据集可行性

1. 2Wiki：`2W-NR-02`、`2W-NR-03`、`2W-NR-07` 已 `FINAL`
2. 下一步依次运行 `MU-NR-02`、`MU-NR-03`、`MU-NR-07`
3. 判断 EFC-RAG 是否至少在两个多跳数据集上超过 IterRetGen。

### 阶段 B：补齐多跳主表

1. 2Wiki 和 MuSiQue 的 Full-QD、IRCoT
2. 验证并运行 Adaptive-RAG classifier
3. 补 No-RAG 下界

### 阶段 C：单跳泛化

1. 在 NQ test 运行 No-RAG、Naive RAG、IterRetGen、Adaptive-RAG、EFC-RAG
2. 检查 EFC-RAG 是否保持接近 Naive RAG 的 EM/F1
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
| 2026-07-12 | 审计并登记 2Wiki EFC-RAG 全量结果 `2W-NR-07`；EFC 在 EM/F1 和 Retrieval Recall 上明显超过 IterRetGen，下一阶段推进到 MuSiQue |
| 2026-07-12 | 审计并登记 2Wiki IterRetGen 全量结果 `2W-NR-03`；记录三轮 top5 与最终检索列表的评测语义，下一项推进到 EFC-RAG |
| 2026-07-10 | 修复 baseline 意外继承 `selective-context` refiner，锁定正式代码基线 `e6b4456` |
| 2026-07-10 | 审计并登记 2Wiki Naive RAG 全量结果 `2W-NR-02`，下一项推进到 IterRetGen |
| 2026-07-10 | 建立正式论文实验计划；纳入 HotpotQA 的 IterRetGen、Full-QD、IRCoT 和 EFC-RAG 四项全量结果；排除全部 smoke 和配置不统一的历史结果 |
| 2026-07-10 | 补充项目目标、算法流程、环境、代码入口、参数、缓存规则、正式命令和跨对话交接信息 |

## 10. 已知问题与不可静默修改的约束

### 10.1 当前已知问题

1. Baseline 默认 refiner 已在 `e6b4456` 修复；Naive 和 Zero-shot 显式关闭 refiner，
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
- NQ 上 EFC 是否接近 Naive RAG，并且 `direct` 比例显著高于多跳数据集；
- 如果 EFC 在 2Wiki/MuSiQue 明显退化，先分析 route 分布、support title recall 和 query
  cache，不要直接进行数据集专属阈值调参并在同一 dev 上汇报。

## 11. 下一轮对话提示词

可直接复制的提示词保存在：

[`NEXT_CONVERSATION_PROMPT.md`](NEXT_CONVERSATION_PROMPT.md)

下一轮对话必须先完整阅读本文档，再检查最新输出、进程和 Git 状态；本文档不是一次性
报告，而是后续所有论文正式实验的唯一进度台账。
