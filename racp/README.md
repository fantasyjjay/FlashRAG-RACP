# RACP / EFC-RAG

这个目录是 RAG 实验的统一入口。`run_racp.py` 当前默认运行 EFC-RAG，并默认执行
HotpotQA dev 的 `prepare` 阶段。常改参数集中在
`run_racp.py` 顶部的 `DEFAULT_RUN_CONFIG`，命令行参数仍具有更高优先级。

默认方法可以通过一个参数切换：

```bash
--method efc      # 默认 EFC-RAG
--method racp     # 原始 gap + refiner 流程
--method rs_mhr   # RS-MHR
--method qd       # 全量 query decomposition
```

## 运行

在 FlashRAG 根目录执行：

```bash
conda activate flashrag
              
python racp/run_racp.py
```

这等价于使用 `DEFAULT_RUN_CONFIG` 中的 HotpotQA、GPU、batch 和 EFC 参数。输出默认
保存在 `racp/output/` 下，并保存本次生成 query 的检索缓存。默认不会绑定已有缓存；
重复实验时可在集中配置中设置 `retrieval_cache_path` 和
`use_retrieval_cache=True`。

原始 RACP 流程需要显式指定：

```bash
python racp/run_racp.py \
  --method racp \
  --stage full \
  --dataset_name nq \
  --split test \
  --gpu_id 2
```

## 分段运行

如果一张卡同时放不下 retriever、reranker、Selective-Context 和 Llama3，可以先只生成最终 prompt：

```bash
python racp/run_racp.py \
  --method racp \
  --stage prepare \
  --dataset_name nq \
  --split test \
  --gpu_id 2
```

这一步会执行：

```text
检索 -> reranker -> RACP 动态 k -> Selective-Context -> final prompt
```

默认保存到当前输出目录的：

```text
prompt_cache.json
```

然后单独加载大模型生成和评测：

```bash
python racp/run_racp.py \
  --stage generate \
  --gpu_id 2 \
  --prompt_cache_path racp/output/nq_YYYY_MM_DD_HH_MM_racp/prompt_cache.json
```

`generate` 阶段不会再新建时间戳目录，`metric_score.txt` 和 `intermediate_data.json` 会写回 `prompt_cache.json` 所在的 prepare 目录。
脚本终端日志会同步追加写入输出目录的 `run.log`。分段运行时，`prepare` 和
`generate` 的日志会保存在同一个文件中。
脚本会自动把 vLLM 的 worker 启动方式切到 `fork`，避免子进程重复执行入口脚本。
如果 cache 同目录存在 `config.yaml`，`generate` 阶段会默认读取那份配置，因此 HotpotQA 等数据集第二阶段只需要传 `--gpu_id` 和 `--prompt_cache_path`。

需要在一个进程中完整跑完时显式使用 `full`：

```bash
python racp/run_racp.py --method racp --stage full --dataset_name nq --split test --gpu_id 2
```

## 全量 QD

使用 `--use_query_decomposition` 时，所有问题都会生成两个独立检索 query，不经过
RS-MHR router。prepare 阶段默认使用 `Llama-3.1-8B-Instruct` planner，并按照
`--planner_batch_size` 分批处理；generate 阶段仍使用配置中的
`Llama-3.1-8B-Instruct`。full-QD parser 允许 planner 输出简短实体 query；
如果只有一条 query 可用，会保留这一条继续检索，只有零条可用时才回退到原问题。
如果 planner 输出超过两条 query，脚本会保留前两条可用 query，控制额外检索成本。

HotpotQA 等多跳数据集可以使用 `title_dedup_topk`，在 reranker 排序后优先选择标题
不同的文档，减少最终 prompt 中同一 Wikipedia 页面重复占用名额：

```bash
--selection_method title_dedup_topk --selection_topk 5
```

如果允许增加 prepare 阶段的 reranker 成本，可以让每条 QD query 额外重排自己
召回的文档，并将归一化分数作为原问题 reranker 分数的补充。这样可以提高第二跳
证据进入最终 top5 的机会，同时保持最终 prompt 仍为 5 篇文档：

```bash
--qd_rerank_with_subqueries --qd_subquery_rerank_weight 0.25
```

使用 `--load_retrieval_topk_cache_path` 读取已有 QD 候选 cache 时，脚本会跳过
planner、dense encoder、corpus 和 FAISS 索引，只加载 reranker。

上述 prepare 输出会保留原问题和 subquery 的 reranker 诊断分数。之后可以离线扫描
多个融合权重，不再重复加载 planner、检索器或 reranker。每个权重会写入独立目录，
避免生成评测时覆盖其他权重的结果：

```bash
python racp/derive_qd_fusion_prompt_cache.py \
  --source_prompt_cache racp/output/hotpotqa_YYYY_MM_DD_HH_MM_full-qd/prompt_cache.json \
  --weights 0.10 0.25 0.50 \
  --selection_topk 5 \
  --prefix qd-fusion
```

例如生成 `0.25` 权重版本：

```bash
python racp/run_racp.py \
  --stage generate \
  --gpu_id 3 \
  --prompt_cache_path racp/output/hotpotqa_YYYY_MM_DD_HH_MM_full-qd/qd-fusion-w0p25-top5/prompt_cache.json
```

同一检索结果可以分别跑默认 prompt 与 strict prompt，然后离线融合预测。融合规则只在
默认输出像长句、逗号解释或并列长串，且 strict 输出更短时采用 strict；否则保留默认：

```bash
python racp/fuse_prediction_outputs.py \
  --default_intermediate racp/output/hotpotqa_YYYY_MM_DD_HH_MM_full-qd/qd-fusion-w0p25-top5/intermediate_data.json \
  --strict_intermediate racp/output/hotpotqa_YYYY_MM_DD_HH_MM_full-qd/qd-fusion-w0p25-top5-strict/intermediate_data.json \
  --output_dir racp/output/hotpotqa_YYYY_MM_DD_HH_MM_full-qd/qd-fusion-w0p25-top5-pred-fusion
```

生成模型容易输出解释或完整句子时，可以增加严格短答案模板。模型仍可内部推理，
但最终只输出最短答案 span：

```bash
--strict_short_answer_prompt
```

如果 strict 的 EM 上升但 F1 下降，可以改用中等严格的 answer-only 模板。它只禁止
解释、引用和无关整句，但要求保留完整实体名、标题和日期：

```bash
--answer_only_prompt
```

如果 answer-only 仍然过长，可以使用更接近 strict 的 exact-answer 模板。它要求
输出精确答案 span，但保留完整实体名、作品标题、日期和多项答案：

```bash
--exact_answer_prompt
```

已有 prompt cache 可以直接派生严格短答案版本，无需重新检索或重排。将输出目录设为
原实验目录后，第二阶段仍会复用同一份 `config.yaml`：

```bash
python racp/derive_prompt_cache.py \
  --source_prompt_cache racp/output/hotpotqa_YYYY_MM_DD_HH_MM_full-qd/prompt_cache.json \
  --topks 5 \
  --output_dir racp/output/hotpotqa_YYYY_MM_DD_HH_MM_full-qd \
  --prefix strict-short-answer \
  --dataset_name hotpotqa \
  --split dev \
  --strict_short_answer_prompt
```

```bash
python racp/run_racp.py \
  --stage prepare \
  --dataset_name hotpotqa \
  --split dev \
  --gpu_id 3 \
  --save_note full-qd-planner8b-reranker-large-top5 \
  --use_query_decomposition \
  --selection_method topk \
  --selection_topk 5 \
  --retrieval_topk 20 \
  --rerank_topk 20 \
  --planner_model Llama-3.1-8B-Instruct \
  --planner_batch_size 16 \
  --planner_gpu_memory_utilization 0.75 \
  --save_retrieval_topk_cache_path racp/output/cache/hotpotqa_full_qd_planner8b_large_top5_retrieval_topk.json
```

## RS-MHR

RS-MHR 的 `static_qd` 和 `missing_hop` 路由需要 LLM planner。默认使用
`Llama-3.1-8B-Instruct`，最终答案生成也使用配置文件中的
`Llama-3.1-8B-Instruct`。planner prompt 默认每批提交 32 条，避免一次提交整个
数据集导致 vLLM 显存溢出。

推荐先分段运行 prepare：

```bash
python racp/run_racp.py \
  --stage prepare \
  --enable_rs_mhr \
  --force_route auto \
  --dataset_name hotpotqa \
  --split dev \
  --gpu_id 4 \
  --save_retrieval_cache
```

RS-MHR 的 `prompt_cache.json` 只保存第二阶段生成和评测所需的数据。需要排查
router 候选池时再增加 `--save_router_debug`，完整候选文档会单独保存到
`router_debug.json`。全量 HotpotQA 的 debug 文件会比较大，常规实验无需开启。

显存仍然紧张时，可以继续降低 planner 批次；8B planner 的 vLLM 显存预留不建议
低于 0.70：

```bash
--planner_batch_size 8 --planner_gpu_memory_utilization 0.70
```

首次成功运行后会在输出目录生成 `retrieval_cache.json`。后续重复实验可以复用
原问题和相同 planner query 的召回结果：

```bash
python racp/run_racp.py \
  --stage prepare \
  --enable_rs_mhr \
  --force_route auto \
  --dataset_name hotpotqa \
  --split dev \
  --gpu_id 4 \
  --use_retrieval_cache \
  --retrieval_cache_path racp/output/hotpotqa_YYYY_MM_DD_HH_MM_racp/retrieval_cache.json
```

如果缓存已经完整，并且希望连 81GB FAISS 索引也不加载，可以增加：

```bash
--retrieval_cache_only
```

cache-only 模式适合重复相同数据集、planner 和检索参数的实验。如果某个 query
没有命中缓存，脚本会直接报错并提示先执行一次普通 prepare 填充缓存。
首次运行仍需读取 Flat FAISS 索引并执行向量检索；降低 `topk` 对 Flat 索引的
扫描耗时改善有限。若首次检索仍然过慢，需要额外构建 IVF/HNSW 等近似索引。

## EFC-RAG

EFC-RAG 是免 cross-encoder reranker 的多跳流程：

```text
原问题 top20 -> top5 probe generation -> 确定性 router
-> generation-guided / static-QD fallback
-> query 级 RRF -> 角色感知 top5 打包 -> 最终生成
```

默认推荐配置使用启发式 missing-hop query。只有 static-QD 路由会加载 8B planner；
EFC 在加载 planner 前会释放 probe，并将 HF 实际生成 batch 限制为 8。单张
RTX 4090 上同时保留 BGE、使用 1024-token 输入和 96-token 输出的压力测试峰值约
18.13 GiB；batch 16 峰值约 20.04 GiB，显存余量较小，因此不作为默认值：

```bash
python racp/run_racp.py \
  --stage prepare \
  --dataset_name hotpotqa \
  --split dev \
  --gpu_id 3 \
  --enable_efc_rag \
  --no_reranker \
  --initial_topk 20 \
  --probe_topk 5 \
  --final_topk 5 \
  --enable_generation_guided \
  --enable_static_qd_fallback \
  --missing_query_mode heuristic \
  --planner_batch_size 32 \
  --title_dedup_soft \
  --test_sample_num 50 \
  --save_note efc-smoke-50
```

prepare 输出会保存 `probe_answer`、路由特征、扩展 query、候选来源、角色分数、
最终标题和成本统计。需要完整候选池时增加 `--save_efc_debug`，另存
`efc_debug.json`。生成与评测仍使用同一目录中的 prompt cache：

```bash
python racp/run_racp.py \
  --stage generate \
  --gpu_id 3 \
  --prompt_cache_path racp/output/hotpotqa_YYYY_MM_DD_HH_MM_efc-smoke-50/prompt_cache.json
```

三个路由可以分别做消融：

```bash
--force_route direct
--force_route static_qd
--force_route generation_guided
```

角色、标题、来源和冗余项可通过 `--role_weight`、`--title_weight`、
`--source_weight`、`--redundancy_weight` 设为 0 做消融；`--rrf_weight 1`
保留基础 RRF 排序。`--missing_query_mode` 还支持 `llm` 和 `raw_y1`。

### EFC-RAG 全量缓存复用

HotpotQA 的 EFC-RAG 会先读取原问题 top20，再检索 router 生成的 QD 或
generation-guided query。推荐使用带真实 dense score 的 query 级缓存：

```text
racp/output/cache/hotpotqa_dev_bge_large_no_rerank_top20_retrieval_cache.json
```

不要把 `*_prompt_cache.json` 传给 `--retrieval_cache_path`。旧 HotpotQA top20
prompt cache 没有保存 dense score，不能无损恢复 EFC router 的分数特征。

全量 prepare：

```bash
python racp/run_racp.py \
  --stage prepare \
  --dataset_name hotpotqa \
  --split dev \
  --gpu_id 3 \
  --enable_efc_rag \
  --no_reranker \
  --initial_topk 20 \
  --probe_topk 5 \
  --final_topk 5 \
  --enable_generation_guided \
  --enable_static_qd_fallback \
  --missing_query_mode heuristic \
  --title_dedup_soft \
  --use_retrieval_cache \
  --retrieval_cache_path racp/output/cache/hotpotqa_dev_bge_large_no_rerank_top20_retrieval_cache.json \
  --save_retrieval_cache \
  --planner_batch_size 16 \
  --save_note efc-hotpotqa-full
```

已有原问题和旧 query 直接命中缓存；新 query 缓存未命中时在线检索。源缓存不会
被改写，合并后的缓存保存在本次实验目录：

```text
racp/output/hotpotqa_YYYY_MM_DD_HH_MM_efc-hotpotqa-full/retrieval_cache.json
```

后续实验优先读取这份扩充后的缓存。生成阶段直接使用同目录的 prompt cache：

```bash
python racp/run_racp.py \
  --stage generate \
  --gpu_id 3 \
  --prompt_cache_path racp/output/hotpotqa_YYYY_MM_DD_HH_MM_efc-hotpotqa-full/prompt_cache.json
```

当 probe、planner 和 query 生成配置不变时，第二次 prepare 可以完全跳过 corpus、
BGE encoder 和 FAISS 索引：

```bash
python racp/run_racp.py \
  --stage prepare \
  --dataset_name hotpotqa \
  --split dev \
  --gpu_id 3 \
  --enable_efc_rag \
  --no_reranker \
  --enable_generation_guided \
  --enable_static_qd_fallback \
  --missing_query_mode heuristic \
  --title_dedup_soft \
  --retrieval_cache_only \
  --retrieval_cache_path racp/output/hotpotqa_YYYY_MM_DD_HH_MM_efc-hotpotqa-full/retrieval_cache.json \
  --save_note efc-hotpotqa-cache-only
```

如果 router 或 planner 产生缓存中不存在的新 query，cache-only 会直接报告缺失
query；此时去掉 `--retrieval_cache_only` 并增加 `--save_retrieval_cache` 再跑一次。

## 常用消融

当前较好的设置：

```bash
python racp/run_racp.py --method racp --dataset_name nq --split test --gpu_id 2 --max_k 8
```

不限制最大 k：

```bash
python racp/run_racp.py --method racp --dataset_name nq --split test --gpu_id 2 --max_k none
```

关闭 reranker：

```bash
python racp/run_racp.py --method racp --dataset_name nq --split test --gpu_id 2 --no_reranker
```

只快速跑 100 条：

```bash
python racp/run_racp.py --method racp --dataset_name nq --split test --gpu_id 2 --test_sample_num 100
```

如果显存紧张，可以降低 vLLM 的显存预留：

```bash
python racp/run_racp.py --method racp --dataset_name nq --split test --gpu_id 2 --gpu_memory_utilization 0.65
```

## 关键参数

- `DEFAULT_RUN_CONFIG`: 代码顶部集中保存默认方法、阶段、数据集、GPU、batch、top-k 和模型。
- `--method`: 选择 `efc`、`racp`、`rs_mhr` 或 `qd`，默认 `efc`。
- `--retrieval_batch_size`: BGE query 编码 batch，4090 默认 1024。
- `--retrieval_topk`: 第一阶段召回数量，默认 20。
- `--rerank_topk`: reranker 后保留数量，默认 20。
- `--buffer`: 最大 gap 位置后额外保留的文档数，默认 5。
- `--max_k`: 最终送入 Selective-Context/LLM 的文档上限，默认 8；传 `none` 表示不限制。
- `--search_ratio`: 用前多少比例的重排分数搜索最大 gap，默认 0.9。
- `--selection_method`: 最终文档选择方式。`title_dedup_topk` 会在 reranker 排序后优先选择不同标题的文档。
- `--reduce_ratio`: Selective-Context 的压缩比例，默认 0.5。
- `--stage`: 运行阶段，默认 `prepare`；`full` 完整跑，`generate` 读取 prompt 后生成和评测。
- `--prompt_cache_path`: `generate` 阶段读取的 prompt cache 路径。
- `--planner_model`: 全量 QD、RS-MHR 或 EFC planner 模型，默认 `Llama-3.1-8B-Instruct`。
- `--planner_batch_size`: Planner 每个外层分块提交的 prompt 数，默认 32。
- `--planner_inference_batch_size`: EFC HF Planner 的真实 GPU batch，4090 默认 8。
- `--qd_rerank_with_subqueries`: 全量 QD 中额外使用每条 subquery 重排其召回文档，并与原问题 reranker 分数融合。
- `--qd_subquery_rerank_weight`: subquery reranker 归一化分数的融合权重，默认 0.25。
- `--strict_short_answer_prompt`: 使用更严格的最短答案生成模板，减少正确答案被解释性文本拖累。
- `--answer_only_prompt`: 使用中等严格的最终答案模板，只禁止解释文本，尽量保留完整答案 span。
- `--exact_answer_prompt`: 使用精确答案 span 模板，介于 strict 与 answer-only 之间。
- `--save_retrieval_cache`: 保存 query 级召回缓存。
- `--use_retrieval_cache`: 读取已有 query 级召回缓存。
- `--retrieval_cache_only`: 缓存完整时跳过 corpus、FAISS 索引和 BGE encoder 加载。
