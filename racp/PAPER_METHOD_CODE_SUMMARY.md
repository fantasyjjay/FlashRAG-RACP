# RACP / EFC-RAG Methodology 代码审计与论文写作说明

> 审计日期：2026-07-15  
> 审计方式：只读检查当前生效代码、四个正式 EFC FINAL 配置与逐样本产物、现有正式台账和 Git 历史。  
> 当前仓库：`HEAD=063e29dde59191c2c249ec1bde669834dc3e88ee`；工作区存在未提交修改。  
> 本文目的：为论文的 Methodology、Algorithm、Implementation Details 和 Appendix 提供可追溯的技术说明，不替代实验结果表。

## 审计范围与版本边界

正式 EFC 结果由 [`racp/PAPER_RESULTS_REFERENCE.md`](PAPER_RESULTS_REFERENCE.md) 约第 349–403 行登记。本文逐项核对的四个目录如下。

| 数据集 | split | 正式目录 | 生效配置 |
|---|---|---|---|
| HotpotQA | dev | [`output/hotpotqa_2026_07_09_15_52_efc-static-bridge-final-top10-full`](output/hotpotqa_2026_07_09_15_52_efc-static-bridge-final-top10-full) | [`config.yaml`](output/hotpotqa_2026_07_09_15_52_efc-static-bridge-final-top10-full/config.yaml)，约第 40–72、123–146、172–212 行 |
| 2WikiMultiHopQA | dev | [`output/2wikimultihopqa_2026_07_12_11_11_2wiki-efc-no-rerank-top10-full`](output/2wikimultihopqa_2026_07_12_11_11_2wiki-efc-no-rerank-top10-full) | [`config.yaml`](output/2wikimultihopqa_2026_07_12_11_11_2wiki-efc-no-rerank-top10-full/config.yaml)，同上 |
| MuSiQue | dev | [`output/musique_2026_07_12_20_15_musique-efc-no-rerank-top10-prepare`](output/musique_2026_07_12_20_15_musique-efc-no-rerank-top10-prepare) | [`config.yaml`](output/musique_2026_07_12_20_15_musique-efc-no-rerank-top10-prepare/config.yaml)，同上；目录名虽含 `prepare`，但已包含完整指标和逐样本预测 |
| NQ | test | [`output/nq_2026_07_14_09_54_nq-efc-no-rerank-top10-full`](output/nq_2026_07_14_09_54_nq-efc-no-rerank-top10-full) | [`config.yaml`](output/nq_2026_07_14_09_54_nq-efc-no-rerank-top10-full/config.yaml)，同上 |

四份 FINAL 配置的 EFC 算法参数一致：`initial_topk=20`、`probe_topk=5`、`final_topk=10`、`qd_num=2`、`qd_topk=5`、`missing_query_num=1`、`gen_topk=10`、`rrf_k=60`、选择权重 `1.0/0.30/0.02/0.05/0.01`、`original_seed_count=4`、无 reranker、无 refiner。数据集行为的唯一显式算法差异是 HotpotQA、2Wiki 和 MuSiQue 被传入 `assume_multihop=True`，NQ 为 `False`（`racp/run_racp.py::retrieve_with_efc`，约第 2157–2164 行）。

需要严格区分三个版本层次：

1. Git commit `ce132a8` 首次加入 EFC；`0044d6b` 加入当前 comparison/contradiction 路由和 original seed；`58dcbe6` 形成当前 generation-guided 优先规则、static bridge prompt/packing；`4f4573b` 锁定正式实验代码基线。相关历史可由 `git log -- racp/efc.py racp/run_racp.py` 和 `git blame` 恢复。
2. 当前 `racp/run_racp.py::DEFAULT_RUN_CONFIG` 约第 72–119 行的源码缺省 `final_topk=6`，但四个 FINAL 的保存配置都明确为 10；论文必须写 FINAL 的 \(K=10\)，不能引用源码缺省值。
3. commit `063e29d` 新增默认关闭的 `rrf_only_selection`；当前未提交 diff 又新增默认关闭的 `title_dedup_only_selection`。它们是后续消融入口，不属于四个既有 FINAL，也不改变两个开关均为 false 时的 canonical EFC 路径（`racp/run_racp.py::retrieve_with_efc`，约第 2531–2586 行）。

# 1. 项目问题定义

## 1.1 输入、输出与固定组件

给定自然语言问题 \(q\) 和固定外部语料库

\[
\mathcal D=\{d_1,\ldots,d_{|\mathcal D|}\},
\]

EFC-RAG 的目标是在不训练新模型、不更新索引的条件下，从 \(\mathcal D\) 中选出至多 \(K\) 篇证据并生成答案 \(\hat a\)。四个 FINAL 共用 `wiki18_100w.jsonl`、BGE-large-en-v1.5 Flat FAISS 索引和 Llama-3.1-8B-Instruct；配置证据位于各 FINAL `config.yaml` 约第 50–72、123–146 行。Dense retriever 负责将 query 编码并对固定索引执行 top-\(k\) 搜索（`flashrag/retriever/retriever.py::DenseRetriever._batch_search`，约第 450–471 行）；generator 同时承担 Probe、条件 Planner 和最终答案生成（`racp/run_racp.py::retrieve_with_efc`，约第 2133–2149、2204–2443 行；`run_generate`，约第 3823–3848 行）。

检索算子记为

\[
\operatorname{Ret}(x,k)=\big[(d_i,s_i,r_i)\big]_{i=1}^{k},
\]

其中 \(s_i\) 是 dense score，\(r_i\) 是从 1 开始的名次。首先执行

\[
R_0=\operatorname{Ret}(q,N_0),\qquad N_0=20.
\]

前 \(N_p=5\) 篇形成 Probe 上下文：

\[
y_p=G\!\left(P_{\mathrm{probe}}(q,R_0[:N_p])\right).
\]

随后由规则特征 \(\phi(q,R_0,y_p)\) 选择路由

\[
z=\pi\!\left(q,R_0,y_p\right)
\in\{\textsc{Direct},\textsc{Gen},\textsc{Static}\}.
\]

如果需要扩展，Planner 产生一个或多个查询 \(q_m\)：

\[
Q_m=P_m(q,y_p,R_0),\qquad
R_m=\{\operatorname{Ret}(q_m,k_m):q_m\in Q_m\}.
\]

注意：generation-guided Planner 实际只读 \(q\) 与完整 \(y_p\)；static bridge Planner 读 \(q\) 与 \(R_0[:5]\)；plain static Planner 只读 \(q\)。上式的参数集合是统一记号，不表示每条路线都实际消费全部三项。

候选池按文档 UID 合并：

\[
\mathcal C=\operatorname{MergeUID}\!\left(R_0\cup\bigcup R_m\right).
\]

最后由路由相关选择器在固定预算 \(K=10\) 下产生

\[
S=\operatorname{Select}_{z}(\mathcal C,q,y_p),\qquad |S|\le K,
\]

并生成

\[
\hat a=G\!\left(P_{\mathrm{ans}}(q,\operatorname{Order}_{z}(S))\right).
\]

代码没有 EFC 专用的最终答案 parser；\(\hat a\) 就是 generator 返回的原始字符串，之后仅由 evaluator 在计算指标时规范化（`racp/run_racp.py::run_generate`，约第 3833–3841 行；`flashrag/evaluator/utils.py::normalize_answer`，约第 5–19 行）。

## 1.2 EFC-RAG 处理的具体问题

单纯按某一个 query 的单文档 dense relevance 取前 \(K\) 篇，无法显式控制以下代码中被分别建模的因素：原问题实体锚点、Probe 暴露的桥接实体、预期答案类型、comparison 两侧覆盖、来源覆盖、标题多样性和文本冗余。EFC 因而先用 Probe 暴露初始证据中的中间实体或失败状态，再决定是否追加一阶段检索，并在固定 final context 预算内融合多 query provenance。该动机与 `compute_role_scores`、`role_aware_pack` 的实际输入相符（`racp/efc.py`，约第 542–698 行），但不应上升为“单文档相关性必然无效”的未经检验定理。

方法属性可准确写为：

- **training-free**：项目中没有 EFC 参数训练、梯度更新或新 checkpoint；
- **inference-time routing and evidence selection**：路由、查询扩展、融合和 packing 均发生在推理时；
- **frozen retriever / generator / index**：不改变 BGE、Llama 或 FAISS index 的参数与内容；
- **no cross-encoder reranker**：EFC 分支在构建配置时强制 `use_reranker=False`（`racp/run_racp.py::build_config_dict`，约第 419–425 行）。

这里的固定预算指最终传给 prompt builder 的文档槽位 \(K=10\)，不是固定的检索 query 数或 raw retrieved-document 数。Direct 的逻辑检索预算为 20 篇原始候选；正常 generation-guided 为 \(20+10\)；正常 static-QD 为 \(20+2\times5\)，随后都压到最多 10 篇。

# 2. 从入口到最终答案的完整调用链

真实主链如下：

```text
python racp/run_racp.py --method efc
→ parse_args / apply_method_defaults
→ build_config_dict / Config
→ run → run_full（或 prepare / generate）
→ load_split
→ get_generator（Probe）与 get_retriever / EFC_CacheOnlyRetriever
→ retrieve_with_efc
   → initial raw top20
   → Probe(top5)
   → compute_router_features → decide_route → availability mapping
   → conditional Planner
   → conditional incremental retrieval
   → normalize → merge_docs → add_rrf_scores
   → route-specific final top10 selection
→ build_final_prompts
→ clean vLLM generate subprocess（full + vLLM）
→ raw final generation
→ Evaluator
```

## 2.1 论文模块—代码实现对应表

| 论文模块 | 文件、类/函数与大致行号 | 关键输入 | 返回/持久化输出 |
|---|---|---|---|
| CLI 与方法选择 | `racp/run_racp.py::parse_args`，约 4120–4386；`apply_method_defaults`，约 4099–4117 | `--method efc`、stage、预算参数 | `args.enable_efc_rag=True` |
| 配置合并 | `racp/run_racp.py::build_config_dict`，约 312–478；`flashrag/config/config.py::Config`，约 9–101 | base YAML + CLI | 保存的完整 `config.yaml` |
| 数据集加载 | `racp/run_racp.py::load_split`，约 3444–3459；`flashrag/utils/utils.py::get_dataset`，约 9–38 | dataset name、split | `Dataset[Item]` |
| 模型/检索器初始化 | `racp/run_racp.py::retrieve_and_prepare`，约 3572–3600；`flashrag/utils/utils.py::get_generator/get_retriever`，约 41–98 | FINAL config | Probe generator、dense retriever 或 cache-only retriever |
| 原始检索 | `racp/run_racp.py::retrieve_with_efc`，约 2114–2125；`rs_mhr_raw_batch_search`，约 1149–1188 | 全部 \(q\)、\(N_0=20\) | 1-based、带 provenance 的 \(R_0\) |
| Probe | 同函数约 2133–2150 | \(q,R_0[:5]\) | 原始 Probe 文本 \(y_p\) |
| 特征 | `racp/efc.py::compute_router_features`，约 363–410 | \(q,R_0,y_p\) | question/probe/可选 centroid 特征 dict |
| Router | `racp/efc.py::decide_route`，约 413–440；`racp/run_racp.py::efc_route_with_availability`，约 2079–2088 | 特征、force/availability | router route 与实际 route |
| Planner | `racp/run_racp.py::get_efc_planner`，约 1927–1965；`generate_rs_mhr_queries`，约 1456–1515 | 路由相关 prompt | 规范化 query、失败原因 |
| 增量检索 | `retrieve_with_efc`，约 2445–2529 | `gen_*` 或 `qd_*` query records | 各 query 的 top-\(k\) 文档与分数 |
| 候选合并 | `racp/efc.py::normalize_doc/merge_docs`，约 129–173 | \(R_0\) 与扩展组 | UID 去重候选池 \(\mathcal C\) |
| RRF | `racp/efc.py::add_rrf_scores`，约 498–510 | 候选的 1-based ranks | `rrf_score` |
| 角色特征 | `racp/efc.py::compute_role_scores`，约 542–581 | \(d,q,y_p,\phi\) | 六维 heuristic role scores |
| Final packing | `racp/run_racp.py::retrieve_with_efc`，约 2531–2586；`racp/efc.py` 约 618–817 | \(\mathcal C,z,K\) | 最终 selected docs |
| Prompt | `racp/run_racp.py::build_final_prompts`，约 3483–3500；`flashrag/prompt/base_prompt.py::PromptTemplate`，约 5–228 | ordered selected docs、\(q\) | 序列化并截断后的 prompt |
| 最终生成 | `racp/run_racp.py::run_full_vllm_generate/run_generate`，约 3866–3897、3823–3848 | prompt cache 中的 prompt | 原始 `pred` |
| 评测 | `flashrag/evaluator/evaluator.py::Evaluator.evaluate`，约 46–81 | dataset + raw `pred` | `metric_score.txt`、逐样本 metric |

## 2.2 调用、副作用与失败回退

| 步骤 | LLM | Retriever | 依赖缓存 | 实际失败行为 |
|---|:---:|:---:|---|---|
| CLI/config/dataset | 否 | 否 | generate stage 可读 prompt cache | 参数非法抛异常；无算法回退 |
| Initial retrieval | 否 | 是 | exact-query retrieval cache | 普通 cache miss 在线检索；cache-only miss 直接抛错 |
| Probe | 是 | 否 | prepare 中不缓存调用；结果会写入 prompt cache | 无重试；异常终止 |
| Router | 否 | 否 | 否 | 被禁用路线由 availability mapping 转 direct/另一扩展路线 |
| Missing-hop Planner | 是 | 否 | 否 | invalid → heuristic repair；repair 空 → late static 或 direct |
| Static Planner | 是 | 否 | 否 | invalid → direct；late static 也失败 → direct |
| Incremental retrieval | 否 | 是 | 同一 exact-query cache | 普通 miss 在线检索；cache-only miss 终止 |
| Merge/RRF/packing | 否 | 否 | 否 | 候选少于 \(K\) 时返回全部候选 |
| Final generation | 是 | 否 | generate stage直接读取完整 prompt cache | 无答案重试或 EFC parser fallback |
| Evaluator | 否 | 否 | 否 | 单个 metric 异常被捕获并跳过，见 `Evaluator.evaluate` 约第 49–59 行 |

# 3. EFC-RAG 总体算法

## 3.1 论文简化版（Algorithm 1）

以下为 35 行 `algorithmic` 正文版伪代码；省略 parser 的容错顺序和 static backfill 的内部循环，但没有引入代码中不存在的步骤。

```latex
\begin{algorithm}[t]
\caption{Evidence-Feedback Controlled RAG}
\label{alg:efc}
\begin{algorithmic}[1]
\Require $q,\mathcal D,\mathrm{Ret},G$; budgets $N_0=20,N_p=5,K=10$
\State $R_0 \gets \mathrm{Ret}(q,N_0)$
\State $y_p \gets G(P_{\mathrm{probe}}(q,R_0[:N_p]))$
\State $x \gets \phi(q,R_0,y_p)$; $z \gets \pi(x)$ \Comment{ordered rules}
\State $\mathcal R \gets \{R_0\}$
\If{$z=\textsc{GenerationGuided}$}
  \State $Q_m \gets \mathrm{Parse}(G(P_m(q,y_p)))$
  \State if $Q_m=\varnothing$, $Q_m\gets\mathrm{HeuristicRepair}(q,y_p,x)$
  \If{$Q_m\neq\varnothing$}
    \State $\mathcal R\gets\mathcal R\cup\{\mathrm{Ret}(u,10):u\in Q_m\}$
  \Else
    \State $z\gets\textsc{Static}$ if enabled, else $\textsc{Direct}$
  \EndIf
\EndIf
\If{$z=\textsc{Static}$}
  \State $Q_d \gets \mathrm{Parse}(G(P_d(q,R_0[:5])))$
  \If{$|Q_d|\ge 2$}
    \State $\mathcal R\gets\mathcal R\cup\{\mathrm{Ret}(u,5):u\in Q_d[:2]\}$
  \Else
    \State $z\gets\textsc{Direct}$
  \EndIf
\EndIf
\State $\mathcal C\gets\mathrm{MergeUID}(\mathcal R)$
\State compute source-weighted $\mathrm{RRF}(d)$ for each $d\in\mathcal C$
\If{$z=\textsc{Direct}$}
  \State $S\gets\mathrm{RankedTitleDiverse}(R_0,K)$
\ElsIf{$z=\textsc{Static}$ and $x.\mathrm{type}\in\{\mathrm{bridge,constraint}\}$}
  \State $S\gets\mathrm{StaticQuotaBackfill}(\mathcal C,K)$
\Else
  \State $S\gets\mathrm{GreedyRolePack}(\mathcal C,q,y_p,K)$
\EndIf
\State $S\gets\mathrm{RouteOrder}(S,z)$ for expansion routes
\State $\hat a\gets G(P_{\mathrm{ans}}(q,S))$
\end{algorithmic}
\end{algorithm}
```

正文配套说明必须补一句：若 generation-guided Planner 失败且 heuristic repair 仍为空，实际实现会再调用一次 static Planner；Algorithm 1 为避免展开异常路径而将其压缩为路线转换。

## 3.2 实现精确版（附录）

```text
INPUT q; FINAL config N0=20, Np=5, K=10

1  R0 ← CACHE_OR_DENSE_RETRIEVE(q, 20)
2  normalize every R0 item with query_id="original", source="original",
   1-based rank, raw dense score and UID
3  yp ← deterministic Llama generation of PROBE_PROMPT(q, R0[:5]), max 128
4  x ← COMPUTE_ROUTER_FEATURES(q, R0, yp,
       assume_multihop = dataset in {HP, 2W, MU})
5  router_route ← first matching rule in DECIDE_ROUTE(x, force_route)
6  route ← AVAILABILITY_MAP(router_route, enabled routes)
7  initialize planner_calls=0, qd_queries=[], gen_queries=[]

8  Partition initial static samples into:
     bridge/constraint → evidence-conditioned static prompt using R0[:5]
     comparison/generic → question-only static prompt
9  For initial static sample:
     output ← deterministic HF planner, max 96, planner stop list
     queries ← tolerant JSON/line parser, normalize, reject invalid/duplicate/q
     if at least 2 valid: keep first 2
     else: planner_failed=True; route=direct

10 For generation-guided with FINAL mode="llm":
     output ← planner(MISSING_PROMPT(q, yp)); planner_calls += 1
     queries ← parser requiring at least 1 valid query; keep first 1
     reject any kept query with SequenceMatcher(query,q) ≥ 0.9
11   if valid:
       gen_queries ← queries; strategy="llm_missing_hop"
12   else:
       planner_failed=True
       repaired ← ENTITY_FOCUS_REPAIR(q, yp, x, titles(R0))
       if repaired nonempty:
         gen_queries=[repaired]; strategy="heuristic_repair"
       else:
         route=static_qd if static enabled else direct

13 For a generation failure newly changed to static with no qd_queries:
     invoke the appropriate static planner a second time
     if at least 2 valid: keep first 2; record fallback strategy
     else: append failure reason; route=direct

14 Build query_records:
     static: each qd_i uses source="qd", query_id="qd_i", topk=5
     generation: each gen_i uses source="generation_guided",
                 query_id="gen_i", topk=10
15 If generation route still has no query:
     static only when qd_queries already exist and static enabled; otherwise direct
16 Batch all extension queries that share topk; cache hit is exact string match
17 Normalize extension lists with 1-based ranks and provenance
18 C ← MERGE_BY_UID([R0] + extension lists), retaining first document body
   and unioning sources/queries/ranks/scores
19 For every d in C:
     RRF(d) ← Σ_j source_weight(j)/(60 + rank_j(d))

20 If canonical route=direct:
     scan R0 by original rank, first pass one lowercased title each
     defer repeated titles; append deferred documents if needed to reach K
     compute role scores only for diagnostics

21 Else if route=static_qd and question_type in {bridge,constraint}:
     A ← up to 4 unseen-title original-provenance docs by original rank
     B ← up to 2 unseen-title qd-provenance docs by qd rank then RRF
     S ← A followed by B
     add unseen-title docs from whole C by (-RRF, best rank) until K
     if still short, add remaining UID-unique docs allowing repeated titles
     compute role coverage only for diagnostics

22 Else:
     compute six heuristic role scores for all d in C
     S ← up to 4 original-provenance seeds using title-diverse first pass
     initialize max role coverage, covered source types and title counts from S
     while |S| < min(K,|C|):
       for every d not in S:
         compute set-marginal role gain, new-title bonus, new-source bonus,
         max lexical Jaccard/same-title redundancy and soft title penalty
       append argmax(total gain, RRF, -best rank)
       update coverage/source/title state

23 For expansion routes, sort original-provenance docs first, then
   expansion-only docs, each by its route-specific rank key
24 Serialize each selected passage as:
   "Doc i(Title: <contents first line>) <contents remaining lines>"
25 Apply chat template and whole-prompt head/tail truncation to 4096 tokens
26 Final answer ← deterministic Llama generation, max 32,
   explicit "<|eot_id|>" stop; runtime model EOS list is UNKNOWN
27 Store raw generator text as pred; run evaluator normalization only for metrics
```

实现依据集中在 `racp/run_racp.py::retrieve_with_efc` 约第 2114–2717 行、`racp/efc.py` 约第 101–891 行，以及 `flashrag/prompt/base_prompt.py::PromptTemplate` 约第 5–228 行。

# 4. Probe 模块

## 4.1 实际 Prompt 与调用参数

实际 system prompt（`racp/run_racp.py::EFC_PROBE_SYSTEM_PROMPT`，约第 184–191 行）为：

```text
You are given several retrieved Wikipedia passages and a question.
Use at most two short evidence sentences. Keep the named bridge entity explicit.
If the passages are insufficient, state the bridge entity and use unknown as the answer.
For yes/no questions, answer yes when the compared attributes are the same and no when
they differ. Always end exactly with: So the answer is <answer>.
The following are given documents.

{reference}
```

实际 user prompt（同文件 `EFC_PROBE_USER_PROMPT`，约第 192 行）为：

```text
Question: {question}
```

Probe 使用原始 top5，不使用 final selector；调用同一正式 Llama-3.1-8B-Instruct checkpoint，`do_sample=False`、`temperature=0`、`max_new_tokens=128`（`retrieve_with_efc`，约第 2133–2148 行）。代码没有显式传方法级 `stop`；vLLM wrapper 显式加入的 stop 只能从代码确认为 `<|eot_id|>`（`flashrag/generator/generator.py::VLLMGenerator.generate`，约第 214–243 行）。模型/vLLM 内部的完整 EOS 枚举未保存在 artifact 中，因而为 `UNKNOWN`。“最多两句”只由 prompt 约束，代码没有句数 validator。

适合正文展示的精简版本可以写成：

```text
Given the question and the top retrieved passages, state at most two short evidence
sentences, keep any bridge entity explicit, and end with
"So the answer is <answer>." Use "unknown" when the evidence is insufficient.
```

## 4.2 Probe 的实际结构化字段

Probe 原文不会被解析成完整的结构化 reasoning object。Router 从原文派生以下字段（`racp/efc.py::compute_router_features`，约第 363–410 行）：

| 字段 | 类型 | 实际计算 |
|---|---|---|
| `y1_final_answer` | string | 最后一个 `So the answer is` 或 `Final answer` 后至句号/换行的片段，小写 |
| `y1_uncertain` | bool | 是否包含固定 uncertainty marker |
| `y1_contradictory` | bool | yes/no Probe 的结论与其 reasoning 词面是否矛盾 |
| `y1_bad` | bool | 短输出、拒答、uncertain、截断尾词、高问题复述比或 contradiction |
| `y1_new_entities` | list[string] | top5 标题命中与大写实体正则的过滤结果 |
| `y1_has_new_entity` | bool | 上述列表是否非空 |
| `answer_type` | enum-like string | number/date/person/location/yesno/entity |
| `answer_type_hit_top5` | bool | 任一 top5 文档是否命中答案类型的词面规则 |

`extract_probe_final_answer` 只供 Router 使用，不是最终答案 parser；没有匹配到固定前缀时返回空字符串，但空字符串本身不会单独触发 bad（`racp/efc.py::extract_probe_final_answer`，约第 270–277 行；`probe_answer_is_bad`，约第 296–328 行）。Probe 也从不直接成为最终预测：包括 direct route 在内，每条样本仍执行一次独立 final generation。

## 4.3 uncertain、bad 与 contradiction

`probe_answer_is_uncertain` 采用不区分大小写的 substring 检查，markers 包括：`no information`、`not enough information`、`does/do/doesn't mention`、`unknown`、`cannot/can't determine`、`unable to determine`、`not provided/specified/available`（`racp/efc.py::PROBE_UNCERTAINTY_MARKERS`，约第 61–74 行；函数约第 265–267 行）。

`probe_answer_is_bad` 依次检查：

1. 少于 2 个空格分词；
2. `I don't/do not know`、`cannot/can't answer`；
3. 上述 uncertain；
4. 末词是 `a/an/and/are/is/of/the/to/was/were`；
5. Probe 词集合中超过 90% 同时出现在问题，且 Probe 长度不超过问题长度加 2；
6. yes/no contradiction。

contradiction 仅对推断为 yes/no 的问题定义：若抽取结论为 `no`，但 reasoning 出现 `both ... same` 或指定国籍词；或者结论为 `yes`，但 reasoning 出现 `different/not the same/differ`，则为真（`racp/efc.py::probe_answer_is_contradictory`，约第 280–293 行）。

## 4.4 新实体、bridge 与 answer type

新实体候选来自两处（`racp/efc.py::extract_new_entities`，约第 331–360 行）：

- top5 标题中同时出现在 Probe、但不出现在原问题的标题；
- Probe 中由大写 span 正则抽出的 1–6 token 实体，允许内部出现 `of/the/and/de/van`。

清理后过滤短于 3 字符、作为小写 substring 已在问题中、属于固定 stop list、重复或纯数字的项。“实体是否在问题中”因此是 substring 规则，不是 entity linker。代码没有另一个 bridge classifier；`y1_has_new_entity=True` 就是 Router 所用的“Probe 暴露 bridge”的操作性判定，不能写成已验证该实体确实连接两个 hop。

`infer_answer_type` 使用问题开头和关键词：how-many/number/capacity/population → number；when/year/date → date；`who` 开头 → person；`where` 开头 → location；若以 is/are/was/were/can/did/does 开头 → yesno；否则 entity（`racp/efc.py`，约第 228–240 行）。它是规则，不调用 embedding 或 LLM。

## 4.5 Probe 如何进入后续模块

- generation-guided：完整 \(y_p\) 与原问题一起进入 missing-hop Planner；
- static bridge：Planner 读取原问题和 \(R_0[:5]\)，不读取 \(y_p\)；
- plain static：Planner 只读取原问题；
- direct：\(y_p\) 只用于路由和诊断 role scores；
- final answer prompt：不包含 Probe reasoning。

对应代码为 `racp/run_racp.py::build_efc_missing_prompt` 约第 1999–2026 行、`build_efc_static_bridge_prompt` 约第 1428–1447 行，以及 `retrieve_with_efc` 约第 2212–2327 行。

## 4.6 真实轨迹的匿名化示例

该示例从 HotpotQA FINAL 的 `intermediate_data.json` 中一条 generation-guided 轨迹匿名化而来；实体名已替换，路由字段与执行关系保持不变。

```text
Question:
在电影 [Film X] 中饰演 [Character Y] 的女士后来担任什么政府职务？

Probe:
饰演者是 [Person A]。但这些文档没有说明她担任的政府职务。
So the answer is unknown.

Derived fields:
question_type=bridge
answer_type=entity
y1_bad=true
y1_uncertain=true
y1_new_entities=["Person A"]

Route:
generation_guided

Planner query:
"government positions held by Person A"
```

这个例子揭示 Router 的真实优先级：尽管 Probe 同时 bad/uncertain，只要先检测到新实体，代码就先进入 generation-guided，而不是 static-QD。原始证据路径为 HotpotQA FINAL `intermediate_data.json` 中相应样本的 `output.probe_answer/router_features/missing_hop_queries`。

# 5. Router 精确规则

## 5.1 特征与 question type

`classify_question` 首先按词面顺序判定 comparison、bridge、constraint；若均未命中，则 `assume_multihop=True` 时返回 bridge，否则返回 generic（`racp/efc.py::classify_question`，约第 176–225 行）。

- comparison markers 包括 `between`、`which came first`、`who had more`、larger/smaller、earlier/later、more/less/same 和宽泛的 `or`；
- bridge markers 包括 `director/author/wife/husband/father/mother of`、`where was`、`arena where`、`what is the`、`who wrote/directed`、`capital/spouse of`；
- constraint markers 包括 `born/located/worked with`、`directed/written/published/founded by`、`member of`。

Router 同时计算 dense-score、title 与 centroid 特征，但当前决策函数并未使用 `s1`、`s5`、`avg_top5_score`、`gap_1_5` 或 `title_unique_ratio`。`q_centroid_sim` 只有启用 centroid router 时才可能影响路线；四个 FINAL 均关闭该开关（`compute_router_features`，约第 363–410 行；四份 FINAL config 约第 172–212 行）。

## 5.2 顺序规则表

规则策略必须按下表顺序描述（`racp/efc.py::decide_route`，约第 413–440 行）。

| 优先级 | 条件 | Router 输出 | 代码位置 |
|---:|---|---|---|
| 1 | `force_route != auto` | 原样返回强制 route | `decide_route` 约 413–415 |
| 2 | `question_type=comparison` 且 comparison 左右侧均在 top5 title+contents 中出现 | direct | 约 416–420 |
| 3 | `y1_contradictory=True` 且 top5 至少一篇命中预期 answer type | direct | 约 421–422 |
| 4 | `question_is_multihop=False` | direct | 约 423–424 |
| 5 | Probe 暴露至少一个新实体 | generation-guided | 约 425–429 |
| 6 | Probe bad 或 uncertain | static-QD | 约 430–431 |
| 7 | `q_centroid_sim` 存在且小于 0.35 | static-QD | 约 432–436 |
| 8 | 以上均不满足 | direct | 约 437–440 |

`force_route` 的判断发生在 Probe 之后，而不是在 Probe 之前；所以 force-direct 消融也固定支付 Probe 和 final generation 两次调用。CLI 对 EFC 强制值的合法集合限制为 `auto/direct/static_qd/generation_guided`（`racp/run_racp.py::run`，约第 3992–3998 行）。

Router 输出之后还有 availability mapping（`racp/run_racp.py::efc_route_with_availability`，约第 2079–2088 行）：

- static 被禁用：若 generation 启用且存在新实体，转 generation-guided；否则 direct；
- generation 被禁用：若 static 启用则 static-QD；否则 direct。

## 5.3 决策树

```text
Probe 已经生成
│
├─ force_route != auto ───────────────────────────────→ forced route
│
├─ comparison 且左右实体均被 top5 覆盖 ─────────────→ direct
│
├─ yes/no Probe 自相矛盾 且 top5 有类型证据 ─────────→ direct
│
├─ question_is_multihop = false ─────────────────────→ direct
│
├─ Probe 暴露新实体 ─────────────────────────────────→ generation-guided
│
├─ Probe bad 或 uncertain ───────────────────────────→ static-QD
│
├─ centroid 已启用且 q-centroid similarity < 0.35 ─→ static-QD
│
└─ default ──────────────────────────────────────────→ direct

随后：若目标 route 被禁用，再执行 availability mapping。
```

三套多跳数据集的 `assume_multihop=True` 会把没有命中任何词面类型的问题也归为 bridge，因此规则 4 基本不会把这类样本送 direct。NQ 的 `assume_multihop=False` 则允许 generic 问题在检查新实体之前直接进入 direct。这不是训练得到的数据集阈值，而是硬编码数据集名称集合（`retrieve_with_efc`，约第 2162–2164 行）。

## 5.4 数学抽象与禁止表述

\[
z=\pi(q,R_0,y_p)=
\operatorname{FirstMatch}\big(\mathcal R_1,\ldots,\mathcal R_8\big).
\]

\(\pi\) 是有序规则策略，不是训练分类器，也没有输出概率、置信度或“证据充分性分数”。论文可以写“probe-conditioned rule router”或“规则化证据反馈路由”，不应写“learned sufficiency estimator”“sufficiency probability”或“calibrated confidence router”。

centroid 分支虽然仍在当前代码中，但四个 FINAL `use_centroid_router=false`，因此 0.35 阈值、`top5_cohesion` 和 `centroid_shift` 都不是正式主方法的有效决策项。若论文不报告独立 centroid 实验，建议只在附录的 inactive implementation option 中说明。

# 6. Generation-Guided Expansion

## 6.1 Planner 输入与 Prompt

四个 FINAL 均采用 `missing_query_mode=llm`、`missing_query_num=1`。实际 system prompt（`racp/run_racp.py::EFC_MISSING_SYSTEM_PROMPT`，约第 199–202 行）为：

```text
You are a retrieval query reformulator for multi-hop question answering.
Output only a JSON array containing one search query.
```

实际 user prompt（`build_efc_missing_prompt`，约第 1999–2012 行）为：

```text
Given the original question and tentative reasoning, generate exactly one
missing-hop search query.

Rules:
1. Use the bridge entity from the tentative reasoning if available.
2. Ask for the missing factual attribute.
3. Do not answer the question.
4. Output only a JSON array with one string.

Question:
{question}

Tentative reasoning:
{probe_answer}

Output:
```

因此可写

\[
q_m=P_m(q,y_p),
\qquad
R_m=\operatorname{Ret}(q_m,k_m),\quad k_m=10.
\]

不要在公式中声称 normal generation-guided Planner 直接读取 \(R_0\)：它只通过 Probe 文本间接接触首轮证据。原始问题不会被字符串附加到最终生成的 search query；它只作为 Planner prompt 的输入。非 FINAL 的 `raw_y1` mode 才会构造 `question + "\n" + probe_answer`，四个正式运行未使用（`retrieve_with_efc`，约第 2456–2486 行）。

Planner checkpoint 与主 generator 相同，都是 Llama-3.1-8B-Instruct，但 EFC Planner 使用 HF backend、最大输入 1024、实际 inference batch 8；外层按 32 个 prompt 分块，`do_sample=False`、`temperature=0`、最大 96 tokens（`build_efc_planner_config`，约第 1927–1935 行；`generate_rs_mhr_queries`，约第 1456–1483 行）。Planner stop list 为 `<|eot_id|>`、两个 `Question:` 变体、`Expected output:`、`Solution:`、`Answer:`（同文件 `PLANNER_STOP_WORDS`，约第 121–128 行）。

## 6.2 Parser、有效性与近似原问题过滤

`parse_planner_output` 的容错顺序（`racp/run_racp.py`，约第 942–995 行）是：

1. 扫描输出中的每个合法 JSON array；
2. 再尝试把完整输出解析为 JSON list 或含 `subqueries/queries/query` 的 dict；
3. 再按行移除项目符号、编号和引号。

`normalize_subquery_list`（约第 912–939 行）执行空白/引号清理、大小写去重，拒绝与原问题规范化后完全相同的 query，并调用 `is_search_like_query`。EFC missing/static parser 默认不允许 entity-only query，所以至少需要 3 个字母数字 token；`yes/no/true/false` 和若干单词答案也被拒绝（`is_search_like_query`，约第 811–868 行）。

missing-hop 的有效条件是规范化后至少 1 条，随后截到前 1 条，而不是要求原始数组只能有一个元素。之后还会用 `SequenceMatcher` 拒绝与原问题相似度 \(\ge0.9\) 的 query（`rs_mhr_query_too_similar`，约第 1450–1453 行；`generate_rs_mhr_queries`，约第 1493–1506 行）。代码没有 embedding 语义相似度检查。

## 6.3 Planner 失败与 heuristic repair

invalid、空输出或 query-too-similar 会设置 `planner_failed=True`，然后调用
`racp/efc.py::build_heuristic_missing_query`（约第 467–495 行）。repair：

1. 只从 Router 已抽取的 `y1_new_entities` 中选择实体；
2. 用“是否匹配 \(R_0\) 标题、Probe 中是否出现 relation pattern、实体词数、字符数”的 tuple 最大化选择一个实体；
3. 从问题的固定正则映射推断 focus，如 `government position`、`birth name`、`year`、`director` 等；未命中时回退 answer-type terms；
4. 连接为 `"<entity> <focus>"`。

repair 结果不会再次通过 `is_search_like_query` 或 0.9 similarity 检查。若 repair 非空，保存 `missing_query_strategy="heuristic_repair"` 并检索 top10；若仍为空，route 改为 static-QD（若启用）或 direct。改为 static 后会进行第二次 Planner pass；该 late static 也失败则 direct（`retrieve_with_efc`，约第 2307–2443 行）。`router_route` 保留原始决策，`route` 保存最终执行路线。

## 6.4 Provenance

每个有效 generation query 形成：

```text
source = "generation_guided"
query_id = "gen_0"        # FINAL 正常为一条
topk = 10
```

每篇返回文档保存 query 字符串、`gen_0` 下的 1-based rank 和 raw dense score，随后按 UID 与原始文档合并（`retrieve_with_efc`，约第 2477–2529 行；`racp/efc.py::normalize_doc`，约第 129–146 行）。

# 7. Static-QD Fallback

## 7.1 进入条件和两种 Prompt

static-QD 可由 Router 的 bad/uncertain 或 centroid 规则触发，也可由 generation-guided query 失败且无法 repair 后触发；force-static 也会在 Probe 之后进入该路线。之后按 question type 再分两种 Planner prompt（`retrieve_with_efc`，约第 2188–2275、2332–2443 行）。

对于 bridge/constraint，实际 evidence-conditioned prompt（`racp/run_racp.py::build_efc_static_bridge_prompt`，约第 1428–1447 行）读取原始 top5；每篇只在 Planner prompt 中保留标题和 `doc_to_text(doc)[:360]`：

```text
Original question:
{question}

Initially retrieved evidence:
[Doc 1]
Title: ...
Evidence: ...                 # 每篇最多前 360 个字符
...

Generate exactly 2 retrieval-oriented bridge queries.

Rules:
1. Do not decompose the question only from its surface wording.
2. Use the initially retrieved titles and evidence to identify likely bridge entities.
3. Each query should combine a concrete bridge entity with the missing target attribute.
4. Prefer concise keyword queries over full questions.
5. Do not answer the original question.
6. Output only a JSON array of strings.
```

对于 comparison/generic，实际 question-only prompt（`racp/run_racp.py::build_rs_mhr_static_prompt`，约第 1391–1403 行）不包含 top5 或 Probe，其逐字模板为：

```text
Given the original question, generate exactly 2 retrieval-oriented search queries.

Rules:
1. Each query should contain a concrete entity from the original question whenever possible.
2. Each query should ask for one missing factual piece needed to answer the original question.
3. Prefer search-engine style keyword queries over conversational questions.
4. Do not answer the question.
5. Do not explain.
6. Output only a JSON array of strings.

Question:
{question}

Output:
```

parser 与 generation-guided 共用 `parse_planner_output`。有效条件是规范化后至少两条，保留前两条；少于两条时 parser 返回空列表，初始 static 直接回退 direct。每条 query 检索 top5：

\[
Q_d=\{q_1,q_2\},\qquad
R_d=\operatorname{Ret}(q_1,5)\cup\operatorname{Ret}(q_2,5).
\]

## 7.2 实际配额与补位

只有 `route=static_qd` 且 question type 为 bridge/constraint 时调用 `racp/efc.py::static_bridge_pack`（约第 710–784 行）。其真实过程是：

1. 原始 provenance 集合按 original rank，**最多新增 4 篇**，这一阶段每个 lowercased title 至多一篇；
2. QD provenance 集合按最小 `qd_*` rank、再按 RRF，**最多新增 2 篇**尚未选择且标题未出现的文档；
3. 全候选池按 `(-RRF, best rank)`，继续加入尚未出现标题的文档直到 \(K\)；
4. 若仍不足 \(\min(K,|\mathcal C|)\)，按同一 RRF 顺序允许重复标题回填。

因此不能写“严格固定 4 original + 2 QD + 4 other”。4 和 2 是前两阶段的新增上限；QD 文档可能因 UID/标题已选而少于 2，后续全池补位也可能再选更多 original 或更多 QD。同一 UID 若同时被 original 和 QD 命中，会同时带两种 provenance。

static-QD 的 comparison/generic 样本不走上述配额，而走 `role_aware_pack`；分派证据为 `racp/run_racp.py::retrieve_with_efc` 约第 2565–2586 行。当前持久化字段把 canonical 所有非-direct 路线统一标作 `selection_method="efc_role_aware"`，即使 bridge static 实际使用 quota pack；该字段命名不能作为算法证据（同函数约第 2682–2690 行）。

## 7.3 与 Full-QD 基线的差异

| 维度 | EFC static-QD | Full-QD |
|---|---|---|
| 触发 | Router 条件触发或 G 失败 fallback | 每个样本无条件执行 |
| Probe | 先执行 Probe | 无 Probe |
| Planner 输入 | bridge/constraint 看 top5；其他只看 \(q\) | 只看 \(q\) |
| query 有效性 | 至少 2 条三 token search-like query，否则 direct | parser 允许 entity-only；0/1/2 条都可继续，1 条为 partial |
| 检索 | original20 + 2×top5 | original20 + 每条 accepted subquery×top5 |
| 融合 | UID provenance merge + source-weighted RRF | UID 去重后保留最大 raw dense score |
| final selection | static bridge quota/backfill；其他 role-aware | FINAL 为按 score 的 title-dedup top10 |

Full-QD 证据路径为 `racp/run_racp.py::generate_subqueries` 约第 1095–1135 行、`retrieve_with_query_decomposition` 约第 3290–3412 行，以及 `select_title_dedup_topk_docs` 约第 622–651 行。其 `subquery_num=2` 在 `generate_subqueries` 和 prompt 文本中实际硬编码为 2，不能据 CLI 形参声称任意可配 query 数。

# 8. Provenance-Aware Candidate Pool

## 8.1 文档身份与合并

候选规范化由 `racp/efc.py::doc_contents/doc_title/doc_uid/normalize_doc` 约第 101–146 行实现。唯一键的优先级是：

```text
existing doc_uid
→ existing doc_id
→ existing id
→ MD5(title + "\n" + contents)
```

正式 wiki18 语料记录通常带 `id`，因此实际 `doc_uid` 通常是语料 ID；只有缺少三种 ID 时才计算 MD5。去重按 `doc_uid`，不按 title：同标题的不同 passage 保持为不同候选，随后由 title-aware selector 处理。反之，同一个 UID 被多个 query 命中时合为一条记录。

`merge_docs` 按 `[R0] + extension groups` 的顺序遍历（`racp/efc.py::merge_docs`，约第 149–173 行；`racp/run_racp.py::retrieve_with_efc`，约第 2516–2531 行）。Python dict 的插入顺序使候选池先保留原始检索顺序，再保留各扩展 query 的首次新 UID 顺序。同 UID 的正文、标题和基础字段保留第一次出现的版本；随后：

- `sources` 和 `source_queries` 以首次出现顺序做 list 去重；
- 同一 `query_id` 的 rank 取最小值；
- `retriever_scores` 用后出现值更新；
- 不把不同 query 的 dense score直接求和、取均值或跨 query 排序。

## 8.2 候选记录字段

| 字段 | 类型 | 实际含义 | 论文是否需要公开 |
|---|---|---|---|
| `doc_uid` | string | UID 去重主键 | 是，说明去重粒度即可 |
| `doc_id` | string/null | 原 `doc_id`，缺失时取 `id` | 可选 |
| `title` | string | 显式 title；否则 contents 第一行；再否则前 80 字符 | 是 |
| `contents` | string | `contents/text/content` 的首个非空值 | 是 |
| `text` | string | 原 `text` 或 contents | 否 |
| `sources` | list[string] | `original`、`qd`、`generation_guided` 的去重集合 | 是 |
| `source_queries` | list[string] | 命中过该文档的 query 原文 | 附录建议公开 |
| `ranks` | dict[string,int] | `original/qd_i/gen_i → 1-based rank` | 是 |
| `retriever_scores` | dict[string,float] | 每个 query 的 raw dense score | 说明保留但不参与 RRF |
| `rrf_score` | float | source-weighted RRF | 是 |
| `role_scores` | dict[string,float] | 六个 heuristic role feature | 是 |

代码来源为 `racp/efc.py::normalize_doc` 约第 129–146 行。`source_queries` 没有直接保存 `query_id → query` 映射；若要逐项重建，需结合外层 query records。`candidate_pool_summary` 对每个 source 做 membership 计数，多源文档会同时计入多列，因此 source count 之和可能大于 `merged_unique`（`racp/run_racp.py::efc_candidate_summary`，约第 2091–2100 行）。

需要区分运行时结构和 FINAL 持久化结构：runtime pool 含完整 ranks/scores/provenance；compact prompt cache 只为最终 selected docs 保留少数字段，且四个 FINAL 默认没有 `candidate_pool_debug`。所以候选结构由代码确定，但旧 FINAL 无法逐文档完整恢复全部候选。

# 9. RRF 融合

## 9.1 精确公式

令 \(J(d)\) 为文档 \(d\) 出现过的 query ID 集合，\(r_j(d)\ge1\) 是 1-based rank。代码计算

\[
s_{\mathrm{RRF}}(d)
=
\sum_{j\in J(d)}
\frac{\alpha_{\tau(j)}}{c+r_j(d)},
\qquad c=60,
\]

其中

\[
\alpha_{\mathrm{original}}=1.00,\qquad
\alpha_{\mathrm{static}}=1.05,\qquad
\alpha_{\mathrm{generation}}=1.15.
\]

query ID 的来源映射为：

\[
\tau(j)=
\begin{cases}
\mathrm{original}, & j=\texttt{original},\\
\mathrm{static}, & j\text{ 以 }\texttt{qd\_}\text{ 开头},\\
\mathrm{generation}, & \text{其他 query ID}.
\end{cases}
\]

常量位于 `racp/efc.py::SOURCE_RRF_WEIGHTS` 约第 6–10 行，融合函数为 `add_rrf_scores` 约第 498–510 行。`git blame` 显示这组三个来源权重从首个 EFC commit `ce132a8` 起未改变。rank 的 1-based 约定由 `racp/run_racp.py::efc_normalize_docs` 约第 2029–2034 行的 `enumerate(..., start=1)` 证实。

同 UID 出现在多个 query 中时，每个 query rank 都贡献一项；不是只保留最佳 query。raw dense score保存在 `retriever_scores`，但不进入 RRF，也不直接进入 canonical greedy selector。dense score只在 Router 中用于记录原始 top-score 特征，而其中大多数当前又未进入决策规则。

## 9.2 数值示例

某文档同时位于 original rank 2 和 generation-guided rank 1，则

\[
\begin{aligned}
s_{\mathrm{RRF}}(d)
&=\frac{1.00}{60+2}+\frac{1.15}{60+1}\\
&=0.0161290+0.0188525\\
&=0.0349815.
\end{aligned}
\]

只出现在 static-QD rank 1 的文档为

\[
\frac{1.05}{61}=0.0172131.
\]

配置项 `source_weight=0.05` 不是这里的 \(\alpha_\tau\)。\(\alpha_\tau\) 在 RRF 中对每个来源 rank 加权；`source_weight` 是后续 greedy packing 对“候选是否带来尚未覆盖的粗粒度 source type”的二元奖励系数（`racp/efc.py::role_aware_pack`，约第 672–681 行）。

# 10. Evidence Role 识别

代码中实际且仅有六个 role 名称：anchor、bridge、answer、comparison-left、comparison-right、constraint（`racp/efc.py::ROLE_NAMES`，约第 11–18 行）。它们可以在同一文档上同时非零，不是互斥标签；全部由字符串/正则/排名启发式计算，不调用 LLM 或 embedding。

令标题为 \(t_d\)，`title + contents` 为 \(x_d\)，\(\mathbf 1[\cdot]\) 为指示函数。

| 角色/特征 | 代码判定与公式 | 分值范围 | 使用阶段 |
|---|---|---:|---|
| anchor | \(\min(1,0.7I_t+0.3I[r_{\rm original}\le5])\)。\(I_t=1\) 当 title 是 question 的规范化 substring，或 question/title token overlap 占 title token 至少 0.5 | \(\{0,.3,.7,1\}\) | role-aware packing、诊断 |
| bridge | \(\min(1,0.7I[t_d\subset y_p\land t_d\not\subset q]+0.3I[\mathrm{generation}\in sources])\) | \(\{0,.3,.7,1\}\) | role-aware packing、诊断 |
| answer | `answer_type_hit(title+contents, answer_type)` | \(\{0,1\}\) | role-aware packing、context 排序第三键 |
| comparison-left | 抽取的左实体是文档 lowercase substring | \(\{0,1\}\) | comparison role gain |
| comparison-right | 抽取的右实体是文档 lowercase substring | \(\{0,1\}\) | comparison role gain |
| constraint | 问题中出现的固定 relation terms 有多少也出现在文档，除以该 terms 数；无 terms 时 0 | \([0,1]\) | role-aware packing、诊断 |

实现为 `racp/efc.py::_token_set/_title_in_text/extract_comparison_sides/compute_role_scores`，约第 513–581 行。

`answer_type_hit` 是宽松词面规则（`racp/efc.py`，约第 243–262 行）：

- number：数字，或 capacity/seats/population/height/score；
- date：1900–2099 年份，或 born/date/founded/released/published；
- location：born in/located in/city/country/state/place；
- person：至少两个首字母大写单词；
- yes/no：出现 is/are/was/were；
- entity：只要文本非空就为 1。

尤其 entity 类型下几乎每篇文档的 answer role 都是 1，所以不能将其描述为“答案包含概率”或精确 answer-bearing classifier。

不同 question type 的内部 role-goal 权重为（`racp/efc.py::role_goal_weights`，约第 584–599 行）：

| question type | 参与边际增益的 role 权重 |
|---|---|
| comparison | left .35、right .35、answer .20、constraint .10 |
| bridge / constraint | anchor .25、bridge .35、answer .30、constraint .10 |
| generic | anchor .40、answer .40、constraint .20 |

“new-source coverage”和“new-title coverage”是 selector 的集合特征，不是 `ROLE_NAMES` 中的证据角色。代码也没有额外的 semantic-role embedding、同义词词典或 role LLM。

# 11. Role-Aware Packing 评分公式

## 11.1 集合边际目标

对当前选择集合 \(S\)，每个 role 的覆盖为

\[
c_r(S)=\max_{s\in S}\rho_r(s),
\]

候选 \(d\) 的 role 边际增益为

\[
g_{\mathrm{role}}(d\mid S)
=
\sum_{r\in G(q)}
\omega_r\max\{0,\rho_r(d)-c_r(S)\}.
\]

标题与来源增益分别是

\[
g_{\mathrm{title}}(d\mid S)=
\mathbf 1[n_{t_d}(S)=0],
\]

\[
g_{\mathrm{src}}(d\mid S)=
\mathbf 1[
sources(d)\setminus\!\!\bigcup_{s\in S}sources(s)\ne\varnothing].
\]

来源增益只区分 original、QD、generation-guided 三种粗标签；新的 `qd_1` 相对 `qd_0` 不构成新来源。

冗余项为

\[
g_{\mathrm{red}}(d,S)=
\max_{s\in S}
\begin{cases}
1, & lower(t_d)=lower(t_s),\\
\operatorname{Jaccard}(T(contents_d),T(contents_s)), & \text{otherwise},
\end{cases}
\]

其中 \(T\) 是由小写字母数字正则构成的 token set；\(S=\varnothing\) 时为 0（`racp/efc.py::lexical_redundancy`，约第 602–615 行）。

soft title penalty 为

\[
p_{\mathrm{title}}(d,S)=
\mathbf 1[
\texttt{title\_dedup\_soft}
\land n_{t_d}(S)\ge\texttt{max\_same\_title}],
\]

FINAL 的 `max_same_title=2`，即第三篇及之后的同标题 passage 才减去整整 1 分；第二篇没有该整分 penalty。它是软惩罚而非硬过滤。

四个 FINAL 的总增益精确为

\[
\begin{aligned}
\Delta(d\mid S)
=&\;1.00\,s_{\mathrm{RRF}}(d)
+0.30\,g_{\mathrm{role}}(d\mid S)\\
&+0.05\,g_{\mathrm{src}}(d\mid S)
+0.02\,g_{\mathrm{title}}(d\mid S)\\
&-0.01\,g_{\mathrm{red}}(d,S)
-p_{\mathrm{title}}(d,S).
\end{aligned}
\]

代码为 `racp/efc.py::role_aware_pack` 约第 618–698 行。helper 内部的缺省 `title_weight=.05`、`redundancy_weight=.05` 不是 FINAL 值；四份保存配置显式覆盖为 .02 和 .01。

## 11.2 Seed、贪心与 tie-breaking

进入 greedy loop 之前，先固定插入

\[
\min(\texttt{original\_seed\_count},K)=4
\]

篇带 original provenance 的文档。它们按 original rank 做 title-diverse first pass；不同 title 不足 4 时再回填延迟的重复标题。Seed 是先插入，不是只加 bonus（`role_aware_pack` 约第 635–652 行；`select_ranked_title_diverse` 约第 796–817 行）。

之后每一轮重新计算依赖 \(S\) 的 role/source/title/redundancy 项，并选择字典序最大的

```text
(total_gain, rrf_score, -minimum_rank_over_all_queries)
```

因此 canonical selector 是 seed 之后的 set-conditioned greedy，而不是一次性逐文档排序；它也不是全局最优集合求解器。三项仍完全相同时，保留候选池首次迭代到的文档，没有额外 UID tie-break。

该 greedy 适用于 generation-guided，以及 static-QD 中非 bridge/constraint 的样本。Direct 和 bridge/constraint static 都绕过它。候选池少于 \(K\) 时，返回全部候选，不制造空文档。

## 11.3 五候选、选三篇的手算示例

为忠实保留 FINAL 的四篇 original seed，下面把 \(S_0\) 视为已经固定插入的 seed 集合，并在剩余 5 个候选中展示接下来 3 个槽位的 greedy 选择。设问题类型为 bridge，当前覆盖

```text
anchor=1, bridge=0, answer=0, constraint=0
covered_sources={original}
```

五个候选如下；`O10/G3` 表示同一 UID 同时位于 original rank 10 和 generation rank 3。

| 候选 | 来源/rank | title | 相关 role | RRF | 与 \(S_0\) 的 redundancy |
|---|---|---|---|---:|---:|
| A | G1 | Alpha | bridge=1, answer=1 | .01885 | .10 |
| B | O5 | Beta | answer=1 | .01538 | .20 |
| C | G2 | Gamma | bridge=1 | .01855 | .05 |
| D | O10+G3 | Delta | bridge=.3, answer=1 | .03254 | .60 |
| E | O6 | Alpha | constraint=1 | .01515 | .15 |

第一轮：

\[
g_{\rm role}(A)=.35(1)+.30(1)=.65,
\]

\[
\Delta(A)=.01885+.30(.65)+.02+.05-.01(.10)=.28285.
\]

同理，\(\Delta(B)\approx.12338\)、\(\Delta(C)\approx.19305\)、\(\Delta(D)\approx.21804\)、\(\Delta(E)\approx.06365\)，因此先选 A。

第二轮，bridge、answer 和 generation source 已覆盖。D 仍凭多 query RRF 与新标题得到

\[
\Delta(D)=.03254+.02-.01(.60)=.04654,
\]

C 为约 .03805，B 为 .03338。E 与已选 A 同 title，title bonus 为 0、same-title redundancy 为 1，但 title count 仅 1，尚不触发整分 penalty；其 constraint 边际仍在，得分约 .03515。因此第二篇选 D。

第三轮，C 的约 .03805 高于 B 与 E，故新增三篇依次为

\[
[A,D,C].
\]

这个例子展示了三点：多 query 命中可提高 RRF；role/source bonus 只在首次覆盖时有效；`max_same_title=2` 不会禁止第二篇同标题文档。

# 12. Direct 路线的证据选择

Direct 的 canonical 逻辑位于 `racp/run_racp.py::retrieve_with_efc` 约第 2555–2564 行和 `racp/efc.py::select_ranked_title_diverse` 约第 796–817 行：

1. 不做增量检索，候选只有 \(R_0\)；
2. 流程仍计算 RRF 和 role scores，但两者都不参与 direct selection；
3. 按 original rank 扫描，第一遍每个 lowercased title 选一篇；
4. 若不同标题不足 \(K=10\)，把延迟的重复标题 passage 按其原始 encounter order 回填；
5. 候选不足 10 时返回全部候选。

Direct 分支没有再调用 `order_context`。因此它不是无条件“原始 top10”：例如 ranks

```text
1:A, 2:A, 3:B, 4:C    and K=4
```

会输出

```text
1:A, 3:B, 4:C, 2:A
```

即 title-diverse first pass 后才回填重复标题。它允许同一 title 多篇 passage，但仅在独立标题不足时。该路线绕过 role-aware selector；role scores 仅用于保存诊断字段。Direct 仍执行 Probe 和 final answer 两次生成，不能用 Standard RAG 的一次生成成本替代。

# 13. 最终上下文顺序和 Prompt

## 13.1 Route-specific context order

Generation-guided 和经过 `order_context` 的 static 路线按以下 key 升序排序（`racp/efc.py::order_context`，约第 867–884 行）：

```python
(
    0 if "original" in sources else 1,
    original_rank if available else minimum_expansion_rank,
    role_scores["answer"],
)
```

因此：

- 带任意 original provenance 的文档全部先于 expansion-only 文档；
- original+expansion 的合并文档按 original 处理；
- original 组按 original rank；
- generation-only 组按最小 `gen_*` rank；
- static-only 组按最小 `qd_*` rank；
- 第三个键是 answer role **升序**，不是降序；只有前两个键相同时才生效，此时 answer=0 会排在 answer=1 之前；
- RRF 与 greedy 选择顺序不会直接成为最终展示顺序。

Direct 不调用该函数，采用第 12 节的 title-diverse-first 输出顺序。

## 13.2 实际 final Prompt

四个 FINAL 的 prompt flags 均为 false，EFC 因 `efc_rag_config.enabled=True` 选择专用 final template（`racp/run_racp.py::build_answer_prompt_template`，约第 3462–3480 行）。源码实际拼接为：

```text
Answer the question based on the given document.Only give me the answer and do not
output any other words.
The following are given documents.

{reference}
```

注意 `document.Only` 中间没有空格。User prompt 是：

```text
Question: {question}
```

`flashrag/prompt/base_prompt.py::PromptTemplate.format_reference` 约第 217–228 行将文档写为：

```text
Doc 1(Title: <contents 的第一行>) <contents 的其余行>
Doc 2(Title: ...)
...
```

实际是 `Doc 1(Title:`，不是 `Doc 1 (Title:`。格式器再次从 `contents` 第一行取 title，而不读取规范化记录的显式 `doc["title"]`。source、rank、RRF 和 role score 都不进入最终 prompt。

## 13.3 输入截断、生成与 parser

四个 FINAL 的 `generator_max_input_len=4096`。对于本地非 OpenAI 模型，完整 chat-template 字符串 token 数若 \(\ge4096\)，`PromptTemplate.truncate_prompt` 会保留前

\[
4096/2-20=2028
\]

个 token 与最后 2028 个 token，直接拼接并删除中间部分（`flashrag/prompt/base_prompt.py`，约第 66–106 行）。这不是逐文档截断，也不尊重文档边界；中间文档可能完全丢失或 passage 被截断，title 也只有落在保留区时才保留。四个 FINAL `run.log` 未出现该函数的 truncation warning，因此现有日志没有证据表明正式运行实际触发过；是否每条 selected top10 都完整可见，严格结论为 `UNKNOWN`，可从已保存 prompt 离线重新 tokenize 验证。

最终生成 `do_sample=False`、temperature 归零、实验 seed 2024、最大 32 tokens。VLLM wrapper 在没有显式 stop 时加入 `<|eot_id|>`（`flashrag/generator/generator.py::VLLMGenerator.generate`，约第 226–248 行）；除此之外没有 EFC 专用换行或 `Answer:` stop。模型内部 EOS 的完整运行时枚举在 artifact 中为 `UNKNOWN`。

EFC 没有最终答案 parser。`run_generate` 直接将 `generator.generate` 的字符串写入 `pred`，不 `.strip()`，不提取 `So the answer is`，也不调用 `flashrag/utils/pred_parse.py`（`racp/run_racp.py`，约第 3833–3841 行）。评测时才执行 lowercase、去 ASCII 标点、去 `a/an/the` 和空白合并（`flashrag/evaluator/utils.py::normalize_answer`，约第 5–19 行）。论文应写“answer-only prompting + standard metric normalization”，不能写“EFC answer parser”。

# 14. 缓存与两阶段执行

## 14.1 Retrieval cache

EFC 调用 `racp/run_racp.py::rs_mhr_raw_batch_search` 约第 1149–1188 行，刻意绕过 FlashRAG 的自动 rerank wrapper。缓存结构是：

```text
key   = exact query string
value = ordered list of documents, each carrying float field "score"
```

命中要求该 query 存在且缓存文档数不少于本次 top-\(k\)；随后切片并恢复 score。miss 或缓存长度不足时，普通模式调用真实 `_batch_search`，并用新结果更新内存 cache。`EFC_CacheOnlyRetriever` 遇到任意 miss 直接抛异常，不会静默在线检索（同文件约第 1861–1885 行）。

| FINAL | retrieval cache 行为 |
|---|---|
| HotpotQA | `retrieval_cache_only=true`；历史完整 EFC cache；不加载 corpus/FAISS/BGE；任何 miss 失败 |
| 2Wiki | 公共原问题 cache 命中，动态 query 可在线 miss；`use/save=true` |
| MuSiQue | 同 2Wiki；prepare 输出保存合并 cache |
| NQ | 同 2Wiki；FINAL 目录保存合并 cache |

`rs_mhr_save_retrieval_cache(..., update_source_cache=False)` 不覆盖公共源 cache；若 `save_cache=True`，写当前输出目录的 `retrieval_cache.json`（同文件约第 1819–1831、2714–2716 行）。

在检索模型、index、corpus、query 完全一致时，cache 设计上复用相同有序文档与 dense score，只应节省速度。但 cache 文件没有模型/index/corpus hash 校验，key 也不含这些配置；旧 FINAL 没有冷/热 bitwise 成对验证。因此：

- 正确匹配的 cache：方法语义上只影响速度；
- stale 或错误来源的 cache：可能改变 \(R_0\)、Probe、Router 和后续结果；
- cache-only 且启用 centroid：`encoder=None` 会跳过 centroid feature，可能改变路线；四个 FINAL 均关闭 centroid，未受此风险影响。

## 14.2 Prompt cache 与 prepare/generate

`prepare` 执行数据加载、检索、Probe、Router、Planner、扩展、融合、packing 和 final prompt 构造，然后保存 prompt cache，不生成最终答案（`racp/run_racp.py::run_prepare`，约第 3774–3821 行）。

`generate` 直接加载 prompt cache，跳过上述全部模块，只初始化 final generator、生成 raw `pred` 并评测（`load_prompt_cache/run_generate`，约第 3765–3848 行）。

EFC 的 `full` + vLLM 实际也采用两阶段：父进程完成 prepare 路径并保存 compact prompt cache，然后启动干净的 `--stage generate` 子进程，以释放 Probe/Planner/retriever 显存和进程状态（`run_full_vllm_generate/run_full`，约第 3866–3919 行）。这是执行工程，不是算法模块。

compact EFC prompt cache 保留 route、Probe、query、selected docs、cost 和完整 final prompt，但丢弃 `planner_raw_outputs` 与完整 candidate pool（`EFC_PROMPT_CACHE_OUTPUT_KEYS` 约第 203–231 行；`compact_efc_prompt_cache_item` 约第 3725–3751 行）。Prompt cache 本身没有问题顺序/config hash 验证；正确 cache 冻结并复用已准备的算法轨迹，只改变执行阶段，错误 cache 则会直接改变输入与结果。论文可写：

> We memoize exact-query retrieval results and optionally separate evidence preparation from final generation for execution efficiency. These caches are implementation mechanisms and are not part of the EFC decision rule or scoring function.

不要把 cache 作为 EFC 的方法创新。

# 15. 实际复杂度与调用成本

## 15.1 符号化复杂度

定义：

- \(N_0=20\)：初始 top-\(k\)；
- \(N_e\)：单个扩展 query 的 top-\(k\)，generation 为 10，static 为 5；
- \(m\)：扩展 query 数，generation 为 1，static 为 2；
- \(M=|\mathcal C|\)：UID 去重后的候选数，正常上界约 \(N_0+mN_e\le30\)；
- \(K=10\)：最终 context 文档数；
- \(L\)：平均 passage 的 token-set 规模；
- \(T_G(P,O)\)：一次 LLM 在输入 \(P\)、输出上限 \(O\) 下的成本；
- \(T_R(k)\)：一次 query 的 dense retrieval 成本；ANN/index 细节保留在该抽象中。

初始检索成本为 \(T_R(N_0)\)；Probe 为 \(T_G(P_{\rm probe},128)\)；条件 Planner 为 \(T_G(P_{\rm plan},96)\)；扩展检索的逻辑 query 成本为 \(mT_R(N_e)\)；UID merge 与 RRF 为 \(O(N_0+mN_e)\) 加 provenance 项数；role feature 为约 \(O(ML)\)；final prompt serialization 为 \(O(KL)\)；final generation 为 \(T_G(P_{\rm final},32)\)。

Direct title-diverse selection含排序，约 \(O(N_0\log N_0)\)。Static bridge packing含若干排序，约 \(O(M\log M)\)。Role-aware selector最多 \(K\) 轮扫描 \(M\) 个候选；不计文本冗余时为 \(O(KM)\)。当前 `lexical_redundancy` 每次又遍历已选集合并重复构造 token sets，朴素上界约

\[
O(MK^2L),
\]

而不是严格线性（`racp/efc.py::role_aware_pack/lexical_redundancy`，约第 602–698 行）。在 FINAL 的 \(K=10,M\le30\) 下这仍是小规模 CPU 后处理。

正式 config `faiss_gpu=false`，所以 index search 不是 GPU FAISS；BGE query encoder 仍使用 CUDA（四个 FINAL `config.yaml` 约第 50–72 行；`flashrag/retriever/retriever.py::DenseRetriever`，约第 352–471 行）。

## 15.2 三条路线的名义调用预算

| 路径 | LLM 逻辑调用 | Retriever 逻辑 query | 顺序检索阶段 |
|---|---:|---:|---:|
| Direct | Probe + final = 2 | original = 1 | 1 |
| Generation-guided 正常 | Probe + missing Planner + final = 3 | original + 1 gen = 2 | 2 |
| Static-QD 正常 | Probe + static Planner + final = 3 | original + 2 QD = 3 | 2；两条 QD 同 batch 阶段 |
| G Planner 失败、repair 成功 | 3 | original + repair = 2 | 2 |
| G Planner 与 repair 均失败、late static 成功 | Probe + 两个 Planner + final = 4 | original + 2 QD = 3 | 2 |
| 两个 Planner 均失败后 direct | 4 | original = 1 | 1 |

最终 route 标签不总能唯一决定已支付成本：static Planner 失败后保存为 direct，但 planner 调用已经发生。四个 FINAL 中理论上的双 Planner fallback 没有实际触发；逐样本 `planner_llm_calls` 只有 0 或 1。

## 15.3 四个数据集的实际平均调用

下表来自各 FINAL `intermediate_data.json` 的逐样本 `efc_cost`，不是只读配置值。Retriever 数是逻辑 query 数；cache hit 可能跳过实际 BGE/FAISS 工作。聚合方法与证据也记录在 [`racp/PAPER_MISSING_INFORMATION_AUDIT.md`](PAPER_MISSING_INFORMATION_AUDIT.md) 约第 123–160 行。

| 数据集 | 样本数 | Avg LLM | Avg planner | Avg retriever query | Avg UID pool | Final docs | final route D / G / S |
|---|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA dev | 7,405 | 2.8718 | 0.8718 | 1.9332 | 25.5246 | 10 | 955 / 5,990 / 460 |
| 2Wiki dev | 12,576 | 2.9051 | 0.9051 | 2.0161 | 26.9002 | 10 | 1,207 / 9,959 / 1,410 |
| MuSiQue dev | 2,417 | 2.9876 | 0.9876 | 2.0426 | 27.6719 | 10 | 35 / 2,244 / 138 |
| NQ test | 3,610 | 2.1008 | 0.1008 | 1.1042 | 20.5006 | 10 | 3,246 / 352 / 12 |

成本字段在 `racp/run_racp.py::retrieve_with_efc` 约第 2692–2707 行按控制流记录 Probe=1、final=1、Planner 次数和 query-record 数。它适合称为“per-question logical calls”，不等于物理 Python/API 调用数，不含 batch amortization、模型加载、token 数或 cache miss telemetry。

四个 FINAL 的 Probe、Planner 和 final 都使用同一个 Llama-3.1-8B-Instruct checkpoint，但 Planner 是另载的 HF 实例；另有独立 BGE-large-en-v1.5 retriever checkpoint。准确表述是“一个生成 checkpoint 承担三种职责，加一个 retriever checkpoint”，而不是“运行时只有一个模型实例”。

# 16. 论文版模块划分建议

## 16.1 推荐 Methodology 结构

下面的结构比把所有路线都概括为“role-aware selection”更忠实：

```text
3 Method
3.1 Problem Formulation
3.2 Overview: Probe-Conditioned Rule Routing
3.3 Conditional Evidence Expansion
    3.3.1 Generation-Guided Missing-Hop Retrieval
    3.3.2 Evidence-Conditioned Static-QD Fallback
3.4 Provenance-Aware Candidate Fusion
3.5 Route-Specific Evidence Packing
    3.5.1 Greedy Role-Coverage Packing
    3.5.2 Direct and Static-Bridge Packing
3.6 Answer Generation

4 Implementation Details
Appendix A Exact Router and Fallbacks
Appendix B Prompts and Parsers
Appendix C Packing Details and Complexity
```

原因是 canonical 主方法并不存在一个覆盖三条 route 的统一 selector：generation-guided 走 greedy role packing，bridge/constraint static 走 quota/RRF backfill，direct 走 ranked title-diverse first pass（`racp/run_racp.py::retrieve_with_efc`，约第 2531–2586 行）。把第 3.5 节命名为 “Role-Aware Evidence Packing” 而不加 route-specific 限定，会错误暗示 direct 和全部 static 都使用同一目标。

## 16.2 可作为主要贡献的部分

在不作外部文献新颖性判定的前提下，真实代码支持将以下组合写成方法贡献：

1. **Probe-conditioned rule routing**：先在 top5 上生成受约束 Probe，再使用新实体、失败状态、问题类型与 comparison coverage 的有序规则选择 direct、generation-guided 或 static-QD。应明确它是规则策略，不是学习分类器（`racp/efc.py::compute_router_features/decide_route`，约第 363–440 行）。
2. **Conditional evidence expansion**：Probe 暴露实体时用完整 Probe 生成 missing-hop query；Probe 无有效实体且失败时使用 static-QD，避免对每条问题固定执行同一种扩展（`racp/run_racp.py::retrieve_with_efc`，约第 2188–2493 行）。
3. **Provenance-aware fusion under a fixed context budget**：同 UID 多 query 命中合并，按来源加权 RRF；在 generation-guided 等适用路径上，用集合边际 role/source/title coverage 与 lexical redundancy 做 greedy packing（`racp/efc.py::merge_docs/add_rrf_scores/role_aware_pack`，约第 149–173、498–510、618–698 行）。
4. **Route-specific packing**：direct 保留初始 rank 语义，static bridge 先分阶段保留原始/QD证据，再统一补位；这应作为预算控制设计，而非宣称所有路线共享一个优化目标（`racp/efc.py`，约第 710–817 行）。

其中加权 RRF、本身的 query decomposition、title diversity 和 greedy coverage 都是已有通用技术范式。论文贡献宜落在它们如何由 Probe feedback 和 route-specific execution 组合，而不是分别声称这些基础算子为新算法。

## 16.3 只应视为工程组件的部分

- exact-query retrieval cache；
- prepare/generate 两阶段与 clean vLLM 子进程；
- Probe/Planner/retriever 的加载和显存释放顺序；
- batching、tmux、GPU 分配与日志；
- no-reranker 是正式实验设置和计算选择，不是独立算法创新；
- evaluator、metric normalization 和输出目录管理。

相应代码位于 `racp/run_racp.py::rs_mhr_raw_batch_search/run_prepare/run_generate/run_full_vllm_generate` 约第 1149–1188、3774–3897 行。

## 16.4 与相邻方法的表述重合风险

| 相邻方法 | 高层重合点 | EFC 代码支持的边界 | 写作建议 |
|---|---|---|---|
| IterRetGen | 用中间生成反馈下一次检索 | EFC 注释明确称 generation-guided 主干遵循该 principle；但 EFC 先 Probe 路由、最多一阶段扩展并做多来源 packing | 不声称首创 generation-guided retrieval；突出条件触发、provenance fusion 和固定预算 packing |
| Full-QD | 静态分解为多个检索 query | EFC 只在路由/fallback 后执行，bridge 类型还读初始 evidence；parser/packing 也不同 | 称为 evidence-conditioned fallback，不称为新的 query decomposition 原理 |
| IRCoT | reasoning 与 retrieval 交替 | EFC 没有多轮 thought–retrieve 循环，也没有 IRCoT demonstration；Probe 不进入 final answer prompt | 不使用 “iterative chain-of-thought retrieval”；写 one-stage conditional expansion |
| Adaptive-RAG | 按问题状态选择检索策略 | EFC 是未训练规则且始终先做 original retrieval/Probe，没有 no-RAG 分支 | 不称 learned adaptive classifier；明确 direct 仍是 RAG 且有 Probe 成本 |
| S2G-RAG | 可能与“生成信号驱动检索/证据组织”在高层相似 | 本仓库没有可审计的 S2G-RAG 实现、prompt 或正式方法对照，精确差异为 `UNKNOWN` | 投稿前按原论文逐模块做 novelty matrix；当前只能作概念重合提醒 |

IterRetGen 关联的直接代码证据是 `racp/efc.py::decide_route` 约第 425–427 行的实现注释；IRCoT/Adaptive 的对照实现不在 EFC 调用链中。S2G-RAG 的具体差异不能从当前仓库验证，不能在本文中猜测。

正文不宜堆入：全部 uncertainty marker、实体正则、parser 容错分支、late-static 二次 Planner、static backfill 循环、cache key 和模型加载顺序。这些应放 Appendix/Implementation Details；正文保留操作性定义、三路线、RRF 与 greedy 边际目标。

# 17. 代码与论文叙述不一致检查

| 可能的论文表述 | 实际代码 | 是否一致 | 建议修正 |
|---|---|:---:|---|
| “EFC 判断证据是否充分” | 没有 sufficiency label/probability；按 question/probe 词面规则路由 | 否 | 写“根据 Probe 状态和实体暴露进行规则路由” |
| “Router 是训练分类器” | `decide_route` 是 first-match if/else；无训练参数 | 否 | 写 training-free rule policy |
| “dense score confidence 决定 route” | s1/s5/avg/gap 被计算但当前 `decide_route` 不使用 | 否 | 不把这些值写入正式 Router |
| “centroid routing 是主方法” | 四个 FINAL 均 `use_centroid_router=false` | 否 | 省略或列为 inactive option |
| “显式为每篇文档分配 anchor/bridge/answer 标签” | 计算可重叠的启发式连续/二元 scores，不是互斥 assignment | 部分 | 写 heuristic role features / coverage scores |
| “answer-bearing score 表示答案存在概率” | entity 类型下任何非空文档都得 1；无概率校准 | 否 | 只称 answer-type compatibility indicator |
| “执行全局集合优化” | 部分路线在四 seeds 后做 greedy marginal selection；没有全局最优求解 | 部分 | 写 set-conditioned greedy packing |
| “所有路线都使用 role-aware packing” | direct 和 bridge/constraint static 绕过 greedy selector | 否 | 写 route-specific packing |
| “Direct 就是取 original top10” | title-diverse first pass 后重复标题回填，可能改变 rank 顺序 | 否 | 写 ranked title-diverse first pass with backfill |
| “Direct 等于 Standard RAG 成本” | Direct 仍有 Probe + final 两次 LLM | 否 | 单独报告 direct 的 2-call 成本 |
| “Static 固定 4/2/4 配额” | 4/2 是前两阶段新增上限，余位全池补、最后可重复标题 | 否 | 写 up-to-4/up-to-2 + global backfill |
| “所有路线最终严格有 top10” | 算法返回 \(\min(10,|\mathcal C|)\)；四个 FINAL artifact 恰好全部为 10 | 条件一致 | 方法写“at most 10”；Implementation 写正式数据均为10 |
| “title dedup 是硬去重” | direct/static 先硬限制后允许回填；greedy 仅第三篇起减 1 | 否 | 写 title-diverse first pass / soft penalty |
| “证据按 RRF 或 answer role 降序送入 LLM” | expansion 先 original，再按 source rank；answer 第三键还是升序 | 否 | 披露 route-specific presentation order |
| “方法进行多轮迭代扩展” | 正常最多一阶段增量检索；static 可同阶段发两条 query；fallback 可多一次 Planner但仍只做一阶段扩展检索 | 否 | 写 single incremental retrieval stage |
| “EFC 不使用 reranker” | 配置构建强制 `use_reranker=false`，四个 FINAL 一致 | 是 | 作为实现设置披露，不包装为创新 |
| “Probe 可直接作为最终答案” | 所有样本另做 final generation | 否 | 写 Probe only guides routing/query/diagnostics |
| “EFC 使用答案 parser” | raw generator text直接写 `pred`；只在 metric 中 normalize | 否 | 写 answer-only prompt + raw prediction |
| “Retrieval Recall@10 是 supporting-evidence recall” | evaluator 只判断任一 final 文档是否包含任一 gold answer string | 否 | 称 Retrieval Answer-Hit@10，或明确 answer-containment 定义 |
| “cache 已证明不改变结果” | 正确配置下设计为 memoization，但无 hash 校验和成对 bitwise FINAL | 证据不足 | 条件性写法，并把 cache 放实现细节 |
| “四数据集算法行为完全同构” | 核心参数相同，但 `assume_multihop` 和 cache 执行不同 | 部分 | 披露 HP/2W/MU=true、NQ=false |
| “当前 HEAD 是 FINAL 的 bitwise 快照” | 当前有后加消融开关与 dirty worktree；canonical 默认路径未变 | 否 | 以 FINAL config + 历史 canonical blob 描述 |

Retrieval Recall 的直接证据为 `flashrag/evaluator/metrics.py::Retrieval_Recall.calculate_metric` 约第 219–248 行：它读取 `retrieval_result[:K]`，对 `normalize_answer(gold)` 是否为 `normalize_answer(contents)` 的 substring 做任一命中，不读取 supporting title 或 supporting sentence。

# 18. 可直接用于论文的中文方法草稿

> 以下正文约 3000–5000 字，可作为英文 Methodology 的中文底稿。参数密集内容集中放在最后的 Implementation Details；引用 “Algorithm 1” 对应本文第 3.1 节。

## 18.1 问题定义

给定问题 \(q\)、外部语料库 \(\mathcal D\)、冻结的检索器 \(\operatorname{Ret}\) 与生成器 \(G\)，检索增强问答需要从语料库中获得与问题有关的文档，并在有限上下文内生成答案 \(\hat a\)。多跳问题的困难不只在于找到与问题表面词项相似的文档：首轮证据可能只揭示中间实体，后续答案证据需要围绕该实体再次检索；另一方面，如果首轮证据已经覆盖比较对象或问题本身不需要多跳扩展，固定执行额外检索会增加调用开销并引入无关文档。即使多个 query 的检索结果都包含相关文档，直接按单一相关性分数截取前 \(K\) 篇也不能控制不同实体、证据来源和关系角色的覆盖。

本文采用 Evidence-Feedback Controlled RAG（EFC-RAG）。方法不训练新的路由器，不更新检索器、生成器或索引，而是在推理阶段执行三项操作：首先，利用初始证据生成一个受约束的 Probe，并从 Probe 中提取可观察的实体与失败状态；其次，根据有序规则选择是否以及如何进行一次增量证据扩展；最后，将不同 query 的检索结果按文档身份和来源融合，并在固定上下文预算内进行路由相关的证据打包。整体流程见 Algorithm 1。

形式上，初始检索结果为

\[
R_0=\operatorname{Ret}(q,N_0).
\]

我们使用其中前 \(N_p\) 篇构造 Probe：

\[
y_p=G(P_{\mathrm{probe}}(q,R_0[:N_p])).
\]

Probe 不是最终答案，而是对当前证据的短暂推理记录。其提示要求模型保留可能的桥接实体，在证据不足时显式输出 unknown，并以固定答案前缀结束。随后从 \(q\)、\(R_0\) 和 \(y_p\) 中提取规则特征 \(x=\phi(q,R_0,y_p)\)，路由策略

\[
z=\pi(x),\qquad
z\in\{\textsc{Direct},\textsc{GenerationGuided},\textsc{StaticQD}\}
\]

决定后续检索与选择方式。这里 \(\pi\) 是确定性的 first-match 规则，而不是预测证据充分性概率的分类器。

## 18.2 Probe 反馈与路由

Probe 使用初始 top-\(N_p\) 文档，由与最终回答相同的生成 checkpoint 确定性生成。路由器不要求把 Probe 解析为完整逻辑图，而只抽取几类可复现信号：问题被词面规则归为 comparison、bridge、constraint 或 generic；Probe 是否包含 unknown、cannot determine 等不确定性表达；输出是否过短、疑似截断或主要复述原问题；yes/no 结论是否与 Probe 自身的描述矛盾；以及 Probe 是否出现原问题中未出现的命名实体。新实体来自初始标题匹配和大写实体正则，因而是一个操作性桥接信号，不表示系统已经验证了真实知识图中的桥关系。

路由规则按固定优先级执行。强制 route 仅用于实验控制。对于 comparison 问题，如果两侧实体都出现在初始 top5 中，则保留初始证据；对于自相矛盾的 yes/no Probe，如果 top5 仍包含预期答案类型的词面证据，也采用 direct；非多跳问题同样进入 direct。其余问题中，若 Probe 暴露新实体，则进入 generation-guided；该规则先于 bad/uncertain 检查，因此一个包含 unknown 但同时给出桥接实体的 Probe仍会驱动 missing-hop retrieval。只有在没有可用新实体且 Probe 无效或不确定时，才进入 static-QD。其余情况回到 direct。多跳数据集和 NQ 的差异只体现在未分类问题的先验：HotpotQA、2Wiki 和 MuSiQue 把未命中词面模式的问题视为 bridge，NQ 则允许其保持 generic。

这一设计把“是否扩展”和“用什么信号扩展”分开。Direct 表示不追加检索，但仍保留 Probe 和最终回答两次生成；generation-guided 使用 Probe 暴露的实体构造缺失跳查询；static-QD 则在 Probe 未提供可用实体时，以原问题和可选初始证据生成替代查询。因而 direct 不能解释为 No-RAG，也不能解释为与单次 Standard RAG 完全相同的成本。

## 18.3 条件证据扩展

在 generation-guided 路线中，Planner 接收原问题和完整 Probe 文本，并输出一个 JSON search query：

\[
q_m=P_m(q,y_p),\qquad
R_m=\operatorname{Ret}(q_m,k_m).
\]

提示要求复用 tentative reasoning 中的桥接实体，并针对仍缺少的事实属性形成检索式，而不是直接回答原问题。Planner 输出先按 JSON array、JSON object 和逐行文本的顺序容错解析，再执行空白归一化、重复过滤、原问题同文过滤和 search-like 检查；与原问题字符串相似度过高的 query 也会被拒绝。若 Planner 没有产生有效 query，系统从 Probe 新实体中选择一个实体，并连接由问题模式推断的目标属性，形成 heuristic repair。只有 Planner 与 repair 都没有提供 query 时，样本才转入 static-QD 或 direct。

Static-QD 生成两个互补查询。对于 bridge 或 constraint 问题，Planner 同时读取初始 top5 的标题和截短证据，要求把可能的桥接实体与目标属性组合；对于其他问题，Planner 只根据原问题产生 retrieval-oriented queries。两条有效 query 分别检索固定数量的文档。Static-QD 与无条件 Full-QD 的关键区别在于：它由 Probe 路由触发，bridge 版本使用初始 evidence，而且其结果进入来源感知融合与路由相关打包；Full-QD 则对每个问题固定分解，并按另一套 dense-score/title 流程选择。

EFC 的正常执行最多包含一个增量检索阶段。Generation-guided 在该阶段发出一个 query；static-QD 在同一阶段批量发出两个 query。异常路径可能先后调用 missing-hop 和 static Planner，但在最终获得有效 query 前不会执行无效扩展，因此不构成 IRCoT 式的多轮 thought–retrieve 循环。

## 18.4 来源感知的候选融合

每个检索结果被规范化为带文档 UID、来源、query、名次和 dense score 的记录。UID 优先使用语料库已有 ID；缺少 ID 时再由标题和全文计算哈希。同一 UID 被原问题和扩展 query 多次检到时，系统保留一份正文，并合并其来源、query 与各自名次。相同标题但 UID 不同的 passage 不在这一阶段合并，因为同一主题下的不同 passage 可能包含不同事实；标题重复由后续 packing 处理。

令 \(J(d)\) 为文档 \(d\) 的来源 query 集合，\(r_j(d)\) 是从 1 开始的检索名次。候选的融合分数为

\[
s_{\mathrm{RRF}}(d)=
\sum_{j\in J(d)}
\frac{\alpha_{\tau(j)}}{c+r_j(d)},
\]

其中 \(\tau(j)\) 表示 original、static-QD 或 generation-guided 来源。来源权重允许 generation-guided 和 static query 的高位结果在固定候选预算中获得略高贡献，同时保留同一文档被多 query 命中的累加效应。原始 dense score仅作为 provenance 保存，不跨不同 query 直接比较，也不进入上述 RRF 公式。

## 18.5 路由相关证据打包

对于需要集合选择的路线，EFC 为每篇候选计算六类启发式角色特征：anchor 衡量标题与原问题的关联及其是否位于初始前列；bridge 衡量标题是否由 Probe 新暴露以及文档是否来自 generation-guided query；answer 表示文档是否满足问题的粗粒度答案类型；comparison-left 和 comparison-right 表示是否覆盖比较两侧；constraint 表示问题中的关系词是否出现在文档。一个文档可同时具有多个非零角色，这些量是字符串与排名规则产生的 compatibility scores，而不是学习得到的互斥语义标签。

设 \(\rho_r(d)\) 是文档 \(d\) 的角色 \(r\) 分数，当前集合对该角色的覆盖为

\[
c_r(S)=\max_{s\in S}\rho_r(s).
\]

角色边际增益定义为

\[
g_{\mathrm{role}}(d\mid S)=
\sum_{r\in G(q)}
\omega_r\max(0,\rho_r(d)-c_r(S)),
\]

其中 \(G(q)\) 与权重 \(\omega_r\) 由问题类型确定。除角色覆盖外，selector 还奖励尚未出现的标题和来源，并惩罚与已选文档的最大词集合 Jaccard 冗余。实际边际目标写为

\[
\begin{aligned}
\Delta(d\mid S)=&
\lambda_{\mathrm{rrf}}s_{\mathrm{RRF}}(d)
+\lambda_{\mathrm{role}}g_{\mathrm{role}}(d\mid S)\\
&+\lambda_{\mathrm{src}}g_{\mathrm{src}}(d\mid S)
+\lambda_{\mathrm{title}}g_{\mathrm{title}}(d\mid S)
-\lambda_{\mathrm{red}}g_{\mathrm{red}}(d,S)
-p_{\mathrm{title}}(d,S).
\end{aligned}
\]

选择前先按原始 rank 固定保留若干 original seeds，以避免扩展文档完全覆盖初始锚点。之后逐轮最大化 \(\Delta(d\mid S)\)，并在得分相同时依次比较 RRF 与最佳检索名次。因为 role、source、title 和 redundancy 都随 \(S\) 更新，该过程是集合条件的 greedy packing；它不求解全局最优组合。

上述目标主要用于 generation-guided，以及少数非 bridge/constraint 的 static 样本。Route-specific packing 还包含两种独立路径。Direct 不使用 RRF 或角色得分，而按 original rank 先选不同标题，再在槽位不足时回填重复标题。Bridge/constraint static 先保留至多若干 original 文档和 QD 文档，再从全候选池按 RRF 补入新标题，最后才允许重复标题。这里的 original/QD 数量是分阶段上限，不是固定组成配额。

## 18.6 上下文组织与答案生成

完成选择后，扩展路线会重新组织展示顺序：带 original provenance 的文档在前，扩展独有文档在后，各组主要按对应 query 的 rank 排列。这样，selector 决定“哪些文档进入上下文”，presentation order 决定“这些文档如何呈现”，两者不应混为同一次排序。Direct 则保留其 title-diverse first pass 与重复标题回填顺序。

最终上下文把每篇 passage 格式化为带编号和标题的文档块，再与原问题一起输入 answer-only prompt。Probe reasoning、Router 特征、RRF 和角色分数均不写入 final prompt。生成器被要求只返回答案，并在固定输出上限内确定性生成。EFC 不使用独立答案抽取器；模型返回字符串直接作为预测，指标计算阶段再采用标准小写、标点、冠词与空白规范化。因此方法质量来自证据轨迹与 answer-only prompting，而不是比基线更强的答案清洗。

文档预算和 token 预算在实现中是两层约束。Selector 首先产生至多 (K) 个文档槽位；Prompt builder 再对完整 chat prompt 执行 tokenization。若总长度达到输入上限，实现保留 prompt 首部和尾部的对称 token 片段，直接丢弃中间部分，而不在 passage 边界上逐篇裁剪。这保留了靠前的 original 证据和靠后的问题，但也意味着“选入 top10”不必然等价于“模型完整看到十篇”。现有正式日志没有证明哪些样本实际触发该截断，因此论文应披露截断规则，不应宣称所有最终文档都以完整形式输入模型。

## 18.7 Implementation Details

正式实现使用固定的 Wikipedia 2018 百万段落语料、BGE-large-en-v1.5 dense retriever 和 Llama-3.1-8B-Instruct。初始检索、Probe context 和最终 context 的预算分别为 20、5 和 10；generation-guided 生成一个 query 并检索 10 篇，static-QD 生成两个 query、每条检索 5 篇。RRF 常数为 60，original、static 和 generation 来源权重分别为 1.00、1.05 和 1.15。集合打包中 RRF、role、title、source 与 redundancy 系数分别为 1.00、0.30、0.02、0.05 和 0.01；先保留 4 个 original seed，soft same-title threshold 为 2。Probe、Planner 和 final generation 的最大输出分别为 128、96 和 32 tokens，最终输入上限为 4096。所有正式运行关闭 cross-encoder reranker 和 refiner。

检索 cache 与 prepare/generate 拆分仅用于复用 exact-query 结果和控制模型生命周期。它们不参与路由公式、RRF 或 packing 目标，因而应作为实现细节而非方法贡献报告。实验中还应同时披露每问题的平均 LLM 调用、逻辑 retrieval query 数、去重候选池大小和最终文档数，避免只用相同 final \(K\) 暗示不同方法拥有完全相同的检索预算。

## 18.8 方法边界

EFC-RAG 的 generation-guided 路径继承了利用中间生成指导后续检索的基本思想；static-QD 使用标准 query decomposition 作为条件 fallback；RRF 也是通用 rank fusion。本文方法的准确抽象是：以受约束 Probe 产生可观察反馈，用确定性规则选择一次扩展方式，再在固定上下文预算内执行带 provenance 的路由相关 packing。该实现没有学习证据充分性概率，没有执行任意轮数的检索推理循环，也不保证 greedy selector 得到全局最优集合。因此论文比较应聚焦这一组合相对固定 Standard RAG、固定 IterRetGen、固定 Full-QD 和多轮 IRCoT 的质量—调用预算差异，而不应声称全面优于 IRCoT，或把 answer-containment Retrieval Recall 称为 supporting-evidence recall。

此外，Router 的输出表示一条受控的执行路径，而不是对问题难度或证据充分性的标定。论文中应同时报告 route 分布与实际 Planner/检索调用，因为 fallback 到 direct 的样本可能已经支付过一次 Planner 成本。这一区分使方法的算法预算与最终路由标签保持可核查性。

# 19. 输出质量检查

## 可以直接写入论文

- EFC-RAG 是 training-free、inference-time 方法；冻结 BGE、Llama 和 FAISS index，不训练 router。
- 输入/输出、\(R_0,y_p,Q_m,R_m,\mathcal C,S,\hat a\) 的定义及 Algorithm 1。
- 四个 FINAL 的共同预算：initial20、probe5、final at most10；normal generation 1×top10；static 2×top5。
- Probe 的真实职责、实际 prompt、128-token 上限，以及 Probe 从不直接作为 final prediction。
- Router 的完整 first-match 顺序；HP/2W/MU 的 `assume_multihop=True` 与 NQ `False`。
- generation-guided prompt、parser、0.9 string-similarity rejection、heuristic repair 和 late-static fallback。
- static bridge 与 plain static 的输入差异，以及“up to 4 / up to 2 + backfill”的真实语义。
- UID merge、1-based rank、source-specific RRF 公式和 1.00/1.05/1.15 权重。
- 六种实际 role feature、问题类型权重与 greedy marginal objective。
- original seeds 是先固定插入；title penalty 是 soft；direct 绕过 role-aware selector。
- 路由相关 context order、实际 `Doc i(Title: ...)` 格式、全 prompt 首尾截断逻辑。
- EFC 没有 final answer parser；raw prediction 只在指标阶段规范化。
- 无 reranker、无 refiner；Probe/final 为 vLLM、Planner 为同 checkpoint 的 HF 实例。
- 四数据集逐样本逻辑调用均值与 final route 数；这些数有 `efc_cost` artifact 支持。
- Retrieval Recall 的实际含义是 final context 中的 gold-answer string hit，不是 supporting evidence recall。

## 需要作者决定的论文抽象

- 主标题继续使用 “Evidence-Feedback Controlled RAG”，还是把 “Controlled” 弱化为更直接的 “Probe-Guided Conditional RAG”。
- 将 Router 作为主要贡献模块，还是作为条件 evidence expansion 的辅助控制策略；代码更支持后者而非独立 learned router。
- 正文是否使用 “role-aware” 一词。若使用，必须限定为 heuristic role-coverage packing，且说明并非所有 route 适用。
- 将 static-QD 称为第三条并列路线，还是称为 generation failure/uncertain Probe 的 fallback；代码同时支持两种来源。
- 最终符号使用 \(Q_m/R_m\) 统一表示所有扩展，还是为 generation \(q_g/R_g\) 与 static \(Q_d/R_d\) 分开。
- 贡献表述聚焦 “conditional expansion + provenance-aware route-specific packing” 的组合，还是进一步拆成两个贡献点。
- exact prompts、marker lists、正则、tie-breaking 和 cache 细节在 Appendix 中公开到何种粒度。
- 是否修订实现中 `selection_method="efc_role_aware"` 对 static bridge 的不准确诊断命名；这不影响既有结果，但影响未来 artifact 可读性。

## 仍需代码作者确认

- 参数选择的研究性依据：现有 Git 只显示演化，不能验证每个权重/阈值是否经过独立搜索、如何避免对 HotpotQA dev 过拟合。
- HP、2Wiki、MuSiQue 三个 FINAL 的实际完整 shell command 与运行时完整未提交 diff；输出目录未保存 command manifest，现状为 `UNKNOWN`。
- retrieval cache 在相同配置下与冷检索是否做过逐 query bitwise 对照；现有 artifact 只能确认设计语义，不能确认实验等价性。
- 四个 FINAL 中 final prompt 是否有任何样本实际触发 4096-token首尾截断；日志无 warning，严格逐样本结论尚未离线重算。
- 完整 Planner 原始输出和严格 output-token 成本；compact prompt cache 已丢弃 `planner_raw_outputs`，旧 FINAL 无法完全恢复。
- vLLM/model 除显式 `<|eot_id|>` 外的完整运行时 EOS 行为；保存配置未枚举，记为 `UNKNOWN`。
- S2G-RAG 与本方法的逐模块新颖性边界；本仓库没有其可审计实现或对照材料。
- 是否在论文中另报 robust supporting-title/sentence recall；当前正式 Retrieval Recall 不能承担该含义。
- 当前 dirty worktree 的消融代码何时锁定为新的实验 commit；它们不得追溯性写入四个既有 FINAL 的主方法描述。
