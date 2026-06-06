下面这份可以直接发给 Codex，让它按 README 设计实现。重点是：**主方法不使用 reranker**，QD 只做 fallback，核心创新是 **证据前沿构造 + 角色感知上下文打包**。设计参考了 adaptive routing、retrieval quality evaluation 和 generation-guided retrieval 这几类思路；例如 Adaptive-RAG 做 query complexity routing，CRAG 做 retrieval evaluator 触发纠正动作，ITER-RETGEN 用上一轮生成辅助下一轮检索。([Ocade Fusion][1])

````markdown
# EFC-RAG: Evidence Frontier Construction and Role-Aware Context Packing for Reranker-free Multi-hop RAG

## 1. Goal

Implement a reranker-free multi-hop RAG pipeline in FlashRAG.

The method should NOT rely on cross-encoder reranker in the main setting.  
The core goal is to improve multi-hop evidence coverage under a fixed final context budget.

Main idea:

1. Retrieve an initial evidence frontier with the original question.
2. Generate a probe answer `y1`.
3. Estimate whether the current context is sufficient.
4. If insufficient, expand the evidence frontier using either:
   - static QD fallback, or
   - generation-guided missing-hop retrieval.
5. Select final context with role-aware budgeted packing instead of reranker.
6. Generate the final answer.

Method name:

```text
EFC-RAG
Evidence Frontier Construction + Role-Aware Context Packing
````

Chinese name:

```text
面向多跳 RAG 的免重排器证据前沿构造与角色感知上下文选择方法
```

---

## 2. High-level Pipeline

```text
Input question q
↓
1. Initial retrieval
   q → retriever top20 = R0
↓
2. Probe context
   C0 = R0 top5
↓
3. Probe generation
   q + C0 → generator → y1
↓
4. Sufficient-context router
   Use R0 + y1 features to choose one route:
   A. Sufficient / Direct
   B. Insufficient-no-anchor / Static QD fallback
   C. Insufficient-with-anchor / Generation-guided retrieval
↓
5. Evidence frontier expansion
   Route A: no expansion
   Route B: static QD retrieval
   Route C: missing-hop retrieval from y1
↓
6. Candidate pool merge
   original docs + qd docs + generation-guided docs
↓
7. Role-aware context packing
   RRF + role coverage gain + title diversity + source coverage
↓
8. Context ordering
   anchor/original first, bridge/answer evidence later
↓
9. Final generation
   final context + q → final answer
```

---

## 3. Files to Modify

Main target file:

```text
racp/run_racp.py
```

If needed, create helper files:

```text
racp/efc_router.py
racp/efc_roles.py
racp/efc_packing.py
```

Keep the implementation simple and compatible with existing FlashRAG retrieval and generation code.

---

## 4. New CLI Arguments

Add these arguments:

```python
--enable_efc_rag

--initial_topk 20
--probe_topk 5
--final_topk 5

--enable_static_qd_fallback
--qd_num 2
--qd_topk 5

--enable_generation_guided
--gen_topk 10
--missing_query_mode heuristic_or_llm
# choices: heuristic, llm, raw_y1

--rrf_k 60

--role_weight 0.30
--rrf_weight 1.00
--title_weight 0.05
--source_weight 0.05
--redundancy_weight 0.05

--title_dedup_soft
--max_same_title 2

--use_centroid_router
--save_efc_debug

--force_route auto
# choices: auto, direct, static_qd, generation_guided

--no_reranker
```

Main default setting:

```text
--enable_efc_rag
--no_reranker
--initial_topk 20
--probe_topk 5
--final_topk 5
--enable_generation_guided
--enable_static_qd_fallback
```

---

## 5. Data Structures

Each retrieved document should be normalized into this structure:

```python
doc = {
    "doc_uid": str,                # stable unique id, use doc_id if available, otherwise hash(title + contents)
    "doc_id": str,
    "title": str,
    "contents": str,

    "sources": list[str],          # ["original"], ["qd"], ["generation_guided"], or multiple
    "source_queries": list[str],

    "ranks": dict,                 # {"original": 1, "qd_0": 3, "gen_0": 2}
    "retriever_scores": dict,      # {"original": 0.83, "qd_0": 0.78}

    "embedding": optional vector,  # use if available for centroid/redundancy
}
```

When the same document appears from multiple queries, merge it by `doc_uid`.

Example merged doc:

```python
{
    "doc_uid": "wiki_123",
    "title": "Androscoggin Bank Colisée",
    "contents": "...",
    "sources": ["generation_guided", "qd"],
    "source_queries": [
        "Androscoggin Bank Colisée seating capacity",
        "Lewiston Maineiacs home arena capacity"
    ],
    "ranks": {
        "gen_0": 1,
        "qd_1": 4
    },
    "retriever_scores": {
        "gen_0": 0.87,
        "qd_1": 0.74
    }
}
```

---

## 6. Stage 1: Initial Retrieval

Input:

```python
question = q
```

Run:

```python
R0 = retrieve(q, topk=initial_topk)
```

Save each doc with:

```python
source = "original"
source_query = q
rank = original rank
retriever_score = dense retriever score
```

Use `R0[:probe_topk]` as the probe context.

Default:

```text
initial_topk = 20
probe_topk = 5
```

---

## 7. Stage 2: Probe Generation

Use `R0[:probe_topk]` to generate a probe answer `y1`.

Prompt:

```text
You are given several retrieved Wikipedia passages and a question.
Use the passages to answer the question.
Reason briefly if needed.
End your response with:
So the answer is <answer>.

[Documents]
{docs}

[Question]
{question}
```

The probe answer `y1` is used for:

1. Direct answer if the context is sufficient.
2. Extracting bridge entities for generation-guided retrieval.
3. Router features.

Save:

```python
probe_answer = y1
probe_context_doc_uids = [...]
```

---

## 8. Stage 3: Router Features

Implement a function:

```python
def compute_router_features(question, R0, y1, retriever=None):
    return features
```

Features:

```python
features = {
    # retrieval score features
    "s1": float,
    "s5": float,
    "avg_top5_score": float,
    "gap_1_5": float,

    # title diversity
    "title_unique_ratio": float,

    # centroid features, if embeddings are available
    "q_centroid_sim": float or None,
    "top5_cohesion": float or None,
    "centroid_shift": float or None,

    # question pattern
    "question_is_multihop": bool,
    "question_type": "bridge" | "comparison" | "constraint" | "generic",

    # y1 features
    "y1_bad": bool,
    "y1_has_new_entity": bool,
    "y1_new_entities": list[str],
    "y1_new_entity_count": int,

    # answer type
    "answer_type": "number" | "date" | "person" | "location" | "yesno" | "entity" | "unknown",
    "answer_type_hit_top5": bool
}
```

### 8.1 Title diversity

```python
title_unique_ratio = len(set(top5_titles)) / len(top5_titles)
```

Interpretation:

```text
Low title_unique_ratio:
top5 may be concentrated around one entity, often only first-hop evidence.

High title_unique_ratio:
top5 may cover more entities or evidence aspects.
```

### 8.2 Centroid features

Use dense embeddings if available.

```python
q_emb = encode(question)
e_i = embedding(doc_i)
C5 = mean(e_1 ... e_5)
C20 = mean(e_1 ... e_20)

q_centroid_sim = cosine(q_emb, C5)
top5_cohesion = mean(cosine(e_i, C5) for i in top5)
centroid_shift = 1 - cosine(C5, C20)
```

If embeddings are not easily available, skip centroid features and set them to `None`.

### 8.3 Multihop pattern detection

Implement lightweight rules.

Comparison patterns:

```text
between A and B
which came first
who had more
which is larger
which is smaller
earlier
later
more
less
```

Bridge patterns:

```text
the X of Y
the director of
the author of
the wife of
the husband of
the father of
the mother of
where was ... born
what is the ... of ...
```

Constraint patterns:

```text
born in
located in
worked with
directed by
written by
published by
founded by
member of
```

### 8.4 Extract answer type

Simple rules:

```python
if "how many" in q or "number" in q or "capacity" in q:
    answer_type = "number"
elif "when" in q or "what year" in q or "date" in q:
    answer_type = "date"
elif q.startswith("who"):
    answer_type = "person"
elif q.startswith("where"):
    answer_type = "location"
elif q.lower().startswith(("is ", "are ", "was ", "were ", "can ", "did ", "does ")):
    answer_type = "yesno"
else:
    answer_type = "entity"
```

### 8.5 y1_bad

Set `y1_bad = True` if:

```text
empty output
too short
contains "I don't know"
contains "cannot answer"
mostly repeats question
JSON/parser failure if JSON mode is used
```

### 8.6 Extract new entities from y1

Do not use heavy NER. Use simple heuristics:

1. Extract capitalized spans from `y1`.
2. Extract document titles mentioned in `y1`.
3. Remove spans already appearing in question.
4. Remove common stop entities.

Return:

```python
y1_new_entities = [...]
```

---

## 9. Stage 4: Router Decision

Implement:

```python
def decide_route(features, force_route="auto"):
    return route
```

Routes:

```text
direct
static_qd
generation_guided
```

### 9.1 Direct route

Meaning:

```text
Initial context is likely sufficient.
Do not expand retrieval.
Use y1 or final generation over original top5.
```

Trigger:

```python
if not features["y1_bad"] \
   and features["title_unique_ratio"] >= 0.6 \
   and features["answer_type_hit_top5"]:
    return "direct"
```

Also direct if the question is not multihop-like:

```python
if not features["question_is_multihop"] and not features["y1_bad"]:
    return "direct"
```

### 9.2 Static-QD route

Meaning:

```text
Initial retrieval is weak and there is no reliable anchor.
Use static question decomposition as fallback.
```

Trigger:

```python
if features["y1_bad"] and not r0_confident:
    return "static_qd"

if not features["y1_has_new_entity"] \
   and features["title_unique_ratio"] < 0.4 \
   and not features["answer_type_hit_top5"]:
    return "static_qd"
```

If centroid features exist:

```python
if features["q_centroid_sim"] is not None:
    if features["q_centroid_sim"] < tau_qsim_low \
       and not features["y1_has_new_entity"]:
        return "static_qd"
```

### 9.3 Generation-guided route

Meaning:

```text
Initial retrieval has some anchor evidence, and y1 exposes a possible bridge entity.
Use y1 to retrieve missing-hop evidence.
```

Trigger:

```python
if features["question_is_multihop"] \
   and features["y1_has_new_entity"] \
   and features["title_unique_ratio"] < 0.8:
    return "generation_guided"
```

If centroid features exist:

```python
if features["top5_cohesion"] is not None:
    if features["top5_cohesion"] > tau_cohesion_high \
       and features["y1_has_new_entity"]:
        return "generation_guided"
```

### 9.4 Default

```python
return "direct"
```

Important:

```text
Router must NOT call another LLM.
Router must be cheap and deterministic.
```

---

## 10. Stage 5: Evidence Frontier Expansion

### 10.1 Route A: Direct

No expansion.

```python
candidate_pool = R0
final_context = role_aware_pack(candidate_pool, route="direct")
```

In direct route, role-aware packing can be minimal:

```text
use original top5
apply only soft title dedup if needed
```

### 10.2 Route B: Static QD fallback

Run existing QD planner if available.

Prompt:

```text
Generate exactly 2 retrieval-oriented search queries for the question.

Rules:
1. Each query should ask for one missing factual piece.
2. Include concrete entities when possible.
3. Do not answer the question.
4. Output only a JSON array of strings.

Question:
{question}
```

Then:

```python
subqueries = generate_qd_queries(q, qd_num=2)
qd_docs = []
for i, sq in enumerate(subqueries):
    docs = retrieve(sq, topk=qd_topk)
    mark source as "qd"
    mark source_query as sq
    mark rank key as f"qd_{i}"
    qd_docs.extend(docs)

candidate_pool = merge_docs(R0 + qd_docs)
```

Failure protection:

```python
if qd_docs bring no new title
and qd_docs bring no new role gain
and qd_docs have low retriever scores:
    ignore qd_docs
    candidate_pool = R0
```

Do NOT output refusal in benchmark setting.

### 10.3 Route C: Generation-guided missing-hop retrieval

Construct a missing-hop query.

Implement three modes:

```text
heuristic
llm
raw_y1
```

Default:

```text
heuristic
```

#### Heuristic mode

Use:

```python
best_entity = select_best_new_entity(y1_new_entities, R0_titles, question)
answer_type_terms = get_answer_type_terms(answer_type, question)
missing_query = best_entity + " " + answer_type_terms
```

Examples:

```text
answer_type = number + question contains "seat/capacity"
→ terms = "capacity seating seats"

answer_type = date/year
→ terms = "birth year date"

answer_type = location
→ terms = "birthplace location place"

answer_type = person
→ terms = "person name"
```

Example:

```text
q:
The arena where the Lewiston Maineiacs played their home games can seat how many people?

y1:
The Lewiston Maineiacs played at Androscoggin Bank Colisée.

missing_query:
Androscoggin Bank Colisée seating capacity
```

#### LLM mode

Use small planner model only if already available.

Prompt:

```text
You are a retrieval query reformulator for multi-hop question answering.

Given the original question and a tentative reasoning result,
generate one missing-hop search query.

Rules:
1. Use the bridge entity from the tentative reasoning if available.
2. Ask for the missing factual attribute.
3. Do not answer the question.
4. Output only a JSON array with one string.

Question:
{question}

Tentative reasoning:
{y1}
```

#### raw_y1 mode

```python
missing_query = question + "\n" + y1
```

This is closest to ITER-RETGEN, but may be noisy.

Then retrieve:

```python
gen_docs = retrieve(missing_query, topk=gen_topk)
mark source as "generation_guided"
mark source_query as missing_query
mark rank key as "gen_0"

candidate_pool = merge_docs(R0 + gen_docs)
```

Optional drift check:

```python
gen_shift = 1 - cosine(centroid(R0_top5), centroid(gen_docs_top5))
```

If:

```text
gen_shift too small → duplicate retrieval
gen_shift too large + low scores → possible retrieval drift
```

Then reduce or ignore gen_docs.

Do not hard fail in v1; just log `gen_shift`.

---

## 11. Stage 6: RRF Scoring

Do not compare raw dense retriever scores across different queries directly.

Use RRF:

```python
rrf_score(d) = sum(
    source_weight[source_key] / (rrf_k + rank_source_key(d))
)
```

Default:

```python
rrf_k = 60
source_weight = {
    "original": 1.0,
    "qd": 1.05,
    "generation_guided": 1.15
}
```

For direct route:

```python
source_weight = {"original": 1.0}
```

---

## 12. Stage 7: Soft Evidence Role Scoring

Do NOT call LLM to classify document roles.

Compute soft role scores with heuristics.

Implement:

```python
def compute_role_scores(doc, question, y1, features):
    return {
        "anchor": float,
        "bridge": float,
        "answer": float,
        "comparison_left": float,
        "comparison_right": float,
        "constraint": float
    }
```

All scores should be in `[0, 1]`.

### 12.1 Anchor score

High if doc title or content overlaps with explicit entities in the question.

Signals:

```text
doc.title appears in question
question entity appears in doc.title
doc comes from original retrieval and rank is high
```

Example:

```python
anchor = 0.0
if title_in_question(doc.title, question):
    anchor += 0.7
if "original" in doc.sources and original_rank <= 5:
    anchor += 0.3
anchor = min(anchor, 1.0)
```

### 12.2 Bridge score

High if doc title appears in `y1` but not in question.

Signals:

```text
doc.title appears in y1
doc.title not in question
doc comes from generation-guided retrieval
doc matches missing-hop query
```

Example:

```python
bridge = 0.0
if title_in_text(doc.title, y1) and not title_in_text(doc.title, question):
    bridge += 0.7
if "generation_guided" in doc.sources:
    bridge += 0.3
bridge = min(bridge, 1.0)
```

### 12.3 Answer score

High if doc contains answer-type signals.

For number questions:

```text
digits
capacity
seats
population
height
score
```

For date/year questions:

```text
year regex: \b(1[0-9]{3}|20[0-9]{2})\b
born
date
founded
released
published
```

For location questions:

```text
born in
located in
city
country
state
place
```

Example:

```python
answer = answer_type_regex_hit(doc.contents, features["answer_type"])
```

### 12.4 Comparison scores

If question is comparison type, extract two sides.

Example patterns:

```text
Between A and B
Which came first, A or B
Who had more ..., A or B
```

Then:

```python
comparison_left = 1.0 if left_entity appears in title/content else 0.0
comparison_right = 1.0 if right_entity appears in title/content else 0.0
```

### 12.5 Constraint score

High if doc contains relation/constraint terms from the question.

Constraint terms:

```text
born
directed
written
published
founded
located
member
spouse
wife
husband
father
mother
worked with
```

Example:

```python
constraint = overlap_count(question_relation_terms, doc.contents) / len(question_relation_terms)
```

---

## 13. Stage 8: Role-aware Budgeted Packing

Implement greedy selection:

```python
def role_aware_pack(candidate_pool, question, y1, features, final_topk, route):
    selected = []
    while len(selected) < final_topk:
        choose doc d with max gain(d | selected)
        selected.append(d)
    return selected
```

Gain function:

```python
gain(d | S) =
    rrf_weight * rrf_score(d)
  + role_weight * role_coverage_gain(d | S)
  + title_weight * title_novelty_gain(d | S)
  + source_weight * source_coverage_gain(d | S)
  - redundancy_weight * redundancy_penalty(d | S)
```

### 13.1 Role coverage gain

Maintain current covered role scores:

```python
covered_roles = {
    "anchor": max score among selected docs,
    "bridge": max score among selected docs,
    "answer": max score among selected docs,
    "comparison_left": max score among selected docs,
    "comparison_right": max score among selected docs,
    "constraint": max score among selected docs
}
```

For a new doc:

```python
role_coverage_gain = sum(
    max(0, role_score[d][role] - covered_roles[role]) * role_goal_weight[role]
)
```

Role goal weights depend on question type.

Bridge question:

```python
role_goal_weight = {
    "anchor": 0.25,
    "bridge": 0.35,
    "answer": 0.30,
    "constraint": 0.10
}
```

Comparison question:

```python
role_goal_weight = {
    "comparison_left": 0.35,
    "comparison_right": 0.35,
    "answer": 0.20,
    "constraint": 0.10
}
```

Generic question:

```python
role_goal_weight = {
    "anchor": 0.40,
    "answer": 0.40,
    "constraint": 0.20
}
```

### 13.2 Title novelty gain

```python
if doc.title not in selected_titles:
    title_novelty_gain = 1.0
else:
    title_novelty_gain = 0.0
```

Soft duplicate control:

```python
if count_same_title >= max_same_title:
    heavily penalize doc
```

Default:

```text
max_same_title = 2
```

### 13.3 Source coverage gain

Encourage useful source diversity.

```python
if doc adds a source not yet represented in selected:
    source_coverage_gain = 1.0
else:
    source_coverage_gain = 0.0
```

Route-specific preference:

Route C:

```text
ensure at least one original doc and one generation-guided doc if available
```

Route B:

```text
ensure at least one original doc and one QD doc if QD docs show role gain
```

Do not force bad docs into context.

### 13.4 Redundancy penalty

If embeddings available:

```python
redundancy_penalty = max cosine(doc.embedding, selected_doc.embedding)
```

If not:

```python
redundancy_penalty = 1.0 if same title and high lexical overlap else 0.0
```

---

## 14. Stage 9: Context Ordering

After final docs are selected, order them before generation.

Route A:

```text
original rank order
```

Route B:

```text
original/anchor docs first
QD docs second
answer/constraint docs later
```

Route C:

```text
original/anchor docs first
generation-guided/bridge docs second
answer docs later
```

Reason:

```text
Small generators are sensitive to context order.
For multi-hop QA, first-hop evidence should appear before second-hop evidence.
```

---

## 15. Stage 10: Final Generation

For Route A:

```python
final_answer = y1
```

Optional:

```python
regenerate with selected final context
```

For Route B and Route C:

```python
final_answer = generator.generate(final_prompt)
```

Final prompt:

```text
You are given several retrieved Wikipedia passages and a question.
Use only the useful evidence from the passages.
For multi-hop questions, combine evidence from different passages.
Output only the final answer.

[Documents]
{final_context}

[Question]
{question}

Final answer:
```

For cleaner EM/F1, avoid long chain-of-thought in final output.

---

## 16. Logging and Debug Output

For each sample, save JSONL record:

```json
{
  "qid": "...",
  "question": "...",
  "gold_answer": "...",

  "route": "direct | static_qd | generation_guided",
  "router_features": {...},

  "probe_answer": "...",
  "probe_context_doc_uids": [...],

  "qd_queries": [...],
  "missing_hop_query": "...",

  "candidate_pool_size": 37,
  "candidate_pool_summary": {
    "original": 20,
    "qd": 10,
    "generation_guided": 10,
    "merged_unique": 34
  },

  "selected_doc_uids": [...],
  "selected_titles": [...],
  "selected_sources": [...],

  "role_scores_selected": [...],
  "role_coverage": {
    "anchor": 0.9,
    "bridge": 0.8,
    "answer": 0.7,
    "comparison_left": 0.0,
    "comparison_right": 0.0,
    "constraint": 0.4
  },

  "final_context": [...],
  "final_answer": "...",

  "llm_calls": 1 or 2,
  "retrieval_calls": 1 or more
}
```

Also print aggregate stats:

```text
Route distribution:
direct: xx%
static_qd: xx%
generation_guided: xx%

Average LLM calls
Average retrieval calls
Average candidate pool size
Average final context docs
Average role coverage
Average title diversity
```

If HotpotQA support titles are available, also compute:

```text
support_title_recall@5
both_support_title_hit@5
answer_hit@5
```

---

## 17. Metrics

Main QA metrics:

```text
EM
F1
Accuracy
Precision
Recall
```

Retrieval/context metrics:

```text
Retrieval Recall@5
Answer Hit@5
Both-support-title hit rate
Selected title unique ratio
Role Coverage@5
Average context tokens
Average LLM calls
Average retrieval calls
Latency if available
```

Define Role Coverage@k:

```python
RoleCoverage@k = sum(covered goal roles) / sum(target goal roles)
```

For bridge questions, target roles:

```text
anchor + bridge + answer
```

For comparison questions:

```text
comparison_left + comparison_right + answer/constraint
```

---

## 18. Required Experiments

Main no-reranker comparison:

```text
1. RAG top5
2. RAG top10
3. QD-RAG top5
4. ITER-RETGEN T=2
5. MMR or title-dedup baseline
6. EFC-RAG full
```

Ablation:

```text
Full EFC-RAG
w/o router: always generation-guided
w/o static QD fallback
w/o generation-guided expansion
w/o role-aware packing: use RRF only
w/o title novelty
w/o role coverage gain
w/o centroid features
```

Optional strong reference:

```text
RAG + bge-reranker-large top5
```

This is not the main baseline. It is only a strong-cost reference.

---

## 19. Implementation Order

Implement in this order:

### Step 1

Add no-reranker RAG top5/top10 baseline.

### Step 2

Add probe generation:

```text
R0 top5 → y1
```

Save y1.

### Step 3

Implement generation-guided retrieval:

```text
q + y1 or missing-hop query → retrieve
```

### Step 4

Implement candidate merge with provenance.

### Step 5

Implement RRF scoring.

### Step 6

Implement role scoring heuristics.

### Step 7

Implement role-aware greedy packing.

### Step 8

Implement router.

### Step 9

Implement QD fallback.

### Step 10

Add debug logging and metrics.

---

## 20. Smoke Test Commands

### 50-sample smoke test

```bash
python racp/run_racp.py \
  --dataset_name hotpotqa \
  --split dev \
  --gpu_id 0 \
  --enable_efc_rag \
  --no_reranker \
  --initial_topk 20 \
  --probe_topk 5 \
  --final_topk 5 \
  --enable_generation_guided \
  --enable_static_qd_fallback \
  --missing_query_mode heuristic \
  --save_efc_debug \
  --test_sample_num 50 \
  --save_note efc-smoke-50
```

### 1000-sample validation

```bash
python racp/run_racp.py \
  --dataset_name hotpotqa \
  --split dev \
  --gpu_id 0 \
  --enable_efc_rag \
  --no_reranker \
  --initial_topk 20 \
  --probe_topk 5 \
  --final_topk 5 \
  --enable_generation_guided \
  --enable_static_qd_fallback \
  --missing_query_mode heuristic \
  --save_efc_debug \
  --test_sample_num 1000 \
  --save_note efc-dev1000
```

### Force route tests

```bash
# Direct only
python racp/run_racp.py \
  --dataset_name hotpotqa \
  --split dev \
  --gpu_id 0 \
  --enable_efc_rag \
  --no_reranker \
  --force_route direct \
  --test_sample_num 1000 \
  --save_note efc-force-direct

# Static QD only
python racp/run_racp.py \
  --dataset_name hotpotqa \
  --split dev \
  --gpu_id 0 \
  --enable_efc_rag \
  --no_reranker \
  --force_route static_qd \
  --test_sample_num 1000 \
  --save_note efc-force-static-qd

# Generation-guided only
python racp/run_racp.py \
  --dataset_name hotpotqa \
  --split dev \
  --gpu_id 0 \
  --enable_efc_rag \
  --no_reranker \
  --force_route generation_guided \
  --test_sample_num 1000 \
  --save_note efc-force-gen-guided
```

---

## 21. Important Constraints

1. Do NOT use cross-encoder reranker in the main EFC-RAG pipeline.
2. Do NOT call LLM to classify every document role.
3. Router must be deterministic and cheap.
4. QD is only fallback, not the main engine.
5. Generation-guided retrieval is the main expansion route.
6. Final selection must be budgeted: final top5 or top6 only.
7. Save enough debug information for ablation and error analysis.
8. Do not force low-quality expanded docs into final context.
9. If expansion fails, fallback to original R0 top5/top6.
10. Keep all modules optional through CLI flags.

```

最关键的实现优先级是：**先做 generation-guided retrieval + candidate merge + RRF + role-aware packing**，router 和 QD fallback 可以后加。这样即使 router 暂时不准，也能先验证“角色感知打包”是否真的比普通 top-k / title-dedup 强。
```

[1]: https://www.ocadefusion.fr/rag/adaptive-rag?utm_source=chatgpt.com "Adaptive RAG : routing intelligent selon la complexité (2026) | Ocade Fusion"
