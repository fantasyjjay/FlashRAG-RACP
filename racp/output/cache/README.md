# RACP Native RAG Cache Guide

This directory stores reusable caches for native RAG top-k baseline experiments.
The examples below use `2wikimultihopqa` `dev`, but the same commands work for
other datasets under `data_dir` by changing `--dataset_name` and `--split`.

Run commands from the FlashRAG repository root:

```bash
conda activate flashrag
cd /home/guanjunjie/jayj/FlashRAG
```

## Naming Convention

Cache filenames should include the dataset, split, retriever, reranker setting,
and top-k value:

```text
{dataset}_{split}_bge_large_{no_rerank|rerank}_top{k}_prompt_cache.json
{dataset}_{split}_bge_large_{no_rerank|rerank}_top{k}_retrieval_cache.json
```

Current HotpotQA native RAG caches in this directory use the same convention:

```text
hotpotqa_dev_bge_large_no_rerank_top5_prompt_cache.json
hotpotqa_dev_bge_large_no_rerank_top10_prompt_cache.json
hotpotqa_dev_bge_large_no_rerank_top15_prompt_cache.json
hotpotqa_dev_bge_large_no_rerank_top20_prompt_cache.json
hotpotqa_dev_bge_large_rerank_top5_prompt_cache.json
hotpotqa_dev_bge_large_rerank_top10_prompt_cache.json
hotpotqa_dev_bge_large_rerank_top15_prompt_cache.json
hotpotqa_dev_bge_large_rerank_top20_prompt_cache.json
```

`hotpotqa_dev_bge_large_no_rerank_prepared_top15_prompt_cache.json` is kept as
an older top15 cache built by a separate prepare run. Prefer the regular
`hotpotqa_dev_bge_large_no_rerank_top15_prompt_cache.json` for top-k curves
derived from top20.

## Cache Types

There are two different cache files:

- `retrieval_cache.json`: maps each question string to retrieved documents and
  scores. It avoids repeating dense retrieval when rebuilding prompt caches.
- `*_prompt_cache.json`: stores dataset items, selected documents, and complete
  prompts. It is consumed directly by `--stage generate`, so generation does
  not load the retriever or reranker.

For top5/top10/top15/top20 baseline experiments, the fastest workflow is:

1. Retrieve top20 once and save a top20 prompt cache.
2. Derive top5/top10/top15 prompt caches locally from the top20 prompt cache.
3. Run generation separately for each prompt cache.

The derivation step does not load any model and does not repeat retrieval.

## No-Reranker Baseline

### 1. Build top20 caches

```bash
CUDA_VISIBLE_DEVICES=0 python racp/run_exp.py \
  --method_name naive \
  --stage prepare \
  --dataset_name 2wikimultihopqa \
  --split dev \
  --gpu_id 0 \
  --retrieval_topk 20 \
  --no_reranker \
  --save_retrieval_cache \
  --prompt_cache_path racp/output/cache/2wikimultihopqa_dev_bge_large_no_rerank_top20_prompt_cache.json \
  --save_note 2wiki-no-rerank-top20-prepare
```

The prompt cache is written to the explicit `--prompt_cache_path`. The
retrieval cache is written to the timestamped experiment directory:

```text
racp/output/2wikimultihopqa_<timestamp>_2wiki-no-rerank-top20-prepare/retrieval_cache.json
```

Keep a stable copy in this directory:

```bash
cp racp/output/2wikimultihopqa_<timestamp>_2wiki-no-rerank-top20-prepare/retrieval_cache.json \
  racp/output/cache/2wikimultihopqa_dev_bge_large_no_rerank_top20_retrieval_cache.json
```

### 2. Derive smaller prompt caches

```bash
python racp/derive_prompt_cache.py \
  --source_prompt_cache racp/output/cache/2wikimultihopqa_dev_bge_large_no_rerank_top20_prompt_cache.json \
  --dataset_name 2wikimultihopqa \
  --split dev \
  --topks 5 10 15 \
  --prefix 2wikimultihopqa_dev_bge_large_no_rerank
```

This creates:

```text
racp/output/cache/2wikimultihopqa_dev_bge_large_no_rerank_top5_prompt_cache.json
racp/output/cache/2wikimultihopqa_dev_bge_large_no_rerank_top10_prompt_cache.json
racp/output/cache/2wikimultihopqa_dev_bge_large_no_rerank_top15_prompt_cache.json
```

### 3. Generate answers from a prompt cache

Example for top10:

```bash
CUDA_VISIBLE_DEVICES=0 python racp/run_exp.py \
  --method_name naive \
  --stage generate \
  --dataset_name 2wikimultihopqa \
  --split dev \
  --gpu_id 0 \
  --prompt_cache_path racp/output/cache/2wikimultihopqa_dev_bge_large_no_rerank_top10_prompt_cache.json \
  --save_note 2wiki-native-no-rerank-top10-generate \
  --gpu_memory_utilization 0.85
```

Change `top10` to `top5`, `top15`, or `top20` to run the other baselines.

## Reuse a Retrieval Cache

Use retrieval cache when a prompt cache needs to be rebuilt without repeating
dense retrieval:

```bash
CUDA_VISIBLE_DEVICES=0 python racp/run_exp.py \
  --method_name naive \
  --stage prepare \
  --dataset_name 2wikimultihopqa \
  --split dev \
  --gpu_id 0 \
  --retrieval_topk 10 \
  --no_reranker \
  --use_retrieval_cache \
  --retrieval_cache_path racp/output/cache/2wikimultihopqa_dev_bge_large_no_rerank_top20_retrieval_cache.json \
  --prompt_cache_path racp/output/cache/2wikimultihopqa_dev_bge_large_no_rerank_top10_prompt_cache.json \
  --save_note 2wiki-no-rerank-top10-from-retrieval-cache
```

For ordinary top-k truncation, prefer `derive_prompt_cache.py`: it is faster
because it does not initialize retrieval models.

## Reranker Baseline

Reranker experiments need a separate top20 preparation run:

```bash
CUDA_VISIBLE_DEVICES=0 python racp/run_exp.py \
  --method_name naive \
  --stage prepare \
  --dataset_name 2wikimultihopqa \
  --split dev \
  --gpu_id 0 \
  --retrieval_topk 20 \
  --rerank_topk 20 \
  --use_reranker \
  --save_retrieval_cache \
  --prompt_cache_path racp/output/cache/2wikimultihopqa_dev_bge_large_rerank_top20_prompt_cache.json \
  --save_note 2wiki-rerank-top20-prepare
```

Copy the resulting `retrieval_cache.json` to a stable filename:

```bash
cp racp/output/2wikimultihopqa_<timestamp>_2wiki-rerank-top20-prepare/retrieval_cache.json \
  racp/output/cache/2wikimultihopqa_dev_bge_large_rerank_top20_retrieval_cache.json
```

Derive reranked top5/top10/top15 prompt caches:

```bash
python racp/derive_prompt_cache.py \
  --source_prompt_cache racp/output/cache/2wikimultihopqa_dev_bge_large_rerank_top20_prompt_cache.json \
  --dataset_name 2wikimultihopqa \
  --split dev \
  --topks 5 10 15 \
  --prefix 2wikimultihopqa_dev_bge_large_rerank
```

Then generate from a derived reranker prompt cache:

```bash
CUDA_VISIBLE_DEVICES=0 python racp/run_exp.py \
  --method_name naive \
  --stage generate \
  --dataset_name 2wikimultihopqa \
  --split dev \
  --gpu_id 0 \
  --prompt_cache_path racp/output/cache/2wikimultihopqa_dev_bge_large_rerank_top5_prompt_cache.json \
  --save_note 2wiki-native-rerank-top5-generate \
  --gpu_memory_utilization 0.85
```

## Important Cache Rules

- Build the largest required top-k first. A top20 cache can be truncated to
  top5/top10/top15, but a top5 cache cannot reconstruct top20.
- Do not use a no-reranker retrieval cache for reranker experiments.
  FlashRAG retrieval caches store the final returned ordering. With reranker
  enabled, that ordering is already the reranked ordering.
- Reuse caches only when dataset split, question text, corpus, index,
  retriever, and reranker settings match.
- A retrieval cache lookup is keyed by the exact question string. Cache misses
  fall back to live retrieval.
- `--stage generate` requires only a prompt cache. It does not need
  `--use_retrieval_cache`.
- Choose a free GPU before generation with `nvidia-smi`. Replace GPU `0` in
  the examples when needed.
