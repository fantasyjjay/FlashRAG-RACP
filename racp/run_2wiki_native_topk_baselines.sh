#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-1}"
PYTHON_BIN="${PYTHON_BIN:-/home/guanjunjie/.conda/envs/flashrag/bin/python}"
DATASET="2wikimultihopqa"
SPLIT="dev"
CACHE_DIR="racp/output/cache"
LOG_DIR="racp/output/logs"

mkdir -p "${CACHE_DIR}" "${LOG_DIR}"

NO_RERANK_PREFIX="${DATASET}_${SPLIT}_bge_large_no_rerank"
RERANK_PREFIX="${DATASET}_${SPLIT}_bge_large_rerank"

NO_RERANK_TOP20_PROMPT="${CACHE_DIR}/${NO_RERANK_PREFIX}_top20_prompt_cache.json"
NO_RERANK_TOP20_RETRIEVAL="${CACHE_DIR}/${NO_RERANK_PREFIX}_top20_retrieval_cache.json"
RERANK_TOP20_PROMPT="${CACHE_DIR}/${RERANK_PREFIX}_top20_prompt_cache.json"
RERANK_TOP20_RETRIEVAL="${CACHE_DIR}/${RERANK_PREFIX}_top20_retrieval_cache.json"

run_on_gpu() {
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "$@"
}

echo "[1/6] Build no-rerank top20 prompt/retrieval cache on GPU ${GPU_ID}"
if [[ ! -f "${NO_RERANK_TOP20_PROMPT}" ]]; then
  run_on_gpu "${PYTHON_BIN}" racp/run_exp.py \
    --method_name naive \
    --stage prepare \
    --dataset_name "${DATASET}" \
    --split "${SPLIT}" \
    --gpu_id "${GPU_ID}" \
    --retrieval_topk 20 \
    --no_reranker \
    --save_retrieval_cache \
    --prompt_cache_path "${NO_RERANK_TOP20_PROMPT}" \
    --save_note "2wiki-no-rerank-top20-prepare"

  latest_prepare_dir="$(ls -td racp/output/${DATASET}_*_2wiki-no-rerank-top20-prepare | head -n 1)"
  cp "${latest_prepare_dir}/retrieval_cache.json" "${NO_RERANK_TOP20_RETRIEVAL}"
else
  echo "Skip: ${NO_RERANK_TOP20_PROMPT} already exists"
fi
if [[ ! -f "${NO_RERANK_TOP20_RETRIEVAL}" ]]; then
  latest_prepare_dir="$(ls -td racp/output/${DATASET}_*_2wiki-no-rerank-top20-prepare | head -n 1)"
  cp "${latest_prepare_dir}/retrieval_cache.json" "${NO_RERANK_TOP20_RETRIEVAL}"
fi

echo "[2/6] Derive no-rerank top5/top10/top15 prompt caches"
if [[ -f "${CACHE_DIR}/${NO_RERANK_PREFIX}_top5_prompt_cache.json" \
   && -f "${CACHE_DIR}/${NO_RERANK_PREFIX}_top10_prompt_cache.json" \
   && -f "${CACHE_DIR}/${NO_RERANK_PREFIX}_top15_prompt_cache.json" ]]; then
  echo "Skip: no-rerank top5/top10/top15 prompt caches already exist"
else
  "${PYTHON_BIN}" racp/derive_prompt_cache.py \
    --source_prompt_cache "${NO_RERANK_TOP20_PROMPT}" \
    --dataset_name "${DATASET}" \
    --split "${SPLIT}" \
    --topks 5 10 15 \
    --prefix "${NO_RERANK_PREFIX}"
fi

echo "[3/6] Build rerank top20 prompt/retrieval cache from no-rerank top20 candidates"
if [[ ! -f "${RERANK_TOP20_PROMPT}" ]]; then
  run_on_gpu "${PYTHON_BIN}" racp/run_exp.py \
    --method_name naive \
    --stage prepare \
    --dataset_name "${DATASET}" \
    --split "${SPLIT}" \
    --gpu_id "${GPU_ID}" \
    --retrieval_topk 20 \
    --rerank_topk 20 \
    --use_reranker \
    --save_retrieval_cache \
    --source_prompt_cache "${NO_RERANK_TOP20_PROMPT}" \
    --prompt_cache_path "${RERANK_TOP20_PROMPT}" \
    --save_note "2wiki-rerank-top20-prepare"

  latest_prepare_dir="$(ls -td racp/output/${DATASET}_*_2wiki-rerank-top20-prepare | head -n 1)"
  cp "${latest_prepare_dir}/retrieval_cache.json" "${RERANK_TOP20_RETRIEVAL}"
else
  echo "Skip: ${RERANK_TOP20_PROMPT} already exists"
fi
if [[ ! -f "${RERANK_TOP20_RETRIEVAL}" ]]; then
  latest_prepare_dir="$(ls -td racp/output/${DATASET}_*_2wiki-rerank-top20-prepare | head -n 1)"
  cp "${latest_prepare_dir}/retrieval_cache.json" "${RERANK_TOP20_RETRIEVAL}"
fi

echo "[4/6] Derive rerank top5/top10/top15 prompt caches"
if [[ -f "${CACHE_DIR}/${RERANK_PREFIX}_top5_prompt_cache.json" \
   && -f "${CACHE_DIR}/${RERANK_PREFIX}_top10_prompt_cache.json" \
   && -f "${CACHE_DIR}/${RERANK_PREFIX}_top15_prompt_cache.json" ]]; then
  echo "Skip: rerank top5/top10/top15 prompt caches already exist"
else
  "${PYTHON_BIN}" racp/derive_prompt_cache.py \
    --source_prompt_cache "${RERANK_TOP20_PROMPT}" \
    --dataset_name "${DATASET}" \
    --split "${SPLIT}" \
    --topks 5 10 15 \
    --prefix "${RERANK_PREFIX}"
fi

run_generation() {
  local prefix="$1"
  local note_prefix="$2"
  local topk="$3"
  local prompt_cache="${CACHE_DIR}/${prefix}_top${topk}_prompt_cache.json"

  echo "Generate ${note_prefix} top${topk}"
  run_on_gpu "${PYTHON_BIN}" racp/run_exp.py \
    --method_name naive \
    --stage generate \
    --dataset_name "${DATASET}" \
    --split "${SPLIT}" \
    --gpu_id "${GPU_ID}" \
    --prompt_cache_path "${prompt_cache}" \
    --save_note "${note_prefix}-top${topk}-generate" \
    --gpu_memory_utilization 0.85
}

echo "[5/6] Generate no-rerank top5/top10/top15/top20"
for topk in 5 10 15 20; do
  run_generation "${NO_RERANK_PREFIX}" "2wiki-native-no-rerank" "${topk}"
done

echo "[6/6] Generate rerank top5/top10/top15/top20"
for topk in 5 10 15 20; do
  run_generation "${RERANK_PREFIX}" "2wiki-native-rerank" "${topk}"
done

echo "Done."
