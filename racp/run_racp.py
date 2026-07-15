import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path


os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def ensure_conda_runtime_libs():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix and Path(sys.argv[0]).resolve() == Path(__file__).resolve():
        conda_prefix = sys.prefix
    if not conda_prefix:
        return

    conda_lib = str(Path(conda_prefix) / "lib")
    if not Path(conda_lib).is_dir():
        return

    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    paths = [path for path in ld_library_path.split(":") if path]
    if paths and paths[0] == conda_lib:
        return

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join([conda_lib] + [path for path in paths if path != conda_lib])
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)


ensure_conda_runtime_libs()

PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_DIR))

from flashrag.config import Config
from flashrag.dataset import Dataset
from flashrag.evaluator import Evaluator
from flashrag.prompt import PromptTemplate
from flashrag.utils import get_dataset
from racp.efc import (
    add_rrf_scores,
    build_heuristic_missing_query,
    compute_router_features,
    decide_route,
    doc_title as efc_doc_title,
    merge_docs as efc_merge_docs,
    normalize_doc as efc_normalize_doc,
    role_aware_pack,
    role_coverage_ratio,
    select_rrf_only,
    score_selected_roles,
    select_ranked_title_diverse,
    static_bridge_pack,
)


DEFAULT_CONFIG_PATH = PROJECT_DIR / "config.yaml"
DEFAULT_SAVE_DIR = PROJECT_DIR / "output"

# Main experiment defaults. Edit this block to change the no-argument run.
# Explicit CLI arguments always take precedence.
DEFAULT_RUN_CONFIG = {
    "method": "efc",
    "stage": "prepare",
    "dataset_name": "hotpotqa",
    "split": "dev",
    "gpu_id": "3",
    "save_note": "efc-hotpotqa-full",
    "test_sample_num": None,
    # GPU and batching
    "retrieval_batch_size": 1024,
    "prepare_gpu_memory_utilization": 0.75,
    "generate_gpu_memory_utilization": 0.85,
    "planner_batch_size": 32,
    "planner_inference_batch_size": 8,
    # Retrieval cache. Set a path and enable reuse for repeated experiments.
    "use_retrieval_cache": False,
    "save_retrieval_cache": True,
    "retrieval_cache_path": None,
    # EFC retrieval and generation
    "initial_topk": 20,
    "probe_topk": 5,
    "probe_max_tokens": 128,
    "final_topk": 6,
    "qd_num": 2,
    "missing_query_num": 1,
    "qd_topk": 5,
    "gen_topk": 10,
    "planner_model": "Llama-3.1-8B-Instruct",
    "planner_max_tokens": 96,
    "missing_query_mode": "llm",
    "enable_generation_guided": True,
    "enable_static_qd_fallback": True,
    "title_dedup_soft": True,
    "rrf_weight": 1.0,
    "role_weight": 0.30,
    "title_weight": 0.02,
    "source_weight": 0.05,
    "redundancy_weight": 0.01,
    "max_same_title": 2,
    "original_seed_count": 4,
    "rrf_only_selection": False,
    "static_bridge_evidence_topk": 5,
    "static_bridge_original_count": 4,
    "static_bridge_qd_count": 2,
}

PLANNER_STOP_WORDS = [
    "<|eot_id|>",
    "\nQuestion:",
    "\n\nQuestion:",
    "Expected output:",
    "Solution:",
    "Answer:",
]
PLANNER_SYSTEM_PROMPT = "You are a retrieval query rewriter. Output only a JSON array of search queries."
STRICT_SHORT_ANSWER_SYSTEM_PROMPT = (
    "Answer the question based on the given documents. Reason internally before answering, "
    "but output only the shortest final answer span and no explanation. "
    "For yes/no questions, output only yes or no. "
    "For comparison questions, output only the requested entity name. "
    "For year questions, output only the year."
    "\nThe following are given documents.\n\n{reference}"
)
STRICT_SHORT_ANSWER_USER_PROMPT = "Question: {question}\nShortest answer:"
ANSWER_ONLY_SYSTEM_PROMPT = (
    "Answer the question based on the given documents. "
    "Output only the final answer, with no explanation, no citations, and no full sentence "
    "unless the answer itself is a sentence. Keep complete names, titles, and dates intact."
    "\nThe following are given documents.\n\n{reference}"
)
ANSWER_ONLY_USER_PROMPT = "Question: {question}\nAnswer:"
EXACT_ANSWER_SYSTEM_PROMPT = (
    "Answer the question based on the given documents. Reason internally before answering, "
    "but output only the exact final answer span. Do not include explanation, citations, "
    "or surrounding sentence text. Keep the answer concise while preserving complete "
    "entity names, work titles, dates, and all required answer items. "
    "For yes/no questions, output only yes or no."
    "\nThe following are given documents.\n\n{reference}"
)
EXACT_ANSWER_USER_PROMPT = "Question: {question}\nExact answer:"
RS_MHR_STATIC_SYSTEM_PROMPT = (
    "You are a retrieval query reformulator for multi-hop question answering. "
    "Output only a JSON array of search queries."
)
RS_MHR_MISSING_SYSTEM_PROMPT = (
    "You are a retrieval query reformulator for multi-hop question answering. "
    "Your task is to generate missing-hop search queries based on initially retrieved evidence. "
    "Output only a JSON array of search queries."
)
RS_MHR_RRF_WEIGHTS = {
    "original": 1.0,
    "static_qd": 1.0,
    "missing_hop": 1.2,
}
RS_MHR_PROVENANCE_BONUS = 0.01
RS_MHR_PROMPT_CACHE_OUTPUT_KEYS = {
    "rs_mhr_enabled",
    "route",
    "router_features",
    "planner_failed",
    "rs_mhr_cost",
    "retrieval_result",
    "retrieval_score",
    "adaptive_k",
    "adaptive_gap_index",
    "selection_method",
    "query_decomposition_enabled",
    "prompt",
}
EFC_PROBE_SYSTEM_PROMPT = (
    "You are given several retrieved Wikipedia passages and a question. "
    "Use at most two short evidence sentences. Keep the named bridge entity explicit. "
    "If the passages are insufficient, state the bridge entity and use unknown as the answer. "
    "For yes/no questions, answer yes when the compared attributes are the same and no when "
    "they differ. Always end exactly with: So the answer is <answer>."
    "\nThe following are given documents.\n\n{reference}"
)
EFC_PROBE_USER_PROMPT = "Question: {question}"
EFC_FINAL_SYSTEM_PROMPT = (
    "Answer the question based on the given document."
    "Only give me the answer and do not output any other words."
    "\nThe following are given documents.\n\n{reference}"
)
EFC_FINAL_USER_PROMPT = "Question: {question}"
EFC_MISSING_SYSTEM_PROMPT = (
    "You are a retrieval query reformulator for multi-hop question answering. "
    "Output only a JSON array containing one search query."
)
EFC_PROMPT_CACHE_OUTPUT_KEYS = {
    "efc_rag_enabled",
    "router_route",
    "route",
    "router_features",
    "probe_answer",
    "probe_context_doc_uids",
    "qd_queries",
    "missing_hop_query",
    "missing_hop_queries",
    "missing_query_strategy",
    "planner_failure_reason",
    "candidate_pool_count",
    "candidate_pool_summary",
    "selected_doc_uids",
    "selected_titles",
    "selected_sources",
    "role_scores_selected",
    "role_coverage",
    "role_coverage_ratio",
    "support_title_recall",
    "both_support_title_hit",
    "retrieval_result",
    "retrieval_score",
    "adaptive_k",
    "selection_method",
    "query_decomposition_enabled",
    "efc_cost",
    "prompt",
}


class TeeStream:
    def __init__(self, terminal_stream, log_stream):
        self.terminal_stream = terminal_stream
        self.log_stream = log_stream

    def write(self, data):
        self.terminal_stream.write(data)
        if not self.log_stream.closed:
            try:
                self.log_stream.write(data)
            except ValueError:
                pass
        return len(data)

    def flush(self):
        self.terminal_stream.flush()
        if not self.log_stream.closed:
            try:
                self.log_stream.flush()
            except ValueError:
                pass

    def isatty(self):
        return self.terminal_stream.isatty()

    def fileno(self):
        return self.terminal_stream.fileno()

    def __getattr__(self, name):
        return getattr(self.terminal_stream, name)


@contextmanager
def tee_run_output(log_path, stage):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        print(f"\n=== RACP {stage} started at {datetime.now().isoformat(timespec='seconds')} ===")
        print(f"Run log: {log_path}")
        status = "finished"
        try:
            yield
        except BaseException:
            status = "failed"
            traceback.print_exc()
            raise
        finally:
            print(
                f"=== RACP {stage} {status} at "
                f"{datetime.now().isoformat(timespec='seconds')} ==="
            )
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def get_run_log_path(config, args):
    if args.stage == "generate" and args.prompt_cache_path is not None:
        output_dir = Path(args.prompt_cache_path).resolve().parent
    else:
        output_dir = Path(config["save_dir"]).resolve()
    return output_dir / "run.log"


def none_or_int(value):
    if isinstance(value, int):
        return value
    value = value.strip().lower()
    if value in {"none", "null", "-1"}:
        return None
    return int(value)


def build_config_dict(args):
    if args.stage == "generate":
        config_dict = {"disable_save": True}
        if args.gpu_id is not None:
            config_dict["gpu_id"] = args.gpu_id
        if args.gpu_memory_utilization is not None:
            config_dict["gpu_memory_utilization"] = args.gpu_memory_utilization
        return config_dict

    config_dict = {
        "experiment_method": args.method,
        "refiner_name": None,
        "refiner_model_path": None,
        "use_reranker": not args.no_reranker,
        "retrieval_topk": args.retrieval_topk,
        "rerank_topk": args.rerank_topk,
        "racp_config": {
            "buffer": args.buffer,
            "max_k": args.max_k,
            "search_ratio": args.search_ratio,
            "selection_method": args.selection_method,
            "selection_topk": args.selection_topk,
            "mmr_lambda": args.mmr_lambda,
            "mmr_embedding_model_path": str(args.mmr_embedding_model_path)
            if args.mmr_embedding_model_path is not None
            else None,
        },
        "selection_method": args.selection_method,
        "selection_topk": args.selection_topk,
        "mmr_lambda": args.mmr_lambda,
        "mmr_embedding_model_path": str(args.mmr_embedding_model_path)
        if args.mmr_embedding_model_path is not None
        else None,
        "decomposition_config": {
            "enabled": args.use_query_decomposition,
            "subquery_num": 2,
            "subquery_topk": args.subquery_topk,
            "rerank_with_subqueries": args.qd_rerank_with_subqueries,
            "subquery_rerank_weight": args.qd_subquery_rerank_weight,
            "planner_max_tokens": args.planner_max_tokens,
            "planner_model": args.planner_model,
            "planner_model_path": str(args.planner_model_path)
            if args.planner_model_path is not None
            else None,
            "planner_gpu_memory_utilization": args.planner_gpu_memory_utilization,
            "planner_batch_size": args.planner_batch_size,
        },
        "save_note": args.save_note,
        "save_dir": str(args.save_dir),
        "strict_short_answer_prompt": args.strict_short_answer_prompt,
        "answer_only_prompt": args.answer_only_prompt,
        "exact_answer_prompt": args.exact_answer_prompt,
        "load_retrieval_topk_cache_path": str(args.load_retrieval_topk_cache_path)
        if args.load_retrieval_topk_cache_path is not None
        else None,
        "save_retrieval_topk_cache_path": str(args.save_retrieval_topk_cache_path)
        if args.save_retrieval_topk_cache_path is not None
        else None,
        "save_retrieval_cache": args.save_retrieval_cache,
        "use_retrieval_cache": args.use_retrieval_cache or args.retrieval_cache_only,
        "retrieval_batch_size": args.retrieval_batch_size,
        "gpu_id": args.gpu_id,
    }
    if args.rerank_model_name is not None:
        config_dict["rerank_model_name"] = args.rerank_model_name
        config_dict["rerank_model_path"] = (
            str(args.rerank_model_path) if args.rerank_model_path is not None else None
        )
    elif args.rerank_model_path is not None:
        config_dict["rerank_model_path"] = str(args.rerank_model_path)
    if args.retrieval_cache_path is not None:
        config_dict["retrieval_cache_path"] = str(args.retrieval_cache_path)
    if args.selection_method in {"topk", "title_dedup_topk"}:
        config_dict["metric_setting"] = {"retrieval_recall_topk": args.selection_topk}

    if args.enable_rs_mhr:
        config_dict["retrieval_topk"] = args.initial_topk
        config_dict["rerank_topk"] = args.initial_topk
        config_dict["metric_setting"] = {"retrieval_recall_topk": args.final_topk}
        config_dict["rs_mhr_config"] = {
            "enabled": True,
            "force_route": args.force_route,
            "initial_topk": args.initial_topk,
            "final_topk": args.final_topk,
            "seed_topk": args.seed_topk,
            "extra_topk": args.extra_topk,
            "static_qd_num": args.static_qd_num,
            "missing_query_num": args.missing_query_num or 1,
            "original_pool_topk": args.original_pool_topk,
            "rrf_k": args.rrf_k,
            "route_low_conf_gap": args.route_low_conf_gap,
            "route_avg_top5_thr": args.route_avg_top5_thr,
            "route_anchor_top1_thr": args.route_anchor_top1_thr,
            "route_unique_title_thr": args.route_unique_title_thr,
            "route_c_seed_quota": args.route_c_seed_quota,
            "route_c_missing_quota": args.route_c_missing_quota,
            "planner_model": args.planner_model,
            "planner_model_path": str(args.planner_model_path)
            if args.planner_model_path is not None
            else None,
            "planner_gpu_memory_utilization": args.planner_gpu_memory_utilization,
            "planner_batch_size": args.planner_batch_size,
            "planner_max_tokens": args.planner_max_tokens,
            "retrieval_cache_only": args.retrieval_cache_only,
            "save_router_debug": args.save_router_debug,
        }

    if args.enable_efc_rag:
        config_dict["use_reranker"] = False
        config_dict["refiner_name"] = None
        config_dict["refiner_model_path"] = None
        config_dict["retrieval_topk"] = args.initial_topk
        config_dict["rerank_topk"] = args.initial_topk
        config_dict["metric_setting"] = {"retrieval_recall_topk": args.final_topk}
        config_dict["efc_rag_config"] = {
            "enabled": True,
            "force_route": args.force_route,
            "initial_topk": args.initial_topk,
            "probe_topk": args.probe_topk,
            "final_topk": args.final_topk,
            "enable_static_qd_fallback": args.enable_static_qd_fallback,
            "qd_num": args.qd_num,
            "qd_topk": args.qd_topk,
            "enable_generation_guided": args.enable_generation_guided,
            "gen_topk": args.gen_topk,
            "missing_query_num": (
                args.missing_query_num or DEFAULT_RUN_CONFIG["missing_query_num"]
            ),
            "missing_query_mode": args.missing_query_mode,
            "rrf_k": args.rrf_k,
            "rrf_weight": args.rrf_weight,
            "role_weight": args.role_weight,
            "title_weight": args.title_weight,
            "source_weight": args.source_weight,
            "redundancy_weight": args.redundancy_weight,
            "title_dedup_soft": args.title_dedup_soft,
            "max_same_title": args.max_same_title,
            "original_seed_count": args.original_seed_count,
            "rrf_only_selection": args.rrf_only_selection,
            "static_bridge_evidence_topk": args.static_bridge_evidence_topk,
            "static_bridge_original_count": args.static_bridge_original_count,
            "static_bridge_qd_count": args.static_bridge_qd_count,
            "use_centroid_router": args.use_centroid_router,
            "save_efc_debug": args.save_efc_debug,
            "retrieval_cache_only": args.retrieval_cache_only,
            "planner_model": args.planner_model,
            "planner_model_path": str(args.planner_model_path)
            if args.planner_model_path is not None
            else None,
            "planner_gpu_memory_utilization": args.planner_gpu_memory_utilization,
            "planner_batch_size": args.planner_batch_size,
            "planner_inference_batch_size": args.planner_inference_batch_size,
            "planner_max_tokens": args.planner_max_tokens,
            "probe_max_tokens": args.probe_max_tokens,
        }

    if args.dataset_name is not None:
        config_dict["dataset_name"] = args.dataset_name
    if args.split is not None:
        config_dict["split"] = args.split
    if args.test_sample_num is not None:
        config_dict["test_sample_num"] = args.test_sample_num
    if args.gpu_memory_utilization is not None:
        config_dict["gpu_memory_utilization"] = args.gpu_memory_utilization

    return config_dict


def doc_to_text(doc):
    if isinstance(doc, str):
        return doc
    if not isinstance(doc, dict):
        return str(doc)

    parts = []
    for key in ("title", "contents", "text", "content"):
        value = doc.get(key)
        if value and value not in parts:
            parts.append(str(value))
    return "\n".join(parts)


def doc_title(doc):
    if isinstance(doc, dict):
        title = str(doc.get("title") or "").strip()
        if title:
            return title
        contents = ""
        for key in ("contents", "text", "content"):
            value = doc.get(key)
            if value:
                contents = str(value)
                break
    else:
        contents = str(doc)

    first_line, separator, _ = contents.partition("\n")
    if separator:
        first_line = first_line.strip().strip("\"'")
        if first_line:
            return first_line
    return contents[:80]


def normalized_doc_title(doc):
    return " ".join(doc_title(doc).lower().split())


def doc_dedupe_key(doc):
    if isinstance(doc, dict):
        return doc.get("id") or doc.get("contents") or doc.get("text") or doc.get("content") or str(doc)
    return str(doc)


def l2_normalize(embeddings):
    import numpy as np

    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {embeddings.shape}.")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-12, None)


def validate_docs_and_scores(docs, scores):
    docs = [] if docs is None else list(docs)
    if not docs:
        return docs, []
    if scores is None:
        raise ValueError("RACP document selection requires scores, but got None.")
    if len(scores) != len(docs):
        raise ValueError(
            f"RACP document selection got {len(docs)} docs but {len(scores)} scores."
        )

    import numpy as np

    try:
        score_array = np.asarray(scores, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("RACP document scores must be numeric.") from exc
    if score_array.ndim != 1:
        raise ValueError(f"RACP document scores must be 1D, got shape {score_array.shape}.")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("RACP document scores contain NaN or infinite values.")
    return docs, score_array.tolist()


def compute_gap_selection(docs, scores, racp_config):
    import numpy as np

    docs, scores = validate_docs_and_scores(docs, scores)
    if not docs:
        return [], [], [], 0, None

    buffer = racp_config.get("buffer", 5)
    search_ratio = racp_config.get("search_ratio", 0.9)
    max_k = racp_config.get("max_k", None)

    ranked = sorted(enumerate(zip(docs, scores)), key=lambda x: x[1][1], reverse=True)
    ranked_indices = [idx for idx, _ in ranked]
    ranked_docs = [doc for _, (doc, _) in ranked]
    ranked_scores = [float(score) for _, (_, score) in ranked]

    if len(ranked_docs) == 1:
        return ranked_docs, ranked_scores, ranked_indices, 1, None

    search_doc_num = int(np.ceil(len(ranked_scores) * search_ratio))
    search_doc_num = min(len(ranked_scores), max(2, search_doc_num))
    gap_search_scores = ranked_scores[:search_doc_num]
    gaps = [
        gap_search_scores[idx] - gap_search_scores[idx + 1]
        for idx in range(len(gap_search_scores) - 1)
    ]

    gap_idx = int(np.argmax(gaps))
    selected_k = max(1, gap_idx + 1 + buffer)
    if max_k is not None:
        selected_k = min(selected_k, max_k)
    selected_k = min(len(ranked_docs), max(1, selected_k))

    return ranked_docs, ranked_scores, ranked_indices, selected_k, gap_idx


def select_adaptive_docs(docs, scores, racp_config):
    ranked_docs, _, _, selected_k, gap_idx = compute_gap_selection(docs, scores, racp_config)
    if not ranked_docs:
        return [], 0, None

    return ranked_docs[:selected_k], selected_k, gap_idx


def select_topk_docs(docs, scores, racp_config):
    docs, scores = validate_docs_and_scores(docs, scores)
    if not docs:
        return [], 0, None, []

    selection_topk = int(racp_config.get("selection_topk", 8))
    if selection_topk < 1:
        raise ValueError(f"selection_topk must be positive, got {selection_topk}.")

    ranked = sorted(enumerate(zip(docs, scores)), key=lambda x: x[1][1], reverse=True)
    selected_k = min(len(ranked), selection_topk)
    selected = ranked[:selected_k]
    selected_docs = [doc for _, (doc, _) in selected]
    selected_indices = [idx for idx, _ in selected]
    return selected_docs, selected_k, None, selected_indices


def select_title_dedup_topk_docs(docs, scores, racp_config):
    docs, scores = validate_docs_and_scores(docs, scores)
    if not docs:
        return [], 0, None, []

    selection_topk = int(racp_config.get("selection_topk", 8))
    if selection_topk < 1:
        raise ValueError(f"selection_topk must be positive, got {selection_topk}.")

    ranked = sorted(enumerate(zip(docs, scores)), key=lambda x: x[1][1], reverse=True)
    selected = []
    deferred = []
    seen_titles = set()
    for item in ranked:
        _, (doc, _) = item
        title = normalized_doc_title(doc)
        if title and title not in seen_titles:
            selected.append(item)
            seen_titles.add(title)
        else:
            deferred.append(item)
        if len(selected) == selection_topk:
            break

    if len(selected) < selection_topk:
        selected.extend(deferred[: selection_topk - len(selected)])

    selected_docs = [doc for _, (doc, _) in selected]
    selected_indices = [idx for idx, _ in selected]
    return selected_docs, len(selected), None, selected_indices


def normalize_relevance_scores(scores):
    import numpy as np

    score_array = np.asarray(scores, dtype=np.float32)
    score_min = float(score_array.min())
    score_max = float(score_array.max())
    if score_max - score_min <= 1e-12:
        return np.ones_like(score_array, dtype=np.float32)
    return (score_array - score_min) / (score_max - score_min)


def encode_mmr_docs_with_retriever(texts, racp_config):
    retriever = racp_config.get("_retriever")
    encoder = getattr(retriever, "encoder", None) if retriever is not None else None
    if encoder is None:
        return None

    batch_size = racp_config.get("retrieval_batch_size") or 64
    old_silent = getattr(encoder, "silent", None)
    try:
        if old_silent is not None:
            encoder.silent = True
        return l2_normalize(encoder.encode(texts, batch_size=batch_size, is_query=False))
    except Exception as exc:
        raise ValueError(
            "MMR tried to reuse the current retriever encoder for document embeddings, "
            "but encoding failed."
        ) from exc
    finally:
        if old_silent is not None:
            encoder.silent = old_silent


def encode_mmr_docs_with_sentence_transformers(texts, racp_config):
    model_path = racp_config.get("mmr_embedding_model_path")
    if model_path is None:
        retrieval_method = racp_config.get("retrieval_method")
        retrieval_model_path = racp_config.get("retrieval_model_path")
        if retrieval_method not in {"bm25", "splade"} and retrieval_model_path:
            model_path = retrieval_model_path

    if not model_path:
        raise ValueError(
            "MMR needs document embeddings, but the current retriever has no encoder. "
            "Please set --mmr_embedding_model_path or configure mmr_embedding_model_path."
        )

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "MMR fallback embedding requires sentence-transformers. "
            "Install it or use a retriever that exposes an encoder."
        ) from exc

    model = racp_config.get("_mmr_embedding_model")
    if model is None:
        model = SentenceTransformer(model_path)
        racp_config["_mmr_embedding_model"] = model

    batch_size = racp_config.get("retrieval_batch_size") or 64
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return l2_normalize(embeddings)


def encode_mmr_docs(texts, racp_config):
    embeddings = encode_mmr_docs_with_retriever(texts, racp_config)
    if embeddings is not None:
        return embeddings
    return encode_mmr_docs_with_sentence_transformers(texts, racp_config)


def select_adaptive_mmr_docs(docs, scores, racp_config):
    ranked_docs, ranked_scores, ranked_indices, adaptive_k, gap_idx = compute_gap_selection(
        docs, scores, racp_config
    )
    if not ranked_docs:
        return [], 0, None, []

    import numpy as np

    mmr_lambda = float(racp_config.get("mmr_lambda", 0.7))
    if not 0.0 <= mmr_lambda <= 1.0:
        raise ValueError(f"mmr_lambda must be in [0, 1], got {mmr_lambda}.")

    adaptive_k = min(len(ranked_docs), max(1, adaptive_k))
    if adaptive_k == 1:
        return [ranked_docs[0]], adaptive_k, gap_idx, [ranked_indices[0]]

    relevance = normalize_relevance_scores(ranked_scores)
    doc_texts = [doc_to_text(doc) for doc in ranked_docs]
    doc_embeddings = encode_mmr_docs(doc_texts, racp_config)
    if len(doc_embeddings) != len(ranked_docs):
        raise ValueError(
            f"MMR embedding count mismatch: got {len(doc_embeddings)} embeddings "
            f"for {len(ranked_docs)} docs."
        )

    doc_doc_sim = doc_embeddings @ doc_embeddings.T
    selected = [0]
    candidates = set(range(1, len(ranked_docs)))

    while candidates and len(selected) < adaptive_k:
        best_idx = None
        best_score = None
        for candidate_idx in candidates:
            max_selected_sim = (
                float(np.max(doc_doc_sim[candidate_idx, selected])) if selected else 0.0
            )
            mmr_score = (
                mmr_lambda * float(relevance[candidate_idx])
                - (1.0 - mmr_lambda) * max_selected_sim
            )
            if best_score is None or mmr_score > best_score:
                best_score = mmr_score
                best_idx = candidate_idx

        selected.append(best_idx)
        candidates.remove(best_idx)

    selected_docs = [ranked_docs[idx] for idx in selected]
    selected_indices = [ranked_indices[idx] for idx in selected]
    return selected_docs, adaptive_k, gap_idx, selected_indices


def build_planner_prompt(question, subquery_num):
    return (
        "You are a retrieval query rewriter.\n\n"
        "Task:\n"
        "Rewrite the input question into exactly 2 independent search queries for retrieving evidence.\n\n"
        "Rules:\n"
        "- Output exactly one JSON array of strings.\n"
        "- Do not output markdown.\n"
        "- Do not output explanations.\n"
        "- Do not output labels.\n"
        "- Do not output examples.\n"
        "- Do not answer the question.\n"
        "- Each string must be a search query, not an answer.\n"
        "- Each query should be useful for searching Wikipedia-style passages.\n"
        "- Output exactly 2 strings even if the question mentions more than 2 entities.\n"
        "- Prefer concise entity + relation keywords when the relation is useful.\n"
        "- A named entity by itself is allowed when it is a useful Wikipedia lookup term.\n"
        "- Avoid yes/no or comparison queries such as \"Are they the same?\"\n"
        "- Use entity names or entity descriptions from the original question.\n"
        "- If the question is simple, rewrite it into 2 complementary search queries.\n\n"
        "Question:\n"
        f"{question}\n\n"
        "JSON array:"
    )


def is_search_like_query(query, allow_entity_only=False):
    import re

    normalized = " ".join(str(query).strip().split())
    if not normalized:
        return False

    lowered = normalized.lower().strip(" .?!")
    if lowered in {"yes", "no", "true", "false"}:
        return False

    one_word_answers = {
        "american",
        "british",
        "french",
        "german",
        "russian",
        "canadian",
        "australian",
        "english",
        "china",
        "japan",
        "india",
        "yes",
        "no",
    }
    if lowered in one_word_answers:
        return False

    tokens = re.findall(r"[A-Za-z0-9]+", normalized)
    min_token_count = 1 if allow_entity_only else 3
    if len(tokens) < min_token_count:
        return False
    if allow_entity_only and len(tokens) == 1:
        low_information_queries = {
            "actor",
            "actress",
            "album",
            "author",
            "book",
            "city",
            "country",
            "date",
            "director",
            "film",
            "location",
            "movie",
            "nationality",
            "school",
            "song",
            "university",
            "writer",
            "year",
        }
        if lowered in low_information_queries:
            return False

    return True


def extract_json_arrays(text):
    import json as json_module

    arrays = []
    start = None
    depth = 0
    in_string = False
    escape = False

    for idx, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "[":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "]" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : idx + 1]
                try:
                    parsed = json_module.loads(candidate)
                except json_module.JSONDecodeError:
                    start = None
                    continue
                if isinstance(parsed, list):
                    arrays.append(parsed)
                start = None

    return arrays


def normalize_subquery_list(candidates, question, subquery_num, min_query_num=None):
    min_query_num = subquery_num if min_query_num is None else min_query_num
    normalized_question = " ".join(question.lower().split())
    subqueries = []
    seen = set()

    for item in candidates:
        if isinstance(item, dict):
            item = (
                item.get("query")
                or item.get("subquery")
                or item.get("text")
                or item.get("search_query")
            )
        if not isinstance(item, str):
            continue
        subquery = " ".join(item.strip().strip("\"'`").split())
        normalized = subquery.lower()
        if not subquery or normalized == normalized_question or normalized in seen:
            continue
        if not is_search_like_query(subquery):
            continue
        seen.add(normalized)
        subqueries.append(subquery)
        if len(subqueries) == subquery_num:
            break

    return subqueries if len(subqueries) >= min_query_num else []


def parse_planner_output(output, question, subquery_num, min_query_num=None):
    min_query_num = subquery_num if min_query_num is None else min_query_num
    import json as json_module
    import re

    text = (output or "").strip()
    for array in extract_json_arrays(text):
        if len(array) < min_query_num:
            continue
        if any(not isinstance(item, str) or not item.strip() for item in array):
            continue
        subqueries = normalize_subquery_list(
            array,
            question,
            subquery_num,
            min_query_num=min_query_num,
        )
        if subqueries:
            return subqueries

    try:
        parsed = json_module.loads(text)
        if isinstance(parsed, list):
            candidates = parsed
        elif isinstance(parsed, dict):
            candidates = (
                parsed.get("subqueries")
                or parsed.get("queries")
                or parsed.get("query")
                or []
            )
            if isinstance(candidates, str):
                candidates = [candidates]
    except json_module.JSONDecodeError:
        candidates = []

    if not candidates:
        candidates = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("```"):
                continue
            line = re.sub(r"^\s*(?:[-*]|\d+[\).\:]?)\s*", "", line).strip()
            line = line.strip("\"'` ,")
            if not line or re.fullmatch(r"[\[\]\{\}]+", line):
                continue
            candidates.append(line)

    return normalize_subquery_list(
        candidates[:subquery_num],
        question,
        subquery_num,
        min_query_num=min_query_num,
    )


def normalize_qd_subquery_list(candidates, question, subquery_num):
    normalized_question = " ".join(question.lower().split())
    subqueries = []
    seen = set()

    for item in candidates:
        if not isinstance(item, str):
            continue
        subquery = " ".join(item.strip().strip("\"'`").split())
        normalized = subquery.lower()
        if (
            not subquery
            or any(char in subquery for char in "[]{}")
            or normalized == normalized_question
            or normalized in seen
        ):
            continue
        if not is_search_like_query(subquery, allow_entity_only=True):
            continue
        seen.add(normalized)
        subqueries.append(subquery)
        if len(subqueries) == subquery_num:
            break

    return subqueries


def parse_qd_planner_output(output, question, subquery_num):
    import json as json_module
    import re

    text = (output or "").strip()
    candidates = []
    try:
        parsed = json_module.loads(text)
        if isinstance(parsed, list):
            candidates = parsed
        elif isinstance(parsed, dict):
            candidates = parsed.get("subqueries") or parsed.get("queries") or []
    except json_module.JSONDecodeError:
        pass

    if not candidates:
        arrays = extract_json_arrays(text)
        candidates = [
            array[item_idx]
            for item_idx in range(max((len(array) for array in arrays), default=0))
            for array in arrays
            if item_idx < len(array)
        ]

    if not candidates:
        candidates = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', text)

    if not candidates:
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("```"):
                continue
            line = re.sub(r"^\s*(?:[-*]|\d+[\).\:]?)\s*", "", line).strip()
            line = line.strip("\"'` ,")
            if not line or re.fullmatch(r"[\[\]\{\}]+", line):
                continue
            candidates.append(line)

    return normalize_qd_subquery_list(candidates, question, subquery_num)


def build_chat_planner_prompts(prompts, planner_model_path, system_prompt=PLANNER_SYSTEM_PROMPT):
    if not planner_model_path:
        raise ValueError("planner_model_path is required for chat-template planner generation.")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(planner_model_path)
    chat_prompts = []
    for prompt in prompts:
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        chat_prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return chat_prompts


def generate_subqueries(questions, generator, decomposition_config):
    if generator is None:
        raise ValueError("Query decomposition requires a planner generator, but got None.")

    subquery_num = 2

    raw_prompts = [build_planner_prompt(question, subquery_num) for question in questions]
    planner_prompts = build_chat_planner_prompts(
        raw_prompts,
        decomposition_config.get("planner_model_path"),
    )
    planner_batch_size = int(decomposition_config.get("planner_batch_size", 32))
    planner_max_tokens = int(decomposition_config.get("planner_max_tokens", 96))
    if planner_batch_size < 1:
        raise ValueError(f"planner_batch_size must be positive, got {planner_batch_size}.")
    raw_outputs = []
    for start_idx in range(0, len(planner_prompts), planner_batch_size):
        raw_outputs.extend(
            generator.generate(
                planner_prompts[start_idx : start_idx + planner_batch_size],
                do_sample=False,
                temperature=0,
                max_new_tokens=planner_max_tokens,
                stop=list(PLANNER_STOP_WORDS),
            )
        )
    records = []
    for raw_output, question in zip(raw_outputs, questions):
        parsed_subqueries = parse_qd_planner_output(raw_output, question, subquery_num)
        valid = len(parsed_subqueries) == subquery_num
        partial = 0 < len(parsed_subqueries) < subquery_num
        records.append(
            {
                "raw_planner_output": raw_output,
                "parsed_subqueries": parsed_subqueries,
                "valid": valid,
                "partial": partial,
                "fallback": not parsed_subqueries,
            }
        )
    return records


def raw_batch_search(retriever, queries, topk):
    if not queries:
        return [], []
    results, scores = retriever._batch_search(query=queries, num=topk, return_score=True)
    if len(results) != len(scores):
        raise ValueError(
            f"Retriever returned {len(results)} raw result lists but {len(scores)} score lists."
        )
    return results, scores


def rs_mhr_raw_batch_search(retriever, queries, topk):
    """Raw retrieval with cache support, intentionally bypassing automatic reranking."""
    if not queries:
        return [], []

    use_cache = bool(getattr(retriever, "use_cache", False))
    save_cache = bool(getattr(retriever, "save_cache", False))
    cache = getattr(retriever, "cache", {}) if (use_cache or save_cache) else {}
    results = [None] * len(queries)
    scores = [None] * len(queries)
    missing_queries = []
    missing_indices = []

    for idx, query in enumerate(queries):
        cached_docs = cache.get(query) if use_cache else None
        if cached_docs is not None and len(cached_docs) >= topk:
            sliced_docs = cached_docs[:topk]
            results[idx] = sliced_docs
            scores[idx] = [float(doc["score"]) for doc in sliced_docs]
        else:
            missing_queries.append(query)
            missing_indices.append(idx)

    fetched_results, fetched_scores = raw_batch_search(retriever, missing_queries, topk)
    for idx, query, docs, doc_scores in zip(
        missing_indices, missing_queries, fetched_results, fetched_scores
    ):
        results[idx] = docs
        scores[idx] = doc_scores
        if use_cache or save_cache:
            cached_docs = []
            for doc, score in zip(docs, doc_scores):
                cached_doc = dict(doc) if isinstance(doc, dict) else {"contents": str(doc)}
                cached_doc["score"] = float(score)
                cached_docs.append(cached_doc)
            cache[query] = cached_docs

    if use_cache or save_cache:
        retriever.cache = cache
    return results, scores


def rs_mhr_doc_contents(doc):
    if isinstance(doc, str):
        return doc
    for key in ("contents", "text", "content"):
        value = doc.get(key)
        if value:
            return str(value)
    return ""


def rs_mhr_title(doc):
    return doc_title(doc)


def rs_mhr_doc_uid(doc):
    if isinstance(doc, dict):
        doc_id = doc.get("doc_id")
        if doc_id is None:
            doc_id = doc.get("id")
        if doc_id is not None:
            return str(doc_id)
        title = str(doc.get("title") or "")
    else:
        title = ""
    digest_input = title + rs_mhr_doc_contents(doc)
    return hashlib.md5(digest_input.encode("utf-8")).hexdigest()


def rs_mhr_source_query(
    source_type,
    query_text,
    query_id,
    retrieval_rank=None,
    retriever_score=None,
    rerank_rank=None,
    reranker_score=None,
):
    return {
        "source_type": source_type,
        "query_text": query_text,
        "query_id": query_id,
        "retrieval_rank": retrieval_rank,
        "retriever_score": retriever_score,
        "rerank_rank": rerank_rank,
        "reranker_score": reranker_score,
    }


def rs_mhr_normalize_doc(
    doc,
    retriever_score=None,
    retrieval_rank=None,
    source_query=None,
    initial_reranker_score=None,
    initial_rerank_rank=None,
):
    normalized = dict(doc) if isinstance(doc, dict) else {"contents": str(doc)}
    contents = rs_mhr_doc_contents(normalized)
    doc_id = normalized.get("doc_id")
    if doc_id is None:
        doc_id = normalized.get("id")
    normalized.update(
        {
            "doc_uid": rs_mhr_doc_uid(normalized),
            "doc_id": doc_id,
            "title": rs_mhr_title(normalized),
            "contents": contents,
            "text": str(normalized.get("text") or contents),
            "retriever_score": retriever_score,
            "retrieval_rank": retrieval_rank,
            "initial_reranker_score": initial_reranker_score,
            "initial_rerank_rank": initial_rerank_rank,
            "source_queries": [],
        }
    )
    if source_query is not None:
        normalized["source_queries"].append(source_query)
    return normalized


def rs_mhr_merge_doc(pool, doc):
    uid = doc["doc_uid"]
    if uid not in pool:
        pool[uid] = dict(doc)
        pool[uid]["source_queries"] = list(doc.get("source_queries", []))
        return

    existing = pool[uid]
    source_keys = {
        (source.get("source_type"), source.get("query_id"), source.get("query_text"))
        for source in existing["source_queries"]
    }
    for source in doc.get("source_queries", []):
        source_key = (source.get("source_type"), source.get("query_id"), source.get("query_text"))
        if source_key not in source_keys:
            existing["source_queries"].append(source)
            source_keys.add(source_key)


def rs_mhr_rerank(reranker, queries, doc_groups):
    if not queries:
        return [], []
    if any(not docs for docs in doc_groups):
        raise ValueError("RS-MHR reranking received an empty document group.")
    topk = max(len(docs) for docs in doc_groups)
    return reranker.rerank(queries, doc_groups, topk=topk)


def route_rs_mhr(question, r0_docs, rs_config):
    if not r0_docs:
        raise ValueError("RS-MHR routing requires at least one initially retrieved document.")

    scores = [float(doc["initial_reranker_score"]) for doc in r0_docs[:20]]
    score_min = min(scores)
    score_max = max(scores)
    if score_max > score_min:
        norm_scores = [(score - score_min) / (score_max - score_min) for score in scores]
    else:
        norm_scores = [0.5] * len(scores)

    def score_at(index):
        if index < len(norm_scores):
            return norm_scores[index]
        return norm_scores[-1]

    norm_s1 = score_at(0)
    norm_s5 = score_at(4)
    norm_s10 = score_at(9)
    avg_norm_top5 = sum(norm_scores[:5]) / min(5, len(norm_scores))
    top5_titles = [rs_mhr_title(doc) for doc in r0_docs[:5]]
    unique_title_ratio = len(set(top5_titles)) / max(1, len(top5_titles))

    lowered_question = f" {question.lower()} "
    comparison_markers = (
        " or ",
        " between ",
        "which",
        "more",
        "less",
        "earlier",
        "later",
        "first",
        "larger",
        "smaller",
        "same",
    )
    relation_markers = (
        "the director of",
        "the author of",
        "the wife of",
        "the husband of",
        "the father of",
        "the mother of",
        "who wrote",
        "who directed",
        "was born",
        "located in",
        "capital of",
        "spouse of",
    )
    is_comparison = any(marker in lowered_question for marker in comparison_markers)
    has_relation_chain = any(marker in lowered_question for marker in relation_markers)
    has_multihop_pattern = is_comparison or has_relation_chain
    norm_gap_5_10 = norm_s5 - norm_s10
    reliable_anchor = (
        norm_s1 >= float(rs_config["route_anchor_top1_thr"])
        or avg_norm_top5 >= float(rs_config["route_avg_top5_thr"])
    )
    evidence_concentrated = unique_title_ratio <= float(rs_config["route_unique_title_thr"])

    force_route = rs_config.get("force_route", "auto")
    if force_route != "auto":
        route = force_route
    elif (
        avg_norm_top5 >= float(rs_config["route_avg_top5_thr"])
        and norm_gap_5_10 >= float(rs_config["route_low_conf_gap"])
        and not (evidence_concentrated and has_multihop_pattern)
    ):
        route = "confident"
    elif reliable_anchor and has_multihop_pattern:
        route = "missing_hop"
    else:
        route = "static_qd"

    return {
        "norm_s1": norm_s1,
        "norm_s5": norm_s5,
        "norm_s10": norm_s10,
        "norm_gap_5_10": norm_gap_5_10,
        "avg_norm_top5": avg_norm_top5,
        "unique_title_ratio": unique_title_ratio,
        "is_comparison": is_comparison,
        "has_relation_chain": has_relation_chain,
        "has_multihop_pattern": has_multihop_pattern,
        "reliable_anchor": reliable_anchor,
        "evidence_concentrated": evidence_concentrated,
        "route": route,
    }


def build_rs_mhr_static_prompt(question, query_num):
    return (
        f"Given the original question, generate exactly {query_num} retrieval-oriented search queries.\n\n"
        "Rules:\n"
        "1. Each query should contain a concrete entity from the original question whenever possible.\n"
        "2. Each query should ask for one missing factual piece needed to answer the original question.\n"
        "3. Prefer search-engine style keyword queries over conversational questions.\n"
        "4. Do not answer the question.\n"
        "5. Do not explain.\n"
        "6. Output only a JSON array of strings.\n\n"
        f"Question:\n{question}\n\n"
        "Output:"
    )


def build_rs_mhr_missing_prompt(question, seed_docs, query_num):
    seed_parts = []
    for idx, doc in enumerate(seed_docs, start=1):
        evidence = rs_mhr_doc_contents(doc)[:300]
        seed_parts.append(f"[Doc {idx}]\nTitle: {rs_mhr_title(doc)}\nEvidence: {evidence}")
    seed_text = "\n\n".join(seed_parts)
    return (
        f"Original question:\n{question}\n\n"
        f"Initially retrieved evidence:\n{seed_text}\n\n"
        f"Generate exactly {query_num} missing-hop search query.\n\n"
        "Rules:\n"
        "1. Do not decompose the original question from scratch.\n"
        "2. Use the initially retrieved evidence to identify what fact is still missing.\n"
        "3. If the evidence reveals a bridge entity, include that entity in the query.\n"
        "4. The query should be retrieval-oriented and concise.\n"
        "5. Do not answer the question.\n"
        "6. Do not explain.\n"
        "7. Output only a JSON array of strings.\n\n"
        "Output:"
    )


def build_efc_static_bridge_prompt(question, seed_docs, query_num):
    seed_parts = []
    for idx, doc in enumerate(seed_docs, start=1):
        title = efc_doc_title(doc)
        evidence = doc_to_text(doc)[:360]
        seed_parts.append(f"[Doc {idx}]\nTitle: {title}\nEvidence: {evidence}")
    seed_text = "\n\n".join(seed_parts)
    return (
        f"Original question:\n{question}\n\n"
        f"Initially retrieved evidence:\n{seed_text}\n\n"
        f"Generate exactly {query_num} retrieval-oriented bridge queries.\n\n"
        "Rules:\n"
        "1. Do not decompose the question only from its surface wording.\n"
        "2. Use the initially retrieved titles and evidence to identify likely bridge entities.\n"
        "3. Each query should combine a concrete bridge entity with the missing target attribute.\n"
        "4. Prefer concise keyword queries over full questions.\n"
        "5. Do not answer the original question.\n"
        "6. Output only a JSON array of strings.\n\n"
        "Output:"
    )


def rs_mhr_query_too_similar(query, question):
    normalized_query = " ".join(query.lower().split())
    normalized_question = " ".join(question.lower().split())
    return SequenceMatcher(None, normalized_query, normalized_question).ratio() >= 0.9


def generate_rs_mhr_queries(
    questions,
    prompts,
    query_num,
    generator,
    planner_model_path,
    system_prompt,
    reject_similar=False,
    min_query_num=None,
    planner_batch_size=32,
    planner_max_tokens=96,
):
    min_query_num = query_num if min_query_num is None else min_query_num
    if not questions:
        return []
    if planner_batch_size < 1:
        raise ValueError(f"planner_batch_size must be positive, got {planner_batch_size}.")
    planner_prompts = build_chat_planner_prompts(prompts, planner_model_path, system_prompt)
    raw_outputs = []
    for start_idx in range(0, len(planner_prompts), planner_batch_size):
        raw_outputs.extend(
            generator.generate(
                planner_prompts[start_idx : start_idx + planner_batch_size],
                do_sample=False,
                temperature=0,
                max_new_tokens=planner_max_tokens,
                stop=list(PLANNER_STOP_WORDS),
            )
        )
    records = []
    for question, raw_output in zip(questions, raw_outputs):
        queries = parse_planner_output(
            raw_output,
            question,
            query_num,
            min_query_num=min_query_num,
        )
        failure_reason = ""
        if reject_similar:
            before_count = len(queries)
            queries = [
                query
                for query in queries
                if not rs_mhr_query_too_similar(query, question)
            ]
            if before_count and len(queries) < min_query_num:
                failure_reason = "query_too_similar"
        if not failure_reason and len(queries) < min_query_num:
            failure_reason = (
                "empty_or_invalid_output" if raw_output.strip() else "empty_output"
            )
        records.append(
            {
                "raw_planner_output": raw_output,
                "queries": queries,
                "valid": len(queries) >= min_query_num,
                "failure_reason": failure_reason,
            }
        )
    return records


def rs_mhr_initial_r0(question, raw_docs, raw_scores, reranked_docs, reranked_scores):
    retrieval_metadata = {}
    for rank, (doc, score) in enumerate(zip(raw_docs, raw_scores), start=1):
        retrieval_metadata[rs_mhr_doc_uid(doc)] = (rank, float(score))

    r0_docs = []
    for rerank_rank, (doc, reranker_score) in enumerate(
        zip(reranked_docs, reranked_scores), start=1
    ):
        uid = rs_mhr_doc_uid(doc)
        retrieval_rank, retriever_score = retrieval_metadata[uid]
        source_query = rs_mhr_source_query(
            "original",
            question,
            "original",
            retrieval_rank=retrieval_rank,
            retriever_score=retriever_score,
            rerank_rank=rerank_rank,
            reranker_score=float(reranker_score),
        )
        r0_docs.append(
            rs_mhr_normalize_doc(
                doc,
                retriever_score=retriever_score,
                retrieval_rank=retrieval_rank,
                source_query=source_query,
                initial_reranker_score=float(reranker_score),
                initial_rerank_rank=rerank_rank,
            )
        )
    return r0_docs


def rs_mhr_extra_docs(query_record, docs, scores, reranked_docs, reranked_scores):
    retrieval_metadata = {}
    for rank, (doc, score) in enumerate(zip(docs, scores), start=1):
        retrieval_metadata[rs_mhr_doc_uid(doc)] = (rank, float(score))

    normalized_docs = []
    for rerank_rank, (doc, reranker_score) in enumerate(
        zip(reranked_docs, reranked_scores), start=1
    ):
        uid = rs_mhr_doc_uid(doc)
        retrieval_rank, retriever_score = retrieval_metadata[uid]
        source_query = rs_mhr_source_query(
            query_record["source_type"],
            query_record["query_text"],
            query_record["query_id"],
            retrieval_rank=retrieval_rank,
            retriever_score=retriever_score,
            rerank_rank=rerank_rank,
            reranker_score=float(reranker_score),
        )
        normalized_docs.append(
            rs_mhr_normalize_doc(
                doc,
                retriever_score=retriever_score,
                retrieval_rank=retrieval_rank,
                source_query=source_query,
            )
        )
    return normalized_docs


def rs_mhr_candidate_summary(doc):
    rank_extra = {
        source["query_id"]: source.get("rerank_rank")
        for source in doc["source_queries"]
        if source["source_type"] != "original"
    }
    return {
        "doc_uid": doc["doc_uid"],
        "title": doc["title"],
        "source_types": sorted({source["source_type"] for source in doc["source_queries"]}),
        "source_queries": doc["source_queries"],
        "rrf_score": doc.get("rrf_score"),
        "rank_original": doc.get("rank_original"),
        "rank_extra": rank_extra,
    }


def rs_mhr_add_unique(selected_docs, docs, limit=None):
    selected_uids = {doc["doc_uid"] for doc in selected_docs}
    for doc in docs:
        if doc["doc_uid"] in selected_uids:
            continue
        selected_docs.append(doc)
        selected_uids.add(doc["doc_uid"])
        if limit is not None and len(selected_docs) >= limit:
            break


def rs_mhr_source_rank(doc, source_type):
    ranks = [
        source.get("rerank_rank")
        for source in doc["source_queries"]
        if source["source_type"] == source_type and source.get("rerank_rank") is not None
    ]
    return min(ranks) if ranks else 10**9


def rs_mhr_context_order(route, selected_docs):
    if route == "confident":
        return sorted(selected_docs, key=lambda doc: doc.get("initial_rerank_rank") or 10**9)
    if route in {"static_qd", "missing_hop_fallback_static_qd"}:
        def static_key(doc):
            original_rank = doc.get("initial_rerank_rank")
            static_sources = [
                source
                for source in doc["source_queries"]
                if source["source_type"] == "static_qd"
            ]
            if original_rank is not None:
                return (0, original_rank, 0)
            if static_sources:
                source = min(
                    static_sources,
                    key=lambda item: (item["query_id"], item.get("rerank_rank") or 10**9),
                )
                return (1, source["query_id"], source.get("rerank_rank") or 10**9)
            return (2, -float(doc.get("rrf_score", 0.0)), 0)

        return sorted(selected_docs, key=static_key)

    return sorted(
        selected_docs,
        key=lambda doc: (
            0 if doc.get("initial_rerank_rank") is not None else 1,
            doc.get("initial_rerank_rank")
            if doc.get("initial_rerank_rank") is not None
            else rs_mhr_source_rank(doc, "missing_hop"),
            -float(doc.get("rrf_score", 0.0)),
        ),
    )


def rs_mhr_fuse_record(record, rs_config):
    route = record["route"]
    r0_docs = record["r0_docs"]
    final_topk = int(rs_config["final_topk"])
    if route == "confident":
        selected_docs = r0_docs[:final_topk]
        for rank, doc in enumerate(r0_docs, start=1):
            doc["rank_original"] = rank
            doc["rrf_score"] = 1.0 / (int(rs_config["rrf_k"]) + rank)
        candidate_docs = r0_docs
    else:
        pool = {}
        for doc in r0_docs[: int(rs_config["original_pool_topk"])]:
            rs_mhr_merge_doc(pool, doc)
        for query_docs in record["extra_ranked_docs"]:
            for doc in query_docs:
                rs_mhr_merge_doc(pool, doc)
        candidate_docs = list(pool.values())

        reranked_pool, _ = rs_mhr_rerank(
            record["reranker"], [record["question"]], [candidate_docs]
        )
        rank_original = {
            doc["doc_uid"]: rank for rank, doc in enumerate(reranked_pool[0], start=1)
        }
        rrf_k = int(rs_config["rrf_k"])
        for doc in candidate_docs:
            doc["rank_original"] = rank_original[doc["doc_uid"]]
            rrf_score = RS_MHR_RRF_WEIGHTS["original"] / (rrf_k + doc["rank_original"])
            for source in doc["source_queries"]:
                if source["source_type"] == "original":
                    continue
                rerank_rank = source.get("rerank_rank")
                if rerank_rank is not None:
                    rrf_score += RS_MHR_RRF_WEIGHTS[source["source_type"]] / (
                        rrf_k + rerank_rank
                    )
            if len(doc["source_queries"]) > 1:
                rrf_score += RS_MHR_PROVENANCE_BONUS
            doc["rrf_score"] = rrf_score

        rrf_ranked_docs = sorted(
            candidate_docs,
            key=lambda doc: (-float(doc["rrf_score"]), doc["rank_original"]),
        )
        selected_docs = []
        if route == "missing_hop":
            rs_mhr_add_unique(
                selected_docs,
                [
                    pool[doc["doc_uid"]]
                    for doc in r0_docs[: int(rs_config["route_c_seed_quota"])]
                    if doc["doc_uid"] in pool
                ],
            )
            missing_ranked_docs = [
                pool[doc["doc_uid"]]
                for query_docs in record["extra_ranked_docs"]
                for doc in query_docs
                if doc["doc_uid"] in pool
                if any(
                    source["source_type"] == "missing_hop"
                    for source in doc["source_queries"]
                )
            ]
            rs_mhr_add_unique(
                selected_docs,
                missing_ranked_docs,
                limit=min(
                    final_topk,
                    len(selected_docs) + int(rs_config["route_c_missing_quota"]),
                ),
            )
        else:
            rs_mhr_add_unique(
                selected_docs,
                [pool[doc["doc_uid"]] for doc in r0_docs[:1] if doc["doc_uid"] in pool],
            )
        rs_mhr_add_unique(selected_docs, rrf_ranked_docs, limit=final_topk)
        selected_docs = selected_docs[:final_topk]

    selected_before_order = [doc["doc_uid"] for doc in selected_docs]
    context_docs = rs_mhr_context_order(route, selected_docs)
    record["candidate_docs"] = candidate_docs
    record["selected_docs"] = context_docs
    record["selected_doc_uids_before_order"] = selected_before_order
    record["selected_doc_uids_context_order"] = [doc["doc_uid"] for doc in context_docs]


def rs_mhr_plan_static_queries(records, planner_generator, planner_model_path, rs_config):
    if not records:
        return
    query_num = int(rs_config["static_qd_num"])
    prompts = [build_rs_mhr_static_prompt(record["question"], query_num) for record in records]
    planner_records = generate_rs_mhr_queries(
        [record["question"] for record in records],
        prompts,
        query_num,
        planner_generator,
        planner_model_path,
        RS_MHR_STATIC_SYSTEM_PROMPT,
        planner_batch_size=int(rs_config["planner_batch_size"]),
        planner_max_tokens=int(rs_config["planner_max_tokens"]),
    )
    for record, planner_record in zip(records, planner_records):
        record["planner_call_count"] += 1
        record["raw_planner_outputs"].append(planner_record["raw_planner_output"])
        if planner_record["valid"]:
            static_queries = planner_record["queries"]
        else:
            static_queries = [record["question"]]
            record["planner_failed"] = True
        record["static_queries"] = static_queries
        record["extra_queries"].extend(
            {
                "source_type": "static_qd",
                "query_text": query,
                "query_id": f"static_qd_{idx}",
            }
            for idx, query in enumerate(static_queries)
        )


def rs_mhr_plan_missing_queries(records, planner_generator, planner_model_path, rs_config):
    if not records:
        return []
    query_num = int(rs_config["missing_query_num"])
    for record in records:
        record["seed_docs"] = record["r0_docs"][: int(rs_config["seed_topk"])]
    prompts = [
        build_rs_mhr_missing_prompt(record["question"], record["seed_docs"], query_num)
        for record in records
    ]
    planner_records = generate_rs_mhr_queries(
        [record["question"] for record in records],
        prompts,
        query_num,
        planner_generator,
        planner_model_path,
        RS_MHR_MISSING_SYSTEM_PROMPT,
        reject_similar=True,
        planner_batch_size=int(rs_config["planner_batch_size"]),
        planner_max_tokens=int(rs_config["planner_max_tokens"]),
    )
    fallback_records = []
    for record, planner_record in zip(records, planner_records):
        record["planner_call_count"] += 1
        record["raw_planner_outputs"].append(planner_record["raw_planner_output"])
        if planner_record["valid"]:
            record["missing_hop_queries"] = planner_record["queries"]
            record["extra_queries"].extend(
                {
                    "source_type": "missing_hop",
                    "query_text": query,
                    "query_id": f"missing_hop_{idx}",
                }
                for idx, query in enumerate(planner_record["queries"])
            )
        else:
            record["planner_failed"] = True
            record["route"] = "missing_hop_fallback_static_qd"
            fallback_records.append(record)
    return fallback_records


def rs_mhr_save_retrieval_cache(retriever, update_source_cache=True):
    if getattr(retriever, "cache_only", False):
        return
    if (
        update_source_cache
        and getattr(retriever, "use_cache", False)
        and getattr(retriever, "cache_path", None)
    ):
        with Path(retriever.cache_path).open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(retriever.cache), f, ensure_ascii=False)
        print(f"Updated retrieval cache: {retriever.cache_path}")
    if getattr(retriever, "save_cache", False):
        retriever._save_cache()


class RS_MHR_CacheOnlyRetriever:
    """Use a complete raw retrieval cache without loading the corpus or FAISS index."""

    def __init__(self, config):
        from flashrag.utils import get_reranker

        cache_path = config["retrieval_cache_path"]
        if not cache_path:
            raise ValueError("--retrieval_cache_only requires --retrieval_cache_path.")
        with Path(cache_path).open("r", encoding="utf-8") as f:
            self.cache = json.load(f)
        self.cache_path = cache_path
        self.cache_only = True
        self.use_cache = True
        self.save_cache = False
        self.use_reranker = True
        self.reranker = get_reranker(config)
        print(f"Loaded RS-MHR cache-only retrieval data from: {cache_path}")

    def _batch_search(self, query, num, return_score=True):
        missing_queries = [item for item in query if item not in self.cache]
        raise ValueError(
            "RS-MHR retrieval cache is incomplete. Cache misses: "
            f"{missing_queries[:3]}. Re-run without --retrieval_cache_only to fill the cache."
        )


class EFC_CacheOnlyRetriever:
    """Use a complete EFC query cache without loading the corpus or FAISS index."""

    def __init__(self, config):
        cache_path = config["retrieval_cache_path"]
        if not cache_path:
            raise ValueError("--retrieval_cache_only requires --retrieval_cache_path.")
        with Path(cache_path).open("r", encoding="utf-8") as f:
            self.cache = json.load(f)
        self.cache_path = cache_path
        self.cache_only = True
        self.use_cache = True
        self.save_cache = False
        self.use_reranker = False
        self.reranker = None
        self.encoder = None
        print(f"Loaded EFC cache-only retrieval data from: {cache_path}")

    def _batch_search(self, query, num, return_score=True):
        missing_queries = [item for item in query if item not in self.cache]
        raise ValueError(
            "EFC retrieval cache is incomplete. Cache misses: "
            f"{missing_queries[:3]}. Re-run without --retrieval_cache_only and with "
            "--save_retrieval_cache to fill generated-query results."
        )


class QD_CacheOnlyRetriever:
    """Rerank a saved QD candidate bundle without loading the dense retriever."""

    def __init__(self, config):
        from flashrag.utils import get_reranker

        self.use_reranker = bool(config["use_reranker"])
        self.reranker = get_reranker(config) if self.use_reranker else None
        print("Using QD candidate cache without loading the corpus or FAISS index.")


def build_planner_config(config, planner_settings):
    planner_model = planner_settings.get("planner_model", "Llama-3.1-8B-Instruct")
    planner_model_path = planner_settings.get("planner_model_path")
    if planner_model_path is None:
        planner_model_path = config["model2path"].get(planner_model, planner_model)

    planner_config = dict(config.final_config if hasattr(config, "final_config") else config)
    planner_config.update(
        {
            "generator_model": planner_model,
            "generator_model_path": planner_model_path,
            "generator_max_input_len": 1024,
            "gpu_memory_utilization": float(
                planner_settings.get("planner_gpu_memory_utilization", 0.75)
            ),
            "generation_params": {
                "do_sample": False,
                "max_tokens": int(planner_settings.get("planner_max_tokens", 96)),
            },
        }
    )
    return planner_config


def build_rs_mhr_planner_config(config):
    return build_planner_config(config, config["rs_mhr_config"] or {})


def build_efc_planner_config(config):
    planner_config = build_planner_config(config, config["efc_rag_config"] or {})
    # The retriever has already initialized CUDA by this point. HF avoids vLLM
    # switching to spawn and re-executing the experiment entrypoint.
    planner_config["framework"] = "hf"
    planner_config["generator_batch_size"] = int(
        (config["efc_rag_config"] or {}).get("planner_inference_batch_size", 8)
    )
    return planner_config


def build_query_decomposition_planner_config(config):
    return build_planner_config(config, config["decomposition_config"] or {})


def log_planner_settings(label, planner_config, planner_settings):
    print(f"{label} planner settings:")
    print(f"  planner_model: {planner_config['generator_model']}")
    print(f"  planner_model_path: {planner_config['generator_model_path']}")
    print(f"  planner_batch_size: {planner_settings['planner_batch_size']}")
    print(f"  effective_generator_batch_size: {planner_config['generator_batch_size']}")
    print(f"  planner_max_tokens: {planner_settings['planner_max_tokens']}")
    print(f"  planner_gpu_memory_utilization: {planner_config['gpu_memory_utilization']}")


def get_rs_mhr_planner(config):
    from flashrag.utils import get_generator

    planner_config = build_rs_mhr_planner_config(config)
    log_planner_settings("RS-MHR", planner_config, config["rs_mhr_config"])
    return get_generator(planner_config)


def get_efc_planner(config):
    from flashrag.utils import get_generator

    planner_config = build_efc_planner_config(config)
    log_planner_settings("EFC-RAG", planner_config, config["efc_rag_config"])
    return get_generator(planner_config)


def get_query_decomposition_planner(config):
    from flashrag.utils import get_generator

    planner_config = build_query_decomposition_planner_config(config)
    log_planner_settings("Query decomposition", planner_config, config["decomposition_config"])
    return get_generator(planner_config)


def release_generator(generator):
    import gc

    model = getattr(generator, "model", None)
    engine = getattr(model, "llm_engine", None)
    engine_core = getattr(engine, "engine_core", None) if engine is not None else None
    if engine_core is not None and hasattr(engine_core, "shutdown"):
        engine_core.shutdown()
    elif engine is not None and hasattr(engine, "shutdown"):
        engine.shutdown()
    if hasattr(generator, "model"):
        generator.model = None
    del generator
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def build_efc_missing_prompt(question, probe_answer, query_num=1):
    if query_num == 1:
        return (
            "Given the original question and tentative reasoning, generate exactly one "
            "missing-hop search query.\n\n"
            "Rules:\n"
            "1. Use the bridge entity from the tentative reasoning if available.\n"
            "2. Ask for the missing factual attribute.\n"
            "3. Do not answer the question.\n"
            "4. Output only a JSON array with one string.\n\n"
            f"Question:\n{question}\n\n"
            f"Tentative reasoning:\n{probe_answer}\n\n"
            "Output:"
        )
    noun = "query" if query_num == 1 else "queries"
    return (
        "Given the original question and tentative reasoning, generate exactly "
        f"{query_num} missing-hop search {noun}.\n\n"
        "Rules:\n"
        "1. Use the bridge entity from the tentative reasoning if available.\n"
        "2. Ask for the missing factual attribute.\n"
        "3. Prefer concise keyword queries over full questions.\n"
        "4. Do not answer the question.\n"
        f"5. Output only a JSON array with {query_num} strings.\n\n"
        f"Question:\n{question}\n\n"
        f"Tentative reasoning:\n{probe_answer}\n\n"
        "Output:"
    )


def efc_normalize_docs(docs, scores, source, query, query_id):
    docs, scores = validate_docs_and_scores(docs, scores)
    return [
        efc_normalize_doc(doc, score, rank, source, query, query_id)
        for rank, (doc, score) in enumerate(zip(docs, scores), start=1)
    ]


def compute_efc_centroid_features(retriever, questions, r0_groups, enabled):
    if not enabled:
        return [{} for _ in questions]
    encoder = getattr(retriever, "encoder", None)
    if encoder is None:
        print("EFC centroid router requested, but the retriever exposes no encoder; skipping.")
        return [{} for _ in questions]

    import numpy as np

    batch_size = getattr(retriever, "batch_size", 64)
    question_embeddings = l2_normalize(
        encoder.encode(questions, batch_size=batch_size, is_query=True)
    )
    flat_docs = [doc_to_text(doc) for docs in r0_groups for doc in docs[:20]]
    group_sizes = [min(20, len(docs)) for docs in r0_groups]
    doc_embeddings = l2_normalize(
        encoder.encode(flat_docs, batch_size=batch_size, is_query=False)
    )
    features = []
    cursor = 0
    for question_embedding, group_size in zip(question_embeddings, group_sizes):
        group = doc_embeddings[cursor : cursor + group_size]
        cursor += group_size
        top5 = group[:5]
        if not len(top5):
            features.append({})
            continue
        centroid5 = top5.mean(axis=0)
        centroid5 /= max(float(np.linalg.norm(centroid5)), 1e-12)
        centroid_all = group.mean(axis=0)
        centroid_all /= max(float(np.linalg.norm(centroid_all)), 1e-12)
        features.append(
            {
                "q_centroid_sim": float(question_embedding @ centroid5),
                "top5_cohesion": float(np.mean(top5 @ centroid5)),
                "centroid_shift": float(1.0 - centroid5 @ centroid_all),
            }
        )
    return features


def efc_route_with_availability(route, features, efc_config):
    static_enabled = bool(efc_config.get("enable_static_qd_fallback", False))
    generation_enabled = bool(efc_config.get("enable_generation_guided", False))
    if route == "static_qd" and not static_enabled:
        if generation_enabled and features["y1_has_new_entity"]:
            return "generation_guided"
        return "direct"
    if route == "generation_guided" and not generation_enabled:
        return "static_qd" if static_enabled else "direct"
    return route


def efc_candidate_summary(candidate_pool):
    source_counts = Counter()
    for doc in candidate_pool:
        source_counts.update(doc.get("sources", []))
    return {
        "original": source_counts.get("original", 0),
        "qd": source_counts.get("qd", 0),
        "generation_guided": source_counts.get("generation_guided", 0),
        "merged_unique": len(candidate_pool),
    }


def efc_support_title_metrics(item, selected_docs):
    supporting_facts = item.metadata.get("supporting_facts", {}) if item.metadata else {}
    support_titles = supporting_facts.get("title", []) if isinstance(supporting_facts, dict) else []
    support_titles = {str(title).lower() for title in support_titles}
    if not support_titles:
        return None, None
    selected_titles = {efc_doc_title(doc).lower() for doc in selected_docs}
    hits = len(support_titles & selected_titles)
    return hits / len(support_titles), hits == len(support_titles)


def retrieve_with_efc(config, dataset, retriever, probe_generator):
    efc_config = config["efc_rag_config"] or {}
    questions = list(dataset.question)
    initial_topk = int(efc_config["initial_topk"])
    probe_topk = int(efc_config["probe_topk"])
    final_topk = int(efc_config["final_topk"])

    raw_results, raw_scores = rs_mhr_raw_batch_search(retriever, questions, initial_topk)
    r0_groups = [
        efc_normalize_docs(docs, scores, "original", question, "original")
        for question, docs, scores in zip(questions, raw_results, raw_scores)
    ]
    centroid_features = compute_efc_centroid_features(
        retriever,
        questions,
        r0_groups,
        bool(efc_config.get("use_centroid_router", False)),
    )

    probe_template = PromptTemplate(
        config,
        EFC_PROBE_SYSTEM_PROMPT,
        EFC_PROBE_USER_PROMPT,
    )
    probe_prompts = [
        probe_template.get_string(question=question, retrieval_result=docs[:probe_topk])
        for question, docs in zip(questions, r0_groups)
    ]
    print(f"EFC-RAG probe generation: {len(probe_prompts)} samples")
    probe_answers = probe_generator.generate(
        probe_prompts,
        do_sample=False,
        temperature=0,
        max_new_tokens=int(efc_config.get("probe_max_tokens", 96)),
    )
    release_generator(probe_generator)
    del probe_generator

    records = []
    force_route = efc_config.get("force_route", "auto")
    for item, question, r0_docs, probe_answer, centroid in zip(
        dataset, questions, r0_groups, probe_answers, centroid_features
    ):
        features = compute_router_features(
            question,
            r0_docs,
            probe_answer,
            centroid,
            assume_multihop=config["dataset_name"].lower()
            in {"hotpotqa", "2wikimultihopqa", "musique"},
        )
        router_route = decide_route(features, force_route=force_route)
        route = efc_route_with_availability(router_route, features, efc_config)
        records.append(
            {
                "item": item,
                "question": question,
                "r0_docs": r0_docs,
                "probe_answer": probe_answer,
                "features": features,
                "router_route": router_route,
                "route": route,
                "qd_queries": [],
                "missing_hop_query": "",
                "missing_hop_queries": [],
                "missing_query_strategy": "none",
                "query_records": [],
                "planner_call_count": 0,
                "planner_failed": False,
                "planner_failure_reason": "",
                "planner_raw_outputs": [],
            }
        )

    static_records = [record for record in records if record["route"] == "static_qd"]
    static_bridge_records = [
        record
        for record in static_records
        if record["features"].get("question_type") in {"bridge", "constraint"}
    ]
    static_bridge_record_ids = {id(record) for record in static_bridge_records}
    static_plain_records = [
        record for record in static_records if id(record) not in static_bridge_record_ids
    ]
    llm_missing_records = [
        record
        for record in records
        if record["route"] == "generation_guided"
        and efc_config.get("missing_query_mode", "heuristic") == "llm"
    ]
    planner_generator = None
    if static_records or llm_missing_records:
        planner_generator = get_efc_planner(config)
        planner_model_path = efc_config.get("planner_model_path")
        if planner_model_path is None:
            planner_model = efc_config.get("planner_model", "Llama-3.1-8B-Instruct")
            planner_model_path = config["model2path"].get(planner_model, planner_model)

        if static_plain_records:
            query_num = int(efc_config.get("qd_num", 2))
            planner_outputs = generate_rs_mhr_queries(
                [record["question"] for record in static_plain_records],
                [
                    build_rs_mhr_static_prompt(record["question"], query_num)
                    for record in static_plain_records
                ],
                query_num,
                planner_generator,
                planner_model_path,
                RS_MHR_STATIC_SYSTEM_PROMPT,
                planner_batch_size=int(efc_config["planner_batch_size"]),
                planner_max_tokens=int(efc_config["planner_max_tokens"]),
            )
            for record, planner_output in zip(static_plain_records, planner_outputs):
                record["planner_call_count"] += 1
                record["planner_raw_outputs"].append(
                    planner_output["raw_planner_output"]
                )
                if planner_output["valid"]:
                    record["qd_queries"] = planner_output["queries"]
                    record["missing_query_strategy"] = "static_qd"
                else:
                    record["planner_failed"] = True
                    record["planner_failure_reason"] = (
                        f"static_qd:{planner_output['failure_reason']}"
                    )
                    record["route"] = "direct"

        if static_bridge_records:
            query_num = int(efc_config.get("qd_num", 2))
            evidence_topk = int(efc_config.get("static_bridge_evidence_topk", 5))
            planner_outputs = generate_rs_mhr_queries(
                [record["question"] for record in static_bridge_records],
                [
                    build_efc_static_bridge_prompt(
                        record["question"],
                        record["r0_docs"][:evidence_topk],
                        query_num,
                    )
                    for record in static_bridge_records
                ],
                query_num,
                planner_generator,
                planner_model_path,
                RS_MHR_MISSING_SYSTEM_PROMPT,
                planner_batch_size=int(efc_config["planner_batch_size"]),
                planner_max_tokens=int(efc_config["planner_max_tokens"]),
            )
            for record, planner_output in zip(static_bridge_records, planner_outputs):
                record["planner_call_count"] += 1
                record["planner_raw_outputs"].append(
                    planner_output["raw_planner_output"]
                )
                if planner_output["valid"]:
                    record["qd_queries"] = planner_output["queries"]
                    record["missing_query_strategy"] = "static_qd_bridge_evidence"
                else:
                    record["planner_failed"] = True
                    record["planner_failure_reason"] = (
                        f"static_qd_bridge:{planner_output['failure_reason']}"
                    )
                    record["route"] = "direct"

        if llm_missing_records:
            missing_query_num = int(efc_config.get("missing_query_num", 1))
            planner_outputs = generate_rs_mhr_queries(
                [record["question"] for record in llm_missing_records],
                [
                    build_efc_missing_prompt(
                        record["question"],
                        record["probe_answer"],
                        missing_query_num,
                    )
                    for record in llm_missing_records
                ],
                missing_query_num,
                planner_generator,
                planner_model_path,
                EFC_MISSING_SYSTEM_PROMPT,
                reject_similar=True,
                min_query_num=1,
                planner_batch_size=int(efc_config["planner_batch_size"]),
                planner_max_tokens=int(efc_config["planner_max_tokens"]),
            )
            for record, planner_output in zip(llm_missing_records, planner_outputs):
                record["planner_call_count"] += 1
                record["planner_raw_outputs"].append(
                    planner_output["raw_planner_output"]
                )
                if planner_output["valid"]:
                    record["missing_hop_queries"] = planner_output["queries"]
                    record["missing_hop_query"] = record["missing_hop_queries"][0]
                    record["missing_query_strategy"] = "llm_missing_hop"
                else:
                    record["planner_failed"] = True
                    record["planner_failure_reason"] = (
                        f"missing_hop:{planner_output['failure_reason']}"
                    )
                    repaired_query = build_heuristic_missing_query(
                        record["question"],
                        record["features"],
                        [efc_doc_title(doc) for doc in record["r0_docs"]],
                        record["probe_answer"],
                    )
                    if repaired_query:
                        record["missing_hop_queries"] = [repaired_query]
                        record["missing_hop_query"] = repaired_query
                        record["missing_query_strategy"] = "heuristic_repair"
                    else:
                        record["route"] = (
                            "static_qd"
                            if efc_config.get("enable_static_qd_fallback", False)
                            else "direct"
                        )

        release_generator(planner_generator)
        del planner_generator

    # LLM missing-query failures can fall back to static QD after the first planner pass.
    late_static_records = [
        record
        for record in records
        if record["route"] == "static_qd" and not record["qd_queries"]
    ]
    late_static_bridge_records = [
        record
        for record in late_static_records
        if record["features"].get("question_type") in {"bridge", "constraint"}
    ]
    late_static_bridge_record_ids = {
        id(record) for record in late_static_bridge_records
    }
    late_static_plain_records = [
        record
        for record in late_static_records
        if id(record) not in late_static_bridge_record_ids
    ]
    if late_static_records:
        planner_generator = get_efc_planner(config)
        planner_model_path = efc_config.get("planner_model_path")
        if planner_model_path is None:
            planner_model = efc_config.get("planner_model", "Llama-3.1-8B-Instruct")
            planner_model_path = config["model2path"].get(planner_model, planner_model)
        query_num = int(efc_config.get("qd_num", 2))
        if late_static_plain_records:
            planner_outputs = generate_rs_mhr_queries(
                [record["question"] for record in late_static_plain_records],
                [
                    build_rs_mhr_static_prompt(record["question"], query_num)
                    for record in late_static_plain_records
                ],
                query_num,
                planner_generator,
                planner_model_path,
                RS_MHR_STATIC_SYSTEM_PROMPT,
                planner_batch_size=int(efc_config["planner_batch_size"]),
                planner_max_tokens=int(efc_config["planner_max_tokens"]),
            )
            for record, planner_output in zip(
                late_static_plain_records, planner_outputs
            ):
                record["planner_call_count"] += 1
                record["planner_raw_outputs"].append(
                    planner_output["raw_planner_output"]
                )
                if planner_output["valid"]:
                    record["qd_queries"] = planner_output["queries"]
                    record["missing_query_strategy"] = "static_qd_fallback"
                else:
                    record["planner_failed"] = True
                    late_reason = (
                        f"static_qd_fallback:{planner_output['failure_reason']}"
                    )
                    record["planner_failure_reason"] = ";".join(
                        reason
                        for reason in (
                            record["planner_failure_reason"],
                            late_reason,
                        )
                        if reason
                    )
                    record["route"] = "direct"
        if late_static_bridge_records:
            evidence_topk = int(efc_config.get("static_bridge_evidence_topk", 5))
            planner_outputs = generate_rs_mhr_queries(
                [record["question"] for record in late_static_bridge_records],
                [
                    build_efc_static_bridge_prompt(
                        record["question"],
                        record["r0_docs"][:evidence_topk],
                        query_num,
                    )
                    for record in late_static_bridge_records
                ],
                query_num,
                planner_generator,
                planner_model_path,
                RS_MHR_MISSING_SYSTEM_PROMPT,
                planner_batch_size=int(efc_config["planner_batch_size"]),
                planner_max_tokens=int(efc_config["planner_max_tokens"]),
            )
            for record, planner_output in zip(
                late_static_bridge_records, planner_outputs
            ):
                record["planner_call_count"] += 1
                record["planner_raw_outputs"].append(
                    planner_output["raw_planner_output"]
                )
                if planner_output["valid"]:
                    record["qd_queries"] = planner_output["queries"]
                    record["missing_query_strategy"] = (
                        "static_qd_bridge_evidence_fallback"
                    )
                else:
                    record["planner_failed"] = True
                    late_reason = (
                        f"static_qd_bridge_fallback:"
                        f"{planner_output['failure_reason']}"
                    )
                    record["planner_failure_reason"] = ";".join(
                        reason
                        for reason in (
                            record["planner_failure_reason"],
                            late_reason,
                        )
                        if reason
                    )
                    record["route"] = "direct"
        release_generator(planner_generator)
        del planner_generator

    for record in records:
        if record["route"] == "static_qd":
            record["query_records"] = [
                {
                    "source": "qd",
                    "query": query,
                    "query_id": f"qd_{idx}",
                    "topk": int(efc_config.get("qd_topk", 5)),
                }
                for idx, query in enumerate(record["qd_queries"])
            ]
        elif record["route"] == "generation_guided":
            mode = efc_config.get("missing_query_mode", "heuristic")
            if mode == "heuristic":
                record["missing_hop_query"] = build_heuristic_missing_query(
                    record["question"],
                    record["features"],
                    [efc_doc_title(doc) for doc in record["r0_docs"]],
                    record["probe_answer"],
                )
                record["missing_hop_queries"] = (
                    [record["missing_hop_query"]]
                    if record["missing_hop_query"]
                    else []
                )
                record["missing_query_strategy"] = "heuristic"
            elif mode == "raw_y1":
                record["missing_hop_query"] = (
                    f"{record['question']}\n{record['probe_answer']}"
                )
                record["missing_hop_queries"] = [record["missing_hop_query"]]
                record["missing_query_strategy"] = "raw_y1"
            if record["missing_hop_queries"]:
                record["query_records"] = [
                    {
                        "source": "generation_guided",
                        "query": query,
                        "query_id": f"gen_{idx}",
                        "topk": int(efc_config.get("gen_topk", 10)),
                    }
                    for idx, query in enumerate(record["missing_hop_queries"])
                ]
            else:
                record["route"] = (
                    "static_qd"
                    if efc_config.get("enable_static_qd_fallback", False)
                    and record["qd_queries"]
                    else "direct"
                )

    flat_query_records = [
        query_record for record in records for query_record in record["query_records"]
    ]
    flat_extra_results = []
    flat_extra_scores = []
    for topk in sorted({record["topk"] for record in flat_query_records}):
        indexed = [
            (idx, record)
            for idx, record in enumerate(flat_query_records)
            if record["topk"] == topk
        ]
        results, scores = rs_mhr_raw_batch_search(
            retriever, [record["query"] for _, record in indexed], topk
        )
        for (idx, _), docs, doc_scores in zip(indexed, results, scores):
            while len(flat_extra_results) <= idx:
                flat_extra_results.append(None)
                flat_extra_scores.append(None)
            flat_extra_results[idx] = docs
            flat_extra_scores[idx] = doc_scores

    cursor = 0
    for record in records:
        extra_groups = []
        for query_record in record["query_records"]:
            extra_groups.append(
                efc_normalize_docs(
                    flat_extra_results[cursor],
                    flat_extra_scores[cursor],
                    query_record["source"],
                    query_record["query"],
                    query_record["query_id"],
                )
            )
            cursor += 1
        candidate_pool = efc_merge_docs([record["r0_docs"]] + extra_groups)
        add_rrf_scores(candidate_pool, int(efc_config.get("rrf_k", 60)))
        if efc_config.get("rrf_only_selection", False):
            selected_docs = select_rrf_only(
                candidate_pool, final_topk, record["route"]
            )
            covered_roles = score_selected_roles(
                selected_docs,
                record["question"],
                record["probe_answer"],
                record["features"],
            )
        elif record["route"] == "direct":
            selected_docs = select_ranked_title_diverse(
                record["r0_docs"], final_topk, max_same_title=1
            )
            covered_roles = score_selected_roles(
                selected_docs,
                record["question"],
                record["probe_answer"],
                record["features"],
            )
        elif (
            record["route"] == "static_qd"
            and record["features"].get("question_type") in {"bridge", "constraint"}
        ):
            selected_docs, covered_roles = static_bridge_pack(
                candidate_pool,
                record["question"],
                record["probe_answer"],
                record["features"],
                final_topk,
                efc_config,
            )
        else:
            selected_docs, covered_roles = role_aware_pack(
                candidate_pool,
                record["question"],
                record["probe_answer"],
                record["features"],
                final_topk,
                record["route"],
                efc_config,
            )
        support_recall, both_support_hit = efc_support_title_metrics(
            record["item"], selected_docs
        )
        record.update(
            {
                "candidate_pool": candidate_pool,
                "selected_docs": selected_docs,
                "covered_roles": covered_roles,
                "support_title_recall": support_recall,
                "both_support_title_hit": both_support_hit,
            }
        )

    outputs = {
        "efc_rag_enabled": [],
        "router_route": [],
        "route": [],
        "router_features": [],
        "probe_answer": [],
        "probe_context_doc_uids": [],
        "qd_queries": [],
        "missing_hop_query": [],
        "missing_hop_queries": [],
        "missing_query_strategy": [],
        "planner_failure_reason": [],
        "planner_raw_outputs": [],
        "candidate_pool_count": [],
        "candidate_pool_summary": [],
        "selected_doc_uids": [],
        "selected_titles": [],
        "selected_sources": [],
        "role_scores_selected": [],
        "role_coverage": [],
        "role_coverage_ratio": [],
        "support_title_recall": [],
        "both_support_title_hit": [],
        "retrieval_result_full": [],
        "retrieval_score_full": [],
        "retrieval_result": [],
        "retrieval_score": [],
        "adaptive_k": [],
        "adaptive_gap_index": [],
        "selection_method": [],
        "query_decomposition_enabled": [],
        "efc_cost": [],
    }
    if efc_config.get("save_efc_debug", False):
        outputs["candidate_pool_debug"] = []

    for record in records:
        selected_docs = record["selected_docs"]
        outputs["efc_rag_enabled"].append(True)
        outputs["router_route"].append(record["router_route"])
        outputs["route"].append(record["route"])
        outputs["router_features"].append(record["features"])
        outputs["probe_answer"].append(record["probe_answer"])
        outputs["probe_context_doc_uids"].append(
            [doc["doc_uid"] for doc in record["r0_docs"][:probe_topk]]
        )
        outputs["qd_queries"].append(record["qd_queries"])
        outputs["missing_hop_query"].append(record["missing_hop_query"])
        outputs["missing_hop_queries"].append(record["missing_hop_queries"])
        outputs["missing_query_strategy"].append(record["missing_query_strategy"])
        outputs["planner_failure_reason"].append(record["planner_failure_reason"])
        outputs["planner_raw_outputs"].append(record["planner_raw_outputs"])
        outputs["candidate_pool_count"].append(len(record["candidate_pool"]))
        outputs["candidate_pool_summary"].append(
            efc_candidate_summary(record["candidate_pool"])
        )
        outputs["selected_doc_uids"].append([doc["doc_uid"] for doc in selected_docs])
        outputs["selected_titles"].append([efc_doc_title(doc) for doc in selected_docs])
        outputs["selected_sources"].append(
            [doc.get("sources", []) for doc in selected_docs]
        )
        outputs["role_scores_selected"].append(
            [doc.get("role_scores", {}) for doc in selected_docs]
        )
        outputs["role_coverage"].append(record["covered_roles"])
        outputs["role_coverage_ratio"].append(
            role_coverage_ratio(
                record["covered_roles"], record["features"]["question_type"]
            )
        )
        outputs["support_title_recall"].append(record["support_title_recall"])
        outputs["both_support_title_hit"].append(record["both_support_title_hit"])
        outputs["retrieval_result_full"].append(record["r0_docs"])
        outputs["retrieval_score_full"].append(
            [doc["retriever_scores"]["original"] for doc in record["r0_docs"]]
        )
        outputs["retrieval_result"].append(selected_docs)
        outputs["retrieval_score"].append(
            [float(doc.get("rrf_score", 0.0)) for doc in selected_docs]
        )
        outputs["adaptive_k"].append(len(selected_docs))
        outputs["adaptive_gap_index"].append(None)
        if efc_config.get("rrf_only_selection", False):
            selection_method = "efc_rrf_only"
        elif record["route"] == "direct":
            selection_method = "efc_ranked_title_diverse"
        else:
            selection_method = "efc_role_aware"
        outputs["selection_method"].append(selection_method)
        outputs["query_decomposition_enabled"].append(False)
        outputs["efc_cost"].append(
            {
                "probe_llm_calls": 1,
                "planner_llm_calls": record["planner_call_count"],
                "final_generation_calls": 1,
                "total_llm_calls": 2 + record["planner_call_count"],
                "initial_retrieval_calls": 1,
                "extra_retrieval_calls": len(record["query_records"]),
                "total_retrieval_calls": 1 + len(record["query_records"]),
                "candidate_pool_count": len(record["candidate_pool"]),
                "final_selected_count": len(selected_docs),
                "planner_failure_count": int(record["planner_failed"]),
                "planner_repair_count": int(
                    record["missing_query_strategy"] == "heuristic_repair"
                ),
            }
        )
        if efc_config.get("save_efc_debug", False):
            outputs["candidate_pool_debug"].append(record["candidate_pool"])

    for key, values in outputs.items():
        dataset.update_output(key, values)
    # Preserve the reusable source cache. If saving is enabled, FlashRAG writes
    # the merged source + newly generated query cache into this run's directory.
    rs_mhr_save_retrieval_cache(retriever, update_source_cache=False)
    return dataset


def log_efc_statistics(dataset, include_metrics=False):
    total = len(dataset)
    route_counts = Counter(dataset.route)
    router_route_values = (
        dataset.router_route
        if total and "router_route" in dataset[0].output
        else None
    )
    if router_route_values is not None:
        print("EFC-RAG router decisions:")
        for route, count in Counter(router_route_values).items():
            print(f"  {route}: {count} / {count / max(1, total):.4f}")
    print("EFC-RAG route statistics:")
    for route in ("direct", "static_qd", "generation_guided"):
        count = route_counts.get(route, 0)
        print(f"  {route}: {count} / {count / total if total else 0.0:.4f}")

    costs = dataset.efc_cost
    if total and "missing_query_strategy" in dataset[0].output:
        print("EFC-RAG missing-query strategies:")
        for strategy, count in Counter(dataset.missing_query_strategy).items():
            print(f"  {strategy}: {count} / {count / total:.4f}")
    if total and "planner_failure_reason" in dataset[0].output:
        failure_reasons = Counter(
            reason for reason in dataset.planner_failure_reason if reason
        )
        if failure_reasons:
            print("EFC-RAG planner failure reasons:")
            for reason, count in failure_reasons.items():
                print(f"  {reason}: {count}")
    print("EFC-RAG average costs:")
    for key in (
        "total_llm_calls",
        "total_retrieval_calls",
        "candidate_pool_count",
        "final_selected_count",
        "planner_failure_count",
        "planner_repair_count",
    ):
        average = (
            sum(float(cost.get(key, 0.0)) for cost in costs) / total if total else 0.0
        )
        print(f"  {key}: {average:.4f}")
    average_role_coverage = (
        sum(float(value) for value in dataset.role_coverage_ratio) / total if total else 0.0
    )
    title_diversity = (
        sum(len(set(titles)) / max(1, len(titles)) for titles in dataset.selected_titles) / total
        if total
        else 0.0
    )
    support_values = [value for value in dataset.support_title_recall if value is not None]
    both_values = [value for value in dataset.both_support_title_hit if value is not None]
    print(f"  role_coverage: {average_role_coverage:.4f}")
    print(f"  selected_title_unique_ratio: {title_diversity:.4f}")
    if support_values:
        print(f"  support_title_recall@{len(dataset.retrieval_result[0])}: "
              f"{sum(support_values) / len(support_values):.4f}")
    if both_values:
        print(f"  both_support_title_hit: {sum(bool(value) for value in both_values) / len(both_values):.4f}")

    if include_metrics:
        print("EFC-RAG route-wise metrics:")
        for route in ("direct", "static_qd", "generation_guided"):
            items = [item for item in dataset if item.route == route]
            if not items:
                continue
            scores = [item.output.get("metric_score", {}) for item in items]
            em = sum(float(score.get("em", 0.0)) for score in scores) / len(items)
            f1 = sum(float(score.get("f1", 0.0)) for score in scores) / len(items)
            print(f"  {route}: EM={em:.4f}, F1={f1:.4f}, samples={len(items)}")


def retrieve_with_rs_mhr(config, dataset, retriever, planner_generator):
    rs_config = config["rs_mhr_config"] or {}
    if not getattr(retriever, "use_reranker", False) or retriever.reranker is None:
        raise ValueError("RS-MHR requires an enabled reranker.")

    questions = list(dataset.question)
    initial_topk = int(rs_config["initial_topk"])
    extra_topk = int(rs_config["extra_topk"])
    raw_results, raw_scores = rs_mhr_raw_batch_search(retriever, questions, initial_topk)
    initial_docs, initial_scores = rs_mhr_rerank(retriever.reranker, questions, raw_results)

    records = []
    for question, docs, scores, reranked_docs, reranked_scores in zip(
        questions, raw_results, raw_scores, initial_docs, initial_scores
    ):
        r0_docs = rs_mhr_initial_r0(question, docs, scores, reranked_docs, reranked_scores)
        router_features = route_rs_mhr(question, r0_docs, rs_config)
        records.append(
            {
                "question": question,
                "r0_docs": r0_docs,
                "route": router_features["route"],
                "router_features": router_features,
                "seed_docs": [],
                "static_queries": [],
                "missing_hop_queries": [],
                "extra_queries": [],
                "extra_ranked_docs": [],
                "planner_failed": False,
                "planner_call_count": 0,
                "raw_planner_outputs": [],
                "reranker": retriever.reranker,
                "initial_retrieval_calls": 1,
                "reranker_pairs": len(docs),
            }
        )

    static_records = [record for record in records if record["route"] == "static_qd"]
    missing_records = [record for record in records if record["route"] == "missing_hop"]
    if (static_records or missing_records) and planner_generator is None:
        raise ValueError("RS-MHR routed samples need a planner generator.")
    planner_model_path = rs_config.get("planner_model_path")
    if planner_model_path is None:
        planner_model = rs_config.get("planner_model", "Llama-3.1-8B-Instruct")
        planner_model_path = config["model2path"].get(planner_model, planner_model)
    fallback_records = rs_mhr_plan_missing_queries(
        missing_records, planner_generator, planner_model_path, rs_config
    )
    rs_mhr_plan_static_queries(
        static_records + fallback_records,
        planner_generator,
        planner_model_path,
        rs_config,
    )

    flat_extra_queries = [
        query["query_text"] for record in records for query in record["extra_queries"]
    ]
    flat_extra_results, flat_extra_scores = rs_mhr_raw_batch_search(
        retriever, flat_extra_queries, extra_topk
    )
    if flat_extra_queries:
        flat_extra_reranked_docs, flat_extra_reranked_scores = rs_mhr_rerank(
            retriever.reranker, flat_extra_queries, flat_extra_results
        )
    else:
        flat_extra_reranked_docs, flat_extra_reranked_scores = [], []

    cursor = 0
    for record in records:
        for query_record in record["extra_queries"]:
            docs = flat_extra_results[cursor]
            scores = flat_extra_scores[cursor]
            reranked_docs = flat_extra_reranked_docs[cursor]
            reranked_scores = flat_extra_reranked_scores[cursor]
            record["extra_ranked_docs"].append(
                rs_mhr_extra_docs(query_record, docs, scores, reranked_docs, reranked_scores)
            )
            record["reranker_pairs"] += len(docs)
            cursor += 1

    for record in records:
        rs_mhr_fuse_record(record, rs_config)
        if record["route"] != "confident":
            record["reranker_pairs"] += len(record["candidate_docs"])

    outputs = {
        "rs_mhr_enabled": [],
        "route": [],
        "router_features": [],
        "initial_rerank_top20": [],
        "seed_docs": [],
        "static_queries": [],
        "missing_hop_queries": [],
        "extra_queries": [],
        "raw_planner_outputs": [],
        "planner_failed": [],
        "candidate_pool_count": [],
        "candidate_pool_summary": [],
        "selected_doc_uids_before_order": [],
        "selected_doc_uids_context_order": [],
        "final_selected_count": [],
        "retrieval_result_full": [],
        "retrieval_score_full": [],
        "retrieval_result": [],
        "retrieval_score": [],
        "adaptive_k": [],
        "adaptive_gap_index": [],
        "selection_method": [],
        "query_decomposition_enabled": [],
        "rs_mhr_cost": [],
    }
    if rs_config.get("save_router_debug"):
        outputs["candidate_pool_debug"] = []

    for record in records:
        candidate_docs = record["candidate_docs"]
        selected_docs = record["selected_docs"]
        outputs["rs_mhr_enabled"].append(True)
        outputs["route"].append(record["route"])
        outputs["router_features"].append(record["router_features"])
        outputs["initial_rerank_top20"].append(
            [
                {
                    "doc_uid": doc["doc_uid"],
                    "title": doc["title"],
                    "initial_rerank_rank": doc["initial_rerank_rank"],
                    "initial_reranker_score": doc["initial_reranker_score"],
                }
                for doc in record["r0_docs"][:20]
            ]
        )
        outputs["seed_docs"].append(
            [{"doc_uid": doc["doc_uid"], "title": doc["title"]} for doc in record["seed_docs"]]
        )
        outputs["static_queries"].append(record["static_queries"])
        outputs["missing_hop_queries"].append(record["missing_hop_queries"])
        outputs["extra_queries"].append(record["extra_queries"])
        outputs["raw_planner_outputs"].append(record["raw_planner_outputs"])
        outputs["planner_failed"].append(record["planner_failed"])
        outputs["candidate_pool_count"].append(len(candidate_docs))
        outputs["candidate_pool_summary"].append(
            [rs_mhr_candidate_summary(doc) for doc in candidate_docs]
        )
        outputs["selected_doc_uids_before_order"].append(
            record["selected_doc_uids_before_order"]
        )
        outputs["selected_doc_uids_context_order"].append(
            record["selected_doc_uids_context_order"]
        )
        outputs["final_selected_count"].append(len(selected_docs))
        outputs["retrieval_result_full"].append(record["r0_docs"])
        outputs["retrieval_score_full"].append(
            [doc["initial_reranker_score"] for doc in record["r0_docs"]]
        )
        outputs["retrieval_result"].append(selected_docs)
        outputs["retrieval_score"].append([doc["rrf_score"] for doc in selected_docs])
        outputs["adaptive_k"].append(len(selected_docs))
        outputs["adaptive_gap_index"].append(None)
        outputs["selection_method"].append("rs_mhr")
        outputs["query_decomposition_enabled"].append(False)
        outputs["rs_mhr_cost"].append(
            {
                "initial_retrieval_calls": record["initial_retrieval_calls"],
                "extra_retrieval_calls": len(record["extra_queries"]),
                "total_retrieval_calls": record["initial_retrieval_calls"]
                + len(record["extra_queries"]),
                "reranker_pairs": record["reranker_pairs"],
                "candidate_pool_count": len(candidate_docs),
                "final_selected_count": len(selected_docs),
                "planner_call_count": record["planner_call_count"],
                "planner_failure_count": int(record["planner_failed"]),
            }
        )
        if rs_config.get("save_router_debug"):
            outputs["candidate_pool_debug"].append(candidate_docs)

    for key, values in outputs.items():
        dataset.update_output(key, values)
    rs_mhr_save_retrieval_cache(retriever)
    return dataset


def log_rs_mhr_statistics(dataset, include_metrics=False):
    routes = dataset.route
    total = len(routes)
    route_counts = Counter(routes)
    route_names = [
        "confident",
        "static_qd",
        "missing_hop",
        "missing_hop_fallback_static_qd",
    ]
    print("Route statistics:")
    print(f"  total samples: {total}")
    for route in route_names:
        count = route_counts.get(route, 0)
        ratio = count / total if total else 0.0
        print(f"  {route}: {count} / {ratio:.4f}")

    costs = dataset.rs_mhr_cost
    cost_keys = [
        "initial_retrieval_calls",
        "extra_retrieval_calls",
        "total_retrieval_calls",
        "reranker_pairs",
        "candidate_pool_count",
        "final_selected_count",
    ]
    print("Cost statistics:")
    for key in cost_keys:
        average = sum(float(cost[key]) for cost in costs) / total if total else 0.0
        print(f"  avg {key}: {average:.4f}")
    print(f"  planner call count: {sum(cost['planner_call_count'] for cost in costs)}")
    print(f"  planner failure count: {sum(cost['planner_failure_count'] for cost in costs)}")

    if include_metrics:
        print("Route-wise metrics:")
        for route in route_names:
            items = [item for item in dataset if item.route == route]
            if not items:
                continue
            metric_scores = [item.output.get("metric_score", {}) for item in items]
            em = sum(float(scores.get("em", 0.0)) for scores in metric_scores) / len(items)
            f1 = sum(float(scores.get("f1", 0.0)) for scores in metric_scores) / len(items)
            print(f"  {route}: EM={em:.4f}, F1={f1:.4f}, samples={len(items)}")


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def bundle_cache_metadata(config, dataset, cache_type):
    return {
        "cache_type": cache_type,
        "dataset_name": config["dataset_name"],
        "split": config["split"],
        "test_sample_num": config["test_sample_num"] if "test_sample_num" in config else None,
        "retrieval_method": config["retrieval_method"],
        "retrieval_model_path": config["retrieval_model_path"],
        "index_path": config["index_path"],
        "corpus_path": config["corpus_path"],
        "retrieval_topk": config["retrieval_topk"],
        "rerank_topk": config["rerank_topk"],
        "use_reranker": config["use_reranker"],
        "use_query_decomposition": (config["decomposition_config"] or {}).get("enabled", False),
        "subquery_topk": (config["decomposition_config"] or {}).get("subquery_topk", 5),
        "questions": list(dataset.question),
    }


def save_bundle_cache(bundle, cache_path, config, dataset, cache_type):
    if cache_path is None:
        return
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": bundle_cache_metadata(config, dataset, cache_type),
        "bundle": to_jsonable(bundle),
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"Saved {cache_type} cache to: {cache_path}")


def load_bundle_cache(cache_path, dataset, expected_cache_type):
    cache_path = Path(cache_path)
    with cache_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    metadata = payload.get("metadata", {})
    cache_type = metadata.get("cache_type")
    if cache_type != expected_cache_type:
        raise ValueError(
            f"Expected a {expected_cache_type} cache, but got {cache_type!r} from {cache_path}."
        )

    cached_questions = metadata.get("questions")
    current_questions = list(dataset.question)
    if cached_questions != current_questions:
        raise ValueError(
            f"Cache questions do not match current dataset order/content: {cache_path}"
        )

    bundle = payload.get("bundle")
    if not isinstance(bundle, dict):
        raise ValueError(f"Cache bundle must be a dict: {cache_path}")

    print(f"Loaded {expected_cache_type} cache from: {cache_path}")
    return bundle


def merge_original_and_subquery_docs(original_docs, original_scores, subquery_docs, subquery_scores):
    merged = {}

    def add_docs(docs, scores, source):
        docs, scores = validate_docs_and_scores(docs, scores)
        for rank, (doc, score) in enumerate(zip(docs, scores)):
            key = doc_dedupe_key(doc)
            score = float(score)
            if key not in merged or score > merged[key]["score"]:
                merged[key] = {
                    "doc": doc,
                    "score": score,
                    "source": source,
                    "rank": rank,
                }

    add_docs(original_docs, original_scores, "original")
    for idx, (docs, scores) in enumerate(zip(subquery_docs, subquery_scores)):
        add_docs(docs, scores, f"subquery_{idx}")

    merged_items = sorted(
        merged.values(),
        key=lambda item: (item["score"], -item["rank"], item["source"] == "original"),
        reverse=True,
    )
    return [item["doc"] for item in merged_items], [item["score"] for item in merged_items]


def rerank_all_docs(reranker, queries, doc_groups):
    if not queries:
        return [], []
    if len(queries) != len(doc_groups):
        raise ValueError(
            f"QD reranking received {len(queries)} queries but {len(doc_groups)} document groups."
        )
    if any(not docs for docs in doc_groups):
        raise ValueError("QD reranking received an empty document group.")

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return reranker.rerank(queries, doc_groups, topk=max(len(docs) for docs in doc_groups))


def normalized_score_map(docs, scores):
    docs, scores = validate_docs_and_scores(docs, scores)
    if not docs:
        return {}
    normalized_scores = normalize_relevance_scores(scores)
    return {
        doc_dedupe_key(doc): float(score)
        for doc, score in zip(docs, normalized_scores)
    }


def align_scores_to_docs(docs, scored_docs, scores):
    scored_docs, scores = validate_docs_and_scores(scored_docs, scores)
    score_map = {
        doc_dedupe_key(doc): float(score)
        for doc, score in zip(scored_docs, scores)
    }
    try:
        return [score_map[doc_dedupe_key(doc)] for doc in docs]
    except KeyError as exc:
        raise ValueError("QD reranker score alignment could not find a retrieved document.") from exc


def fuse_qd_rerank_scores(
    original_docs,
    original_scores,
    subquery_doc_groups,
    subquery_score_groups,
    subquery_weight,
):
    original_docs, original_scores = validate_docs_and_scores(original_docs, original_scores)
    if len(subquery_doc_groups) != len(subquery_score_groups):
        raise ValueError(
            "QD score fusion received a different number of subquery document and score groups."
        )
    if subquery_weight < 0:
        raise ValueError(f"QD subquery rerank weight must be non-negative, got {subquery_weight}.")

    original_score_map = normalized_score_map(original_docs, original_scores)
    subquery_score_maps = [
        normalized_score_map(docs, scores)
        for docs, scores in zip(subquery_doc_groups, subquery_score_groups)
    ]
    fused_scores = []
    qd_boosts = []
    for doc in original_docs:
        key = doc_dedupe_key(doc)
        qd_boost = max(
            (score_map.get(key, 0.0) for score_map in subquery_score_maps),
            default=0.0,
        )
        fused_scores.append(original_score_map[key] + subquery_weight * qd_boost)
        qd_boosts.append(qd_boost)

    ranked = sorted(
        zip(original_docs, fused_scores, qd_boosts),
        key=lambda item: item[1],
        reverse=True,
    )
    return (
        [doc for doc, _, _ in ranked],
        [float(score) for _, score, _ in ranked],
        [float(boost) for _, _, boost in ranked],
    )


def rerank_with_qd_subqueries(
    reranker,
    questions,
    merged_results,
    planner_subqueries,
    grouped_subquery_results,
    rerank_topk,
    subquery_weight,
):
    original_docs, original_scores = rerank_all_docs(reranker, questions, merged_results)

    flat_subqueries = []
    flat_subquery_docs = []
    for subqueries, subquery_docs in zip(planner_subqueries, grouped_subquery_results):
        if len(subqueries) != len(subquery_docs):
            raise ValueError(
                "QD subquery reranking received a different number of queries and document groups."
            )
        flat_subqueries.extend(subqueries)
        flat_subquery_docs.extend(subquery_docs)
    flat_subquery_results, flat_subquery_scores = rerank_all_docs(
        reranker, flat_subqueries, flat_subquery_docs
    )

    grouped_subquery_rerank_results = []
    grouped_subquery_rerank_scores = []
    cursor = 0
    for subqueries in planner_subqueries:
        next_cursor = cursor + len(subqueries)
        grouped_subquery_rerank_results.append(flat_subquery_results[cursor:next_cursor])
        grouped_subquery_rerank_scores.append(flat_subquery_scores[cursor:next_cursor])
        cursor = next_cursor

    retrieval_results = []
    retrieval_scores = []
    retrieval_qd_boosts = []
    merged_original_rerank_scores = []
    aligned_subquery_rerank_scores = []
    for sample_idx, (docs, scores, sub_docs, sub_scores) in enumerate(
        zip(
            original_docs,
            original_scores,
            grouped_subquery_rerank_results,
            grouped_subquery_rerank_scores,
        )
    ):
        docs, scores, qd_boosts = fuse_qd_rerank_scores(
            docs,
            scores,
            sub_docs,
            sub_scores,
            subquery_weight,
        )
        retrieval_results.append(docs[:rerank_topk])
        retrieval_scores.append(scores[:rerank_topk])
        retrieval_qd_boosts.append(qd_boosts[:rerank_topk])
        merged_original_rerank_scores.append(
            align_scores_to_docs(
                merged_results[sample_idx],
                original_docs[sample_idx],
                original_scores[sample_idx],
            )
        )
        aligned_subquery_rerank_scores.append(
            [
                align_scores_to_docs(source_docs, reranked_docs, reranked_scores)
                for source_docs, reranked_docs, reranked_scores in zip(
                    grouped_subquery_results[sample_idx],
                    sub_docs,
                    sub_scores,
                )
            ]
        )

    return {
        "retrieval_result_full": retrieval_results,
        "retrieval_score_full": retrieval_scores,
        "retrieval_qd_boost_full": retrieval_qd_boosts,
        "merged_original_rerank_score": merged_original_rerank_scores,
        "subquery_rerank_score": aligned_subquery_rerank_scores,
    }


def retrieve_with_query_decomposition(config, dataset, retriever, planner_generator):
    decomposition_config = config["decomposition_config"] or {}
    decomposition_config["planner_model_path"] = (
        decomposition_config.get("planner_model_path")
        or config["model2path"].get(
            decomposition_config.get("planner_model", "Llama-3.1-8B-Instruct"),
            decomposition_config.get("planner_model", "Llama-3.1-8B-Instruct"),
        )
    )
    subquery_topk = int(decomposition_config.get("subquery_topk", 5))
    if subquery_topk < 1:
        raise ValueError(f"subquery_topk must be positive, got {subquery_topk}.")

    questions = list(dataset.question)
    load_retrieval_topk_cache_path = config["load_retrieval_topk_cache_path"]
    if load_retrieval_topk_cache_path:
        retrieval_bundle = load_bundle_cache(
            load_retrieval_topk_cache_path, dataset, "retrieval_topk"
        )
        merged_results = retrieval_bundle["merged_retrieval_result"]
        merged_scores = retrieval_bundle["merged_retrieval_score"]
    else:
        planner_records = generate_subqueries(questions, planner_generator, decomposition_config)
        planner_subqueries = [record["parsed_subqueries"] for record in planner_records]
        raw_planner_outputs = [record["raw_planner_output"] for record in planner_records]
        planner_valid = [record["valid"] for record in planner_records]
        planner_partial = [record["partial"] for record in planner_records]
        planner_fallback = [record["fallback"] for record in planner_records]
        planner_accepted_subquery_count = [
            len(record["parsed_subqueries"]) for record in planner_records
        ]

        original_results, original_scores = raw_batch_search(
            retriever, questions, int(config["retrieval_topk"])
        )

        flat_subqueries = [subquery for subqueries in planner_subqueries for subquery in subqueries]
        flat_results, flat_scores = raw_batch_search(retriever, flat_subqueries, subquery_topk)

        grouped_subquery_results = []
        grouped_subquery_scores = []
        cursor = 0
        for subqueries in planner_subqueries:
            next_cursor = cursor + len(subqueries)
            grouped_subquery_results.append(flat_results[cursor:next_cursor])
            grouped_subquery_scores.append(flat_scores[cursor:next_cursor])
            cursor = next_cursor

        merged_results = []
        merged_scores = []
        for docs, scores, sub_docs, sub_scores in zip(
            original_results, original_scores, grouped_subquery_results, grouped_subquery_scores
        ):
            docs, scores = merge_original_and_subquery_docs(docs, scores, sub_docs, sub_scores)
            merged_results.append(docs)
            merged_scores.append(scores)

        retrieval_bundle = {
            "raw_planner_output": raw_planner_outputs,
            "parsed_subqueries": planner_subqueries,
            "valid": planner_valid,
            "partial": planner_partial,
            "fallback": planner_fallback,
            "planner_subqueries": planner_subqueries,
            "planner_valid": planner_valid,
            "planner_partial": planner_partial,
            "planner_fallback": planner_fallback,
            "planner_accepted_subquery_count": planner_accepted_subquery_count,
            "original_retrieval_result": original_results,
            "original_retrieval_score": original_scores,
            "subquery_retrieval_result": grouped_subquery_results,
            "subquery_retrieval_score": grouped_subquery_scores,
            "merged_retrieval_result": merged_results,
            "merged_retrieval_score": merged_scores,
        }
        save_bundle_cache(
            retrieval_bundle,
            config["save_retrieval_topk_cache_path"],
            config,
            dataset,
            "retrieval_topk",
        )

    if getattr(retriever, "use_reranker", False):
        if retriever.reranker is None:
            raise ValueError("use_reranker is True, but retriever.reranker is None.")
        rerank_topk = min(int(config["rerank_topk"]), min(len(docs) for docs in merged_results))
        if rerank_topk > 0:
            if decomposition_config.get("rerank_with_subqueries", False):
                reranker_bundle = rerank_with_qd_subqueries(
                    retriever.reranker,
                    questions,
                    merged_results,
                    retrieval_bundle["planner_subqueries"],
                    retrieval_bundle["subquery_retrieval_result"],
                    rerank_topk,
                    float(decomposition_config.get("subquery_rerank_weight", 0.25)),
                )
                retrieval_results = reranker_bundle["retrieval_result_full"]
                retrieval_scores = reranker_bundle["retrieval_score_full"]
            else:
                retrieval_results, retrieval_scores = retriever.reranker.rerank(
                    questions,
                    merged_results,
                    topk=rerank_topk,
                )
                reranker_bundle = {}
        else:
            retrieval_results, retrieval_scores = merged_results, merged_scores
            reranker_bundle = {}
    else:
        retrieval_topk = int(config["retrieval_topk"])
        retrieval_results = [docs[:retrieval_topk] for docs in merged_results]
        retrieval_scores = [scores[:retrieval_topk] for scores in merged_scores]
        reranker_bundle = {}

    reranker_bundle = {
        **retrieval_bundle,
        **reranker_bundle,
        "retrieval_result_full": retrieval_results,
        "retrieval_score_full": retrieval_scores,
    }
    return reranker_bundle


def log_query_decomposition_statistics(dataset):
    total = len(dataset)
    full_count = 0
    partial_count = 0
    fallback_count = 0
    accepted_subquery_count = 0
    for item in dataset:
        output = item.output
        subqueries = output.get("planner_subqueries", output.get("parsed_subqueries", []))
        accepted_count = len(subqueries)
        accepted_subquery_count += accepted_count
        if output.get("planner_valid", output.get("valid", accepted_count == 2)):
            full_count += 1
        elif output.get("planner_partial", output.get("partial", accepted_count > 0)):
            partial_count += 1
        else:
            fallback_count += 1

    print("Query decomposition statistics:")
    print(f"  total samples: {total}")
    print(f"  full: {full_count} / {full_count / total if total else 0.0:.4f}")
    print(f"  partial: {partial_count} / {partial_count / total if total else 0.0:.4f}")
    print(f"  fallback: {fallback_count} / {fallback_count / total if total else 0.0:.4f}")
    print(
        f"  avg accepted_subqueries: "
        f"{accepted_subquery_count / total if total else 0.0:.4f}"
    )


def load_split(config, split):
    if split is None:
        split = config["split"]
    if isinstance(split, (list, tuple)):
        if len(split) != 1:
            raise ValueError(f"Standalone RACP expects exactly one split, got {split}.")
        split = split[0]
    all_split = get_dataset(config)
    data = all_split[split]
    if data is None:
        available = sorted(path.stem for path in Path(config["dataset_path"]).glob("*.jsonl"))
        raise FileNotFoundError(
            f"Split '{split}' not found for dataset '{config['dataset_name']}'. "
            f"Available splits: {available}"
        )
    return data


def build_answer_prompt_template(config):
    strict_short_answer_prompt = (
        config["strict_short_answer_prompt"] if "strict_short_answer_prompt" in config else False
    )
    answer_only_prompt = config["answer_only_prompt"] if "answer_only_prompt" in config else False
    exact_answer_prompt = config["exact_answer_prompt"] if "exact_answer_prompt" in config else False
    if strict_short_answer_prompt:
        return PromptTemplate(
            config,
            STRICT_SHORT_ANSWER_SYSTEM_PROMPT,
            STRICT_SHORT_ANSWER_USER_PROMPT,
        )
    if exact_answer_prompt:
        return PromptTemplate(config, EXACT_ANSWER_SYSTEM_PROMPT, EXACT_ANSWER_USER_PROMPT)
    if answer_only_prompt:
        return PromptTemplate(config, ANSWER_ONLY_SYSTEM_PROMPT, ANSWER_ONLY_USER_PROMPT)
    if (config["efc_rag_config"] or {}).get("enabled", False):
        return PromptTemplate(config, EFC_FINAL_SYSTEM_PROMPT, EFC_FINAL_USER_PROMPT)
    return PromptTemplate(config)


def build_final_prompts(config, dataset):
    prompt_template = build_answer_prompt_template(config)
    if not config["use_fid"]:
        input_prompts = [
            prompt_template.get_string(question=q, retrieval_result=r)
            for q, r in zip(dataset.question, dataset.retrieval_result)
        ]

    if config["use_fid"]:
        print("Use FiD generation")
        input_prompts = []
        for item in dataset:
            q = item.question
            docs = item.retrieval_result
            input_prompts.append([q + " " + doc["contents"] for doc in docs])

    dataset.update_output("prompt", input_prompts)
    return dataset


def log_selection_settings(config, retriever):
    racp_config = config["racp_config"] or {}
    decomposition_config = config["decomposition_config"] or {}
    rs_mhr_config = config["rs_mhr_config"] or {}
    efc_config = config["efc_rag_config"] or {}
    use_reranker = getattr(retriever, "use_reranker", config["use_reranker"])
    retrieval_top_n = config["rerank_topk"] if use_reranker else config["retrieval_topk"]

    print("RACP selection settings:")
    print(f"  selection_method: {racp_config.get('selection_method', 'gap')}")
    print(f"  search_ratio: {racp_config.get('search_ratio', 0.9)}")
    print(f"  buffer: {racp_config.get('buffer', 5)}")
    print(f"  max_k: {racp_config.get('max_k', None)}")
    print(f"  selection_topk: {racp_config.get('selection_topk', 8)}")
    print(f"  mmr_lambda: {racp_config.get('mmr_lambda', 0.7)}")
    print(f"  use_reranker: {use_reranker}")
    print(f"  rerank_model_name: {config['rerank_model_name']}")
    print(f"  rerank_model_path: {config['rerank_model_path']}")
    print(f"  retrieval_top_n: {retrieval_top_n}")
    print(f"  dataset_name: {config['dataset_name']}")
    print(f"  split: {config['split']}")
    print(f"  generator_model: {config['generator_model']}")
    print(
        f"  strict_short_answer_prompt: "
        f"{config['strict_short_answer_prompt'] if 'strict_short_answer_prompt' in config else False}"
    )
    print(
        f"  answer_only_prompt: "
        f"{config['answer_only_prompt'] if 'answer_only_prompt' in config else False}"
    )
    print(
        f"  exact_answer_prompt: "
        f"{config['exact_answer_prompt'] if 'exact_answer_prompt' in config else False}"
    )
    print(f"  retriever_model: {config['retrieval_method']}")
    print(f"  use_query_decomposition: {decomposition_config.get('enabled', False)}")
    print(f"  enable_rs_mhr: {rs_mhr_config.get('enabled', False)}")
    if rs_mhr_config.get("enabled", False):
        print(
            f"  rs_mhr_planner_model: "
            f"{rs_mhr_config.get('planner_model', 'Llama-3.1-8B-Instruct')}"
        )
        print(f"  rs_mhr_planner_batch_size: {rs_mhr_config.get('planner_batch_size', 32)}")
    print(f"  enable_efc_rag: {efc_config.get('enabled', False)}")
    if efc_config.get("enabled", False):
        print(f"  efc_force_route: {efc_config.get('force_route', 'auto')}")
        print(f"  efc_probe_topk: {efc_config.get('probe_topk', 5)}")
        print(f"  efc_final_topk: {efc_config.get('final_topk', 5)}")
        print(f"  efc_missing_query_mode: {efc_config.get('missing_query_mode', 'heuristic')}")
        print(f"  efc_retrieval_cache_only: {efc_config.get('retrieval_cache_only', False)}")
    print(f"  planner_subquery_num: {decomposition_config.get('subquery_num', 2)}")
    print(f"  subquery_topk: {decomposition_config.get('subquery_topk', 5)}")
    print(
        f"  qd_rerank_with_subqueries: "
        f"{decomposition_config.get('rerank_with_subqueries', False)}"
    )
    print(
        f"  qd_subquery_rerank_weight: "
        f"{decomposition_config.get('subquery_rerank_weight', 0.25)}"
    )
    print(f"  load_retrieval_topk_cache_path: {config['load_retrieval_topk_cache_path']}")
    print(f"  save_retrieval_topk_cache_path: {config['save_retrieval_topk_cache_path']}")
    print(f"  save_retrieval_cache: {config['save_retrieval_cache']}")
    print(f"  use_retrieval_cache: {config['use_retrieval_cache']}")
    print(f"  retrieval_cache_path: {config['retrieval_cache_path']}")
    if rs_mhr_config.get("enabled", False):
        print(f"  retrieval_cache_only: {rs_mhr_config.get('retrieval_cache_only', False)}")


def retrieve_and_prepare(config, dataset, planner_generator=None):
    from flashrag.utils import get_generator, get_retriever

    rs_mhr_config = config["rs_mhr_config"] or {}
    efc_config = config["efc_rag_config"] or {}
    decomposition_config = config["decomposition_config"] or {}
    probe_generator = None
    if efc_config.get("enabled", False):
        # vLLM must start before the retriever initializes CUDA; otherwise recent
        # vLLM versions force spawn and execute this entrypoint again.
        probe_generator = get_generator(config)

    if efc_config.get("enabled", False) and efc_config.get(
        "retrieval_cache_only", False
    ):
        retriever = EFC_CacheOnlyRetriever(config)
    elif rs_mhr_config.get("enabled", False) and rs_mhr_config.get(
        "retrieval_cache_only", False
    ):
        retriever = RS_MHR_CacheOnlyRetriever(config)
    elif decomposition_config.get("enabled", False) and config["load_retrieval_topk_cache_path"]:
        retriever = QD_CacheOnlyRetriever(config)
    else:
        retriever = get_retriever(config)
    log_selection_settings(config, retriever)

    if efc_config.get("enabled", False):
        dataset = retrieve_with_efc(config, dataset, retriever, probe_generator)
        return build_final_prompts(config, dataset)

    if rs_mhr_config.get("enabled", False):
        dataset = retrieve_with_rs_mhr(config, dataset, retriever, planner_generator)
        return build_final_prompts(config, dataset)

    use_query_decomposition = bool(decomposition_config.get("enabled", False))
    if use_query_decomposition:
        retrieval_bundle = retrieve_with_query_decomposition(
            config, dataset, retriever, planner_generator
        )
        retrieval_results = retrieval_bundle["retrieval_result_full"]
        retrieval_scores = retrieval_bundle["retrieval_score_full"]
    else:
        retrieval_results, retrieval_scores = retriever.batch_search(dataset.question, return_score=True)
        retrieval_bundle = {
            "retrieval_result_full": retrieval_results,
            "retrieval_score_full": retrieval_scores,
        }

    if len(retrieval_results) != len(retrieval_scores):
        raise ValueError(
            f"Retriever returned {len(retrieval_results)} result lists but "
            f"{len(retrieval_scores)} score lists."
        )
    for key, value in retrieval_bundle.items():
        dataset.update_output(key, value)

    selected_results = []
    adaptive_ks = []
    adaptive_gap_indices = []
    mmr_selected_indices = []
    racp_config = dict(config["racp_config"] or {})
    racp_config["_retriever"] = retriever
    racp_config["retrieval_batch_size"] = config["retrieval_batch_size"]
    racp_config["retrieval_method"] = config["retrieval_method"]
    racp_config["retrieval_model_path"] = config["retrieval_model_path"]
    racp_config["mmr_embedding_model_path"] = (
        racp_config.get("mmr_embedding_model_path") or config["mmr_embedding_model_path"]
    )
    selection_method = racp_config.get("selection_method", "gap")
    if selection_method not in {"gap", "gap_mmr", "topk", "title_dedup_topk"}:
        raise ValueError(
            f"Unsupported selection_method '{selection_method}'. "
            "Choose from: gap, gap_mmr, topk, title_dedup_topk."
        )

    for docs, scores in zip(retrieval_results, retrieval_scores):
        if selection_method == "gap":
            selected_docs, selected_k, gap_idx = select_adaptive_docs(docs, scores, racp_config)
            selected_indices = []
        elif selection_method == "gap_mmr":
            selected_docs, selected_k, gap_idx, selected_indices = select_adaptive_mmr_docs(
                docs, scores, racp_config
            )
        elif selection_method == "title_dedup_topk":
            selected_docs, selected_k, gap_idx, selected_indices = select_title_dedup_topk_docs(
                docs, scores, racp_config
            )
        else:
            selected_docs, selected_k, gap_idx, selected_indices = select_topk_docs(
                docs, scores, racp_config
            )
        selected_results.append(selected_docs)
        adaptive_ks.append(selected_k)
        adaptive_gap_indices.append(gap_idx)
        mmr_selected_indices.append(selected_indices)

    dataset.update_output("retrieval_result", selected_results)
    dataset.update_output("adaptive_k", adaptive_ks)
    dataset.update_output("adaptive_gap_index", adaptive_gap_indices)
    dataset.update_output("selection_method", [selection_method for _ in selected_results])
    dataset.update_output(
        "selection_topk", [int(racp_config.get("selection_topk", 8)) for _ in selected_results]
    )
    dataset.update_output(
        "mmr_lambda", [float(racp_config.get("mmr_lambda", 0.7)) for _ in selected_results]
    )
    dataset.update_output("mmr_selected_indices", mmr_selected_indices)
    dataset.update_output(
        "query_decomposition_enabled", [use_query_decomposition for _ in selected_results]
    )

    dataset = build_final_prompts(config, dataset)

    if config["save_retrieval_cache"]:
        retriever._save_cache()

    return dataset


def save_dataset_streaming(dataset, save_path, transform=None):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        f.write("[\n")
        for idx, item in enumerate(dataset):
            if idx:
                f.write(",\n")
            data = item.to_dict()
            if transform is not None:
                data = transform(data)
            json.dump(data, f, indent=4, ensure_ascii=False)
        f.write("\n]\n")


def compact_rs_mhr_prompt_cache_item(data):
    data = dict(data)
    output = data.get("output", {})
    compact_output = {
        key: value for key, value in output.items() if key in RS_MHR_PROMPT_CACHE_OUTPUT_KEYS
    }
    compact_output["retrieval_result"] = [
        {
            key: doc[key]
            for key in ("id", "doc_id", "doc_uid", "title", "contents")
            if key in doc
        }
        if isinstance(doc, dict)
        else doc
        for doc in compact_output.get("retrieval_result", [])
    ]
    data["output"] = compact_output
    return data


def compact_efc_prompt_cache_item(data):
    data = dict(data)
    output = data.get("output", {})
    compact_output = {
        key: value for key, value in output.items() if key in EFC_PROMPT_CACHE_OUTPUT_KEYS
    }
    compact_output["retrieval_result"] = [
        {
            key: doc[key]
            for key in (
                "id",
                "doc_id",
                "doc_uid",
                "title",
                "contents",
                "sources",
                "rrf_score",
                "role_scores",
            )
            if key in doc
        }
        if isinstance(doc, dict)
        else doc
        for doc in compact_output.get("retrieval_result", [])
    ]
    data["output"] = compact_output
    return data


def save_prompt_cache(dataset, save_path, compact_rs_mhr=False, compact_efc=False):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if compact_rs_mhr:
        save_dataset_streaming(dataset, save_path, transform=compact_rs_mhr_prompt_cache_item)
    elif compact_efc:
        save_dataset_streaming(dataset, save_path, transform=compact_efc_prompt_cache_item)
    else:
        dataset.save(str(save_path))
    print(f"Prompt cache saved to: {save_path}")


def load_prompt_cache(config, cache_path):
    if cache_path is None:
        raise ValueError("Please provide --prompt_cache_path for --stage generate.")
    cache_path = Path(cache_path)
    with cache_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset(config=config, data=data)


def run_prepare(config, args):
    dataset = load_split(config, args.split)
    rs_mhr_config = config["rs_mhr_config"] or {}
    efc_config = config["efc_rag_config"] or {}
    if (
        config["decomposition_config"]["enabled"]
        and not config["load_retrieval_topk_cache_path"]
    ):
        planner_generator = get_query_decomposition_planner(config)
    elif (
        rs_mhr_config.get("enabled", False)
        and rs_mhr_config.get("force_route", "auto") != "confident"
    ):
        planner_generator = get_rs_mhr_planner(config)
    else:
        planner_generator = None
    dataset = retrieve_and_prepare(config, dataset, planner_generator=planner_generator)
    if planner_generator is not None:
        release_generator(planner_generator)
        del planner_generator

    prompt_cache_path = args.prompt_cache_path
    if prompt_cache_path is None:
        prompt_cache_path = Path(config["save_dir"]) / "prompt_cache.json"
    rs_mhr_enabled = (config["rs_mhr_config"] or {}).get("enabled", False)
    efc_enabled = efc_config.get("enabled", False)
    if rs_mhr_enabled and rs_mhr_config.get("save_router_debug", False):
        router_debug_path = Path(config["save_dir"]) / "router_debug.json"
        save_dataset_streaming(dataset, router_debug_path)
        print(f"Router debug saved to: {router_debug_path}")
    if efc_enabled and efc_config.get("save_efc_debug", False):
        efc_debug_path = Path(config["save_dir"]) / "efc_debug.json"
        save_dataset_streaming(dataset, efc_debug_path)
        print(f"EFC debug saved to: {efc_debug_path}")
    save_prompt_cache(
        dataset,
        Path(prompt_cache_path),
        compact_rs_mhr=rs_mhr_enabled,
        compact_efc=efc_enabled,
    )
    if rs_mhr_enabled:
        log_rs_mhr_statistics(dataset)
    elif efc_enabled:
        log_efc_statistics(dataset)
    elif config["decomposition_config"]["enabled"]:
        log_query_decomposition_statistics(dataset)
    return dataset


def run_generate(config, args):
    from flashrag.utils import get_generator

    if args.prompt_cache_path is None:
        raise ValueError("Please provide --prompt_cache_path for --stage generate.")

    prompt_cache_dir = Path(args.prompt_cache_path).resolve().parent
    config["save_dir"] = str(prompt_cache_dir)
    prompt_cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_prompt_cache(config, args.prompt_cache_path)
    input_prompts = dataset.prompt

    generator = get_generator(config)
    pred_answer_list = generator.generate(input_prompts)
    dataset.update_output("pred", pred_answer_list)

    result = Evaluator(config).evaluate(dataset)
    print(result)
    if (config["rs_mhr_config"] or {}).get("enabled", False):
        log_rs_mhr_statistics(dataset, include_metrics=True)
    elif (config["efc_rag_config"] or {}).get("enabled", False):
        log_efc_statistics(dataset, include_metrics=True)
    elif config["decomposition_config"]["enabled"]:
        log_query_decomposition_statistics(dataset)
    return dataset


def build_full_generate_command(args, prompt_cache_path):
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        "generate",
        "--prompt_cache_path",
        str(Path(prompt_cache_path).resolve()),
        "--gpu_id",
        str(args.gpu_id),
        "--gpu_memory_utilization",
        str(args.generate_gpu_memory_utilization),
    ]


def run_full_vllm_generate(config, args, dataset):
    prompt_cache_path = Path(config["save_dir"]) / "prompt_cache.json"
    rs_mhr_enabled = (config["rs_mhr_config"] or {}).get("enabled", False)
    efc_enabled = (config["efc_rag_config"] or {}).get("enabled", False)
    save_prompt_cache(
        dataset,
        prompt_cache_path,
        compact_rs_mhr=rs_mhr_enabled,
        compact_efc=efc_enabled,
    )

    command = build_full_generate_command(args, prompt_cache_path)
    if efc_enabled:
        method_name = "EFC-RAG"
    elif rs_mhr_enabled:
        method_name = "RS-MHR"
    elif config["decomposition_config"]["enabled"]:
        method_name = "Query decomposition"
    else:
        method_name = "RACP"
    print(f"{method_name} final generation: launching a clean vLLM subprocess.")
    print(
        f"  gpu_memory_utilization: {args.generate_gpu_memory_utilization}"
    )
    subprocess.run(command, cwd=str(REPO_DIR), check=True)

    result_path = Path(config["save_dir"]) / "intermediate_data.json"
    if not result_path.exists():
        raise RuntimeError(
            f"vLLM generation finished without producing {result_path}."
        )
    return load_prompt_cache(config, result_path)


def run_full(config, args):
    from flashrag.utils import get_generator

    rs_mhr_config = config["rs_mhr_config"] or {}
    efc_config = config["efc_rag_config"] or {}
    if efc_config.get("enabled", False):
        import gc

        dataset = load_split(config, args.split)
        dataset = retrieve_and_prepare(config, dataset)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        if config["framework"] == "vllm":
            return run_full_vllm_generate(config, args, dataset)
        generator = get_generator(config)
    elif rs_mhr_config.get("enabled", False):
        dataset = load_split(config, args.split)
        planner_generator = (
            None
            if rs_mhr_config.get("force_route", "auto") == "confident"
            else get_rs_mhr_planner(config)
        )
        dataset = retrieve_and_prepare(config, dataset, planner_generator=planner_generator)
        if planner_generator is not None:
            release_generator(planner_generator)
            del planner_generator
        if config["framework"] == "vllm":
            return run_full_vllm_generate(config, args, dataset)
        generator = get_generator(config)
    elif config["decomposition_config"]["enabled"]:
        dataset = load_split(config, args.split)
        planner_generator = (
            None
            if config["load_retrieval_topk_cache_path"]
            else get_query_decomposition_planner(config)
        )
        dataset = retrieve_and_prepare(config, dataset, planner_generator=planner_generator)
        if planner_generator is not None:
            release_generator(planner_generator)
            del planner_generator
        if config["framework"] == "vllm":
            return run_full_vllm_generate(config, args, dataset)
        generator = get_generator(config)
    else:
        # Keep the original startup order for the existing RACP flow.
        generator = get_generator(config)
        dataset = load_split(config, args.split)
        dataset = retrieve_and_prepare(config, dataset)

    pred_answer_list = generator.generate(dataset.prompt)
    dataset.update_output("pred", pred_answer_list)

    result = Evaluator(config).evaluate(dataset)
    print(result)
    if (config["rs_mhr_config"] or {}).get("enabled", False):
        log_rs_mhr_statistics(dataset, include_metrics=True)
    elif (config["efc_rag_config"] or {}).get("enabled", False):
        log_efc_statistics(dataset, include_metrics=True)
    elif config["decomposition_config"]["enabled"]:
        log_query_decomposition_statistics(dataset)
    return dataset


def run(args):
    enabled_special_flows = sum(
        bool(flag)
        for flag in (
            args.enable_rs_mhr,
            args.enable_efc_rag,
            args.use_query_decomposition,
        )
    )
    if enabled_special_flows > 1:
        raise ValueError(
            "--enable_rs_mhr, --enable_efc_rag, and --use_query_decomposition "
            "are separate flows."
        )
    if args.enable_rs_mhr and args.no_reranker:
        raise ValueError("--enable_rs_mhr requires the reranker.")
    if args.enable_rs_mhr and args.force_route not in {
        "auto",
        "confident",
        "static_qd",
        "missing_hop",
    }:
        raise ValueError(f"--force_route {args.force_route} is not valid for RS-MHR.")
    if args.enable_efc_rag and args.force_route not in {
        "auto",
        "direct",
        "static_qd",
        "generation_guided",
    }:
        raise ValueError(f"--force_route {args.force_route} is not valid for EFC-RAG.")
    if args.retrieval_cache_only and not (args.enable_rs_mhr or args.enable_efc_rag):
        raise ValueError(
            "--retrieval_cache_only is supported by --enable_rs_mhr or --enable_efc_rag."
        )
    if args.retrieval_cache_only and args.retrieval_cache_path is None:
        raise ValueError("--retrieval_cache_only requires --retrieval_cache_path.")
    for arg_name in (
        "retrieval_batch_size",
        "planner_batch_size",
        "planner_inference_batch_size",
        "planner_max_tokens",
    ):
        if getattr(args, arg_name) < 1:
            raise ValueError(f"--{arg_name} must be positive.")
    if args.missing_query_num is not None and args.missing_query_num < 1:
        raise ValueError("--missing_query_num must be positive.")
    for arg_name in (
        "gpu_memory_utilization",
        "generate_gpu_memory_utilization",
    ):
        if not 0 < getattr(args, arg_name) <= 1:
            raise ValueError(f"--{arg_name} must be in (0, 1].")
    if args.qd_subquery_rerank_weight < 0:
        raise ValueError("--qd_subquery_rerank_weight must be non-negative.")
    answer_prompt_flags = [
        args.strict_short_answer_prompt,
        args.answer_only_prompt,
        args.exact_answer_prompt,
    ]
    if sum(bool(flag) for flag in answer_prompt_flags) > 1:
        raise ValueError(
            "--strict_short_answer_prompt, --answer_only_prompt, and "
            "--exact_answer_prompt are exclusive."
        )
    if args.enable_rs_mhr:
        positive_args = [
            "initial_topk",
            "final_topk",
            "seed_topk",
            "extra_topk",
            "static_qd_num",
            "original_pool_topk",
            "rrf_k",
            "route_c_seed_quota",
            "route_c_missing_quota",
        ]
        for arg_name in positive_args:
            if getattr(args, arg_name) < 1:
                raise ValueError(f"--{arg_name} must be positive.")
    if args.enable_efc_rag:
        positive_args = (
            "initial_topk",
            "probe_topk",
            "final_topk",
            "qd_num",
            "qd_topk",
            "gen_topk",
            "rrf_k",
            "max_same_title",
            "probe_max_tokens",
        )
        for arg_name in positive_args:
            if getattr(args, arg_name) < 1:
                raise ValueError(f"--{arg_name} must be positive.")
        if args.probe_topk > args.initial_topk:
            raise ValueError("--probe_topk cannot exceed --initial_topk.")
        if not 0 <= args.original_seed_count <= args.final_topk:
            raise ValueError(
                "--original_seed_count must be between 0 and --final_topk."
            )
        for arg_name in (
            "rrf_weight",
            "role_weight",
            "title_weight",
            "source_weight",
            "redundancy_weight",
        ):
            if getattr(args, arg_name) < 0:
                raise ValueError(f"--{arg_name} must be non-negative.")

    if args.stage == "generate" and args.prompt_cache_path is not None:
        prompt_config_path = Path(args.prompt_cache_path).resolve().parent / "config.yaml"
        if Path(args.config_path).resolve() == DEFAULT_CONFIG_PATH.resolve() and prompt_config_path.exists():
            args.config_path = prompt_config_path

    config = Config(str(args.config_path), build_config_dict(args))
    with tee_run_output(get_run_log_path(config, args), args.stage):
        if args.stage == "prepare":
            return run_prepare(config, args)
        if args.stage == "generate":
            return run_generate(config, args)

        return run_full(config, args)


def apply_method_defaults(args):
    legacy_methods = [
        method
        for method, enabled in (
            ("rs_mhr", args.enable_rs_mhr),
            ("efc", args.enable_efc_rag),
            ("qd", args.use_query_decomposition),
        )
        if enabled
    ]
    if legacy_methods:
        if len(legacy_methods) == 1:
            args.method = legacy_methods[0]
        return args

    args.enable_efc_rag = args.method == "efc"
    args.enable_rs_mhr = args.method == "rs_mhr"
    args.use_query_decomposition = args.method == "qd"
    return args


def parse_args():
    parser = argparse.ArgumentParser(description="Run the standalone RACP experiment.")
    parser.add_argument(
        "--method",
        choices=["efc", "racp", "rs_mhr", "qd"],
        default=DEFAULT_RUN_CONFIG["method"],
        help="Experiment flow. The default no-argument run uses EFC-RAG.",
    )
    parser.add_argument(
        "--stage",
        choices=["full", "prepare", "generate"],
        default=DEFAULT_RUN_CONFIG["stage"],
    )
    parser.add_argument("--prompt_cache_path", type=Path, default=None)
    parser.add_argument("--config_path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--dataset_name", type=str, default=DEFAULT_RUN_CONFIG["dataset_name"]
    )
    parser.add_argument("--split", type=str, default=DEFAULT_RUN_CONFIG["split"])
    parser.add_argument("--gpu_id", type=str, default=DEFAULT_RUN_CONFIG["gpu_id"])
    parser.add_argument("--save_dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--save_note", type=str, default=DEFAULT_RUN_CONFIG["save_note"])

    parser.add_argument("--retrieval_topk", type=int, default=20)
    parser.add_argument(
        "--retrieval_batch_size",
        type=int,
        default=DEFAULT_RUN_CONFIG["retrieval_batch_size"],
    )
    parser.add_argument("--rerank_topk", type=int, default=20)
    parser.add_argument("--no_reranker", action="store_true")
    parser.add_argument("--rerank_model_name", type=str, default=None)
    parser.add_argument("--rerank_model_path", type=Path, default=None)

    parser.add_argument("--buffer", type=int, default=5)
    parser.add_argument("--max_k", type=none_or_int, default=8)
    parser.add_argument("--search_ratio", type=float, default=0.9)
    parser.add_argument(
        "--selection_method",
        choices=["gap", "gap_mmr", "topk", "title_dedup_topk"],
        default="gap",
    )
    parser.add_argument("--selection_topk", type=int, default=8)
    parser.add_argument("--mmr_lambda", type=float, default=0.7)
    parser.add_argument("--mmr_embedding_model_path", type=Path, default=None)
    parser.add_argument("--use_query_decomposition", action="store_true")
    parser.add_argument("--planner_subquery_num", type=int, default=2)
    parser.add_argument("--subquery_topk", type=int, default=5)
    parser.add_argument("--qd_rerank_with_subqueries", action="store_true")
    parser.add_argument("--qd_subquery_rerank_weight", type=float, default=0.25)
    parser.add_argument(
        "--planner_max_tokens",
        type=int,
        default=DEFAULT_RUN_CONFIG["planner_max_tokens"],
    )
    parser.add_argument("--load_retrieval_topk_cache_path", type=Path, default=None)
    parser.add_argument("--save_retrieval_topk_cache_path", type=Path, default=None)
    save_cache_group = parser.add_mutually_exclusive_group()
    save_cache_group.add_argument(
        "--save_retrieval_cache", dest="save_retrieval_cache", action="store_true"
    )
    save_cache_group.add_argument(
        "--no_save_retrieval_cache",
        dest="save_retrieval_cache",
        action="store_false",
    )
    parser.set_defaults(
        save_retrieval_cache=DEFAULT_RUN_CONFIG["save_retrieval_cache"]
    )
    use_cache_group = parser.add_mutually_exclusive_group()
    use_cache_group.add_argument(
        "--use_retrieval_cache", dest="use_retrieval_cache", action="store_true"
    )
    use_cache_group.add_argument(
        "--no_use_retrieval_cache",
        dest="use_retrieval_cache",
        action="store_false",
    )
    parser.set_defaults(use_retrieval_cache=DEFAULT_RUN_CONFIG["use_retrieval_cache"])
    parser.add_argument("--retrieval_cache_only", action="store_true")
    parser.add_argument(
        "--retrieval_cache_path",
        type=Path,
        default=DEFAULT_RUN_CONFIG["retrieval_cache_path"],
    )

    parser.add_argument("--enable_rs_mhr", action="store_true")
    parser.add_argument("--enable_efc_rag", action="store_true")
    parser.add_argument(
        "--force_route",
        choices=[
            "auto",
            "confident",
            "direct",
            "static_qd",
            "missing_hop",
            "generation_guided",
        ],
        default="auto",
    )
    parser.add_argument(
        "--initial_topk", type=int, default=DEFAULT_RUN_CONFIG["initial_topk"]
    )
    parser.add_argument("--probe_topk", type=int, default=DEFAULT_RUN_CONFIG["probe_topk"])
    parser.add_argument("--final_topk", type=int, default=DEFAULT_RUN_CONFIG["final_topk"])
    parser.add_argument("--seed_topk", type=int, default=3)
    parser.add_argument("--extra_topk", type=int, default=5)
    parser.add_argument("--static_qd_num", type=int, default=2)
    parser.add_argument("--missing_query_num", type=int, default=None)
    parser.add_argument("--original_pool_topk", type=int, default=20)
    parser.add_argument("--rrf_k", type=int, default=60)
    static_qd_group = parser.add_mutually_exclusive_group()
    static_qd_group.add_argument(
        "--enable_static_qd_fallback",
        dest="enable_static_qd_fallback",
        action="store_true",
    )
    static_qd_group.add_argument(
        "--disable_static_qd_fallback",
        dest="enable_static_qd_fallback",
        action="store_false",
    )
    parser.set_defaults(
        enable_static_qd_fallback=DEFAULT_RUN_CONFIG["enable_static_qd_fallback"]
    )
    parser.add_argument("--qd_num", type=int, default=DEFAULT_RUN_CONFIG["qd_num"])
    parser.add_argument("--qd_topk", type=int, default=DEFAULT_RUN_CONFIG["qd_topk"])
    generation_guided_group = parser.add_mutually_exclusive_group()
    generation_guided_group.add_argument(
        "--enable_generation_guided",
        dest="enable_generation_guided",
        action="store_true",
    )
    generation_guided_group.add_argument(
        "--disable_generation_guided",
        dest="enable_generation_guided",
        action="store_false",
    )
    parser.set_defaults(
        enable_generation_guided=DEFAULT_RUN_CONFIG["enable_generation_guided"]
    )
    parser.add_argument("--gen_topk", type=int, default=DEFAULT_RUN_CONFIG["gen_topk"])
    parser.add_argument(
        "--missing_query_mode",
        choices=["heuristic", "llm", "raw_y1"],
        default=DEFAULT_RUN_CONFIG["missing_query_mode"],
    )
    parser.add_argument(
        "--rrf_weight", type=float, default=DEFAULT_RUN_CONFIG["rrf_weight"]
    )
    parser.add_argument(
        "--role_weight", type=float, default=DEFAULT_RUN_CONFIG["role_weight"]
    )
    parser.add_argument(
        "--title_weight", type=float, default=DEFAULT_RUN_CONFIG["title_weight"]
    )
    parser.add_argument(
        "--source_weight", type=float, default=DEFAULT_RUN_CONFIG["source_weight"]
    )
    parser.add_argument(
        "--redundancy_weight",
        type=float,
        default=DEFAULT_RUN_CONFIG["redundancy_weight"],
    )
    title_dedup_group = parser.add_mutually_exclusive_group()
    title_dedup_group.add_argument(
        "--title_dedup_soft", dest="title_dedup_soft", action="store_true"
    )
    title_dedup_group.add_argument(
        "--no_title_dedup_soft", dest="title_dedup_soft", action="store_false"
    )
    parser.set_defaults(title_dedup_soft=DEFAULT_RUN_CONFIG["title_dedup_soft"])
    parser.add_argument(
        "--max_same_title", type=int, default=DEFAULT_RUN_CONFIG["max_same_title"]
    )
    parser.add_argument(
        "--original_seed_count",
        type=int,
        default=DEFAULT_RUN_CONFIG["original_seed_count"],
    )
    parser.add_argument(
        "--rrf_only_selection",
        action="store_true",
        help=(
            "EFC ablation: select final documents only by fused RRF score for "
            "all routes, bypassing role/diversity/seed/static-quota selection."
        ),
    )
    parser.add_argument(
        "--static_bridge_evidence_topk",
        type=int,
        default=DEFAULT_RUN_CONFIG["static_bridge_evidence_topk"],
    )
    parser.add_argument(
        "--static_bridge_original_count",
        type=int,
        default=DEFAULT_RUN_CONFIG["static_bridge_original_count"],
    )
    parser.add_argument(
        "--static_bridge_qd_count",
        type=int,
        default=DEFAULT_RUN_CONFIG["static_bridge_qd_count"],
    )
    parser.add_argument("--use_centroid_router", action="store_true")
    parser.add_argument("--save_efc_debug", action="store_true")
    parser.add_argument(
        "--probe_max_tokens",
        type=int,
        default=DEFAULT_RUN_CONFIG["probe_max_tokens"],
    )
    parser.add_argument("--route_low_conf_gap", type=float, default=0.10)
    parser.add_argument("--route_avg_top5_thr", type=float, default=0.55)
    parser.add_argument("--route_anchor_top1_thr", type=float, default=0.75)
    parser.add_argument("--route_unique_title_thr", type=float, default=0.60)
    parser.add_argument("--route_c_seed_quota", type=int, default=2)
    parser.add_argument("--route_c_missing_quota", type=int, default=2)
    parser.add_argument(
        "--planner_model", type=str, default=DEFAULT_RUN_CONFIG["planner_model"]
    )
    parser.add_argument("--planner_model_path", type=Path, default=None)
    parser.add_argument("--planner_gpu_memory_utilization", type=float, default=0.75)
    parser.add_argument(
        "--planner_batch_size",
        type=int,
        default=DEFAULT_RUN_CONFIG["planner_batch_size"],
        help="Number of planner prompts submitted by each outer chunk.",
    )
    parser.add_argument(
        "--planner_inference_batch_size",
        type=int,
        default=DEFAULT_RUN_CONFIG["planner_inference_batch_size"],
        help="Actual HF planner batch size on GPU; 8 is tested on a 24GB RTX 4090.",
    )
    parser.add_argument("--save_router_debug", action="store_true")

    parser.add_argument("--gpu_memory_utilization", type=float, default=None)
    parser.add_argument(
        "--generate_gpu_memory_utilization",
        type=float,
        default=DEFAULT_RUN_CONFIG["generate_gpu_memory_utilization"],
        help="vLLM GPU memory utilization for the clean final-generation subprocess in full mode.",
    )
    parser.add_argument("--strict_short_answer_prompt", action="store_true")
    parser.add_argument("--answer_only_prompt", action="store_true")
    parser.add_argument("--exact_answer_prompt", action="store_true")
    parser.add_argument(
        "--test_sample_num",
        type=none_or_int,
        default=DEFAULT_RUN_CONFIG["test_sample_num"],
    )
    args = apply_method_defaults(parser.parse_args())
    if args.gpu_memory_utilization is None:
        utilization_key = (
            "generate_gpu_memory_utilization"
            if args.stage == "generate"
            else "prepare_gpu_memory_utilization"
        )
        args.gpu_memory_utilization = DEFAULT_RUN_CONFIG[utilization_key]
    return args


if __name__ == "__main__":
    run(parse_args())
