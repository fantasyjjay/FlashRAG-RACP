import argparse
import json
import os
import sys
from pathlib import Path


def requested_stage():
    for arg in sys.argv:
        if arg.startswith("--stage="):
            return arg.split("=", 1)[1]
    if "--stage" in sys.argv:
        stage_idx = sys.argv.index("--stage") + 1
        if stage_idx < len(sys.argv):
            return sys.argv[stage_idx]
    return "full"


if requested_stage() == "prepare":
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
else:
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def ensure_conda_runtime_libs():
    conda_prefix = os.environ.get("CONDA_PREFIX")
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


DEFAULT_CONFIG_PATH = PROJECT_DIR / "config.yaml"
DEFAULT_SAVE_DIR = PROJECT_DIR / "output"
PLANNER_STOP_WORDS = [
    "<|eot_id|>",
    "\nQuestion:",
    "\n\nQuestion:",
    "Expected output:",
    "Solution:",
    "Answer:",
]
PLANNER_SYSTEM_PROMPT = "You are a retrieval query rewriter. Output only a JSON array of search queries."


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
            "planner_max_tokens": 96,
            "planner_model_path": None,
        },
        "save_note": args.save_note,
        "save_dir": str(args.save_dir),
        "load_retrieval_topk_cache_path": str(args.load_retrieval_topk_cache_path)
        if args.load_retrieval_topk_cache_path is not None
        else None,
        "save_retrieval_topk_cache_path": str(args.save_retrieval_topk_cache_path)
        if args.save_retrieval_topk_cache_path is not None
        else None,
        "gpu_id": args.gpu_id,
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
        "- Avoid yes/no or comparison queries such as \"Are they the same?\"\n"
        "- Use entity names or entity descriptions from the original question.\n"
        "- If the question is simple, rewrite it into 2 complementary search queries.\n\n"
        "Question:\n"
        f"{question}\n\n"
        "JSON array:"
    )


def is_search_like_query(query):
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
    if len(tokens) < 3:
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


def normalize_subquery_list(candidates, question, subquery_num):
    normalized_question = " ".join(question.lower().split())
    subqueries = []
    seen = set()

    for item in candidates:
        if not isinstance(item, str):
            return []
        subquery = " ".join(item.strip().strip("\"'`").split())
        normalized = subquery.lower()
        if not subquery or normalized == normalized_question or normalized in seen:
            return []
        if not is_search_like_query(subquery):
            return []
        seen.add(normalized)
        subqueries.append(subquery)

    if len(subqueries) != subquery_num:
        return []
    return subqueries


def parse_planner_output(output, question, subquery_num):
    import json as json_module
    import re

    text = (output or "").strip()
    for array in extract_json_arrays(text):
        if len(array) != subquery_num:
            continue
        if any(not isinstance(item, str) or not item.strip() for item in array):
            continue
        subqueries = normalize_subquery_list(array, question, subquery_num)
        if subqueries:
            return subqueries

    try:
        parsed = json_module.loads(text)
        if isinstance(parsed, list):
            candidates = parsed
        elif isinstance(parsed, dict):
            candidates = parsed.get("subqueries") or parsed.get("queries") or []
    except json_module.JSONDecodeError:
        candidates = []

    if not candidates:
        candidates = []
        for line in text.splitlines():
            line = re.sub(r"^\s*(?:[-*]|\d+[\).\:]?)\s*", "", line).strip()
            line = line.strip("\"'` ,")
            if line:
                candidates.append(line)

    return normalize_subquery_list(candidates[:subquery_num], question, subquery_num)


def build_chat_planner_prompts(prompts, planner_model_path):
    if not planner_model_path:
        raise ValueError("planner_model_path is required for chat-template planner generation.")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(planner_model_path)
    chat_prompts = []
    for prompt in prompts:
        messages = [
            {
                "role": "system",
                "content": PLANNER_SYSTEM_PROMPT,
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
    raw_outputs = generator.generate(
        planner_prompts,
        do_sample=False,
        temperature=0,
        max_new_tokens=96,
        stop=list(PLANNER_STOP_WORDS),
    )
    records = []
    for raw_output, question in zip(raw_outputs, questions):
        parsed_subqueries = parse_planner_output(raw_output, question, subquery_num)
        valid = len(parsed_subqueries) == subquery_num
        records.append(
            {
                "raw_planner_output": raw_output,
                "parsed_subqueries": parsed_subqueries,
                "valid": valid,
                "fallback": not valid,
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


def retrieve_with_query_decomposition(config, dataset, retriever, planner_generator):
    decomposition_config = config["decomposition_config"] or {}
    decomposition_config["planner_model_path"] = (
        decomposition_config.get("planner_model_path") or config["generator_model_path"]
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
        planner_subqueries = [
            record["parsed_subqueries"] if record["valid"] else [] for record in planner_records
        ]
        raw_planner_outputs = [record["raw_planner_output"] for record in planner_records]
        planner_valid = [record["valid"] for record in planner_records]
        planner_fallback = [record["fallback"] for record in planner_records]

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
            "fallback": planner_fallback,
            "planner_subqueries": planner_subqueries,
            "planner_valid": planner_valid,
            "planner_fallback": planner_fallback,
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
            retrieval_results, retrieval_scores = retriever.reranker.rerank(
                questions,
                merged_results,
                topk=rerank_topk,
            )
        else:
            retrieval_results, retrieval_scores = merged_results, merged_scores
    else:
        retrieval_topk = int(config["retrieval_topk"])
        retrieval_results = [docs[:retrieval_topk] for docs in merged_results]
        retrieval_scores = [scores[:retrieval_topk] for scores in merged_scores]

    reranker_bundle = {
        **retrieval_bundle,
        "retrieval_result_full": retrieval_results,
        "retrieval_score_full": retrieval_scores,
    }
    return reranker_bundle


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


def build_final_prompts(config, dataset):
    prompt_template = PromptTemplate(config)
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
    print(f"  retrieval_top_n: {retrieval_top_n}")
    print(f"  dataset_name: {config['dataset_name']}")
    print(f"  split: {config['split']}")
    print(f"  generator_model: {config['generator_model']}")
    print(f"  retriever_model: {config['retrieval_method']}")
    print(f"  use_query_decomposition: {decomposition_config.get('enabled', False)}")
    print(f"  planner_subquery_num: {decomposition_config.get('subquery_num', 2)}")
    print(f"  subquery_topk: {decomposition_config.get('subquery_topk', 5)}")
    print(f"  load_retrieval_topk_cache_path: {config['load_retrieval_topk_cache_path']}")
    print(f"  save_retrieval_topk_cache_path: {config['save_retrieval_topk_cache_path']}")


def retrieve_and_prepare(config, dataset, planner_generator=None):
    from flashrag.utils import get_retriever

    retriever = get_retriever(config)
    log_selection_settings(config, retriever)

    decomposition_config = config["decomposition_config"] or {}
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
    if selection_method not in {"gap", "gap_mmr", "topk"}:
        raise ValueError(
            f"Unsupported selection_method '{selection_method}'. "
            "Choose from: gap, gap_mmr, topk."
        )

    for docs, scores in zip(retrieval_results, retrieval_scores):
        if selection_method == "gap":
            selected_docs, selected_k, gap_idx = select_adaptive_docs(docs, scores, racp_config)
            selected_indices = []
        elif selection_method == "gap_mmr":
            selected_docs, selected_k, gap_idx, selected_indices = select_adaptive_mmr_docs(
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


def save_prompt_cache(dataset, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
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
    from flashrag.utils import get_generator

    dataset = load_split(config, args.split)
    planner_generator = get_generator(config) if config["decomposition_config"]["enabled"] else None
    dataset = retrieve_and_prepare(config, dataset, planner_generator=planner_generator)

    prompt_cache_path = args.prompt_cache_path
    if prompt_cache_path is None:
        prompt_cache_path = Path(config["save_dir"]) / "prompt_cache.json"
    save_prompt_cache(dataset, Path(prompt_cache_path))
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
    return dataset


def run_full(config, args):
    from flashrag.utils import get_generator

    # Keep full-stage startup aligned with RACPPipeline/run_exp:
    # initialize the generator before retrieval touches CUDA.
    generator = get_generator(config)
    dataset = load_split(config, args.split)
    planner_generator = generator if config["decomposition_config"]["enabled"] else None
    dataset = retrieve_and_prepare(config, dataset, planner_generator=planner_generator)

    pred_answer_list = generator.generate(dataset.prompt)
    dataset.update_output("pred", pred_answer_list)

    result = Evaluator(config).evaluate(dataset)
    print(result)
    return dataset


def run(args):
    if args.stage == "generate" and args.prompt_cache_path is not None:
        prompt_config_path = Path(args.prompt_cache_path).resolve().parent / "config.yaml"
        if Path(args.config_path).resolve() == DEFAULT_CONFIG_PATH.resolve() and prompt_config_path.exists():
            args.config_path = prompt_config_path

    config = Config(str(args.config_path), build_config_dict(args))
    if args.stage == "prepare":
        return run_prepare(config, args)
    if args.stage == "generate":
        return run_generate(config, args)

    return run_full(config, args)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the standalone RACP experiment.")
    parser.add_argument("--stage", choices=["full", "prepare", "generate"], default="full")
    parser.add_argument("--prompt_cache_path", type=Path, default=None)
    parser.add_argument("--config_path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--gpu_id", type=str, default="2")
    parser.add_argument("--save_dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--save_note", type=str, default="racp")

    parser.add_argument("--retrieval_topk", type=int, default=20)
    parser.add_argument("--rerank_topk", type=int, default=20)
    parser.add_argument("--no_reranker", action="store_true")

    parser.add_argument("--buffer", type=int, default=5)
    parser.add_argument("--max_k", type=none_or_int, default=8)
    parser.add_argument("--search_ratio", type=float, default=0.9)
    parser.add_argument("--selection_method", choices=["gap", "gap_mmr", "topk"], default="gap")
    parser.add_argument("--selection_topk", type=int, default=8)
    parser.add_argument("--mmr_lambda", type=float, default=0.7)
    parser.add_argument("--mmr_embedding_model_path", type=Path, default=None)
    parser.add_argument("--use_query_decomposition", action="store_true")
    parser.add_argument("--planner_subquery_num", type=int, default=2)
    parser.add_argument("--subquery_topk", type=int, default=5)
    parser.add_argument("--planner_max_tokens", type=int, default=96)
    parser.add_argument("--load_retrieval_topk_cache_path", type=Path, default=None)
    parser.add_argument("--save_retrieval_topk_cache_path", type=Path, default=None)

    parser.add_argument("--gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--test_sample_num", type=none_or_int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
