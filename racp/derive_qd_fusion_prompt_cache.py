import argparse
import json
import shutil
import sys
from pathlib import Path


RACP_DIR = Path(__file__).resolve().parent
REPO_DIR = RACP_DIR.parent
sys.path.insert(0, str(REPO_DIR))

from flashrag.config import Config
from flashrag.dataset import Dataset
from racp.run_racp import (
    build_answer_prompt_template,
    fuse_qd_rerank_scores,
    select_title_dedup_topk_docs,
)


CONFIG_PATH = RACP_DIR / "config.yaml"


def load_source_cache(cache_path):
    with Path(cache_path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("QD fusion source prompt cache must be a non-empty JSON list.")
    return data


def require_output(output, key, item_idx):
    value = output.get(key)
    if value is None:
        raise ValueError(
            f"Item {item_idx} has no output.{key}. "
            "Run racp/run_racp.py --stage prepare with --qd_rerank_with_subqueries first."
        )
    return value


def build_prompt_template(
    config,
    strict_short_answer_prompt,
    answer_only_prompt=False,
    exact_answer_prompt=False,
):
    config["strict_short_answer_prompt"] = strict_short_answer_prompt
    config["answer_only_prompt"] = answer_only_prompt
    config["exact_answer_prompt"] = exact_answer_prompt
    return build_answer_prompt_template(config)


def derive_cache(
    source_data,
    output_path,
    config,
    weight,
    topk,
    strict_short_answer_prompt,
    answer_only_prompt=False,
    exact_answer_prompt=False,
):
    prompt_template = build_prompt_template(
        config,
        strict_short_answer_prompt,
        answer_only_prompt,
        exact_answer_prompt,
    )
    derived_data = []
    rerank_topk = int(config["rerank_topk"])

    for item_idx, item in enumerate(source_data):
        output = dict(item.get("output", {}))
        merged_docs = require_output(output, "merged_retrieval_result", item_idx)
        original_scores = require_output(output, "merged_original_rerank_score", item_idx)
        subquery_docs = require_output(output, "subquery_retrieval_result", item_idx)
        subquery_scores = require_output(output, "subquery_rerank_score", item_idx)

        reranked_docs, reranked_scores, qd_boosts = fuse_qd_rerank_scores(
            merged_docs,
            original_scores,
            subquery_docs,
            subquery_scores,
            weight,
        )
        reranked_docs = reranked_docs[:rerank_topk]
        reranked_scores = reranked_scores[:rerank_topk]
        qd_boosts = qd_boosts[:rerank_topk]
        selected_docs, _, _, selected_indices = select_title_dedup_topk_docs(
            reranked_docs,
            reranked_scores,
            {"selection_topk": topk},
        )

        output["retrieval_result_full"] = reranked_docs
        output["retrieval_score_full"] = reranked_scores
        output["retrieval_qd_boost_full"] = qd_boosts
        output["retrieval_result"] = selected_docs
        output["selection_method"] = "title_dedup_topk"
        output["selection_topk"] = topk
        output["mmr_selected_indices"] = selected_indices
        output["prompt"] = prompt_template.get_string(
            question=item.get("question"),
            retrieval_result=selected_docs,
        )

        new_item = dict(item)
        new_item["output"] = output
        derived_data.append(new_item)

    dataset = Dataset(config=config, data=derived_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(str(output_path))
    print(f"Derived QD fusion prompt cache saved to: {output_path}")
    print(f"Items: {len(derived_data)}")
    print(f"QD subquery rerank weight: {weight}")
    print(f"Selection: title_dedup_topk / top{topk}")
    print(f"Strict short answer prompt: {strict_short_answer_prompt}")
    print(f"Answer-only prompt: {answer_only_prompt}")
    print(f"Exact answer prompt: {exact_answer_prompt}")


def weight_label(weight):
    return str(weight).replace(".", "p")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Derive QD fusion prompt caches from saved reranker diagnostics."
    )
    parser.add_argument("--source_prompt_cache", type=Path, required=True)
    parser.add_argument("--weights", type=float, nargs="+", required=True)
    parser.add_argument("--selection_topk", type=int, default=5)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--prefix", type=str, default="qd-fusion")
    parser.add_argument("--dataset_name", type=str, default="hotpotqa")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--strict_short_answer_prompt", action="store_true")
    parser.add_argument("--answer_only_prompt", action="store_true")
    parser.add_argument("--exact_answer_prompt", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.selection_topk < 1:
        raise ValueError("--selection_topk must be positive.")
    if any(weight < 0 for weight in args.weights):
        raise ValueError("--weights must be non-negative.")
    if sum(
        bool(flag)
        for flag in (
            args.strict_short_answer_prompt,
            args.answer_only_prompt,
            args.exact_answer_prompt,
        )
    ) > 1:
        raise ValueError(
            "--strict_short_answer_prompt, --answer_only_prompt, and "
            "--exact_answer_prompt are exclusive."
        )

    source_data = load_source_cache(args.source_prompt_cache)
    output_dir = args.output_dir or args.source_prompt_cache.resolve().parent
    source_config_path = args.source_prompt_cache.resolve().parent / "config.yaml"
    config = Config(
        str(CONFIG_PATH),
        {
            "save_dir": str(output_dir),
            "disable_save": True,
            "dataset_name": args.dataset_name,
            "split": args.split,
        },
    )

    for weight in args.weights:
        if args.strict_short_answer_prompt:
            suffix = "-strict"
        elif args.answer_only_prompt:
            suffix = "-answer-only"
        elif args.exact_answer_prompt:
            suffix = "-exact"
        else:
            suffix = ""
        variant_dir = output_dir / (
            f"{args.prefix}-w{weight_label(weight)}-top{args.selection_topk}{suffix}"
        )
        output_path = variant_dir / "prompt_cache.json"
        derive_cache(
            source_data,
            output_path,
            config,
            weight,
            args.selection_topk,
            args.strict_short_answer_prompt,
            args.answer_only_prompt,
            args.exact_answer_prompt,
        )
        if source_config_path.exists():
            shutil.copy2(source_config_path, variant_dir / "config.yaml")


if __name__ == "__main__":
    main()
