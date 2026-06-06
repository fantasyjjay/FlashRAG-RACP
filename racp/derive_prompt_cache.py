import argparse
import json
import sys
from pathlib import Path


RACP_DIR = Path(__file__).resolve().parent
REPO_DIR = RACP_DIR.parent
sys.path.insert(0, str(REPO_DIR))

from flashrag.config import Config
from flashrag.dataset import Dataset
from run_racp import (
    build_answer_prompt_template,
)


CONFIG_PATH = RACP_DIR / "config.yaml"


def load_source_cache(cache_path):
    with Path(cache_path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Prompt cache must be a JSON list, got {type(data)}.")
    if not data:
        raise ValueError("Prompt cache is empty.")
    return data


def get_retrieval_result(item, item_idx):
    output = item.get("output", {})
    retrieval_result = output.get("retrieval_result")
    if retrieval_result is None:
        raise ValueError(
            f"Item {item_idx} has no output.retrieval_result. "
            "This script needs a prompt cache saved by racp/run_exp.py --stage prepare."
        )
    if not isinstance(retrieval_result, list):
        raise ValueError(f"Item {item_idx} output.retrieval_result must be a list.")
    return retrieval_result


def derive_cache(
    source_data,
    topk,
    output_path,
    config,
    strict_short_answer_prompt=False,
    answer_only_prompt=False,
    exact_answer_prompt=False,
):
    config["strict_short_answer_prompt"] = strict_short_answer_prompt
    config["answer_only_prompt"] = answer_only_prompt
    config["exact_answer_prompt"] = exact_answer_prompt
    prompt_template = build_answer_prompt_template(config)
    derived_data = []
    too_short = 0

    for item_idx, item in enumerate(source_data):
        retrieval_result = get_retrieval_result(item, item_idx)
        if len(retrieval_result) < topk:
            too_short += 1
        sliced_result = retrieval_result[:topk]

        new_item = dict(item)
        new_output = dict(item.get("output", {}))
        new_output["retrieval_result"] = sliced_result
        new_output["prompt"] = prompt_template.get_string(
            question=item.get("question"),
            retrieval_result=sliced_result,
        )
        new_item["output"] = new_output
        derived_data.append(new_item)

    dataset = Dataset(config=config, data=derived_data)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(str(output_path))

    print(f"Derived top{topk} prompt cache saved to: {output_path}")
    print(f"Items: {len(derived_data)}")
    if too_short:
        print(f"Warning: {too_short} items had fewer than {topk} retrieved docs.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Derive smaller top-k prompt caches from a larger RACP/native prompt cache."
    )
    parser.add_argument(
        "--source_prompt_cache",
        type=Path,
        required=True,
        help="Existing prompt cache containing output.retrieval_result, e.g. top20.",
    )
    parser.add_argument(
        "--topks",
        type=int,
        nargs="+",
        required=True,
        help="Target top-k values, e.g. --topks 5 10 15.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=RACP_DIR / "output" / "cache",
        help="Directory to write derived prompt caches.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="native_no_rerank",
        help="Output filename prefix.",
    )
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--strict_short_answer_prompt", action="store_true")
    parser.add_argument("--answer_only_prompt", action="store_true")
    parser.add_argument("--exact_answer_prompt", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
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

    config_overrides = {
        "save_dir": str(RACP_DIR / "output"),
        "disable_save": True,
    }
    if args.dataset_name is not None:
        config_overrides["dataset_name"] = args.dataset_name
    if args.split is not None:
        config_overrides["split"] = args.split

    config = Config(str(CONFIG_PATH), config_overrides)
    max_source_docs = min(len(get_retrieval_result(item, idx)) for idx, item in enumerate(source_data))

    for topk in args.topks:
        if topk <= 0:
            raise ValueError(f"topk must be positive, got {topk}.")
        if topk > max_source_docs:
            raise ValueError(
                f"Requested top{topk}, but at least one source item only has "
                f"{max_source_docs} retrieved docs."
            )
        output_path = args.output_dir / f"{args.prefix}_top{topk}_prompt_cache.json"
        derive_cache(
            source_data,
            topk,
            output_path,
            config,
            strict_short_answer_prompt=args.strict_short_answer_prompt,
            answer_only_prompt=args.answer_only_prompt,
            exact_answer_prompt=args.exact_answer_prompt,
        )


if __name__ == "__main__":
    main()
