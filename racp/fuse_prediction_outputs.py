import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from flashrag.evaluator.utils import normalize_answer


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} must be a non-empty JSON list.")
    return data


def token_count(text):
    return len(normalize_answer(text).split())


def choose_prediction(default_pred, strict_pred):
    if token_count(strict_pred) >= token_count(default_pred):
        return default_pred, "default"
    normalized_default = normalize_answer(default_pred)
    if "." in default_pred or "," in default_pred or " and " in normalized_default:
        return strict_pred, "strict"
    return default_pred, "default"


def exact_match(prediction, golden_answers):
    normalized_prediction = normalize_answer(prediction)
    return float(
        any(normalized_prediction == normalize_answer(answer) for answer in golden_answers)
    )


def sub_exact_match(prediction, golden_answers):
    normalized_prediction = normalize_answer(prediction)
    return float(
        any(normalize_answer(answer) in normalized_prediction for answer in golden_answers)
    )


def token_level_scores(prediction, golden_answers):
    final_metric = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    normalized_prediction = normalize_answer(prediction)
    prediction_tokens = normalized_prediction.split()
    for ground_truth in golden_answers:
        normalized_ground_truth = normalize_answer(ground_truth)
        if (
            normalized_prediction in {"yes", "no", "noanswer"}
            and normalized_prediction != normalized_ground_truth
        ):
            continue
        if (
            normalized_ground_truth in {"yes", "no", "noanswer"}
            and normalized_prediction != normalized_ground_truth
        ):
            continue
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(prediction_tokens)
        recall = num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        if f1 > final_metric["f1"]:
            final_metric = {"f1": f1, "precision": precision, "recall": recall}
    return final_metric


def evaluate(data, retrieval_recall_key=None, retrieval_recall_value=None):
    total = len(data)
    scores = {
        "em": 0.0,
        "f1": 0.0,
        "acc": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    for item in data:
        pred = item["output"]["pred"]
        golden_answers = item["golden_answers"]
        scores["em"] += exact_match(pred, golden_answers)
        scores["acc"] += sub_exact_match(pred, golden_answers)
        token_scores = token_level_scores(pred, golden_answers)
        scores["f1"] += token_scores["f1"]
        scores["precision"] += token_scores["precision"]
        scores["recall"] += token_scores["recall"]

    result = {key: value / total for key, value in scores.items()}
    if retrieval_recall_key is not None:
        result[retrieval_recall_key] = retrieval_recall_value
    return result


def load_retrieval_recall(metric_path):
    if metric_path is None or not Path(metric_path).exists():
        return None, None
    for line in Path(metric_path).read_text(encoding="utf-8").splitlines():
        if line.startswith("retrieval_recall_top"):
            key, value = line.split(":", 1)
            return key.strip(), float(value.strip())
    return None, None


def fuse_outputs(default_data, strict_data):
    if len(default_data) != len(strict_data):
        raise ValueError(
            f"Default and strict data sizes differ: {len(default_data)} vs {len(strict_data)}."
        )

    fused_data = []
    route_counts = Counter()
    for idx, (default_item, strict_item) in enumerate(zip(default_data, strict_data)):
        if default_item.get("id") != strict_item.get("id"):
            raise ValueError(f"Item id mismatch at index {idx}.")
        default_pred = default_item["output"].get("pred", "")
        strict_pred = strict_item["output"].get("pred", "")
        pred, route = choose_prediction(default_pred, strict_pred)
        route_counts[route] += 1

        item = dict(default_item)
        output = dict(default_item.get("output", {}))
        output["pred_default"] = default_pred
        output["pred_strict"] = strict_pred
        output["pred_fusion_route"] = route
        output["pred"] = pred
        item["output"] = output
        fused_data.append(item)
    return fused_data, route_counts


def write_metric_score(path, result):
    with Path(path).open("w", encoding="utf-8") as f:
        for key, value in result.items():
            f.write(f"{key}: {value}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuse default and strict RACP generation outputs without rerunning the model."
    )
    parser.add_argument("--default_intermediate", type=Path, required=True)
    parser.add_argument("--strict_intermediate", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    default_data = load_json(args.default_intermediate)
    strict_data = load_json(args.strict_intermediate)
    fused_data, route_counts = fuse_outputs(default_data, strict_data)

    retrieval_key, retrieval_value = load_retrieval_recall(
        args.default_intermediate.parent / "metric_score.txt"
    )
    result = evaluate(fused_data, retrieval_key, retrieval_value)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "intermediate_data.json").open("w", encoding="utf-8") as f:
        json.dump(fused_data, f, ensure_ascii=False)
    write_metric_score(args.output_dir / "metric_score.txt", result)
    source_config = args.default_intermediate.parent / "config.yaml"
    if source_config.exists():
        shutil.copy2(source_config, args.output_dir / "config.yaml")

    print(f"Prediction fusion saved to: {args.output_dir}")
    print(f"Route counts: {dict(route_counts)}")
    print(result)


if __name__ == "__main__":
    main()
