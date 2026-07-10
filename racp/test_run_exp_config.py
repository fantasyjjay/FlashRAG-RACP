from types import SimpleNamespace
from unittest.mock import patch

from racp import run_exp


class ConfigCaptured(Exception):
    pass


def capture_method_config(method, args):
    captured = {}

    def capture(config_dict):
        captured.update(config_dict)
        raise ConfigCaptured

    with patch.object(run_exp, "load_config", side_effect=capture):
        try:
            method(args)
        except ConfigCaptured:
            pass

    return captured


def baseline_args(**overrides):
    values = {
        "method_name": "naive",
        "save_note": None,
        "gpu_id": "2",
        "dataset_name": "2wikimultihopqa",
        "split": "dev",
        "retrieval_topk": 10,
        "rerank_topk": None,
        "no_reranker": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_shared_racp_config_disables_refiner_by_default():
    config = run_exp.load_config({"disable_save": True})

    assert config["refiner_name"] is None


def test_naive_explicitly_disables_refiner():
    config_dict = capture_method_config(run_exp.naive, baseline_args())

    assert config_dict["refiner_name"] is None


def test_zero_shot_explicitly_disables_refiner():
    config_dict = capture_method_config(
        run_exp.zero_shot,
        baseline_args(method_name="zero-shot"),
    )

    assert config_dict["refiner_name"] is None


def test_selective_context_explicitly_enables_refiner():
    config_dict = run_exp.refiner_method_config(
        baseline_args(method_name="selective-context")
    )

    assert config_dict["refiner_name"] == "selective-context"
    assert config_dict["refiner_model_path"].endswith("gpt2")


def test_sequential_pipeline_skips_refiner_when_disabled():
    config = run_exp.load_config(
        {
            "disable_save": True,
            "gpu_id": "2",
            "dataset_name": "2wikimultihopqa",
            "split": "dev",
            "refiner_name": None,
        }
    )

    with patch("flashrag.pipeline.pipeline.get_generator", return_value=object()), patch(
        "flashrag.pipeline.pipeline.get_retriever", return_value=object()
    ), patch("flashrag.pipeline.pipeline.get_refiner") as get_refiner:
        from flashrag.pipeline.pipeline import SequentialPipeline

        pipeline = SequentialPipeline(config)

    assert pipeline.refiner is None
    get_refiner.assert_not_called()
