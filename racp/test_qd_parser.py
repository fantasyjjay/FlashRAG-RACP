import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_DIR))

from racp import run_racp
from racp import derive_qd_fusion_prompt_cache
from racp import fuse_prediction_outputs


def test_qd_parser_accepts_short_entity_queries():
    output = '["Scott Derrickson", "Ed Wood nationality"]'

    assert run_racp.parse_qd_planner_output(output, "Were they of the same nationality?", 2) == [
        "Scott Derrickson",
        "Ed Wood nationality",
    ]


def test_qd_parser_keeps_one_useful_query_when_the_other_is_an_answer():
    output = '["American", "Ed Wood nationality"]'

    assert run_racp.parse_qd_planner_output(output, "What nationality was Ed Wood?", 2) == [
        "Ed Wood nationality",
    ]


def test_qd_parser_rejects_low_information_single_words():
    output = '["film", "Ed Wood nationality"]'

    assert run_racp.parse_qd_planner_output(output, "What nationality was Ed Wood?", 2) == [
        "Ed Wood nationality",
    ]


def test_qd_parser_truncates_extra_queries_after_filtering():
    output = '["Dirleton", "East Lothian", "coastal area of Scotland"]'

    assert run_racp.parse_qd_planner_output(output, "What coastal area borders Dirleton?", 2) == [
        "Dirleton",
        "East Lothian",
    ]


def test_qd_parser_recovers_multiple_json_arrays():
    output = '["Stenocereus", "tree-like plants"], ["Pachypodium", "tree-like plants"]'

    assert run_racp.parse_qd_planner_output(output, "Can both genera include tree-like plants?", 2) == [
        "Stenocereus",
        "Pachypodium",
    ]


def test_qd_parser_recovers_complete_string_from_truncated_json():
    output = '["Kiss and Tell film cast", "government position'

    assert run_racp.parse_qd_planner_output(output, "What position did the actress hold?", 2) == [
        "Kiss and Tell film cast",
    ]


def test_rs_mhr_parser_keeps_strict_query_validation():
    output = '["Scott Derrickson", "Ed Wood nationality"]'

    assert run_racp.parse_planner_output(output, "Were they of the same nationality?", 2) == []


def test_qd_retrieval_uses_partial_planner_output():
    class FakeDataset:
        question = ["Who was Ed Wood?"]

    class FakeRetriever:
        use_reranker = False

    config = {
        "decomposition_config": {
            "planner_model": "Llama-3.2-3B-Instruct",
            "planner_model_path": "/tmp/planner",
            "subquery_topk": 1,
        },
        "model2path": {},
        "load_retrieval_topk_cache_path": None,
        "save_retrieval_topk_cache_path": None,
        "retrieval_topk": 1,
    }
    planner_records = [
        {
            "raw_planner_output": '["American", "Ed Wood nationality"]',
            "parsed_subqueries": ["Ed Wood nationality"],
            "valid": False,
            "partial": True,
            "fallback": False,
        }
    ]
    search_results = [
        ([[{"id": "original", "contents": "original"}]], [[0.9]]),
        ([[{"id": "subquery", "contents": "subquery"}]], [[0.8]]),
    ]

    with patch.object(run_racp, "generate_subqueries", return_value=planner_records), patch.object(
        run_racp, "raw_batch_search", side_effect=search_results
    ) as raw_batch_search:
        bundle = run_racp.retrieve_with_query_decomposition(
            config, FakeDataset(), FakeRetriever(), planner_generator=object()
        )

    assert raw_batch_search.call_args_list[1].args[1] == ["Ed Wood nationality"]
    assert bundle["planner_partial"] == [True]
    assert bundle["planner_fallback"] == [False]
    assert bundle["planner_accepted_subquery_count"] == [1]
    assert len(bundle["merged_retrieval_result"][0]) == 2


def test_qd_score_fusion_promotes_a_subquery_evidence_doc():
    docs = [
        {"id": "a", "contents": '"Alpha"\nalpha'},
        {"id": "b", "contents": '"Beta"\nbeta'},
        {"id": "c", "contents": '"Bridge"\nbridge'},
    ]

    ranked_docs, ranked_scores, qd_boosts = run_racp.fuse_qd_rerank_scores(
        docs,
        [10.0, 9.0, 8.0],
        [[docs[2]]],
        [[5.0]],
        subquery_weight=0.75,
    )

    assert [doc["id"] for doc in ranked_docs] == ["a", "c", "b"]
    assert ranked_scores == [1.0, 0.75, 0.5]
    assert qd_boosts == [0.0, 1.0, 0.0]


def test_qd_subquery_reranking_saves_aligned_diagnostic_scores():
    docs = [
        {"id": "a", "contents": '"Alpha"\nalpha'},
        {"id": "b", "contents": '"Beta"\nbeta'},
        {"id": "c", "contents": '"Bridge"\nbridge'},
    ]

    class FakeReranker:
        def rerank(self, queries, doc_groups, topk):
            score_table = {
                "question": {"a": 10.0, "b": 9.0, "c": 8.0},
                "bridge query": {"c": 5.0},
            }
            result_docs = []
            result_scores = []
            for query, group in zip(queries, doc_groups):
                ranked = sorted(
                    group,
                    key=lambda doc: score_table[query][doc["id"]],
                    reverse=True,
                )
                result_docs.append(ranked[:topk])
                result_scores.append([score_table[query][doc["id"]] for doc in ranked[:topk]])
            return result_docs, result_scores

    bundle = run_racp.rerank_with_qd_subqueries(
        FakeReranker(),
        ["question"],
        [docs],
        [["bridge query"]],
        [[[docs[2]]]],
        rerank_topk=3,
        subquery_weight=0.75,
    )

    assert [doc["id"] for doc in bundle["retrieval_result_full"][0]] == ["a", "c", "b"]
    assert bundle["merged_original_rerank_score"] == [[10.0, 9.0, 8.0]]
    assert bundle["subquery_rerank_score"] == [[[5.0]]]


def test_qd_retrieval_can_fuse_cached_subquery_reranker_scores():
    docs = [
        {"id": "a", "contents": '"Alpha"\nalpha'},
        {"id": "b", "contents": '"Beta"\nbeta'},
        {"id": "c", "contents": '"Bridge"\nbridge'},
    ]

    class FakeDataset:
        question = ["question"]

    class FakeReranker:
        def rerank(self, queries, doc_groups, topk):
            score_table = {
                "question": {"a": 10.0, "b": 9.0, "c": 8.0},
                "bridge query": {"c": 5.0},
            }
            result_docs = []
            result_scores = []
            for query, group in zip(queries, doc_groups):
                ranked = sorted(group, key=lambda doc: score_table[query][doc["id"]], reverse=True)
                result_docs.append(ranked[:topk])
                result_scores.append([score_table[query][doc["id"]] for doc in ranked[:topk]])
            return result_docs, result_scores

    class FakeRetriever:
        use_reranker = True
        reranker = FakeReranker()

    retrieval_bundle = {
        "planner_subqueries": [["bridge query"]],
        "subquery_retrieval_result": [[[docs[2]]]],
        "merged_retrieval_result": [docs],
        "merged_retrieval_score": [[0.9, 0.8, 0.7]],
    }
    config = {
        "decomposition_config": {
            "planner_model_path": "/tmp/planner",
            "subquery_topk": 1,
            "rerank_with_subqueries": True,
            "subquery_rerank_weight": 0.75,
        },
        "model2path": {},
        "load_retrieval_topk_cache_path": "/tmp/cache.json",
        "save_retrieval_topk_cache_path": None,
        "retrieval_topk": 3,
        "rerank_topk": 3,
    }

    with patch.object(run_racp, "load_bundle_cache", return_value=retrieval_bundle):
        bundle = run_racp.retrieve_with_query_decomposition(
            config, FakeDataset(), FakeRetriever(), planner_generator=None
        )

    assert [doc["id"] for doc in bundle["retrieval_result_full"][0]] == ["a", "c", "b"]
    assert bundle["retrieval_qd_boost_full"] == [[0.0, 1.0, 0.0]]


def test_qd_cache_only_retriever_loads_only_the_reranker():
    fake_reranker = object()

    with patch("flashrag.utils.get_reranker", return_value=fake_reranker) as get_reranker:
        retriever = run_racp.QD_CacheOnlyRetriever({"use_reranker": True})

    assert retriever.use_reranker is True
    assert retriever.reranker is fake_reranker
    get_reranker.assert_called_once()


def test_strict_short_answer_prompt_is_opt_in():
    captured = {}

    class FakePromptTemplate:
        def __init__(self, config, system_prompt="", user_prompt=""):
            captured["system_prompt"] = system_prompt
            captured["user_prompt"] = user_prompt

        def get_string(self, question, retrieval_result):
            return f"{question}:{len(retrieval_result)}"

    class FakeDataset:
        question = ["question"]
        retrieval_result = [[{"id": "a", "contents": "alpha"}]]

        def update_output(self, key, value):
            captured[key] = value

    with patch.object(run_racp, "PromptTemplate", FakePromptTemplate):
        run_racp.build_final_prompts(
            {"strict_short_answer_prompt": True, "use_fid": False},
            FakeDataset(),
        )

    assert "shortest final answer span" in captured["system_prompt"]
    assert captured["user_prompt"] == "Question: {question}\nShortest answer:"
    assert captured["prompt"] == ["question:1"]


def test_answer_only_prompt_is_less_aggressive_than_strict_short_answer():
    captured = {}

    class FakePromptTemplate:
        def __init__(self, config, system_prompt="", user_prompt=""):
            captured["system_prompt"] = system_prompt
            captured["user_prompt"] = user_prompt

    with patch.object(run_racp, "PromptTemplate", FakePromptTemplate):
        run_racp.build_answer_prompt_template(
            {
                "strict_short_answer_prompt": False,
                "answer_only_prompt": True,
            }
        )

    assert "Output only the final answer" in captured["system_prompt"]
    assert "Keep complete names, titles, and dates intact" in captured["system_prompt"]
    assert "shortest final answer span" not in captured["system_prompt"]
    assert captured["user_prompt"] == "Question: {question}\nAnswer:"


def test_exact_answer_prompt_preserves_complete_spans():
    captured = {}

    class FakePromptTemplate:
        def __init__(self, config, system_prompt="", user_prompt=""):
            captured["system_prompt"] = system_prompt
            captured["user_prompt"] = user_prompt

    with patch.object(run_racp, "PromptTemplate", FakePromptTemplate):
        run_racp.build_answer_prompt_template(
            {
                "strict_short_answer_prompt": False,
                "answer_only_prompt": False,
                "exact_answer_prompt": True,
            }
        )

    assert "exact final answer span" in captured["system_prompt"]
    assert "preserving complete" in captured["system_prompt"]
    assert "work titles" in captured["system_prompt"]
    assert captured["user_prompt"] == "Question: {question}\nExact answer:"


def test_prediction_fusion_prefers_strict_only_for_sentence_like_default():
    pred, route = fuse_prediction_outputs.choose_prediction(
        "George Archainbaud, because the documents identify him as the director.",
        "George Archainbaud",
    )
    assert pred == "George Archainbaud"
    assert route == "strict"

    pred, route = fuse_prediction_outputs.choose_prediction(
        "George Archainbaud",
        "George",
    )
    assert pred == "George Archainbaud"
    assert route == "default"


def test_qd_fusion_prompt_cache_can_be_derived_offline():
    docs = [
        {"id": "a", "contents": '"Alpha"\nalpha'},
        {"id": "b", "contents": '"Beta"\nbeta'},
        {"id": "c", "contents": '"Bridge"\nbridge'},
    ]
    source_data = [
        {
            "question": "question",
            "output": {
                "merged_retrieval_result": docs,
                "merged_original_rerank_score": [10.0, 9.0, 8.0],
                "subquery_retrieval_result": [[docs[2]]],
                "subquery_rerank_score": [[5.0]],
            },
        }
    ]

    class FakePromptTemplate:
        def get_string(self, question, retrieval_result):
            return f"{question}:{','.join(doc['id'] for doc in retrieval_result)}"

    class FakeDataset:
        def __init__(self, config, data):
            self.data = data

        def save(self, path):
            Path(path).write_text(json.dumps(self.data), encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
        derive_qd_fusion_prompt_cache,
        "build_prompt_template",
        return_value=FakePromptTemplate(),
    ), patch.object(derive_qd_fusion_prompt_cache, "Dataset", FakeDataset):
        output_path = Path(tmp_dir) / "prompt_cache.json"
        derive_qd_fusion_prompt_cache.derive_cache(
            source_data,
            output_path,
            {"rerank_topk": 3},
            weight=0.75,
            topk=2,
            strict_short_answer_prompt=False,
        )
        derived_data = json.loads(output_path.read_text(encoding="utf-8"))

    output = derived_data[0]["output"]
    assert [doc["id"] for doc in output["retrieval_result"]] == ["a", "c"]
    assert output["prompt"] == "question:a,c"
