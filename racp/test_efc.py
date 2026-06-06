import sys
import json
import tempfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_DIR))

from racp.efc import (
    add_rrf_scores,
    build_heuristic_missing_query,
    classify_question,
    compute_router_features,
    decide_route,
    extract_comparison_sides,
    merge_docs,
    normalize_doc,
    probe_answer_is_bad,
    role_aware_pack,
    select_ranked_title_diverse,
)
from racp.run_racp import EFC_CacheOnlyRetriever, release_generator


def make_doc(doc_id, title, text, rank, source="original", query_id="original"):
    return normalize_doc(
        {"id": doc_id, "title": title, "contents": f"{title}\n{text}"},
        score=1.0 / rank,
        rank=rank,
        source=source,
        query="query",
        query_id=query_id,
    )


def test_router_uses_generation_guided_when_probe_exposes_bridge():
    docs = [
        make_doc("1", "Lewiston Maineiacs", "The team played at Androscoggin Bank Colisee.", 1),
        make_doc("2", "Lewiston Maineiacs", "A junior ice hockey team.", 2),
        make_doc("3", "Lewiston Maineiacs", "The team's history.", 3),
        make_doc("4", "Maineiacs seasons", "Season records.", 4),
        make_doc("5", "Maineiacs players", "Player roster.", 5),
    ]
    question = "The arena where the Lewiston Maineiacs played can seat how many people?"
    probe = "They played at Androscoggin Bank Colisee. So the answer is unknown."

    features = compute_router_features(question, docs, probe)

    assert "Androscoggin Bank Colisee" in features["y1_new_entities"]
    assert decide_route(features) == "generation_guided"
    assert (
        build_heuristic_missing_query(
            question, features, [doc["title"] for doc in docs], probe
        )
        == "Androscoggin Bank Colisee seating capacity"
    )


def test_hotpot_prior_marks_implicit_bridge_question_as_multihop():
    question = "What government position was held by the child star in Bright Eyes?"

    assert classify_question(question) == "generic"
    assert classify_question(question, assume_multihop=True) == "bridge"


def test_uncertain_probe_with_bridge_entity_expands_generation_guided():
    docs = [
        make_doc("1", "Bright Eyes", "The film starred Shirley Temple.", 1),
        make_doc("2", "Bright Eyes soundtrack", "The film includes a famous song.", 2),
    ]
    question = "What government position was held by the child star in Bright Eyes?"
    probe = (
        "The child star was Shirley Temple, but the passages do not mention her "
        "government position. So the answer is unknown."
    )

    features = compute_router_features(
        question, docs, probe, assume_multihop=True
    )

    assert features["y1_uncertain"]
    assert decide_route(features) == "generation_guided"
    assert build_heuristic_missing_query(
        question, features, [doc["title"] for doc in docs], probe
    ) == "Shirley Temple government position"


def test_merge_and_rrf_preserve_multi_query_provenance():
    original = make_doc("1", "Alpha", "Original evidence.", 2)
    expanded = make_doc(
        "1",
        "Alpha",
        "Original evidence.",
        1,
        source="generation_guided",
        query_id="gen_0",
    )
    pool = add_rrf_scores(merge_docs([[original], [expanded]]), rrf_k=60)

    assert pool[0]["sources"] == ["original", "generation_guided"]
    assert pool[0]["ranks"] == {"original": 2, "gen_0": 1}
    assert pool[0]["rrf_score"] > 1 / 62


def test_comparison_side_parser_handles_leading_auxiliary_verb():
    assert extract_comparison_sides(
        "Were Scott Derrickson and Ed Wood of the same nationality?"
    ) == ("Scott Derrickson", "Ed Wood")


def test_truncated_probe_answer_is_bad():
    assert probe_answer_is_bad(
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        "Both people were American. So the answer is No, they were",
    )


def test_contradictory_yes_no_probe_is_bad():
    question = "Were Scott Derrickson and Ed Wood of the same nationality?"
    probe = "Scott Derrickson and Ed Wood were both American. So the answer is No."
    docs = [
        make_doc("1", "Scott Derrickson", "Scott Derrickson is American.", 1),
        make_doc("2", "Ed Wood", "Ed Wood was American.", 2),
    ]

    assert probe_answer_is_bad(question, probe)
    features = compute_router_features(
        question, docs, probe, assume_multihop=True
    )
    assert features["y1_contradictory"]
    assert decide_route(features) == "direct"


def test_comparison_with_both_sides_in_top5_stays_direct():
    question = "Who is older, Annie Morton or Terry Richardson?"
    probe = "The dates are not provided. So the answer is unknown."
    docs = [
        make_doc("1", "Annie Morton", "Annie Morton was born in 1970.", 1),
        make_doc("2", "Terry Richardson", "Terry Richardson was born in 1965.", 2),
    ]

    features = compute_router_features(
        question, docs, probe, assume_multihop=True
    )

    assert features["comparison_side_coverage_top5"] == 2
    assert decide_route(features) == "direct"


def test_direct_selection_preserves_rank_and_limits_repeated_titles():
    docs = [
        make_doc("1", "Alpha", "First Alpha passage.", 1),
        make_doc("2", "Alpha", "Second Alpha passage.", 2),
        make_doc("3", "Beta", "Beta passage.", 3),
        make_doc("4", "Gamma", "Gamma passage.", 4),
    ]

    selected = select_ranked_title_diverse(docs, final_topk=3, max_same_title=1)

    assert [doc["title"] for doc in selected] == ["Alpha", "Beta", "Gamma"]
    assert [doc["ranks"]["original"] for doc in selected] == [1, 3, 4]


def test_release_generator_shuts_down_vllm_v1_engine_core():
    class EngineCore:
        def __init__(self):
            self.closed = False

        def shutdown(self):
            self.closed = True

    engine_core = EngineCore()
    generator = type(
        "Generator",
        (),
        {
            "model": type(
                "Model",
                (),
                {"llm_engine": type("Engine", (), {"engine_core": engine_core})()},
            )()
        },
    )()

    release_generator(generator)

    assert engine_core.closed
    assert generator.model is None


def test_efc_cache_only_retriever_reports_generated_query_miss():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_path = Path(tmp_dir) / "retrieval_cache.json"
        cache_path.write_text(
            json.dumps({"original question": [{"id": "1", "contents": "doc", "score": 1.0}]}),
            encoding="utf-8",
        )
        retriever = EFC_CacheOnlyRetriever({"retrieval_cache_path": str(cache_path)})

        try:
            retriever._batch_search(["new generated query"], 5)
        except ValueError as exc:
            assert "new generated query" in str(exc)
        else:
            raise AssertionError("Expected an incomplete EFC cache to report the missing query.")


def test_role_packing_covers_anchor_and_answer_evidence():
    anchor = make_doc(
        "a",
        "Lewiston Maineiacs",
        "The Lewiston Maineiacs played at Androscoggin Bank Colisee.",
        1,
    )
    answer = make_doc(
        "b",
        "Androscoggin Bank Colisee",
        "The arena has a seating capacity of 4,000.",
        1,
        source="generation_guided",
        query_id="gen_0",
    )
    duplicate = make_doc(
        "c",
        "Lewiston Maineiacs",
        "The team was based in Lewiston.",
        2,
    )
    question = "The arena where the Lewiston Maineiacs played can seat how many people?"
    probe = "They played at Androscoggin Bank Colisee. So the answer is unknown."
    pool = add_rrf_scores([anchor, answer, duplicate])
    features = compute_router_features(question, [anchor, duplicate], probe)

    selected, covered = role_aware_pack(
        pool,
        question,
        probe,
        features,
        final_topk=2,
        route="generation_guided",
        config={
            "rrf_weight": 1.0,
            "role_weight": 0.30,
            "title_weight": 0.05,
            "source_weight": 0.05,
            "redundancy_weight": 0.05,
            "title_dedup_soft": True,
            "max_same_title": 1,
        },
    )

    assert "b" in {doc["doc_uid"] for doc in selected}
    assert any("original" in doc["sources"] for doc in selected)
    assert covered["anchor"] > 0
    assert covered["bridge"] > 0
    assert covered["answer"] == 1.0


def test_role_packing_can_reserve_original_evidence():
    originals = [
        make_doc(str(index), f"Original {index}", "Original evidence.", index)
        for index in range(1, 5)
    ]
    expansions = [
        make_doc(
            f"e{index}",
            f"Expansion {index}",
            "New answer evidence.",
            index,
            source="generation_guided",
            query_id="gen_0",
        )
        for index in range(1, 5)
    ]
    pool = add_rrf_scores(originals + expansions)
    features = compute_router_features(
        "What entity supplies the answer?",
        originals,
        "The answer is unknown.",
        assume_multihop=True,
    )

    selected, _ = role_aware_pack(
        pool,
        "What entity supplies the answer?",
        "The answer is unknown.",
        features,
        final_topk=6,
        route="generation_guided",
        config={"original_seed_count": 3},
    )

    assert sum("original" in doc["sources"] for doc in selected) >= 3
