import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_DIR))

from racp.run_racp import select_title_dedup_topk_docs


def make_doc(doc_id, title):
    return {"id": doc_id, "contents": f'"{title}"\n{title} contents'}


def test_title_dedup_topk_prefers_distinct_titles():
    docs = [
        make_doc("a1", "Alpha"),
        make_doc("a2", "Alpha"),
        make_doc("b1", "Beta"),
        make_doc("c1", "Gamma"),
    ]

    selected_docs, selected_k, gap_idx, selected_indices = select_title_dedup_topk_docs(
        docs, [4.0, 3.0, 2.0, 1.0], {"selection_topk": 3}
    )

    assert [doc["id"] for doc in selected_docs] == ["a1", "b1", "c1"]
    assert selected_k == 3
    assert gap_idx is None
    assert selected_indices == [0, 2, 3]


def test_title_dedup_topk_fills_with_duplicates_when_needed():
    docs = [
        make_doc("a1", "Alpha"),
        make_doc("a2", "Alpha"),
        make_doc("b1", "Beta"),
    ]

    selected_docs, selected_k, _, selected_indices = select_title_dedup_topk_docs(
        docs, [4.0, 3.0, 2.0], {"selection_topk": 3}
    )

    assert [doc["id"] for doc in selected_docs] == ["a1", "b1", "a2"]
    assert selected_k == 3
    assert selected_indices == [0, 2, 1]


def test_title_dedup_topk_uses_explicit_title_when_available():
    docs = [
        {"id": "a1", "title": "Alpha", "contents": "first passage"},
        {"id": "a2", "title": "alpha", "contents": "second passage"},
        {"id": "b1", "title": "Beta", "contents": "third passage"},
    ]

    selected_docs, _, _, _ = select_title_dedup_topk_docs(
        docs, [4.0, 3.0, 2.0], {"selection_topk": 2}
    )

    assert [doc["id"] for doc in selected_docs] == ["a1", "b1"]
