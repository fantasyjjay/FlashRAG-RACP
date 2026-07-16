import json
import unittest

from racp.run_racp import RACP_CacheOnlyRetriever, compute_gap_selection


class AdaptiveKTest(unittest.TestCase):
    def test_modified_adaptive_k_applies_buffer_and_max_k(self):
        docs = [{"id": str(idx), "contents": f"doc {idx}"} for idx in range(20)]
        scores = [1.0 - 0.01 * idx for idx in range(11)] + [
            0.10 - 0.001 * idx for idx in range(9)
        ]

        _, _, _, selected_k, gap_idx = compute_gap_selection(
            docs,
            scores,
            {"buffer": 5, "search_ratio": 0.9, "max_k": 8},
        )

        self.assertEqual(gap_idx, 10)
        self.assertEqual(selected_k, 8)

    def test_modified_adaptive_k_without_cap_keeps_gap_plus_buffer(self):
        docs = [{"id": str(idx), "contents": f"doc {idx}"} for idx in range(20)]
        scores = [1.0 - 0.01 * idx for idx in range(11)] + [
            0.10 - 0.001 * idx for idx in range(9)
        ]

        _, _, _, selected_k, gap_idx = compute_gap_selection(
            docs,
            scores,
            {"buffer": 5, "search_ratio": 0.9, "max_k": None},
        )

        self.assertEqual(gap_idx, 10)
        self.assertEqual(selected_k, 16)

    def test_racp_cache_only_retriever_reuses_dense_scores(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "retrieval_cache.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "question": [
                            {"id": "1", "contents": "doc 1", "score": 0.9},
                            {"id": "2", "contents": "doc 2", "score": 0.8},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            retriever = RACP_CacheOnlyRetriever(
                {
                    "retrieval_cache_path": str(cache_path),
                    "retrieval_topk": 2,
                    "use_reranker": False,
                }
            )

            docs, scores = retriever.batch_search(["question"], return_score=True)

            self.assertEqual([doc["id"] for doc in docs[0]], ["1", "2"])
            self.assertEqual(scores, [[0.9, 0.8]])

    def test_racp_cache_only_retriever_rejects_cache_miss(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "retrieval_cache.json"
            cache_path.write_text(json.dumps({"known": []}), encoding="utf-8")
            retriever = RACP_CacheOnlyRetriever(
                {
                    "retrieval_cache_path": str(cache_path),
                    "retrieval_topk": 1,
                    "use_reranker": False,
                }
            )

            with self.assertRaisesRegex(ValueError, "Cache misses"):
                retriever.batch_search(["unknown"], return_score=True)


if __name__ == "__main__":
    unittest.main()
