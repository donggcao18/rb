from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from code_search.common import format_code_document, load_jsonl, normalize_query
from code_search.metrics import evaluate_run
from code_search.mine_hard_negatives import filter_known_positives, search_embeddings
from code_search.prepare_data import prepare_dataset
from code_search.validate_data import validate_prepared_data


def _row(numeric_id: str, text_id: str, text: str) -> dict[str, str]:
    return {
        "numeric_id": numeric_id,
        "text_id": text_id,
        "text": text,
        "repo": "owner/repo",
        "path": f"lib/{text_id}.rb",
        "identifier": text_id,
        "url_based_id": f"owner/repo/{text_id}",
    }


class PrepareDataTests(unittest.TestCase):
    def test_jsonl_multi_positive_and_split_safe_qrels(self) -> None:
        train_rows = [
            _row("q1", "d1", "Query: Shared behavior"),
            _row("q2", "d2", "Query:   shared   behavior "),
            _row("q3", "d3", "Query: Other behavior"),
            _row("c1", "d1", "Code: def one; end"),
            _row("c2", "d2", "Code: def two; end"),
            _row("c3", "d3", "Code: def three; end"),
        ]
        test_rows = [_row("t1", "d3", "Query: SHARED BEHAVIOR")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.json"
            test_path = root / "test.jsonl"
            for path, rows in ((train_path, train_rows), (test_path, test_rows)):
                with path.open("w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")
            audit = prepare_dataset(train_path, root / "prepared", test_path, dev_fraction=0)
            queries = load_jsonl(root / "prepared" / "queries.train.jsonl")
            shared = next(row for row in queries if row["normalized_query"] == "shared behavior")
            self.assertEqual(shared["positive_doc_ids"], ["d1", "d2"])
            self.assertEqual(audit["multi_positive_groups"], 1)
            self.assertEqual(audit["train_test_normalized_query_overlap"], 1)
            self.assertTrue(validate_prepared_data(root / "prepared")["valid"])

            global_root = root / "global"
            prepare_dataset(train_path, global_root, test_path, dev_fraction=0, global_qrels=True)
            global_queries = load_jsonl(global_root / "queries.train.jsonl")
            global_shared = next(row for row in global_queries if row["normalized_query"] == "shared behavior")
            self.assertEqual(global_shared["positive_doc_ids"], ["d1", "d2", "d3"])

    def test_missing_document_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(json.dumps(_row("q1", "missing", "Query: x")) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing from the code corpus"):
                prepare_dataset(path, Path(directory) / "out", dev_fraction=0)

    def test_normalization(self) -> None:
        self.assertEqual(normalize_query(" Query:  A   B\n"), "a b")

    def test_code_format_uses_explicit_fields(self) -> None:
        formatted = format_code_document(
            {"identifier": "find", "path": "lib/a.rb", "code": "def find; end"}
        )
        self.assertEqual(formatted, "identifier: find\npath: lib/a.rb\ncode:\ndef find; end")


class RetrievalUtilityTests(unittest.TestCase):
    def test_all_positives_are_filtered(self) -> None:
        ranking = ["p1", "n1", "p2", "n1", "n2", "n3"]
        self.assertEqual(filter_known_positives(ranking, ["p1", "p2"], 2), ["n1", "n2"])

    def test_multilabel_metrics(self) -> None:
        run = {"q": ["x", "a", "b", "c"]}
        metrics = evaluate_run(run, {"q": {"a", "b"}}, [1, 2, 3])
        self.assertEqual(metrics["hit@1"], 0.0)
        self.assertEqual(metrics["hit@2"], 1.0)
        self.assertEqual(metrics["recall@2"], 0.5)
        self.assertEqual(metrics["recall@3"], 1.0)
        self.assertAlmostEqual(metrics["mrr@3"], 0.5)
        self.assertAlmostEqual(metrics["map@3"], ((1 / 2) + (2 / 3)) / 2)

    def test_numpy_exact_search_fallback(self) -> None:
        queries = np.asarray([[1.0, 0.0]], dtype=np.float32)
        documents = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.5, 0.0]], dtype=np.float32)
        scores, indices = search_embeddings(queries, documents, 2)
        self.assertEqual(indices.tolist(), [[1, 2]])
        self.assertEqual(scores.tolist(), [[1.0, 0.5]])


class TorchLossTests(unittest.TestCase):
    def test_multi_positive_matches_manual_value(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed in the local verification runtime")
        from code_search.losses import multi_positive_listwise_loss

        scores = torch.tensor([[1.0, 2.0, 0.0]])
        positives = torch.tensor([[True, True, False]])
        expected = torch.logsumexp(scores, 1) - torch.logsumexp(scores[:, :2], 1)
        self.assertTrue(torch.allclose(multi_positive_listwise_loss(scores, positives, reduction="none"), expected))


if __name__ == "__main__":
    unittest.main()
