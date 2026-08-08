import json
import math
import tempfile
import unittest
from pathlib import Path

from dpr.data.multilabel_metrics import evaluate_multilabel_retrieval
from dpr.data.thevault_utils import normalize_thevault_query
from scripts.mine_thevault_hard_negatives import mine_hard_negatives
from scripts.prepare_thevault import prepare


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")


def code(doc_id, body, identifier):
    return {
        "text_id": doc_id,
        "text": "Code: " + body,
        "repo": "example/repo",
        "path": "lib/example.rb",
        "identifier": identifier,
        "parameters": [],
    }


def query(doc_id, text):
    return {"text_id": doc_id, "text": "Query: " + text}


class TheVaultPreparationTest(unittest.TestCase):
    def test_normalization_matches_existing_multilabel_builder(self):
        self.assertEqual(
            normalize_thevault_query(" Query:  Parse   A File "),
            "parse a file",
        )

    def test_prepare_groups_all_relevant_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "Ruby_train_r32.0.json"
            test_path = root / "Ruby_test_r32.0.json"
            output = root / "prepared"
            write_jsonl(
                train_path,
                [
                    query("doc-a", "parse a configuration file"),
                    query("doc-b", "Parse   a configuration file"),
                    query("doc-c", "render a template"),
                    code("doc-a", "def parse_a; end", "parse_a"),
                    code("doc-b", "def parse_b; end", "parse_b"),
                    code("doc-c", "def render; end", "render"),
                ],
            )
            write_jsonl(test_path, [query("doc-b", "parse a configuration file")])

            manifest = prepare(str(train_path), str(test_path), str(output), dev_ratio=0)

            train_records = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
            test_records = [json.loads(line) for line in (output / "test.jsonl").read_text().splitlines()]
            parse_train = next(row for row in train_records if row["question"].lower().startswith("parse"))
            self.assertEqual(set(parse_train["positive_ids"]), {"doc-a", "doc-b"})
            self.assertEqual({row["id"] for row in parse_train["positive_ctxs"]}, {"doc-a", "doc-b"})
            self.assertEqual(set(test_records[0]["positive_ids"]), {"doc-a", "doc-b"})
            self.assertFalse(manifest["policy"]["pseudo_queries_used"])
            self.assertEqual(manifest["multilabel"]["train_groups"], 1)

    def test_missing_positive_document_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.json"
            test_path = root / "test.json"
            write_jsonl(train_path, [query("missing", "missing code")])
            write_jsonl(test_path, [])
            with self.assertRaisesRegex(ValueError, "absent from the code corpus"):
                prepare(str(train_path), str(test_path), str(root / "out"), dev_ratio=0)


class MultiLabelMetricsTest(unittest.TestCase):
    def test_multiple_relevant_documents_are_counted(self):
        aggregate, per_query, flags = evaluate_multilabel_retrieval(
            [["doc-a", "doc-b"]],
            [(["wrong-1", "doc-a", "wrong-2", "doc-b"], [4.0, 3.0, 2.0, 1.0])],
            [1, 2, 4],
        )
        self.assertEqual(flags[0], [False, True, False, True])
        self.assertEqual(aggregate["hit@1"], 0.0)
        self.assertEqual(aggregate["hit@2"], 1.0)
        self.assertEqual(aggregate["recall@2"], 0.5)
        self.assertEqual(aggregate["recall@4"], 1.0)
        self.assertEqual(aggregate["mrr@4"], 0.5)
        self.assertTrue(math.isclose(per_query[0]["map@4"], 0.5))


class HardNegativeMiningTest(unittest.TestCase):
    def test_every_multilabel_positive_is_excluded(self):
        train_records = [
            {
                "query_id": "query-1",
                "question": "parse a file",
                "positive_ids": ["doc-a", "doc-b"],
                "positive_ctxs": [
                    {"id": "doc-a", "text": "positive a", "title": "a"},
                    {"id": "doc-b", "text": "positive b", "title": "b"},
                ],
                "negative_ctxs": [],
                "hard_negative_ctxs": [],
            }
        ]
        retrieval_records = [
            {
                "query_id": "query-1",
                "ctxs": [
                    {"id": "doc-a", "text": "positive a", "title": "a", "score": "9"},
                    {"id": "hard-1", "text": "negative 1", "title": "n1", "score": "8"},
                    {"id": "doc-b", "text": "positive b", "title": "b", "score": "7"},
                    {"id": "hard-1", "text": "negative 1", "title": "n1", "score": "6"},
                    {"id": "hard-2", "text": "negative 2", "title": "n2", "score": "5"},
                ],
            }
        ]

        output, stats = mine_hard_negatives(train_records, retrieval_records, 2)

        self.assertEqual(
            [ctx["id"] for ctx in output[0]["hard_negative_ctxs"]],
            ["hard-1", "hard-2"],
        )
        self.assertEqual(stats["known_positives_filtered"], 2)
        self.assertEqual(stats["hard_negatives_selected"], 2)


class MultiPositiveLossTest(unittest.TestCase):
    def test_single_positive_matches_legacy_loss(self):
        try:
            import torch
            from dpr.models.biencoder import BiEncoderMultiPositiveLoss, BiEncoderNllLoss
        except ImportError:
            self.skipTest("PyTorch DPR dependencies are not installed")

        questions = torch.tensor([[1.0, 0.0]])
        contexts = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
        legacy_loss, _ = BiEncoderNllLoss().calc(questions, contexts, [0], [[]])
        multilabel_loss, _ = BiEncoderMultiPositiveLoss().calc(
            questions, contexts, [[0]], ["doc-a", "wrong"], [["doc-a"]]
        )
        self.assertTrue(torch.allclose(legacy_loss, multilabel_loss))

    def test_every_matching_document_id_is_positive(self):
        try:
            import torch
            from dpr.models.biencoder import BiEncoderMultiPositiveLoss
        except ImportError:
            self.skipTest("PyTorch DPR dependencies are not installed")

        questions = torch.tensor([[1.0, 0.0]])
        contexts = torch.tensor([[2.0, 0.0], [1.5, 0.0], [0.0, 1.0]])
        loss, correct = BiEncoderMultiPositiveLoss().calc(
            questions,
            contexts,
            [[0]],
            ["doc-a", "doc-b", "wrong"],
            [["doc-a", "doc-b"]],
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(correct.item(), 1)


if __name__ == "__main__":
    unittest.main()
