import json
import math
import tempfile
import unittest
from pathlib import Path

from dpr.data.multilabel_metrics import evaluate_multilabel_retrieval
from dpr.data.thevault_utils import normalize_thevault_query
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


class BiEncoderLossTest(unittest.TestCase):
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

    def test_legacy_loss_masks_other_multilabel_positives(self):
        try:
            import torch
            from dpr.models.biencoder import BiEncoderNllLoss
        except ImportError:
            self.skipTest("PyTorch DPR dependencies are not installed")

        questions = torch.tensor([[1.0, 0.0]])
        contexts = torch.tensor([[1.0, 0.0], [4.0, 0.0], [0.5, 0.0]])
        context_ids = ["doc-a", "doc-b", "wrong"]
        positive_ids = [["doc-a", "doc-b"]]

        unmasked_loss, unmasked_correct = BiEncoderNllLoss().calc(
            questions,
            contexts,
            [0],
            [[]],
            ctx_ids=context_ids,
            positive_doc_ids=positive_ids,
        )
        masked_loss, masked_correct = BiEncoderNllLoss(mask_false_negatives=True).calc(
            questions,
            contexts,
            [0],
            [[]],
            ctx_ids=context_ids,
            positive_doc_ids=positive_ids,
        )

        self.assertEqual(unmasked_correct.item(), 0)
        self.assertEqual(masked_correct.item(), 1)
        self.assertLess(masked_loss.item(), unmasked_loss.item())


class HFRobertaEncoderTest(unittest.TestCase):
    def test_codebert_adapter_uses_first_token_representation(self):
        try:
            import torch
            from transformers import RobertaConfig

            from dpr.models.hf_models import HFRobertaEncoder
        except (ImportError, OSError):
            self.skipTest("PyTorch and Transformers are not installed")

        config = RobertaConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=32,
            max_position_embeddings=32,
            pad_token_id=1,
        )
        encoder = HFRobertaEncoder(config)
        encoder.eval()

        input_ids = torch.tensor([[0, 7, 8, 2, 1], [0, 9, 2, 1, 1]])
        segment_ids = torch.zeros_like(input_ids)
        attention_mask = input_ids.ne(config.pad_token_id)

        with torch.no_grad():
            sequence, pooled, _ = encoder(input_ids, segment_ids, attention_mask)

        self.assertEqual(tuple(sequence.shape), (2, 5, 16))
        self.assertEqual(tuple(pooled.shape), (2, 16))
        self.assertTrue(torch.allclose(pooled, sequence[:, 0, :]))
        self.assertEqual(encoder.get_out_size(), 16)


if __name__ == "__main__":
    unittest.main()
