from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .common import format_code_document, load_jsonl


def load_corpus(path: str | Path) -> dict[str, dict[str, Any]]:
    corpus: dict[str, dict[str, Any]] = {}
    for document in load_jsonl(path):
        doc_id = str(document["doc_id"])
        if doc_id in corpus:
            raise ValueError(f"Duplicate corpus document: {doc_id}")
        corpus[doc_id] = document
    return corpus


def load_queries(path: str | Path) -> list[dict[str, Any]]:
    queries = load_jsonl(path)
    for query in queries:
        query["query_id"] = str(query["query_id"])
        query["positive_doc_ids"] = [str(value) for value in query["positive_doc_ids"]]
    return queries


def load_hard_negatives(path: str | Path | None) -> dict[str, list[str]]:
    if path is None or not Path(path).exists():
        return {}
    output: dict[str, list[str]] = {}
    for row in load_jsonl(path):
        output[str(row["query_id"])] = [str(value) for value in row.get("hard_negative_doc_ids", [])]
    return output


@dataclass
class CandidateExample:
    query_id: str
    query: str
    all_positive_ids: list[str]
    selected_positive_ids: list[str]
    selected_negative_ids: list[str]


class CandidateDataset:
    """Dependency-free candidate selector usable by PyTorch DataLoader."""

    def __init__(
        self,
        queries: list[dict[str, Any]],
        corpus: Mapping[str, dict[str, Any]],
        hard_negatives: Mapping[str, list[str]] | None = None,
        max_positives: int = 1,
        max_negatives: int = 15,
        seed: int = 13,
    ) -> None:
        self.queries = queries
        self.corpus = corpus
        self.hard_negatives = hard_negatives or {}
        self.max_positives = max_positives
        self.max_negatives = max_negatives
        self.seed = seed
        for query in queries:
            missing = [doc_id for doc_id in query["positive_doc_ids"] if doc_id not in corpus]
            if missing:
                raise ValueError(f"Query {query['query_id']} has missing positive documents: {missing}")

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, index: int) -> CandidateExample:
        query = self.queries[index]
        rng = random.Random(self.seed + index + random.randint(0, 2**20))
        all_positives = list(dict.fromkeys(query["positive_doc_ids"]))
        selected_positives = list(all_positives)
        rng.shuffle(selected_positives)
        selected_positives = selected_positives[: self.max_positives]

        positive_set = set(all_positives)
        negatives = [
            doc_id
            for doc_id in dict.fromkeys(self.hard_negatives.get(query["query_id"], []))
            if doc_id in self.corpus and doc_id not in positive_set
        ]
        rng.shuffle(negatives)
        negatives = negatives[: self.max_negatives]
        return CandidateExample(
            query_id=query["query_id"],
            query=query["query"],
            all_positive_ids=all_positives,
            selected_positive_ids=selected_positives,
            selected_negative_ids=negatives,
        )


class RetrieverCollator:
    def __init__(
        self,
        tokenizer: Any,
        corpus: Mapping[str, dict[str, Any]],
        query_max_length: int,
        code_max_length: int,
        include_metadata: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.query_max_length = query_max_length
        self.code_max_length = code_max_length
        self.include_metadata = include_metadata

    def __call__(self, examples: list[CandidateExample]) -> dict[str, Any]:
        import torch

        document_ids = list(
            dict.fromkeys(
                doc_id
                for example in examples
                for doc_id in [*example.selected_positive_ids, *example.selected_negative_ids]
            )
        )
        if not document_ids:
            raise ValueError("Retriever batch has no documents")
        index = {doc_id: offset for offset, doc_id in enumerate(document_ids)}
        positive_mask = torch.zeros((len(examples), len(document_ids)), dtype=torch.bool)
        for row, example in enumerate(examples):
            for doc_id in example.all_positive_ids:
                if doc_id in index:
                    positive_mask[row, index[doc_id]] = True
        if not bool(positive_mask.any(dim=1).all()):
            raise ValueError("Every retriever query must include at least one positive in the batch")

        query_tokens = self.tokenizer(
            [example.query for example in examples],
            padding=True,
            truncation=True,
            max_length=self.query_max_length,
            return_tensors="pt",
        )
        code_tokens = self.tokenizer(
            [format_code_document(self.corpus[doc_id], self.include_metadata) for doc_id in document_ids],
            padding=True,
            truncation=True,
            max_length=self.code_max_length,
            return_tensors="pt",
        )
        return {
            "query_tokens": query_tokens,
            "code_tokens": code_tokens,
            "positive_mask": positive_mask,
            "query_ids": [example.query_id for example in examples],
            "doc_ids": document_ids,
        }


class AR2Collator:
    """Create aligned per-query candidates for ranker and AR2 updates."""

    def __init__(
        self,
        tokenizer: Any,
        corpus: Mapping[str, dict[str, Any]],
        query_max_length: int,
        code_max_length: int,
        pair_max_length: int,
        include_metadata: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.query_max_length = query_max_length
        self.code_max_length = code_max_length
        self.pair_max_length = pair_max_length
        self.include_metadata = include_metadata

    def __call__(self, examples: list[CandidateExample]) -> dict[str, Any]:
        import torch

        candidates = [
            list(dict.fromkeys([*example.selected_positive_ids, *example.selected_negative_ids]))
            for example in examples
        ]
        width = max((len(values) for values in candidates), default=0)
        if width < 2:
            raise ValueError("Ranker/AR2 batches require at least one positive and one negative per query")
        candidate_mask = torch.zeros((len(examples), width), dtype=torch.bool)
        positive_mask = torch.zeros((len(examples), width), dtype=torch.bool)
        pair_queries: list[str] = []
        pair_codes: list[str] = []
        padded_doc_ids: list[list[str]] = []
        for row, (example, doc_ids) in enumerate(zip(examples, candidates)):
            padded = list(doc_ids) + [doc_ids[0]] * (width - len(doc_ids))
            padded_doc_ids.append(padded)
            positive_set = set(example.all_positive_ids)
            for column, doc_id in enumerate(padded):
                pair_queries.append(example.query)
                pair_codes.append(format_code_document(self.corpus[doc_id], self.include_metadata))
                if column < len(doc_ids):
                    candidate_mask[row, column] = True
                    positive_mask[row, column] = doc_id in positive_set
        if not bool(positive_mask.any(dim=1).all()):
            raise ValueError("Every ranker query must include a positive candidate")

        query_tokens = self.tokenizer(
            [example.query for example in examples], padding=True, truncation=True,
            max_length=self.query_max_length, return_tensors="pt"
        )
        code_tokens = self.tokenizer(
            pair_codes, padding=True, truncation=True,
            max_length=self.code_max_length, return_tensors="pt"
        )
        pair_tokens = self.tokenizer(
            pair_queries, pair_codes, padding=True, truncation="only_second",
            max_length=self.pair_max_length, return_tensors="pt"
        )
        return {
            "query_tokens": query_tokens,
            "code_tokens": code_tokens,
            "pair_tokens": pair_tokens,
            "candidate_mask": candidate_mask,
            "positive_mask": positive_mask,
            "doc_ids": padded_doc_ids,
            "width": width,
        }


def move_tokens(tokens: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in tokens.items() if hasattr(value, "to")}
