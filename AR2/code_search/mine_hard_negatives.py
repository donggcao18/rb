from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .common import format_code_document, load_config, write_jsonl
from .datasets import load_corpus, load_queries
from .training_utils import (
    autocast_context,
    load_dual_encoder_checkpoint,
    require_training_dependencies,
    select_device,
    token_inputs,
)


def filter_known_positives(
    ranked_doc_ids: list[str], positive_doc_ids: list[str], limit: int
) -> list[str]:
    positives = set(str(value) for value in positive_doc_ids)
    output: list[str] = []
    seen: set[str] = set()
    for doc_id in ranked_doc_ids:
        doc_id = str(doc_id)
        if doc_id in positives or doc_id in seen:
            continue
        seen.add(doc_id)
        output.append(doc_id)
        if len(output) >= limit:
            break
    return output


def _encode_texts(
    texts: list[str], tokenizer: Any, encoder: Any, batch_size: int, max_length: int,
    device: Any, precision: str
) -> np.ndarray:
    import torch

    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            tokens = tokenizer(
                texts[start : start + batch_size], padding=True, truncation=True,
                max_length=max_length, return_tensors="pt"
            )
            input_ids, attention_mask = token_inputs(tokens, device)
            with autocast_context(torch, device, precision):
                batch = encoder(input_ids, attention_mask)
            vectors.append(batch.float().cpu().numpy())
    if not vectors:
        hidden = getattr(getattr(encoder, "config", None), "hidden_size", 0)
        return np.empty((0, hidden), dtype=np.float32)
    return np.concatenate(vectors, axis=0).astype(np.float32, copy=False)


def encode_corpus(
    corpus: Mapping[str, dict[str, Any]], model: Any, tokenizer: Any, batch_size: int,
    max_length: int, device: Any, precision: str, include_metadata: bool
) -> tuple[list[str], np.ndarray]:
    model.eval()
    doc_ids = list(corpus)
    texts = [format_code_document(corpus[doc_id], include_metadata) for doc_id in doc_ids]
    embeddings = _encode_texts(
        texts, tokenizer, model.encode_code, batch_size, max_length, device, precision
    )
    return doc_ids, embeddings


def encode_queries(
    queries: list[dict[str, Any]], model: Any, tokenizer: Any, batch_size: int,
    max_length: int, device: Any, precision: str
) -> np.ndarray:
    model.eval()
    return _encode_texts(
        [query["query"] for query in queries], tokenizer, model.encode_query,
        batch_size, max_length, device, precision
    )


def search_embeddings(query_embeddings: np.ndarray, document_embeddings: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    if document_embeddings.shape[0] == 0:
        raise ValueError("Cannot search an empty corpus")
    k = min(top_k, document_embeddings.shape[0])
    try:
        import faiss

        index = faiss.IndexFlatIP(document_embeddings.shape[1])
        index.add(document_embeddings)
        scores, indices = index.search(query_embeddings, k)
        return scores, indices
    except ImportError:
        if document_embeddings.shape[0] > 100_000:
            raise RuntimeError("FAISS is required for corpora larger than 100,000 documents")
        score_matrix = query_embeddings @ document_embeddings.T
        partial = np.argpartition(-score_matrix, kth=k - 1, axis=1)[:, :k]
        partial_scores = np.take_along_axis(score_matrix, partial, axis=1)
        order = np.argsort(-partial_scores, axis=1)
        indices = np.take_along_axis(partial, order, axis=1)
        scores = np.take_along_axis(score_matrix, indices, axis=1)
        return scores, indices


def retrieve(
    corpus: Mapping[str, dict[str, Any]], queries: list[dict[str, Any]], model: Any,
    tokenizer: Any, config: Mapping[str, Any], device: Any
) -> dict[str, list[dict[str, Any]]]:
    mining = config["mining"]
    retriever = config["retriever"]
    precision = config.get("runtime", {}).get("precision", "bf16")
    doc_ids, doc_embeddings = encode_corpus(
        corpus, model, tokenizer, mining.get("corpus_batch_size", 512),
        retriever["code_max_length"], device, precision,
        retriever.get("include_metadata", True),
    )
    query_embeddings = encode_queries(
        queries, model, tokenizer, mining.get("query_batch_size", 512),
        retriever["query_max_length"], device, precision,
    )
    scores, indices = search_embeddings(query_embeddings, doc_embeddings, mining["top_k"])
    output: dict[str, list[dict[str, Any]]] = {}
    for row, query in enumerate(queries):
        output[query["query_id"]] = [
            {"doc_id": doc_ids[int(index)], "score": float(scores[row, rank]), "rank": rank + 1}
            for rank, index in enumerate(indices[row])
        ]
    return output


def write_mined_candidates(
    path: str | Path, queries: list[dict[str, Any]], rankings: Mapping[str, list[dict[str, Any]]],
    negative_pool_size: int
) -> None:
    rows = []
    for query in queries:
        retrieved = rankings[query["query_id"]]
        hard_negatives = filter_known_positives(
            [item["doc_id"] for item in retrieved], query["positive_doc_ids"], negative_pool_size
        )
        rows.append(
            {
                "query_id": query["query_id"],
                "positive_doc_ids": query["positive_doc_ids"],
                "hard_negative_doc_ids": hard_negatives,
                "retrieved": retrieved,
            }
        )
    write_jsonl(path, rows)


def mine_from_checkpoint(config: Mapping[str, Any], checkpoint_path: str | Path, split: str, output: str | Path) -> None:
    torch, _, transformers = require_training_dependencies()
    AutoTokenizer, _ = transformers
    device = select_device(torch)
    data_dir = Path(config["data"]["prepared_dir"])
    corpus = load_corpus(data_dir / "corpus.jsonl")
    queries = load_queries(data_dir / f"queries.{split}.jsonl")
    model, checkpoint = load_dual_encoder_checkpoint(checkpoint_path, device)
    tokenizer_name = checkpoint.get("tokenizer_name", checkpoint["model_config"]["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    rankings = retrieve(corpus, queries, model, tokenizer, config, device)
    write_mined_candidates(
        output, queries, rankings,
        int(config["mining"].get("negative_pool_size", config["mining"]["top_k"])),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine AR2 hard negatives from a retriever checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    mine_from_checkpoint(load_config(args.config), args.checkpoint, args.split, args.output)
    print(f"Hard negatives written to {args.output}")


if __name__ == "__main__":
    main()
