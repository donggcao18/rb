from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from .common import format_code_document, load_config, write_json
from .datasets import load_corpus, load_queries
from .metrics import evaluate_run, qrels_from_queries
from .mine_hard_negatives import retrieve
from .training_utils import (
    autocast_context,
    load_dual_encoder_checkpoint,
    load_ranker_checkpoint,
    require_training_dependencies,
    select_device,
    token_inputs,
)


def _rerank(
    rankings: Mapping[str, list[dict[str, Any]]], queries: list[dict[str, Any]],
    corpus: Mapping[str, dict[str, Any]], ranker: Any, tokenizer: Any,
    config: Mapping[str, Any], device: Any
) -> dict[str, list[str]]:
    import torch

    query_by_id = {query["query_id"]: query for query in queries}
    output: dict[str, list[str]] = {}
    batch_size = int(config["ranker"].get("inference_batch_size", 128))
    precision = config.get("runtime", {}).get("precision", "bf16")
    ranker.eval()
    with torch.no_grad():
        for query_id, candidates in rankings.items():
            query_text = query_by_id[query_id]["query"]
            scored: list[tuple[str, float]] = []
            for start in range(0, len(candidates), batch_size):
                chunk = candidates[start : start + batch_size]
                codes = [
                    format_code_document(corpus[item["doc_id"]], config["retriever"].get("include_metadata", True))
                    for item in chunk
                ]
                tokens = tokenizer(
                    [query_text] * len(chunk), codes, padding=True, truncation="only_second",
                    max_length=config["ranker"]["max_length"], return_tensors="pt"
                )
                input_ids, attention_mask = token_inputs(tokens, device)
                with autocast_context(torch, device, precision):
                    scores = ranker(input_ids, attention_mask).float().cpu().tolist()
                scored.extend((item["doc_id"], score) for item, score in zip(chunk, scores))
            output[query_id] = [doc_id for doc_id, _ in sorted(scored, key=lambda pair: pair[1], reverse=True)]
    return output


def evaluate(
    config: Mapping[str, Any], retriever_checkpoint: str | Path, split: str,
    output_path: str | Path, ranker_checkpoint: str | Path | None = None
) -> dict[str, Any]:
    torch, _, transformers = require_training_dependencies()
    AutoTokenizer, _ = transformers
    device = select_device(torch)
    data_dir = Path(config["data"]["prepared_dir"])
    corpus = load_corpus(data_dir / "corpus.jsonl")
    queries = load_queries(data_dir / f"queries.{split}.jsonl")
    retriever, state = load_dual_encoder_checkpoint(retriever_checkpoint, device)
    retriever_tokenizer = AutoTokenizer.from_pretrained(state.get("tokenizer_name", state["model_config"]["model_name"]))
    rankings = retrieve(corpus, queries, retriever, retriever_tokenizer, config, device)
    retriever_run = {query_id: [item["doc_id"] for item in items] for query_id, items in rankings.items()}
    qrels = qrels_from_queries(queries)
    cutoffs = config.get("evaluation", {}).get("cutoffs", [1, 5, 10, 100])
    result: dict[str, Any] = {
        "split": split,
        "retriever": evaluate_run(retriever_run, qrels, cutoffs),
        "retriever_run": retriever_run,
    }
    if ranker_checkpoint:
        ranker, ranker_state = load_ranker_checkpoint(ranker_checkpoint, device)
        ranker_tokenizer = AutoTokenizer.from_pretrained(
            ranker_state.get("tokenizer_name", ranker_state["model_config"]["model_name"])
        )
        reranked_run = _rerank(rankings, queries, corpus, ranker, ranker_tokenizer, config, device)
        result["reranker"] = evaluate_run(reranked_run, qrels, cutoffs)
        result["reranked_run"] = reranked_run
    write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multi-positive AR2 retrieval")
    parser.add_argument("--config", required=True)
    parser.add_argument("--retriever-checkpoint", required=True)
    parser.add_argument("--ranker-checkpoint")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate(
        load_config(args.config), args.retriever_checkpoint, args.split, args.output, args.ranker_checkpoint
    )
    print(result.get("reranker", result["retriever"]))


if __name__ == "__main__":
    main()
