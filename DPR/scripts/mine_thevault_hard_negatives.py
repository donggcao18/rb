#!/usr/bin/env python3
"""Attach retrieved hard negatives while excluding every known positive ID."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def read_jsonl(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def mine_hard_negatives(
    train_records: List[Dict],
    retrieval_records: List[Dict],
    num_hard_negatives: int,
) -> Tuple[List[Dict], Dict[str, int]]:
    if num_hard_negatives < 1:
        raise ValueError("num_hard_negatives must be at least 1")

    retrieval_by_query = {}
    for result in retrieval_records:
        query_id = str(result.get("query_id", ""))
        if not query_id:
            raise ValueError("A retrieval result is missing query_id")
        if query_id in retrieval_by_query:
            raise ValueError("Duplicate retrieval result for query_id {}".format(query_id))
        retrieval_by_query[query_id] = result

    output = []
    positives_filtered = 0
    selected_total = 0
    queries_without_negatives = 0
    missing_query_ids = []

    for record in train_records:
        query_id = str(record.get("query_id", ""))
        retrieval = retrieval_by_query.get(query_id)
        if retrieval is None:
            missing_query_ids.append(query_id)
            continue

        positive_ids = {str(value) for value in record.get("positive_ids", [])}
        positive_ids.update(str(ctx["id"]) for ctx in record.get("positive_ctxs", []))
        if not positive_ids:
            raise ValueError("Training query {} has no positive document IDs".format(query_id))

        selected = []
        seen_ids = set()
        for context in retrieval.get("ctxs", []):
            document_id = str(context.get("id", ""))
            if not document_id:
                continue
            if document_id in positive_ids:
                positives_filtered += 1
                continue
            if document_id in seen_ids:
                continue
            if "text" not in context:
                raise ValueError(
                    "Retrieved document {} for query {} has no text".format(document_id, query_id)
                )

            hard_negative = {
                "id": document_id,
                "text": context["text"],
                "title": context.get("title"),
            }
            if "score" in context:
                hard_negative["score"] = context["score"]
            selected.append(hard_negative)
            seen_ids.add(document_id)
            if len(selected) >= num_hard_negatives:
                break

        if not selected:
            queries_without_negatives += 1

        mined_record = dict(record)
        mined_record["hard_negative_ctxs"] = selected
        output.append(mined_record)
        selected_total += len(selected)

    if missing_query_ids:
        preview = ", ".join(missing_query_ids[:10])
        raise ValueError(
            "Retrieval results are missing {} training queries (first: {})".format(
                len(missing_query_ids), preview
            )
        )

    for record in output:
        positives = {str(value) for value in record.get("positive_ids", [])}
        overlap = positives.intersection(str(ctx["id"]) for ctx in record["hard_negative_ctxs"])
        if overlap:
            raise AssertionError(
                "Hard-negative/positive overlap for query {}: {}".format(
                    record.get("query_id"), sorted(overlap)
                )
            )

    stats = {
        "train_samples": len(train_records),
        "retrieval_results": len(retrieval_records),
        "output_samples": len(output),
        "hard_negatives_selected": selected_total,
        "known_positives_filtered": positives_filtered,
        "queries_without_hard_negatives": queries_without_negatives,
        "hard_negative_pool_size": num_hard_negatives,
    }
    return output, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path, help="Prepared train.jsonl")
    parser.add_argument(
        "--retrieval-results",
        required=True,
        type=Path,
        help="dense_retriever.py JSON output for ruby_train_for_mining",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-hard-negatives", type=int, default=20)
    args = parser.parse_args()

    train_records = read_jsonl(args.train)
    with args.retrieval_results.open(encoding="utf-8") as stream:
        retrieval_payload = json.load(stream)
    retrieval_records = retrieval_payload.get("results")
    if not isinstance(retrieval_records, list):
        raise ValueError("Retrieval JSON must contain a results list")

    output, stats = mine_hard_negatives(
        train_records,
        retrieval_records,
        args.num_hard_negatives,
    )
    write_jsonl(args.output, output)
    print(json.dumps(stats, indent=2, sort_keys=True))
    print("Wrote {}".format(args.output))


if __name__ == "__main__":
    main()
