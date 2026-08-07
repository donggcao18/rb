from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .common import write_json
from .datasets import load_corpus, load_queries


def validate_prepared_data(prepared_dir: str | Path) -> dict[str, Any]:
    root = Path(prepared_dir)
    corpus = load_corpus(root / "corpus.jsonl")
    report: dict[str, Any] = {
        "corpus_documents": len(corpus),
        "splits": {},
        "cross_split_normalized_query_overlap": {},
    }
    normalized_by_split: dict[str, set[str]] = {}
    failures: list[str] = []
    for split in ("train", "dev", "test"):
        path = root / f"queries.{split}.jsonl"
        queries = load_queries(path) if path.exists() else []
        query_ids = [query["query_id"] for query in queries]
        missing = {
            query["query_id"]: [doc_id for doc_id in query["positive_doc_ids"] if doc_id not in corpus]
            for query in queries
        }
        missing = {query_id: values for query_id, values in missing.items() if values}
        if len(query_ids) != len(set(query_ids)):
            failures.append(f"duplicate query IDs in {split}")
        if missing:
            failures.append(f"missing positive documents in {split}")
        normalized_by_split[split] = {str(query.get("normalized_query", "")) for query in queries}
        report["splits"][split] = {
            "queries": len(queries),
            "multi_positive_queries": sum(len(query["positive_doc_ids"]) > 1 for query in queries),
            "max_positives": max((len(query["positive_doc_ids"]) for query in queries), default=0),
            "missing_positive_documents": missing,
        }
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = sorted(normalized_by_split[left].intersection(normalized_by_split[right]) - {""})
        report["cross_split_normalized_query_overlap"][f"{left}_{right}"] = overlap[:100]
        if left == "train" and right == "dev" and overlap:
            failures.append("normalized queries overlap between train and dev")
    report["valid"] = not failures
    report["failures"] = failures
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate canonical multi-positive AR2 data")
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = validate_prepared_data(args.prepared_dir)
    if args.output:
        write_json(args.output, report)
    print(report)
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
