from __future__ import annotations

import argparse
import hashlib
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .common import (
    iter_json_records,
    normalize_query,
    record_kind,
    strip_code_prefix,
    strip_query_prefix,
    write_json,
    write_jsonl,
)


def _string_id(row: dict[str, Any], field: str, source: str) -> str:
    value = row.get(field)
    if value is None or str(value) == "":
        raise ValueError(f"Missing {field} in {source} record: {row}")
    return str(value)


def _stable_query_id(split: str, rows: list[dict[str, Any]], normalized: str) -> str:
    for row in rows:
        value = row.get("numeric_id")
        if value is not None and str(value):
            return f"{split}:{value}"
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{split}:sha1:{digest}"


def _read_source(path: str | Path, source_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    queries: list[dict[str, Any]] = []
    codes: list[dict[str, Any]] = []
    kinds: Counter = Counter()
    for row in iter_json_records(path):
        kind = record_kind(row.get("text", ""))
        kinds[kind] += 1
        enriched = dict(row)
        enriched["_source"] = source_name
        if kind == "query":
            queries.append(enriched)
        elif kind == "code":
            codes.append(enriched)
    return queries, codes, kinds


def build_corpus(code_rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    by_id: dict[str, dict[str, Any]] = {}
    conflicts: list[str] = []
    for row in code_rows:
        doc_id = _string_id(row, "text_id", row.get("_source", "input"))
        document = {
            "doc_id": doc_id,
            "code": strip_code_prefix(row.get("text", "")),
            "repo": row.get("repo", ""),
            "path": row.get("path", ""),
            "identifier": row.get("identifier", ""),
            "url_based_id": row.get("url_based_id", row.get("url_id", "")),
        }
        previous = by_id.get(doc_id)
        if previous is None:
            by_id[doc_id] = document
        elif previous["code"] != document["code"]:
            conflicts.append(doc_id)
    return list(by_id.values()), sorted(set(conflicts))


def group_queries(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        normalized = normalize_query(row.get("text", ""))
        if normalized:
            groups[normalized].append(row)
    return dict(groups)


def _positive_ids(rows: Iterable[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(_string_id(row, "text_id", row.get("_source", "input")) for row in rows))


def materialize_queries(
    split: str,
    groups: dict[str, list[dict[str, Any]]],
    positive_scope: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for normalized in sorted(groups):
        rows = groups[normalized]
        positives = positive_scope.get(normalized) if positive_scope is not None else _positive_ids(rows)
        positives = list(dict.fromkeys(positives or []))
        first = rows[0]
        output.append(
            {
                "query_id": _stable_query_id(split, rows, normalized),
                "query": strip_query_prefix(first.get("text", "")),
                "normalized_query": normalized,
                "positive_doc_ids": positives,
                "source_numeric_ids": [str(row.get("numeric_id", "")) for row in rows],
            }
        )
    return output


def split_train_dev(
    train_groups: dict[str, list[dict[str, Any]]], dev_fraction: float, seed: int
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    if not 0 <= dev_fraction < 1:
        raise ValueError("dev_fraction must be in [0, 1)")
    keys = sorted(train_groups)
    if dev_fraction == 0 or len(keys) < 2:
        return dict(train_groups), {}
    rng = random.Random(seed)
    rng.shuffle(keys)
    dev_count = max(1, round(len(keys) * dev_fraction))
    dev_keys = set(keys[:dev_count])
    train = {key: train_groups[key] for key in keys if key not in dev_keys}
    dev = {key: train_groups[key] for key in keys if key in dev_keys}
    return train, dev


def _write_qrels(path: Path, queries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("query_id\tdoc_id\trelevance\n")
        for query in queries:
            for doc_id in query["positive_doc_ids"]:
                handle.write(f"{query['query_id']}\t{doc_id}\t1\n")


def _length_summary(values: Iterable[str]) -> dict[str, float]:
    lengths = sorted(len(value.split()) for value in values)
    if not lengths:
        return {"count": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}

    def percentile(fraction: float) -> int:
        return lengths[min(len(lengths) - 1, max(0, math.ceil(len(lengths) * fraction) - 1))]

    return {
        "count": len(lengths),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": lengths[-1],
    }


def prepare_dataset(
    train_path: str | Path,
    output_dir: str | Path,
    test_path: str | Path | None = None,
    dev_fraction: float = 0.1,
    seed: int = 13,
    global_qrels: bool = False,
    allow_missing_documents: bool = False,
) -> dict[str, Any]:
    train_queries_raw, train_codes, train_kinds = _read_source(train_path, "train")
    if test_path:
        test_queries_raw, test_codes, test_kinds = _read_source(test_path, "test")
    else:
        test_queries_raw, test_codes, test_kinds = [], [], Counter()

    corpus, conflicts = build_corpus([*train_codes, *test_codes])
    if conflicts:
        raise ValueError(f"Conflicting code content for document IDs: {conflicts[:10]}")
    corpus_ids = {row["doc_id"] for row in corpus}

    train_groups_all = group_queries(train_queries_raw)
    test_groups = group_queries(test_queries_raw)
    overlap = sorted(set(train_groups_all).intersection(test_groups))

    global_positive_scope: dict[str, list[str]] | None = None
    if global_qrels:
        global_positive_scope = {}
        for normalized in set(train_groups_all).union(test_groups):
            global_positive_scope[normalized] = _positive_ids(
                [*train_groups_all.get(normalized, []), *test_groups.get(normalized, [])]
            )

    train_groups, dev_groups = split_train_dev(train_groups_all, dev_fraction, seed)
    train_queries = materialize_queries("train", train_groups, global_positive_scope)
    dev_queries = materialize_queries("dev", dev_groups, global_positive_scope)
    test_queries = materialize_queries("test", test_groups, global_positive_scope)
    all_queries = [*train_queries, *dev_queries, *test_queries]

    missing_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, queries in (("train", train_queries), ("dev", dev_queries), ("test", test_queries)):
        missing = []
        for query in queries:
            absent = [doc_id for doc_id in query["positive_doc_ids"] if doc_id not in corpus_ids]
            if absent:
                missing.append({"query_id": query["query_id"], "missing_doc_ids": absent})
        missing_by_split[split] = missing

    missing_count = sum(len(item["missing_doc_ids"]) for values in missing_by_split.values() for item in values)
    if missing_count and not allow_missing_documents:
        preview = [item for values in missing_by_split.values() for item in values][:5]
        raise ValueError(
            f"{missing_count} positive document references are missing from the code corpus. "
            f"Examples: {preview}. Use --allow-missing-documents only for incomplete samples."
        )

    target = Path(output_dir)
    write_jsonl(target / "corpus.jsonl", corpus)
    for split, queries in (("train", train_queries), ("dev", dev_queries), ("test", test_queries)):
        write_jsonl(target / f"queries.{split}.jsonl", queries)
        _write_qrels(target / f"qrels.{split}.tsv", queries)

    group_sizes = [len(query["positive_doc_ids"]) for query in all_queries]
    audit = {
        "inputs": {
            "train": str(train_path),
            "test": str(test_path) if test_path else None,
            "global_qrels": global_qrels,
        },
        "raw_record_kinds": {"train": dict(train_kinds), "test": dict(test_kinds)},
        "corpus_documents": len(corpus),
        "query_groups": {
            "train": len(train_queries),
            "dev": len(dev_queries),
            "test": len(test_queries),
        },
        "multi_positive_groups": sum(size > 1 for size in group_sizes),
        "max_positive_group_size": max(group_sizes, default=0),
        "train_test_normalized_query_overlap": len(overlap),
        "overlap_examples": overlap[:20],
        "missing_positive_references": missing_count,
        "missing_by_split": missing_by_split,
        "whitespace_token_lengths": {
            "queries": _length_summary(query["query"] for query in all_queries),
            "code": _length_summary(document["code"] for document in corpus),
        },
    }
    write_json(target / "audit.json", audit)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare TheVault Ruby data for multi-positive AR2")
    parser.add_argument("--train", required=True, help="Original Ruby train JSON/JSONL")
    parser.add_argument("--test", help="Original Ruby test JSON/JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--global-qrels", action="store_true")
    parser.add_argument("--allow-missing-documents", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    audit = prepare_dataset(
        train_path=args.train,
        test_path=args.test,
        output_dir=args.output_dir,
        dev_fraction=args.dev_fraction,
        seed=args.seed,
        global_qrels=args.global_qrels,
        allow_missing_documents=args.allow_missing_documents,
    )
    print(f"Prepared corpus with {audit['corpus_documents']} documents")
    print(f"Query groups: {audit['query_groups']}")
    print(f"Multi-positive groups: {audit['multi_positive_groups']}")
    print(f"Missing positive references: {audit['missing_positive_references']}")


if __name__ == "__main__":
    main()
