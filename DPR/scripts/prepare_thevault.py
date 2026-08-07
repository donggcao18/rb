#!/usr/bin/env python3
"""Convert mixed TheVault Query/Code JSONL files into DPR artifacts."""

import argparse
import csv
import hashlib
import json
import logging
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dpr.data.thevault_utils import (  # noqa: E402
    CODE_PREFIX_RE,
    QUERY_PREFIX_RE,
    iter_json_records,
    normalize_thevault_query,
    strip_code_prefix,
    strip_query_prefix,
)

logger = logging.getLogger("prepare_thevault")


class UnionFind:
    def __init__(self):
        self.parent = {}

    def add(self, item):
        self.parent.setdefault(item, item)

    def find(self, item):
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def stable_query_id(normalized_query: str) -> str:
    return hashlib.sha1(normalized_query.encode("utf-8")).hexdigest()[:20]


def build_title(record: Dict) -> str:
    params = []
    for parameter in record.get("parameters") or []:
        if isinstance(parameter, dict):
            params.append(str(parameter.get("param", "")))
        else:
            params.append(str(parameter))
    signature = "{}({})".format(record.get("identifier", ""), ",".join(params))
    return " | ".join(
        value for value in [str(record.get("repo", "")), str(record.get("path", "")), signature] if value
    )


def read_original(path: str) -> Tuple[List[Dict], List[Dict], int]:
    queries = []
    codes = []
    ignored = 0
    for line_number, record in enumerate(iter_json_records(path), start=1):
        text = str(record.get("text", ""))
        if QUERY_PREFIX_RE.match(text):
            queries.append(record)
        elif CODE_PREFIX_RE.match(text):
            codes.append(record)
        else:
            ignored += 1
            logger.warning("Ignoring untyped row %s:%d", path, line_number)
    return queries, codes, ignored


def build_corpus(code_rows: Iterable[Dict]) -> OrderedDict:
    corpus = OrderedDict()
    for row in code_rows:
        if "text_id" not in row:
            raise ValueError("Code row is missing text_id: {}".format(row))
        doc_id = str(row["text_id"])
        passage = {
            "id": doc_id,
            "text": strip_code_prefix(row.get("text", "")),
            "title": build_title(row),
            "metadata": {
                key: row.get(key)
                for key in ["repo", "path", "identifier", "parameters", "language", "url_based_id", "hexsha"]
            },
        }
        if not passage["text"]:
            raise ValueError("Code document {} has empty text".format(doc_id))
        previous = corpus.get(doc_id)
        if previous and (previous["text"], previous["title"]) != (passage["text"], passage["title"]):
            raise ValueError("Conflicting code rows share text_id {}".format(doc_id))
        corpus.setdefault(doc_id, passage)
    return corpus


def group_queries(query_rows: Iterable[Dict]) -> OrderedDict:
    groups = OrderedDict()
    for row in query_rows:
        if "text_id" not in row:
            raise ValueError("Query row is missing text_id: {}".format(row))
        original_query = strip_query_prefix(row.get("text", ""))
        normalized = normalize_thevault_query(original_query)
        if not normalized:
            raise ValueError("Query row has empty text: {}".format(row))
        if normalized not in groups:
            groups[normalized] = {
                "query_id": stable_query_id(normalized),
                "question": original_query,
                "normalized_query": normalized,
                "positive_ids": [],
            }
        doc_id = str(row["text_id"])
        if doc_id not in groups[normalized]["positive_ids"]:
            groups[normalized]["positive_ids"].append(doc_id)
    return groups


def connected_split(groups: OrderedDict, dev_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    if not 0 <= dev_ratio < 1:
        raise ValueError("dev_ratio must be in [0, 1)")
    values = list(groups.values())
    if dev_ratio == 0 or len(values) < 2:
        return values, []

    union_find = UnionFind()
    doc_owner = {}
    for group in values:
        query_key = "q:" + group["query_id"]
        union_find.add(query_key)
        for doc_id in group["positive_ids"]:
            doc_key = "d:" + doc_id
            union_find.union(query_key, doc_key)
            if doc_id in doc_owner:
                union_find.union(query_key, doc_owner[doc_id])
            else:
                doc_owner[doc_id] = query_key

    components = defaultdict(list)
    for group in values:
        components[union_find.find("q:" + group["query_id"])].append(group)
    component_values = list(components.values())
    component_values.sort(
        key=lambda component: hashlib.sha1(
            (str(seed) + ":" + component[0]["query_id"]).encode("utf-8")
        ).hexdigest()
    )
    dev_count = max(1, round(len(component_values) * dev_ratio))
    dev_count = min(dev_count, len(component_values) - 1)
    dev_components = component_values[:dev_count]
    train_components = component_values[dev_count:]
    return (
        [group for component in train_components for group in component],
        [group for component in dev_components for group in component],
    )


def materialize_training(groups: Iterable[Dict], corpus: Dict[str, Dict]) -> List[Dict]:
    result = []
    for group in groups:
        positive_ctxs = []
        for doc_id in group["positive_ids"]:
            passage = corpus.get(doc_id)
            if passage:
                positive_ctxs.append({key: passage[key] for key in ["id", "text", "title"]})
        if positive_ctxs:
            result.append(
                {
                    "query_id": group["query_id"],
                    "question": group["question"],
                    "positive_ids": group["positive_ids"],
                    "positive_ctxs": positive_ctxs,
                    "negative_ctxs": [],
                    "hard_negative_ctxs": [],
                }
            )
    return result


def write_jsonl(path: Path, records: Iterable[Dict]):
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare(
    train_path: str,
    test_path: str,
    output_dir: str,
    dev_ratio: float = 0.05,
    seed: int = 12345,
    test_relevance_scope: str = "global",
    allow_missing_positive_ids: bool = False,
) -> Dict:
    train_queries, train_codes, train_ignored = read_original(train_path)
    test_queries, test_codes, test_ignored = read_original(test_path)
    corpus = build_corpus(train_codes + test_codes)
    train_groups = group_queries(train_queries)
    test_groups = group_queries(test_queries)

    if test_relevance_scope == "global":
        all_groups = group_queries(train_queries + test_queries)
        for normalized, group in test_groups.items():
            group["positive_ids"] = list(all_groups[normalized]["positive_ids"])
    elif test_relevance_scope != "test":
        raise ValueError("test_relevance_scope must be 'global' or 'test'")

    all_required_ids: Set[str] = set()
    for group in list(train_groups.values()) + list(test_groups.values()):
        all_required_ids.update(group["positive_ids"])
    missing_ids = sorted(all_required_ids.difference(corpus.keys()))
    if missing_ids and not allow_missing_positive_ids:
        preview = ", ".join(missing_ids[:10])
        raise ValueError(
            "{} positive text_id values are absent from the code corpus (first: {}). "
            "Provide the complete files or pass --allow-missing-positive-ids for diagnostics.".format(
                len(missing_ids), preview
            )
        )

    train_groups_list, dev_groups_list = connected_split(train_groups, dev_ratio, seed)
    train_records = materialize_training(train_groups_list, corpus)
    dev_records = materialize_training(dev_groups_list, corpus)
    test_records = [
        {
            "query_id": group["query_id"],
            "question": group["question"],
            "positive_ids": [doc_id for doc_id in group["positive_ids"] if doc_id in corpus],
        }
        for group in test_groups.values()
    ]
    test_records = [record for record in test_records if record["positive_ids"]]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "corpus.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "text", "title"])
        for passage in corpus.values():
            writer.writerow([passage["id"], passage["text"], passage["title"]])
    write_jsonl(output / "train.jsonl", train_records)
    write_jsonl(output / "dev.jsonl", dev_records)
    write_jsonl(output / "test.jsonl", test_records)

    train_normalized = set(train_groups)
    test_normalized = set(test_groups)
    manifest = {
        "inputs": {"train": str(Path(train_path)), "test": str(Path(test_path))},
        "policy": {
            "query_grouping": "strip Query prefix, lowercase, trim, collapse whitespace",
            "test_relevance_scope": test_relevance_scope,
            "dev_ratio": dev_ratio,
            "seed": seed,
            "pseudo_queries_used": False,
        },
        "counts": {
            "train_query_rows": len(train_queries),
            "test_query_rows": len(test_queries),
            "code_documents": len(corpus),
            "unique_train_queries": len(train_groups),
            "unique_test_queries": len(test_groups),
            "train_samples": len(train_records),
            "dev_samples": len(dev_records),
            "test_samples": len(test_records),
            "train_ignored_rows": train_ignored,
            "test_ignored_rows": test_ignored,
            "missing_positive_ids": len(missing_ids),
            "train_test_normalized_query_overlap": len(train_normalized.intersection(test_normalized)),
        },
        "multilabel": {
            "train_groups": sum(len(group["positive_ids"]) > 1 for group in train_groups.values()),
            "test_groups": sum(len(group["positive_ids"]) > 1 for group in test_groups.values()),
            "max_train_positives": max((len(group["positive_ids"]) for group in train_groups.values()), default=0),
            "max_test_positives": max((len(group["positive_ids"]) for group in test_groups.values()), default=0),
        },
        "missing_positive_ids": missing_ids,
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Mixed Query/Code training JSON or JSONL")
    parser.add_argument("--test", required=True, help="Test query JSON or JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dev-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--test-relevance-scope", choices=["global", "test"], default="global")
    parser.add_argument("--allow-missing-positive-ids", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manifest = prepare(
        args.train,
        args.test,
        args.output_dir,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
        test_relevance_scope=args.test_relevance_scope,
        allow_missing_positive_ids=args.allow_missing_positive_ids,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
