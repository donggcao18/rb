from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence


def _deduplicate(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in items))


def evaluate_run(
    run: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Iterable[str]],
    cutoffs: Sequence[int] = (1, 5, 10, 100),
) -> dict[str, float]:
    """Evaluate binary multi-positive qrels against ranked document IDs."""
    if not qrels:
        raise ValueError("qrels is empty")
    ks = sorted(set(int(k) for k in cutoffs if int(k) > 0))
    totals = {f"hit@{k}": 0.0 for k in ks}
    totals.update({f"recall@{k}": 0.0 for k in ks})
    totals.update({f"mrr@{k}": 0.0 for k in ks})
    totals.update({f"map@{k}": 0.0 for k in ks})
    totals.update({f"ndcg@{k}": 0.0 for k in ks})

    for query_id, relevant_values in qrels.items():
        relevant = {str(value) for value in relevant_values}
        if not relevant:
            raise ValueError(f"Query {query_id} has no relevant documents")
        ranking = _deduplicate(run.get(query_id, []))
        for k in ks:
            top = ranking[:k]
            hits = [1 if doc_id in relevant else 0 for doc_id in top]
            hit_count = sum(hits)
            totals[f"hit@{k}"] += float(hit_count > 0)
            totals[f"recall@{k}"] += hit_count / len(relevant)

            first = next((rank for rank, value in enumerate(hits, start=1) if value), None)
            totals[f"mrr@{k}"] += 0.0 if first is None else 1.0 / first

            correct = 0
            precision_sum = 0.0
            for rank, value in enumerate(hits, start=1):
                if value:
                    correct += 1
                    precision_sum += correct / rank
            totals[f"map@{k}"] += precision_sum / min(len(relevant), k)

            dcg = sum(value / math.log2(rank + 1) for rank, value in enumerate(hits, start=1))
            ideal_hits = min(len(relevant), k)
            idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            totals[f"ndcg@{k}"] += dcg / idcg if idcg else 0.0

    count = len(qrels)
    return {name: value / count for name, value in totals.items()} | {"queries": float(count)}


def qrels_from_queries(queries: Iterable[Mapping[str, object]]) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    for query in queries:
        query_id = str(query["query_id"])
        output[query_id] = {str(value) for value in query["positive_doc_ids"]}  # type: ignore[index]
    return output
