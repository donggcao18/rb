"""Document-ID retrieval metrics for queries with one or more relevant documents."""

import math
from typing import Dict, Iterable, List, Sequence, Tuple


def _average_precision_at_k(ranked_ids: Sequence[str], positives: set, k: int) -> float:
    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in positives:
            hits += 1
            precision_sum += hits / rank
    denominator = min(len(positives), k)
    return precision_sum / denominator if denominator else 0.0


def _ndcg_at_k(ranked_ids: Sequence[str], positives: set, k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked_ids[:k], start=1)
        if doc_id in positives
    )
    ideal_hits = min(len(positives), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal if ideal else 0.0


def evaluate_multilabel_retrieval(
    positive_ids: Sequence[Iterable[str]],
    results: Sequence[Tuple[Sequence[object], Sequence[float]]],
    cutoffs: Sequence[int],
) -> Tuple[Dict[str, float], List[Dict[str, float]], List[List[bool]]]:
    if len(positive_ids) != len(results):
        raise ValueError("positive_ids and results must have the same length")
    normalized_cutoffs = sorted(set(int(value) for value in cutoffs if int(value) > 0))
    if not normalized_cutoffs:
        raise ValueError("At least one positive metric cutoff is required")

    totals = {"hit@{}".format(k): 0.0 for k in normalized_cutoffs}
    for k in normalized_cutoffs:
        for name in ["precision", "recall", "mrr", "map", "ndcg"]:
            totals["{}@{}".format(name, k)] = 0.0

    per_query = []
    relevance_flags = []
    for relevant_values, result in zip(positive_ids, results):
        relevant = set(str(value) for value in relevant_values)
        if not relevant:
            raise ValueError("Every evaluation query must have at least one positive document ID")
        ranked = [str(value) for value in result[0]]
        flags = [doc_id in relevant for doc_id in ranked]
        relevance_flags.append(flags)
        query_metrics = {}
        for k in normalized_cutoffs:
            top_flags = flags[:k]
            hit_count = sum(top_flags)
            first_rank = next((rank for rank, value in enumerate(top_flags, start=1) if value), None)
            values = {
                "hit@{}".format(k): float(hit_count > 0),
                "precision@{}".format(k): hit_count / float(min(k, len(ranked)) or 1),
                "recall@{}".format(k): hit_count / float(len(relevant)),
                "mrr@{}".format(k): 1.0 / first_rank if first_rank else 0.0,
                "map@{}".format(k): _average_precision_at_k(ranked, relevant, k),
                "ndcg@{}".format(k): _ndcg_at_k(ranked, relevant, k),
            }
            query_metrics.update(values)
            for key, value in values.items():
                totals[key] += value
        per_query.append(query_metrics)

    count = len(results)
    aggregate = {key: value / count if count else 0.0 for key, value in totals.items()}
    aggregate["queries"] = count
    return aggregate, per_query, relevance_flags
