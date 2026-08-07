from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from .common import load_config, seed_everything
from .datasets import AR2Collator, CandidateDataset, load_corpus, load_hard_negatives, load_queries
from .losses import ar2_retriever_loss, multi_positive_listwise_loss
from .mine_hard_negatives import retrieve, write_mined_candidates
from .training_utils import (
    autocast_context,
    infinite_batches,
    load_dual_encoder_checkpoint,
    load_ranker_checkpoint,
    require_training_dependencies,
    save_checkpoint,
    set_trainable,
    token_inputs,
)


def _make_loader(
    DataLoader: Any,
    queries: list[dict[str, Any]],
    corpus: Mapping[str, dict[str, Any]],
    negatives_path: str | Path,
    tokenizer: Any,
    config: Mapping[str, Any],
) -> Any:
    hard_negatives = load_hard_negatives(negatives_path)
    if not hard_negatives:
        raise ValueError(f"No hard negatives found in {negatives_path}")
    ar2_config = config["ar2"]
    dataset = CandidateDataset(
        queries, corpus, hard_negatives,
        max_positives=int(ar2_config.get("max_positives_per_query", 2)),
        max_negatives=int(config["mining"]["negatives_per_query"]),
        seed=int(config.get("runtime", {}).get("seed", 13)),
    )
    collator = AR2Collator(
        tokenizer, corpus, config["retriever"]["query_max_length"],
        config["retriever"]["code_max_length"], config["ranker"]["max_length"],
        config["retriever"].get("include_metadata", True),
    )
    return DataLoader(
        dataset, batch_size=int(ar2_config["per_device_batch_size"]), shuffle=True,
        collate_fn=collator, num_workers=int(config.get("runtime", {}).get("num_workers", 4)),
        pin_memory=True,
    )


def _ranker_phase(
    torch: Any, loader: Any, model: Any, optimizer: Any, steps: int, accumulation: int,
    device: Any, precision: str, max_grad_norm: float
) -> float:
    set_trainable(model, True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batches = infinite_batches(loader)
    running = 0.0
    for step in range(steps):
        for _ in range(accumulation):
            batch = next(batches)
            pair_ids, pair_mask = token_inputs(batch["pair_tokens"], device)
            positive_mask = batch["positive_mask"].to(device)
            candidate_mask = batch["candidate_mask"].to(device)
            with autocast_context(torch, device, precision):
                scores = model(pair_ids, pair_mask).reshape(positive_mask.shape)
                loss = multi_positive_listwise_loss(scores, positive_mask, candidate_mask) / accumulation
            loss.backward()
            running += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return running / max(steps, 1)


def _retriever_phase(
    torch: Any, loader: Any, retriever: Any, ranker: Any, optimizer: Any, steps: int,
    accumulation: int, device: Any, precision: str, ar2_config: Mapping[str, Any], max_grad_norm: float
) -> dict[str, float]:
    set_trainable(ranker, False)
    ranker.eval()
    set_trainable(retriever, True)
    retriever.train()
    optimizer.zero_grad(set_to_none=True)
    batches = infinite_batches(loader)
    totals = {"loss": 0.0, "adversarial_loss": 0.0, "distillation_loss": 0.0}
    for _ in range(steps):
        for _ in range(accumulation):
            batch = next(batches)
            query_ids, query_mask = token_inputs(batch["query_tokens"], device)
            code_ids, code_mask = token_inputs(batch["code_tokens"], device)
            pair_ids, pair_mask = token_inputs(batch["pair_tokens"], device)
            positive_mask = batch["positive_mask"].to(device)
            candidate_mask = batch["candidate_mask"].to(device)
            width = int(batch["width"])
            with autocast_context(torch, device, precision):
                query_vectors = retriever.encode_query(query_ids, query_mask)
                code_vectors = retriever.encode_code(code_ids, code_mask).reshape(query_vectors.shape[0], width, -1)
                retriever_scores = torch.einsum("bh,bmh->bm", query_vectors, code_vectors)
                with torch.no_grad():
                    ranker_scores = ranker(pair_ids, pair_mask).reshape(positive_mask.shape)
                loss, parts = ar2_retriever_loss(
                    retriever_scores, ranker_scores, positive_mask, candidate_mask,
                    adv_lambda=float(ar2_config.get("adv_lambda", 0.5)),
                    retriever_temperature=float(ar2_config.get("retriever_temperature", 1.0)),
                    ranker_temperature=float(ar2_config.get("ranker_temperature", 1.0)),
                )
                scaled_loss = loss / accumulation
            scaled_loss.backward()
            totals["loss"] += scaled_loss.item()
            totals["adversarial_loss"] += parts["adversarial_loss"].item() / accumulation
            totals["distillation_loss"] += parts["distillation_loss"].item() / accumulation
        torch.nn.utils.clip_grad_norm_(retriever.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    set_trainable(ranker, True)
    return {key: value / max(steps, 1) for key, value in totals.items()}


def train(
    config: Mapping[str, Any], retriever_checkpoint: str | Path,
    ranker_checkpoint: str | Path, hard_negative_path: str | Path,
    start_round: int = 1,
) -> tuple[Path, Path]:
    torch, DataLoader, transformers = require_training_dependencies()
    AutoTokenizer, _ = transformers
    if int(__import__("os").environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("train_ar2 currently supports one process; use gradient accumulation on one A100")
    runtime = config.get("runtime", {})
    seed_everything(int(runtime.get("seed", 13)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = runtime.get("precision", "bf16")
    data_dir = Path(config["data"]["prepared_dir"])
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(data_dir / "corpus.jsonl")
    queries = load_queries(data_dir / "queries.train.jsonl")
    retriever, retriever_state = load_dual_encoder_checkpoint(retriever_checkpoint, device)
    ranker, ranker_state = load_ranker_checkpoint(ranker_checkpoint, device)
    retriever_tokenizer = AutoTokenizer.from_pretrained(
        retriever_state.get("tokenizer_name", retriever_state["model_config"]["model_name"])
    )
    ranker_tokenizer = AutoTokenizer.from_pretrained(
        ranker_state.get("tokenizer_name", ranker_state["model_config"]["model_name"])
    )
    retriever_tokenizer_name = retriever_state.get(
        "tokenizer_name", retriever_state["model_config"]["model_name"]
    )
    ranker_tokenizer_name = ranker_state.get(
        "tokenizer_name", ranker_state["model_config"]["model_name"]
    )
    if retriever_tokenizer_name != ranker_tokenizer_name:
        raise ValueError(
            "Current aligned AR2 collator requires retriever and ranker to use the same tokenizer"
        )
    tokenizer = ranker_tokenizer

    retriever_optimizer = torch.optim.AdamW(
        retriever.parameters(), lr=float(config["ar2"].get("retriever_learning_rate", 1e-5)),
        weight_decay=float(config["retriever"].get("weight_decay", 0.01)),
    )
    ranker_optimizer = torch.optim.AdamW(
        ranker.parameters(), lr=float(config["ar2"].get("ranker_learning_rate", 1e-6)),
        weight_decay=float(config["ranker"].get("weight_decay", 0.01)),
    )
    current_negatives = Path(hard_negative_path)
    retriever_out = output_dir / "retriever-ar2-final.pt"
    ranker_out = output_dir / "ranker-ar2-final.pt"
    rounds = int(config["ar2"].get("rounds", 5))
    if not 1 <= start_round <= rounds:
        raise ValueError(f"start_round must be between 1 and configured rounds ({rounds})")
    if start_round > 1:
        if retriever_state.get("optimizer"):
            retriever_optimizer.load_state_dict(retriever_state["optimizer"])
        if ranker_state.get("optimizer"):
            ranker_optimizer.load_state_dict(ranker_state["optimizer"])

    for round_index in range(start_round, rounds + 1):
        loader = _make_loader(DataLoader, queries, corpus, current_negatives, tokenizer, config)
        ranker_loss = _ranker_phase(
            torch, loader, ranker, ranker_optimizer,
            int(config["ar2"].get("ranker_steps_per_round", 500)),
            int(config["ar2"].get("ranker_gradient_accumulation", 1)),
            device, precision, float(config["ranker"].get("max_grad_norm", 1.0)),
        )
        retriever_losses = _retriever_phase(
            torch, loader, retriever, ranker, retriever_optimizer,
            int(config["ar2"].get("retriever_steps_per_round", 1500)),
            int(config["ar2"].get("retriever_gradient_accumulation", 1)),
            device, precision, config["ar2"], float(config["retriever"].get("max_grad_norm", 1.0)),
        )
        print(f"round={round_index} ranker_loss={ranker_loss:.6f} retriever={retriever_losses}")

        round_retriever = output_dir / f"retriever-ar2-round-{round_index}.pt"
        round_ranker = output_dir / f"ranker-ar2-round-{round_index}.pt"
        save_checkpoint(
            round_retriever, retriever, retriever_optimizer, round_index,
            {"model_config": retriever.export_config(), "tokenizer_name": retriever_tokenizer_name},
        )
        save_checkpoint(
            round_ranker, ranker, ranker_optimizer, round_index,
            {"model_config": ranker.export_config(), "tokenizer_name": ranker_tokenizer_name},
        )
        rankings = retrieve(corpus, queries, retriever, retriever_tokenizer, config, device)
        current_negatives = output_dir / f"hard-negatives-round-{round_index}.jsonl"
        write_mined_candidates(
            current_negatives, queries, rankings,
            int(config["mining"].get("negative_pool_size", config["mining"]["top_k"])),
        )
        retriever_out, ranker_out = round_retriever, round_ranker

    return retriever_out, ranker_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-positive AR2 alternating co-training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--retriever-checkpoint", required=True)
    parser.add_argument("--ranker-checkpoint", required=True)
    parser.add_argument("--hard-negatives", required=True)
    parser.add_argument("--start-round", type=int, default=1)
    args = parser.parse_args()
    train(
        load_config(args.config), args.retriever_checkpoint, args.ranker_checkpoint,
        args.hard_negatives, args.start_round,
    )


if __name__ == "__main__":
    main()
