from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping

from .common import load_config, seed_everything
from .datasets import AR2Collator, CandidateDataset, load_corpus, load_hard_negatives, load_queries
from .losses import multi_positive_listwise_loss, multi_positive_top1_accuracy
from .models import create_cross_encoder
from .training_utils import autocast_context, require_training_dependencies, save_checkpoint, token_inputs


def train(config: Mapping[str, Any], hard_negative_path: str | Path | None = None) -> Path:
    torch, DataLoader, transformers = require_training_dependencies()
    AutoTokenizer, get_scheduler = transformers
    runtime = config.get("runtime", {})
    ranker_config = config["ranker"]
    data_dir = Path(config["data"]["prepared_dir"])
    seed_everything(int(runtime.get("seed", 13)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corpus = load_corpus(data_dir / "corpus.jsonl")
    queries = load_queries(data_dir / "queries.train.jsonl")
    negative_path = hard_negative_path or config["data"].get("initial_hard_negatives")
    hard_negatives = load_hard_negatives(negative_path)
    if not hard_negatives:
        raise ValueError("Ranker warm-up requires mined hard negatives")
    dataset = CandidateDataset(
        queries, corpus, hard_negatives,
        max_positives=int(ranker_config.get("max_positives_per_query", 2)),
        max_negatives=int(ranker_config.get("negatives_per_query", 15)),
        seed=int(runtime.get("seed", 13)),
    )
    tokenizer = AutoTokenizer.from_pretrained(ranker_config["model_name"])
    collator = AR2Collator(
        tokenizer, corpus, config["retriever"]["query_max_length"],
        config["retriever"]["code_max_length"], ranker_config["max_length"],
        config["retriever"].get("include_metadata", True),
    )
    loader = DataLoader(
        dataset, batch_size=int(ranker_config["per_device_batch_size"]), shuffle=True,
        collate_fn=collator, num_workers=int(runtime.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )
    model = create_cross_encoder(model_name=ranker_config["model_name"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(ranker_config.get("learning_rate", 1e-5)),
        weight_decay=float(ranker_config.get("weight_decay", 0.01)),
    )
    accumulation = int(ranker_config.get("gradient_accumulation", 1))
    epochs = int(ranker_config.get("warmup_epochs", 2))
    max_steps = int(ranker_config.get("max_steps", 0)) or epochs * max(1, math.ceil(len(loader) / accumulation))
    scheduler = get_scheduler(optimizer, int(max_steps * ranker_config.get("warmup_ratio", 0.1)), max_steps)
    precision = runtime.get("precision", "bf16")
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and precision == "fp16")

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    for _ in range(epochs if not ranker_config.get("max_steps") else 10**9):
        for micro_step, batch in enumerate(loader, start=1):
            pair_ids, pair_mask = token_inputs(batch["pair_tokens"], device)
            positive_mask = batch["positive_mask"].to(device)
            candidate_mask = batch["candidate_mask"].to(device)
            with autocast_context(torch, device, precision):
                scores = model(pair_ids, pair_mask).reshape(positive_mask.shape)
                loss = multi_positive_listwise_loss(scores, positive_mask, candidate_mask) / accumulation
            scaler.scale(loss).backward()
            if micro_step % accumulation != 0 and micro_step != len(loader):
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(ranker_config.get("max_grad_norm", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % int(runtime.get("log_every", 50)) == 0:
                accuracy = multi_positive_top1_accuracy(scores.detach(), positive_mask, candidate_mask).item()
                print(f"ranker step={global_step} loss={loss.item() * accumulation:.6f} top1={accuracy:.4f}")
            if global_step >= max_steps:
                break
        if global_step >= max_steps:
            break

    output_dir = Path(config["output"]["dir"])
    checkpoint_path = output_dir / "ranker-warm.pt"
    save_checkpoint(
        checkpoint_path, model, optimizer, global_step,
        {"model_config": model.export_config(), "tokenizer_name": ranker_config["model_name"]},
    )
    tokenizer.save_pretrained(output_dir / "ranker-tokenizer")
    print(f"Saved warm ranker to {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm up the multi-positive AR2 ranker")
    parser.add_argument("--config", required=True)
    parser.add_argument("--hard-negatives")
    args = parser.parse_args()
    train(load_config(args.config), args.hard_negatives)


if __name__ == "__main__":
    main()
