from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Mapping

from .common import load_config, seed_everything
from .datasets import CandidateDataset, RetrieverCollator, load_corpus, load_hard_negatives, load_queries
from .losses import multi_positive_listwise_loss, multi_positive_top1_accuracy
from .models import create_dual_encoder
from .training_utils import autocast_context, require_training_dependencies, save_checkpoint, token_inputs


def train(config: Mapping[str, Any]) -> Path:
    torch, DataLoader, transformers = require_training_dependencies()
    AutoTokenizer, get_scheduler = transformers
    runtime = config.get("runtime", {})
    retriever_config = config["retriever"]
    data_config = config["data"]
    seed_everything(int(runtime.get("seed", 13)))

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    rank = torch.distributed.get_rank() if distributed else 0

    data_dir = Path(data_config["prepared_dir"])
    corpus = load_corpus(data_dir / "corpus.jsonl")
    queries = load_queries(data_dir / "queries.train.jsonl")
    hard_negatives = load_hard_negatives(data_config.get("initial_hard_negatives"))
    dataset = CandidateDataset(
        queries, corpus, hard_negatives,
        max_positives=int(retriever_config.get("max_positives_per_query", 1)),
        max_negatives=int(retriever_config.get("hard_negatives_per_query", 0)),
        seed=int(runtime.get("seed", 13)),
    )
    tokenizer = AutoTokenizer.from_pretrained(retriever_config["model_name"])
    collator = RetrieverCollator(
        tokenizer, corpus, retriever_config["query_max_length"], retriever_config["code_max_length"],
        retriever_config.get("include_metadata", True),
    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset, batch_size=int(retriever_config["per_device_batch_size"]),
        shuffle=sampler is None, sampler=sampler, collate_fn=collator,
        num_workers=int(runtime.get("num_workers", 4)), pin_memory=device.type == "cuda",
    )

    model = create_dual_encoder(
        model_name=retriever_config["model_name"],
        share_weights=bool(retriever_config.get("share_weights", False)),
        pooling=retriever_config.get("pooling", "cls"),
        normalize_embeddings=bool(retriever_config.get("normalize_embeddings", False)),
    ).to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(retriever_config.get("learning_rate", 2e-5)),
        weight_decay=float(retriever_config.get("weight_decay", 0.01)),
    )
    accumulation = int(retriever_config.get("gradient_accumulation", 1))
    epochs = int(retriever_config.get("warmup_epochs", 3))
    steps_per_epoch = max(1, math.ceil(len(loader) / accumulation))
    max_steps = int(retriever_config.get("max_steps", 0)) or epochs * steps_per_epoch
    warmup_steps = int(max_steps * float(retriever_config.get("warmup_ratio", 0.1)))
    scheduler = get_scheduler(optimizer, warmup_steps, max_steps)
    precision = runtime.get("precision", "bf16")
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and precision == "fp16")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    for epoch in range(epochs if not retriever_config.get("max_steps") else 10**9):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for micro_step, batch in enumerate(loader, start=1):
            query_ids, query_mask = token_inputs(batch["query_tokens"], device)
            code_ids, code_mask = token_inputs(batch["code_tokens"], device)
            positive_mask = batch["positive_mask"].to(device)
            with autocast_context(torch, device, precision):
                scores = model(query_ids, query_mask, code_ids, code_mask)
                loss = multi_positive_listwise_loss(scores, positive_mask) / accumulation
            scaler.scale(loss).backward()
            if micro_step % accumulation != 0 and micro_step != len(loader):
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(retriever_config.get("max_grad_norm", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if rank == 0 and global_step % int(runtime.get("log_every", 50)) == 0:
                accuracy = multi_positive_top1_accuracy(scores.detach(), positive_mask).item()
                print(f"retriever step={global_step} loss={loss.item() * accumulation:.6f} top1={accuracy:.4f}")
            if global_step >= max_steps:
                break
        if global_step >= max_steps:
            break

    output_dir = Path(config["output"]["dir"])
    checkpoint_path = output_dir / "retriever-warm.pt"
    if rank == 0:
        unwrapped = model.module if hasattr(model, "module") else model
        save_checkpoint(
            checkpoint_path, unwrapped, optimizer, global_step,
            {"model_config": unwrapped.export_config(), "tokenizer_name": retriever_config["model_name"]},
        )
        tokenizer.save_pretrained(output_dir / "retriever-tokenizer")
        print(f"Saved warm retriever to {checkpoint_path}")
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm up the multi-positive AR2 retriever")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
