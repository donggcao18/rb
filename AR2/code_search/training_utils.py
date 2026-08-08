from __future__ import annotations

import contextlib
import math
from pathlib import Path
from typing import Any, Iterable


def require_training_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise RuntimeError(
            "Training requires torch and transformers. Install requirements-code-search.txt on the GPU machine."
        ) from exc
    return torch, DataLoader, (AutoTokenizer, get_linear_schedule_with_warmup)


def select_device(torch: Any) -> Any:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_context(torch: Any, device: Any, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def optimizer_steps_per_epoch(dataset_size: int, batch_size: int, accumulation: int) -> int:
    return max(1, math.ceil(dataset_size / batch_size / accumulation))


def infinite_batches(loader: Iterable[Any]):
    while True:
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        if not yielded:
            raise ValueError("Cannot iterate an empty data loader")


def save_checkpoint(path: str | Path, model: Any, optimizer: Any, step: int, extra: dict[str, Any]) -> None:
    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        **extra,
    }
    torch.save(payload, target)


def load_dual_encoder_checkpoint(path: str | Path, device: Any) -> tuple[Any, dict[str, Any]]:
    import torch

    from .models import create_dual_encoder

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    model = create_dual_encoder(**config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    return model, checkpoint


def load_ranker_checkpoint(path: str | Path, device: Any) -> tuple[Any, dict[str, Any]]:
    import torch

    from .models import create_cross_encoder

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    model = create_cross_encoder(**config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    return model, checkpoint


def token_inputs(tokens: dict[str, Any], device: Any) -> tuple[Any, Any]:
    return tokens["input_ids"].to(device), tokens["attention_mask"].to(device)


def set_trainable(model: Any, trainable: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(trainable)
