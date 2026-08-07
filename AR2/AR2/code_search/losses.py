from __future__ import annotations

from typing import Any


def _torch_modules() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("AR2 losses require PyTorch") from exc
    return torch, functional


def multi_positive_listwise_loss(
    scores: Any,
    positive_mask: Any,
    valid_mask: Any | None = None,
    reduction: str = "mean",
) -> Any:
    """Marginal log-likelihood of selecting any relevant document.

    ``scores`` and masks have shape ``[queries, candidates]``. Each row must
    contain at least one valid positive.
    """
    torch, _ = _torch_modules()
    if scores.ndim != 2 or positive_mask.shape != scores.shape:
        raise ValueError("scores and positive_mask must have matching [Q, D] shapes")
    positive_mask = positive_mask.bool()
    valid_mask = torch.ones_like(positive_mask) if valid_mask is None else valid_mask.bool()
    if valid_mask.shape != scores.shape:
        raise ValueError("valid_mask must match scores")
    effective_positive = positive_mask & valid_mask
    if not bool(effective_positive.any(dim=1).all()):
        raise ValueError("Every query must have at least one valid positive candidate")
    if not bool(valid_mask.any(dim=1).all()):
        raise ValueError("Every query must have at least one valid candidate")

    negative_infinity = torch.finfo(scores.dtype).min
    denominator = torch.logsumexp(scores.masked_fill(~valid_mask, negative_infinity), dim=1)
    numerator = torch.logsumexp(scores.masked_fill(~effective_positive, negative_infinity), dim=1)
    losses = denominator - numerator
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


def multi_positive_top1_accuracy(scores: Any, positive_mask: Any, valid_mask: Any | None = None) -> Any:
    torch, _ = _torch_modules()
    valid_mask = torch.ones_like(positive_mask, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
    masked = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
    predictions = masked.argmax(dim=1)
    return positive_mask.bool().gather(1, predictions.unsqueeze(1)).float().mean()


def ar2_retriever_loss(
    retriever_scores: Any,
    ranker_scores: Any,
    positive_mask: Any,
    valid_mask: Any | None = None,
    adv_lambda: float = 0.5,
    retriever_temperature: float = 1.0,
    ranker_temperature: float = 1.0,
) -> tuple[Any, dict[str, Any]]:
    """Multi-positive AR2 retriever objective.

    The distillation term matches the ranker's listwise distribution. The
    adversarial policy-gradient term encourages probability mass on negatives
    that the frozen ranker finds difficult. All positives are excluded from
    the adversarial negative distribution.
    """
    torch, functional = _torch_modules()
    if not 0 <= adv_lambda <= 1:
        raise ValueError("adv_lambda must be in [0, 1]")
    if retriever_scores.shape != ranker_scores.shape or retriever_scores.shape != positive_mask.shape:
        raise ValueError("retriever_scores, ranker_scores, and positive_mask must match")
    valid_mask = torch.ones_like(positive_mask, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
    positive_mask = positive_mask.bool() & valid_mask
    negative_mask = valid_mask & ~positive_mask
    if not bool(positive_mask.any(dim=1).all()):
        raise ValueError("Every query must have a positive candidate")
    if not bool(negative_mask.any(dim=1).all()):
        raise ValueError("Every query must have an adversarial negative candidate")

    neg_inf = torch.finfo(retriever_scores.dtype).min
    student_logits = (retriever_scores / retriever_temperature).masked_fill(~valid_mask, neg_inf)
    student_log_probs = functional.log_softmax(student_logits, dim=1)
    with torch.no_grad():
        teacher_logits = (ranker_scores / ranker_temperature).masked_fill(~valid_mask, neg_inf)
        teacher_probs = functional.softmax(teacher_logits, dim=1)
    distillation = -(teacher_probs * student_log_probs).sum(dim=1).mean()

    with torch.no_grad():
        positive_count = positive_mask.sum(dim=1).clamp_min(1)
        positive_reference = torch.logsumexp(
            ranker_scores.masked_fill(~positive_mask, torch.finfo(ranker_scores.dtype).min), dim=1
        ) - positive_count.log()
        reward = functional.logsigmoid(positive_reference.unsqueeze(1) - ranker_scores)

    negative_logits = (retriever_scores / retriever_temperature).masked_fill(~negative_mask, neg_inf)
    negative_log_probs = functional.log_softmax(negative_logits, dim=1)
    negative_probs = negative_log_probs.exp().detach()
    adversarial = (negative_probs * negative_log_probs * reward.detach()).sum(dim=1).mean()

    total = adv_lambda * adversarial + (1.0 - adv_lambda) * distillation
    return total, {
        "adversarial_loss": adversarial.detach(),
        "distillation_loss": distillation.detach(),
    }
