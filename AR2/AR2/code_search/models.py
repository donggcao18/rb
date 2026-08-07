from __future__ import annotations

from typing import Any


def _dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError("Models require torch and transformers; install requirements-code-search.txt") from exc
    return torch, nn, AutoModel


def mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)


def build_dual_encoder_class() -> type:
    torch, nn, AutoModel = _dependencies()

    class DualEncoder(nn.Module):
        def __init__(
            self,
            model_name: str,
            share_weights: bool = False,
            pooling: str = "cls",
            normalize_embeddings: bool = False,
        ) -> None:
            super().__init__()
            if pooling not in {"cls", "mean"}:
                raise ValueError("pooling must be 'cls' or 'mean'")
            self.model_name = model_name
            self.share_weights = share_weights
            self.pooling = pooling
            self.normalize_embeddings = normalize_embeddings
            self.query_encoder = AutoModel.from_pretrained(model_name)
            self.code_encoder = self.query_encoder if share_weights else AutoModel.from_pretrained(model_name)

        def _pool(self, output: Any, attention_mask: Any) -> Any:
            vector = output.last_hidden_state[:, 0] if self.pooling == "cls" else mean_pool(
                output.last_hidden_state, attention_mask
            )
            return torch.nn.functional.normalize(vector, dim=-1) if self.normalize_embeddings else vector

        def encode_query(self, input_ids: Any, attention_mask: Any) -> Any:
            return self._pool(self.query_encoder(input_ids=input_ids, attention_mask=attention_mask), attention_mask)

        def encode_code(self, input_ids: Any, attention_mask: Any) -> Any:
            return self._pool(self.code_encoder(input_ids=input_ids, attention_mask=attention_mask), attention_mask)

        def forward(
            self,
            query_input_ids: Any,
            query_attention_mask: Any,
            code_input_ids: Any,
            code_attention_mask: Any,
        ) -> Any:
            query = self.encode_query(query_input_ids, query_attention_mask)
            code = self.encode_code(code_input_ids, code_attention_mask)
            return query @ code.transpose(0, 1)

        def export_config(self) -> dict[str, Any]:
            return {
                "model_name": self.model_name,
                "share_weights": self.share_weights,
                "pooling": self.pooling,
                "normalize_embeddings": self.normalize_embeddings,
            }

    return DualEncoder


def build_cross_encoder_class() -> type:
    _, nn, AutoModel = _dependencies()

    class CrossEncoderRanker(nn.Module):
        def __init__(self, model_name: str) -> None:
            super().__init__()
            self.model_name = model_name
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden_size = self.encoder.config.hidden_size
            self.scorer = nn.Linear(hidden_size, 1)

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            return self.scorer(output.last_hidden_state[:, 0]).squeeze(-1)

        def export_config(self) -> dict[str, Any]:
            return {"model_name": self.model_name}

    return CrossEncoderRanker


def create_dual_encoder(**kwargs: Any) -> Any:
    return build_dual_encoder_class()(**kwargs)


def create_cross_encoder(**kwargs: Any) -> Any:
    return build_cross_encoder_class()(**kwargs)
