from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


QUERY_PREFIX_RE = re.compile(r"^\s*Query\s*:\s*", re.IGNORECASE)
CODE_PREFIX_RE = re.compile(r"^\s*Code\s*:\s*", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")


def iter_json_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield objects from a JSON array or a JSONL file."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        first = ""
        while True:
            char = handle.read(1)
            if not char:
                return
            if not char.isspace():
                first = char
                break
        handle.seek(0)
        if first == "[":
            payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError(f"{source} must contain a JSON list")
            for index, row in enumerate(payload, start=1):
                if not isinstance(row, dict):
                    raise ValueError(f"{source}: item {index} is not an object")
                yield row
            return

        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {source}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{source}:{line_number} is not a JSON object")
            yield row


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_json_records(path))


def normalize_query(text: Any) -> str:
    stripped = QUERY_PREFIX_RE.sub("", str(text))
    return SPACE_RE.sub(" ", stripped.strip().lower())


def strip_query_prefix(text: Any) -> str:
    return QUERY_PREFIX_RE.sub("", str(text)).strip()


def strip_code_prefix(text: Any) -> str:
    return CODE_PREFIX_RE.sub("", str(text)).strip()


def record_kind(text: Any) -> str:
    value = str(text)
    if QUERY_PREFIX_RE.match(value):
        return "query"
    if CODE_PREFIX_RE.match(value):
        return "code"
    return "other"


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    suffix = source.suffix.lower()
    with source.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            payload = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("YAML configuration requires PyYAML") from exc
            payload = yaml.safe_load(handle)
        else:
            raise ValueError("Configuration must be JSON or YAML")
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration root in {source} must be an object")
    return payload


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def format_code_document(document: Mapping[str, Any], include_metadata: bool = True) -> str:
    code = str(document.get("code", "")).strip()
    if not include_metadata:
        return code
    fields = []
    identifier = str(document.get("identifier", "")).strip()
    path = str(document.get("path", "")).strip()
    if identifier:
        fields.append(f"identifier: {identifier}")
    if path:
        fields.append(f"path: {path}")
    fields.append(f"code:\n{code}")
    return "\n".join(fields)
