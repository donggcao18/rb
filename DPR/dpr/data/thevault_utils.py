"""Dependency-free parsing and normalization helpers for TheVault data."""

import json
import re
from pathlib import Path
from typing import Dict, Iterable

QUERY_PREFIX_RE = re.compile(r"^\s*Query\s*:\s*", re.IGNORECASE)
CODE_PREFIX_RE = re.compile(r"^\s*Code\s*:\s*", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")


def iter_json_records(path: str) -> Iterable[Dict]:
    """Read either a JSON array or a JSONL file, independent of extension."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        first = ""
        while True:
            char = stream.read(1)
            if not char:
                return
            if not char.isspace():
                first = char
                break
        stream.seek(0)

        if first == "[":
            records = json.load(stream)
            if not isinstance(records, list):
                raise ValueError("{} must contain a JSON list".format(path))
            for record in records:
                if record:
                    yield record
            return

        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid JSON at {}:{}: {}".format(path, line_number, exc)) from exc


def normalize_thevault_query(text: str) -> str:
    """Match the grouping behavior in the existing build_multilable.py script."""
    text = QUERY_PREFIX_RE.sub("", str(text))
    return SPACE_RE.sub(" ", text.strip().lower())


def strip_query_prefix(text: str) -> str:
    return QUERY_PREFIX_RE.sub("", str(text)).strip()


def strip_code_prefix(text: str) -> str:
    return CODE_PREFIX_RE.sub("", str(text)).strip()

