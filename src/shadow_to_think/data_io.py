from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]], append: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def infer_prompt(row: Dict[str, Any]) -> str:
    for key in ("prompt", "question", "input", "problem", "query"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    raise KeyError("Dataset row must contain one of: prompt, question, input, problem, query")


def infer_gold(row: Dict[str, Any]) -> Optional[str]:
    for key in ("answer", "gold", "target", "label", "final_answer"):
        val = row.get(key)
        if val is not None:
            return str(val).strip()
    return None


def load_shadow_records(paths: List[str | Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in paths:
        records.extend(read_jsonl(path))
    return records
