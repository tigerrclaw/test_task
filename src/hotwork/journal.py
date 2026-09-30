from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class JournalSession:
    """Собирает записи одного прогона и пишет один файл с датой и временем в имени."""

    def __init__(self, directory: Path, kind: str):
        if kind not in {"request", "manual_request", "eval_request"}:
            raise ValueError(f"Неизвестный тип журнала: {kind}")
        self.directory = directory
        self.kind = kind
        self.records: list[dict] = []
        self.started_at = datetime.now(timezone.utc)
        self.path: Path | None = None

    def add(self, record: dict) -> None:
        self.records.append(record)

    def flush(self) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = self.started_at.astimezone().strftime("%Y%m%d_%H%M%S")
        path = self.directory / f"{self.kind}_{stamp}.json"
        payload = {
            "kind": self.kind,
            "started_at": self.started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "count": len(self.records),
            "applications": self.records,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.path = path
        return path
