from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from hotwork.config import ConfigError, Regulation, submission_root
from hotwork.extract import Extraction
from hotwork.journal import JournalSession
from hotwork.models import Decision
from hotwork.rules import evaluate, decide


def needs_model(text: object, limit: int) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    return len(text) <= limit


def display_path(path: Path) -> str:
    """Короткое имя для лога: results.json, data/output.json, eval_report.md."""
    try:
        return path.resolve().relative_to(submission_root().resolve()).as_posix()
    except (OSError, ValueError):
        return path.name


class Extractor(Protocol):
    prompt_version: str

    def extract(self, text: str) -> Extraction:
        ...


def process_application(
    application: dict,
    *,
    regulation: Regulation,
    extractor: Extractor,
    journal: JournalSession | None = None,
) -> dict:
    app_id = application["id"]
    text = application.get("text")
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)

    record = {
        "id": app_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_text": text[:100_000],
        "input_truncated_in_journal": len(text) > 100_000,
        "input_length": len(text),
        "regulation_version": regulation.version,
        "regulation": regulation.as_dict(),
        "prompt_version": extractor.prompt_version,
        "model": None,
        "used_fallback": False,
        "primary_error": None,
        "llm_calls": [],
        "raw_llm_response": None,
        "facts": None,
        "parse_error": None,
        "rule_results": [],
        "latency_ms": None,
        "usage": None,
        "skipped_llm": None,
        "decision": None,
        "rules": [],
        "explanation": None,
        "missing": [],
    }

    def finish(decision: Decision, rules: list[str], explanation: str, missing: list[str]) -> dict:
        record["decision"] = decision.value
        record["rules"] = rules
        record["explanation"] = explanation
        record["missing"] = missing
        if journal is not None:
            journal.add(record)
        _log_application(app_id, record)
        return {
            "id": app_id,
            "decision": decision.value,
            "rules": rules,
            "explanation": explanation,
            "missing": missing,
        }

    if not text.strip():
        record["skipped_llm"] = "empty"
        return finish(
            Decision.ESCALATE,
            ["П7"],
            "Заявка пустая, проверить требования регламента по тексту нельзя.",
            ["текст заявки"],
        )

    if len(text) > regulation.max_text_chars:
        record["skipped_llm"] = "too_long"
        return finish(
            Decision.ESCALATE,
            ["П7"],
            f"Заявка длиннее {regulation.max_text_chars} символов и целиком не проверялась.",
            ["текст заявки в пределах лимита"],
        )

    try:
        extraction = extractor.extract(text)
    except ConfigError:
        raise
    except Exception as exc:
        record["parse_error"] = type(exc).__name__
        return finish(
            Decision.ESCALATE,
            ["П7"],
            "Модель не вернула пригодный разбор заявки.",
            ["разбор заявки"],
        )

    record["model"] = extraction.model
    record["used_fallback"] = extraction.used_fallback
    record["primary_error"] = extraction.primary_error
    record["llm_calls"] = extraction.attempts or []
    record["raw_llm_response"] = extraction.raw
    record["latency_ms"] = extraction.latency_ms
    record["usage"] = extraction.usage
    record["prompt_version"] = extraction.prompt_version or extractor.prompt_version
    if extraction.facts is None:
        record["parse_error"] = extraction.error
        return finish(
            Decision.ESCALATE,
            ["П7"],
            "Ответ модели не прошёл проверку схемы фактов, решение по заявке не принимается.",
            ["разбор заявки"],
        )

    record["facts"] = extraction.facts.model_dump(mode="json")
    outcomes = evaluate(extraction.facts, regulation)
    record["rule_results"] = [
        {
            "rule_id": item.rule_id,
            "status": item.status.value,
            "reason": item.reason,
            "missing": list(item.missing),
        }
        for item in outcomes
    ]
    decision, rules, explanation, missing = decide(outcomes)
    return finish(decision, rules, explanation, missing)


def _log_application(app_id: object, record: dict) -> None:
    if record.get("skipped_llm") == "empty":
        print(f"заявка {app_id}: модель не вызывалась, заявка пустая", file=sys.stderr)
        return
    if record.get("skipped_llm") == "too_long":
        print(f"заявка {app_id}: модель не вызывалась, заявка длиннее лимита", file=sys.stderr)
        return
    for item in record.get("llm_calls") or []:
        error = item.get("error")
        if not error:
            continue
        line = (
            f"заявка {app_id}: {item['provider']} {item['model']}, "
            f"попытка {item['attempt']}, ошибка: {error}"
        )
        wait = item.get("retry_in_seconds")
        if wait:
            line += f", следующая попытка через {wait:.1f} с"
        elif wait == 0:
            line += ", повтор сразу"
        print(line, file=sys.stderr)
    model = record.get("model") or "не определена"
    if record.get("facts") is None and record.get("parse_error"):
        outcome = "разбор не получен"
    elif record.get("skipped_llm"):
        outcome = "модель не вызывалась"
    else:
        outcome = "разбор получен"
    switch = ", было переключение на запасную модель" if record.get("used_fallback") else ""
    print(f"заявка {app_id}: {outcome}, модель {model}{switch}", file=sys.stderr)
