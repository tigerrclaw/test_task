from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hotwork.cli import build_extractor
from hotwork.config import ConfigError, data_dir, llm_settings_from_env, load_env_file, load_regulation, submission_root
from hotwork.journal import JournalSession
from hotwork.pipeline import display_path, needs_model, process_application

CASES_PATH = Path(__file__).with_name("cases.json")


def case_text(case: dict) -> str:
    text = case.get("text") or ""
    pad_to = case.get("pad_to")
    if pad_to:
        unit = case.get("pad") or "заявка "
        while len(text) < pad_to:
            text += unit
    return text


def evaluate_cases(cases: list[dict], extractor, regulation, journal: JournalSession) -> list[dict]:
    rows = []
    for case in cases:
        text = case_text(case)
        result = process_application(
            {"id": case["id"], "text": text},
            regulation=regulation,
            extractor=extractor,
            journal=journal,
        )
        expected_rules = case.get("expected_rules") or []
        rules_ok = set(expected_rules) <= set(result["rules"])
        decision_ok = result["decision"] == case["expected_decision"]
        record = journal.records[-1]
        rows.append(
            {
                "id": case["id"],
                "source": case.get("source", ""),
                "why": case.get("why") or "",
                "expected_decision": case["expected_decision"],
                "actual_decision": result["decision"],
                "expected_rules": expected_rules,
                "actual_rules": result["rules"],
                "explanation": result["explanation"],
                "missing": result["missing"],
                "ok": decision_ok and rules_ok,
                "latency_ms": record.get("latency_ms"),
                "usage": record.get("usage"),
                "model": record.get("model"),
                "skipped_llm": record.get("skipped_llm"),
            }
        )
    return rows


def usage_totals(rows: list[dict]) -> tuple[int, int, float]:
    prompt = 0
    completion = 0
    latency = 0.0
    for row in rows:
        usage = row.get("usage") or {}
        prompt += int(usage.get("prompt_tokens") or 0)
        completion += int(usage.get("completion_tokens") or 0)
        if row.get("latency_ms"):
            latency += float(row["latency_ms"])
    return prompt, completion, latency


def cost_line(prompt_tokens: int, completion_tokens: int) -> str:
    raw_in = os.environ.get("LLM_INPUT_USD_PER_1M", "").strip()
    raw_out = os.environ.get("LLM_OUTPUT_USD_PER_1M", "").strip()
    if not raw_in or not raw_out:
        return (
            "Стоимость не посчитана: задайте LLM_INPUT_USD_PER_1M и LLM_OUTPUT_USD_PER_1M. "
            f"Токены прогона: вход {prompt_tokens}, выход {completion_tokens}."
        )
    price_in = float(raw_in)
    price_out = float(raw_out)
    total = prompt_tokens / 1_000_000 * price_in + completion_tokens / 1_000_000 * price_out
    return (
        f"Оценка стоимости прогона: ${total:.6f} "
        f"(вход ${price_in}/1M, выход ${price_out}/1M; токены {prompt_tokens}/{completion_tokens})."
    )


def render_report(
    rows: list[dict],
    *,
    model: str,
    regulation_version: str,
    prompt_version: str,
) -> str:
    passed = sum(1 for row in rows if row["ok"])
    failed = [row for row in rows if not row["ok"]]
    called = [row for row in rows if row.get("latency_ms")]
    prompt_tokens, completion_tokens, latency = usage_totals(rows)
    avg_ms = latency / len(called) if called else 0.0
    lines = [
        "# Отчёт eval",
        "",
        f"- Модель: `{model}`",
        f"- Версия промпта: `{prompt_version}`",
        f"- Версия регламента: `{regulation_version}`",
        f"- Результат: {passed} из {len(rows)}, ошибок {len(failed)}",
        "",
        "## Время и стоимость",
        "",
        f"- Заявок с вызовом модели: {len(called)}",
        f"- Суммарная задержка вызовов: {latency / 1000:.2f} с",
        f"- Средняя задержка одного вызова: {avg_ms / 1000:.2f} с",
        f"- {cost_line(prompt_tokens, completion_tokens)}",
        "",
        "Если до провайдера не достучаться, между попытками паузы 10, 20 и 30 секунд с джиттером, затем запасной провайдер, если он задан.",
        "",
        "## Ошибки",
        "",
    ]
    if not failed:
        lines.append("Расхождений с ожидаемым решением нет.")
    else:
        for row in failed:
            lines.append(
                f"- `{row['id']}`: ожидалось {row['expected_decision']} {row['expected_rules']}, "
                f"получено {row['actual_decision']} {row['actual_rules']}. {row['explanation']}"
            )
    lines.extend(
        [
            "",
            "## По каждому запросу",
            "",
            "| Случай | Вход, токены | Выход, токены | Задержка, с |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        if row.get("skipped_llm"):
            lines.append(f"| `{row['id']}` | не вызывалась | не вызывалась | не вызывалась |")
            continue
        usage = row.get("usage") or {}
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        latency_ms = row.get("latency_ms")
        prompt_cell = "—" if prompt is None else str(int(prompt))
        completion_cell = "—" if completion is None else str(int(completion))
        latency_cell = "—" if not latency_ms else f"{float(latency_ms) / 1000:.2f}"
        lines.append(f"| `{row['id']}` | {prompt_cell} | {completion_cell} | {latency_cell} |")
    lines.append("")
    lines.append("Пустая и слишком длинная заявка в модель не отправляются, поэтому у них нет токенов и задержки.")
    lines.extend(["", "## Свои случаи", ""])
    for row in rows:
        if row["source"] != "custom":
            continue
        mark = "совпало" if row["ok"] else "ошибка"
        lines.append(f"- `{row['id']}` ({mark}, {row['actual_decision']}): {row['why']}")
    lines.append("")
    return "\n".join(lines)


def print_rows(rows: list[dict]) -> None:
    for row in rows:
        status = "OK" if row["ok"] else "FAIL"
        print(f"{row['id']}\t{row['expected_decision']}\t{row['actual_decision']}\t{status}")
        if not row["ok"]:
            print(f"  rules expected={row['expected_rules']} actual={row['actual_rules']}")
            print(f"  {row['explanation']}")


def report_path() -> Path:
    return submission_root() / "eval_report.md"


def eval_journal_dir() -> Path:
    return data_dir() / "eval-journal"


def write_report(text: str) -> Path:
    path = report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def main() -> int:
    load_env_file()
    regulation = load_regulation()
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    model_required = any(needs_model(case_text(case), regulation.max_text_chars) for case in cases)
    model = os.environ.get("LLM_MODEL", "").strip() or "не задана"
    try:
        if model_required:
            _, _, model, _ = llm_settings_from_env()
            extractor = build_extractor()
        else:
            extractor = None
    except ConfigError as exc:
        custom = [case for case in cases if case.get("source") == "custom"]
        lines = [
            "# Отчёт eval",
            "",
            "Живой прогон не выполнен. " + str(exc) + ".",
            "",
            "Повторите `python eval/run_eval.py`, когда заданы `LLM_BASE_URL`, `LLM_API_KEY` и `LLM_MODEL`.",
            "Юнит-тесты правил, разбора ответа и сбоев от ключа не зависят: `python -m pytest tests -q`.",
            "",
            "## Свои случаи",
            "",
        ]
        for case in custom:
            lines.append(
                f"- `{case['id']}` → {case['expected_decision']}: {case.get('why', '')}"
            )
        lines.append("")
        write_report("\n".join(lines))
        print(str(exc), file=sys.stderr)
        return 1

    journal = JournalSession(eval_journal_dir(), "eval_request")
    rows = evaluate_cases(cases, extractor, regulation, journal)
    journal_path = journal.flush()
    print_rows(rows)

    from hotwork.extract import PROMPT_VERSION

    written = write_report(
        render_report(
            rows,
            model=model,
            regulation_version=regulation.version,
            prompt_version=PROMPT_VERSION,
        )
    )
    print(f"\n{display_path(written)}")
    print(f"журнал: {display_path(journal_path)}")
    passed = sum(1 for row in rows if row["ok"])
    print(f"Итог: {passed}/{len(rows)}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
