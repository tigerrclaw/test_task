"""Отдельный прогон сравнения моделей. Основной агент и .env не меняет."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from hotwork.config import ConfigError, llm_settings_from_env, load_env_file, load_regulation
from hotwork.extract import LLMExtractor, PROMPT_VERSION
from hotwork.journal import JournalSession
from hotwork.llm import LLMClient
from hotwork.pipeline import process_application
from eval.run_eval import CASES_PATH, evaluate_cases, render_report

APPLICATIONS_PATH = ROOT / "data" / "applications.json"
# Grok на том же OpenAI-совместимом эндпоинте, что и Astra (darkapi).
GROK_MODEL = "grok-4.7"


def profiles() -> list[tuple[str, str, str, str]]:
    load_env_file()
    base_url, api_key, astra_model, _timeout = llm_settings_from_env()
    return [
        ("grok", base_url, api_key, GROK_MODEL),
        ("astra", base_url, api_key, astra_model),
    ]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_profile(slug: str, base_url: str, api_key: str, model: str, regulation, applications: list[dict], cases: list[dict]) -> dict:
    timeout = 120.0
    extractor = LLMExtractor(LLMClient(base_url, api_key, model, timeout))
    journal_dir = ROOT / "build" / "compare" / slug / "journal"
    print(f"\n=== {model} ===", flush=True)
    batch = JournalSession(journal_dir, "request")
    results = []
    for application in applications:
        print(f"заявка {application['id']}...", flush=True)
        results.append(
            process_application(
                application,
                regulation=regulation,
                extractor=extractor,
                journal=batch,
            )
        )
    batch.flush()
    results_path = ROOT / f"results_{slug}.json"
    write_json(results_path, results)
    print(f"результаты: {results_path}", flush=True)

    print("eval...", flush=True)
    eval_batch = JournalSession(journal_dir, "eval_request")
    rows = evaluate_cases(cases, extractor, regulation, eval_batch)
    eval_batch.flush()
    report = render_report(
        rows,
        model=model,
        regulation_version=regulation.version,
        prompt_version=PROMPT_VERSION,
    )
    report_path = ROOT / f"eval_report_{slug}.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"eval: {report_path}", flush=True)
    return {"slug": slug, "model": model, "results": results, "rows": rows}


def render_comparison(runs: list[dict]) -> str:
    left, right = runs
    left_label = left["slug"].capitalize()
    right_label = right["slug"].capitalize()
    lines = [
        "# Сравнение моделей",
        "",
        f"Один и тот же промпт `{PROMPT_VERSION}`, регламент не менялся. Запасная модель на время прогона отключена.",
        "Основной код агента и `.env` не менялись — отдельный прогон `eval/compare_models.py`.",
        "",
        f"- {left['slug']}: `{left['model']}`",
        f"- {right['slug']}: `{right['model']}`",
        "",
        "## Eval",
        "",
        f"| Случай | Ожидание | {left_label} | {right_label} |",
        "|---|---|---|---|",
    ]
    left_rows = {row["id"]: row for row in left["rows"]}
    right_rows = {row["id"]: row for row in right["rows"]}
    for case_id in left_rows:
        a = left_rows[case_id]
        b = right_rows[case_id]
        lines.append(
            f"| `{case_id}` | {a['expected_decision']} | {a['actual_decision']} | {b['actual_decision']} |"
        )
    left_ok = sum(1 for row in left["rows"] if row["ok"])
    right_ok = sum(1 for row in right["rows"] if row["ok"])
    lines.extend(
        [
            "",
            f"Совпадений с ожиданием: {left_label} {left_ok} из {len(left['rows'])}, {right_label} {right_ok} из {len(right['rows'])}.",
            "",
            "## Заявки 1–11",
            "",
            f"| № | {left_label} | {right_label} | Сравнение |",
            "|---|---|---|---|",
        ]
    )
    for a, b in zip(left["results"], right["results"]):
        same = "совпало" if a["decision"] == b["decision"] else "разные решения"
        lines.append(
            f"| {a['id']} | {a['decision']} ({', '.join(a['rules']) or '—'}) | {b['decision']} ({', '.join(b['rules']) or '—'}) | {same} |"
        )
    lines.extend(
        [
            "",
            f"Подробности: `results_{left['slug']}.json`, `results_{right['slug']}.json`, "
            f"`eval_report_{left['slug']}.md`, `eval_report_{right['slug']}.md`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    try:
        chosen = profiles()
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    regulation = load_regulation()
    applications = json.loads(APPLICATIONS_PATH.read_text(encoding="utf-8"))
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    runs = [run_profile(*profile, regulation, applications, cases) for profile in chosen]
    report_path = ROOT / "model_comparison.md"
    report_path.write_text(render_comparison(runs), encoding="utf-8")
    print(f"\nСравнение: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
