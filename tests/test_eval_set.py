import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_report_lists_tokens_and_latency_for_each_case():
    from eval.run_eval import render_report

    text = render_report(
        [
            {
                "id": 1,
                "source": "official",
                "why": "",
                "expected_decision": "ДОПУСТИТЬ",
                "actual_decision": "ДОПУСТИТЬ",
                "expected_rules": [],
                "actual_rules": [],
                "explanation": "",
                "missing": [],
                "ok": True,
                "latency_ms": 1500,
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
                "model": "m",
                "skipped_llm": None,
            },
            {
                "id": "c14",
                "source": "custom",
                "why": "Пустая заявка",
                "expected_decision": "ЭСКАЛИРОВАТЬ",
                "actual_decision": "ЭСКАЛИРОВАТЬ",
                "expected_rules": [],
                "actual_rules": ["П7"],
                "explanation": "",
                "missing": [],
                "ok": True,
                "latency_ms": None,
                "usage": None,
                "model": None,
                "skipped_llm": "empty",
            },
        ],
        model="m",
        regulation_version="fire-works-1",
        prompt_version="v5",
    )
    assert "| `1` | 10 | 4 | 1.50 |" in text
    assert "| `c14` | не вызывалась | не вызывалась | не вызывалась |" in text
    assert "в модель не отправляются" in text


def test_delivery_files_cover_required_cases():
    applications = json.loads((ROOT / "data" / "applications.json").read_text(encoding="utf-8"))
    cases = json.loads((ROOT / "eval" / "cases.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in applications] == list(range(1, 12))
    official = [case for case in cases if case["source"] == "official"]
    custom = [case for case in cases if case["source"] == "custom"]
    assert [case["id"] for case in official] == list(range(1, 9))
    assert len(custom) >= 10
    assert all(case.get("why") for case in custom)
    assert all(case["expected_decision"] in {"ДОПУСТИТЬ", "ОТКАЗАТЬ", "ЭСКАЛИРОВАТЬ"} for case in cases)
