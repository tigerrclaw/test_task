"""Прогон eval-кейсов с заготовленными ответами модели — без сети и ключа."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hotwork.config import load_regulation
from hotwork.extract import LLMExtractor
from hotwork.journal import JournalSession
from hotwork.llm import LLMCall
from hotwork.pipeline import process_application

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "eval" / "cases.json"
FIXTURES_PATH = Path(__file__).parent / "fixtures" / "mock_llm_responses.json"


@pytest.fixture(scope="module")
def mock_responses() -> dict:
    return json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


class FixtureLLMClient:
    """Подменяет LLMClient: отдаёт JSON из fixtures по id заявки в user-сообщении."""

    def __init__(self, responses: dict[str, dict], model: str = "mock-llm"):
        self.responses = responses
        self.model = model
        self.calls = 0
        self.last_user = ""
        self._next_id: str | int | None = None

    def set_case_id(self, case_id: object) -> None:
        self._next_id = case_id

    def complete(self, system: str, user: str) -> LLMCall:
        del system
        self.calls += 1
        self.last_user = user
        key = str(self._next_id)
        if key not in self.responses:
            raise AssertionError(f"Нет mock-ответа для заявки {key}")
        raw = json.dumps(self.responses[key], ensure_ascii=False)
        return LLMCall(
            raw=raw,
            model=self.model,
            latency_ms=1.0,
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )


@pytest.mark.mock_llm
def test_official_and_custom_cases_with_mocked_llm(tmp_path, mock_responses, cases):
    regulation = load_regulation()
    client = FixtureLLMClient(mock_responses)
    extractor = LLMExtractor(client, sleeper=lambda _s: None, uniform=lambda _a, _b: 0)
    journal = JournalSession(tmp_path, "eval_request")

    runnable = [case for case in cases if case["id"] not in {"c14", "c15"}]
    for case in runnable:
        client.set_case_id(case["id"])
        result = process_application(
            {"id": case["id"], "text": case["text"]},
            regulation=regulation,
            extractor=extractor,
            journal=journal,
        )
        assert result["decision"] == case["expected_decision"], case["id"]
        expected_rules = case.get("expected_rules") or []
        assert set(expected_rules) <= set(result["rules"]), case["id"]

    path = journal.flush()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["count"] == len(runnable)
    assert client.calls == len(runnable)
    assert "ДОПУСТИТЬ" in client.last_user or "НД" in client.last_user


@pytest.mark.mock_llm
def test_mocked_garbage_response_escalates(tmp_path):
    class GarbageClient:
        model = "mock-garbage"

        def complete(self, system: str, user: str) -> LLMCall:
            del system, user
            return LLMCall(raw="ДОПУСТИТЬ без JSON", model=self.model, latency_ms=1.0, usage=None)

    journal = JournalSession(tmp_path, "request")
    extractor = LLMExtractor(GarbageClient(), sleeper=lambda _s: None, uniform=lambda _a, _b: 0)
    # extract retries garbage once then fails facts
    result = process_application(
        {"id": "g1", "text": "НД-1, сварка в цехе"},
        regulation=load_regulation(),
        extractor=extractor,
        journal=journal,
    )
    assert result["decision"] == "ЭСКАЛИРОВАТЬ"
    assert result["rules"] == ["П7"]
    journal.flush()


@pytest.mark.mock_llm
def test_empty_and_long_skip_mocked_llm(tmp_path, mock_responses):
    client = FixtureLLMClient(mock_responses)
    extractor = LLMExtractor(client, sleeper=lambda _s: None, uniform=lambda _a, _b: 0)
    journal = JournalSession(tmp_path, "request")
    regulation = load_regulation()

    empty = process_application(
        {"id": "c14", "text": ""},
        regulation=regulation,
        extractor=extractor,
        journal=journal,
    )
    long = process_application(
        {"id": "c15", "text": "x" * (regulation.max_text_chars + 1)},
        regulation=regulation,
        extractor=extractor,
        journal=journal,
    )
    assert empty["decision"] == "ЭСКАЛИРОВАТЬ"
    assert long["decision"] == "ЭСКАЛИРОВАТЬ"
    assert client.calls == 0
    journal.flush()
