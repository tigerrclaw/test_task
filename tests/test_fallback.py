import json

from hotwork.extract import LLMExtractor
from hotwork.llm import LLMCall, LLMError, is_retryable


def valid_json() -> str:
    return json.dumps(
        {
            "permit_number": "НД-1",
            "permit_explicitly_absent": False,
            "instruction_date": "2026-09-01",
            "hazard_present": False,
            "distance_m": None,
            "distance_precision": "unknown",
            "approval": "absent",
            "extinguisher_present": True,
            "observer_present": True,
            "observer_is_worker": False,
            "location": "indoor",
            "wind_speed_ms": None,
            "wind_precision": "unknown",
            "height_m": None,
            "height_precision": "unknown",
            "height_is_unclear": False,
            "height_permit_present": None,
        },
        ensure_ascii=False,
    )


class ScriptedClient:
    def __init__(self, model: str, outcomes: list):
        self.model = model
        self.outcomes = list(outcomes)
        self.calls = 0

    def complete(self, system: str, user: str) -> LLMCall:
        del system, user
        self.calls += 1
        item = self.outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMCall(raw=item, model=self.model, latency_ms=5.0, usage=None)


def extractor(primary, fallback=None, slept=None):
    return LLMExtractor(
        primary,
        fallback,
        sleeper=(slept.append if slept is not None else (lambda _seconds: None)),
        uniform=lambda _start, _end: 0,
    )


def test_unreachable_primary_waits_then_uses_fallback():
    slept = []
    primary = ScriptedClient("primary", [LLMError("timeout", retryable=True)] * 4)
    fallback = ScriptedClient("backup", [valid_json()])
    result = extractor(primary, fallback, slept).extract("НД-1, сварка в цехе")
    assert result.facts is not None
    assert result.model == "backup"
    assert result.used_fallback is True
    assert primary.calls == 4
    assert fallback.calls == 1
    assert slept == [10, 20, 30]
    assert [item["error"] for item in result.attempts if item["provider"] == "основной"] == ["timeout"] * 4
    assert result.attempts[-1]["provider"] == "запасной"
    assert result.attempts[-1]["error"] is None


def test_reachable_again_on_second_try_does_not_switch():
    slept = []
    primary = ScriptedClient("primary", [LLMError("timeout", retryable=True), valid_json()])
    fallback = ScriptedClient("backup", [valid_json()])
    result = extractor(primary, fallback, slept).extract("НД-1")
    assert result.model == "primary"
    assert result.used_fallback is False
    assert fallback.calls == 0
    assert slept == [10]


def test_primary_only_does_not_require_fallback():
    slept = []
    primary = ScriptedClient("primary", [LLMError("timeout", retryable=True)] * 4)
    result = extractor(primary, None, slept).extract("НД-1")
    assert result.facts is None
    assert result.used_fallback is False
    assert result.model == "primary"
    assert slept == [10, 20, 30]


def test_garbage_json_retries_once_without_long_pause():
    slept = []
    primary = ScriptedClient("primary", ["ДОПУСТИТЬ", "ДОПУСТИТЬ"])
    fallback = ScriptedClient("backup", [valid_json()])
    result = extractor(primary, fallback, slept).extract("НД-1")
    assert result.model == "backup"
    assert result.used_fallback is True
    assert primary.calls == 2
    assert slept == []


def test_non_retryable_error_switches_without_backoff():
    slept = []
    primary = ScriptedClient("primary", [LLMError("unauthorized", retryable=False)])
    fallback = ScriptedClient("backup", [valid_json()])
    result = extractor(primary, fallback, slept).extract("НД-1")
    assert result.model == "backup"
    assert primary.calls == 1
    assert slept == []


def test_journal_and_terminal_show_errors_switch_and_model(tmp_path, capsys):
    from hotwork.config import load_regulation
    from hotwork.journal import JournalSession
    from hotwork.pipeline import process_application

    slept: list[float] = []
    primary = ScriptedClient("primary-model", [LLMError("timeout", retryable=True), valid_json()])
    journal = JournalSession(tmp_path, "request")
    result = process_application(
        {"id": 4, "text": "НД-1, сварка в цехе"},
        regulation=load_regulation(),
        extractor=extractor(primary, None, slept),
        journal=journal,
    )
    path = journal.flush()
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = payload["applications"][0]
    assert result["decision"]
    assert record["model"] == "primary-model"
    assert record["used_fallback"] is False
    assert record["llm_calls"][0]["error"] == "timeout"
    assert record["llm_calls"][0]["retry_in_seconds"] == 10
    assert record["llm_calls"][1]["error"] is None
    err = capsys.readouterr().err
    assert "заявка 4: основной primary-model, попытка 1, ошибка: timeout" in err
    assert "следующая попытка через 10.0 с" in err
    assert "заявка 4: разбор получен, модель primary-model" in err
    assert "переключение" not in err
    assert path.name.startswith("request_")


def test_timeout_exception_is_retryable():
    class APITimeoutError(Exception):
        pass

    assert is_retryable(APITimeoutError("request timed out"))
    assert not is_retryable(type("BadRequestError", (Exception,), {})("invalid model"))
