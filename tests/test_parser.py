import json
from datetime import date

import pytest

from hotwork.models import Approval, Facts, Location, ParseError, Precision, parse_facts


def full_payload(**overrides) -> dict:
    payload = {
        "permit_number": "НД-1",
        "permit_explicitly_absent": False,
        "instruction_date": "2026-09-03",
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
        "decision": "ДОПУСТИТЬ",
    }
    payload.update(overrides)
    return payload


def test_parse_ignores_decision_and_reads_fenced_json():
    raw = "```json\n" + json.dumps(full_payload(), ensure_ascii=False) + "\n```"
    facts = parse_facts(raw)
    assert facts.permit_number == "НД-1"
    assert facts.instruction_date == date(2026, 9, 3)
    assert facts.location is Location.INDOOR
    assert not hasattr(facts, "decision")


def test_parse_embedded_json_and_russian_date():
    raw = "Вот факты: " + json.dumps(full_payload(instruction_date="03.09.2026", approval="письменно"))
    facts = parse_facts(raw)
    assert facts.instruction_date == date(2026, 9, 3)
    assert facts.approval is Approval.WRITTEN


def test_qualitative_date_and_number_become_null():
    facts = parse_facts(
        json.dumps(
            full_payload(
                instruction_date="начало весны",
                wind_speed_ms="умеренный",
                distance_m="рядом",
            )
        )
    )
    assert facts.instruction_date is None
    assert facts.wind_speed_ms is None
    assert facts.distance_m is None


def test_nested_facts_and_decimal_comma():
    facts = parse_facts(json.dumps({"facts": full_payload(distance_m="9,99", distance_precision="точно")}))
    assert facts.distance_m == 9.99
    assert facts.distance_precision is Precision.EXACT


def test_garbage_and_short_object_do_not_pass_schema():
    with pytest.raises(ParseError):
        parse_facts("ДОПУСТИТЬ")
    with pytest.raises(ParseError):
        parse_facts('{"decision": "ДОПУСТИТЬ"}')
    with pytest.raises(ParseError):
        parse_facts("")
    with pytest.raises(ParseError):
        parse_facts("[1, 2, 3]")


def test_unknown_enum_does_not_crash():
    facts = parse_facts(json.dumps(full_payload(location="на крыше", approval="как-нибудь")))
    assert facts.location is Location.UNKNOWN
    assert facts.approval is Approval.UNKNOWN
    assert isinstance(facts, Facts)


def test_prompt_treats_nd_as_permit_number():
    from hotwork.extract import SYSTEM_PROMPT

    assert "НД-114" in SYSTEM_PROMPT
    assert "номер наряда-допуска" in SYSTEM_PROMPT
    assert "ОП-5" in SYSTEM_PROMPT
    assert "эстакад" in SYSTEM_PROMPT
    assert "location=outdoor" in SYSTEM_PROMPT


def test_request_uses_strict_json_schema():
    from hotwork.llm import LLMClient
    from hotwork.models import Facts, facts_response_format

    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)

            class Message:
                content = "{}"

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]
                usage = None

            return Response()

    class Chat:
        completions = Completions()

    client = LLMClient("http://localhost/v1", "key", "model")
    client._client = type("Fake", (), {"chat": Chat()})()
    client.complete("system", "user")
    response_format = captured["response_format"]
    assert response_format == facts_response_format()
    assert response_format["type"] == "json_schema"
    schema = response_format["json_schema"]
    assert schema["strict"] is True
    body = schema["schema"]
    assert body["additionalProperties"] is False
    assert set(body["required"]) == set(Facts.model_fields)
    assert set(body["properties"]) == set(Facts.model_fields)
