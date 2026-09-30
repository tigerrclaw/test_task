from __future__ import annotations

import json
import re
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


class Decision(str, Enum):
    ALLOW = "ДОПУСТИТЬ"
    REJECT = "ОТКАЗАТЬ"
    ESCALATE = "ЭСКАЛИРОВАТЬ"


class RuleStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class Precision(str, Enum):
    EXACT = "exact"
    APPROXIMATE = "approximate"
    UNKNOWN = "unknown"


class Location(str, Enum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    UNKNOWN = "unknown"
    CONTRADICTORY = "contradictory"


class Approval(str, Enum):
    WRITTEN = "written"
    ORAL = "oral"
    ABSENT = "absent"
    UNKNOWN = "unknown"


MIN_FACT_FIELDS = 8

_APPROVAL_ALIASES = {
    "written": Approval.WRITTEN,
    "письменное": Approval.WRITTEN,
    "письменно": Approval.WRITTEN,
    "oral": Approval.ORAL,
    "устное": Approval.ORAL,
    "устно": Approval.ORAL,
    "absent": Approval.ABSENT,
    "none": Approval.ABSENT,
    "нет": Approval.ABSENT,
    "unknown": Approval.UNKNOWN,
    "неизвестно": Approval.UNKNOWN,
}

_LOCATION_ALIASES = {
    "indoor": Location.INDOOR,
    "помещение": Location.INDOOR,
    "цех": Location.INDOOR,
    "outdoor": Location.OUTDOOR,
    "улица": Location.OUTDOOR,
    "площадка": Location.OUTDOOR,
    "unknown": Location.UNKNOWN,
    "неизвестно": Location.UNKNOWN,
    "contradictory": Location.CONTRADICTORY,
    "противоречие": Location.CONTRADICTORY,
}

_PRECISION_ALIASES = {
    "exact": Precision.EXACT,
    "точно": Precision.EXACT,
    "approximate": Precision.APPROXIMATE,
    "примерно": Precision.APPROXIMATE,
    "около": Precision.APPROXIMATE,
    "unknown": Precision.UNKNOWN,
    "неизвестно": Precision.UNKNOWN,
}


class ParseError(ValueError):
    """Ответ модели не стал пригодным объектом фактов."""


class Facts(BaseModel):
    model_config = ConfigDict(extra="ignore")

    permit_number: str | None = None
    permit_explicitly_absent: bool = False
    instruction_date: date | None = None
    hazard_present: bool | None = None
    distance_m: float | None = None
    distance_precision: Precision = Precision.UNKNOWN
    approval: Approval = Approval.UNKNOWN
    extinguisher_present: bool | None = None
    observer_present: bool | None = None
    observer_is_worker: bool | None = None
    location: Location = Location.UNKNOWN
    wind_speed_ms: float | None = None
    wind_precision: Precision = Precision.UNKNOWN
    height_m: float | None = None
    height_precision: Precision = Precision.UNKNOWN
    height_is_unclear: bool = False
    height_permit_present: bool | None = None

    @field_validator("permit_number", mode="before")
    @classmethod
    def _permit_number(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() in {"null", "none", "нет", "отсутствует"}:
            return None
        return text

    @field_validator("permit_explicitly_absent", "height_is_unclear", mode="before")
    @classmethod
    def _flag(cls, value: Any) -> bool:
        return bool(coerce_tri_bool(value))

    @field_validator("instruction_date", mode="before")
    @classmethod
    def _instruction_date(cls, value: Any) -> date | None:
        return coerce_optional_date(value)

    @field_validator("distance_m", "wind_speed_ms", "height_m", mode="before")
    @classmethod
    def _numbers(cls, value: Any) -> float | None:
        return coerce_optional_float(value)

    @field_validator(
        "hazard_present",
        "extinguisher_present",
        "observer_present",
        "observer_is_worker",
        "height_permit_present",
        mode="before",
    )
    @classmethod
    def _tri_bool(cls, value: Any) -> bool | None:
        return coerce_tri_bool(value)

    @field_validator("distance_precision", "wind_precision", "height_precision", mode="before")
    @classmethod
    def _precision(cls, value: Any) -> Precision:
        return _match_alias(_PRECISION_ALIASES, value, Precision.UNKNOWN)

    @field_validator("approval", mode="before")
    @classmethod
    def _approval(cls, value: Any) -> Approval:
        return _match_alias(_APPROVAL_ALIASES, value, Approval.UNKNOWN)

    @field_validator("location", mode="before")
    @classmethod
    def _location(cls, value: Any) -> Location:
        return _match_alias(_LOCATION_ALIASES, value, Location.UNKNOWN)


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


def _enum_schema(enum_cls: type[Enum]) -> dict[str, Any]:
    return {"type": "string", "enum": [item.value for item in enum_cls]}


def facts_json_schema() -> dict[str, Any]:
    """Строгая схема для response_format. Все поля обязательны, лишние запрещены."""
    properties: dict[str, Any] = {
        "permit_number": _nullable({"type": "string"}),
        "permit_explicitly_absent": {"type": "boolean"},
        "instruction_date": _nullable({"type": "string"}),
        "hazard_present": _nullable({"type": "boolean"}),
        "distance_m": _nullable({"type": "number"}),
        "distance_precision": _enum_schema(Precision),
        "approval": _enum_schema(Approval),
        "extinguisher_present": _nullable({"type": "boolean"}),
        "observer_present": _nullable({"type": "boolean"}),
        "observer_is_worker": _nullable({"type": "boolean"}),
        "location": _enum_schema(Location),
        "wind_speed_ms": _nullable({"type": "number"}),
        "wind_precision": _enum_schema(Precision),
        "height_m": _nullable({"type": "number"}),
        "height_precision": _enum_schema(Precision),
        "height_is_unclear": {"type": "boolean"},
        "height_permit_present": _nullable({"type": "boolean"}),
    }
    if set(properties) != set(Facts.model_fields):
        raise RuntimeError("JSON-схема фактов разъехалась с моделью Facts")
    return {
        "name": "hotwork_facts",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": list(properties),
        },
    }


def facts_response_format() -> dict[str, Any]:
    return {"type": "json_schema", "json_schema": facts_json_schema()}


def coerce_optional_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"null", "none", "unknown"}:
            return None
        for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d.%m.%y"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
    return None


def coerce_optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", ".")
        if not text or text.lower() in {"null", "none", "unknown"}:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def coerce_tri_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if not text or text in {"null", "none", "unknown", "неизвестно"}:
            return None
        if text in {"true", "yes", "да", "1"}:
            return True
        if text in {"false", "no", "нет", "0"}:
            return False
    return None


def _match_alias(aliases: dict[str, Any], value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, type(default)):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in aliases:
            return aliases[key]
    return default


def parse_facts(raw: str) -> Facts:
    payload = load_json_object(raw)
    if isinstance(payload.get("facts"), dict):
        payload = payload["facts"]
    known = set(Facts.model_fields)
    if len(known & set(payload)) < MIN_FACT_FIELDS:
        raise ParseError("В ответе модели слишком мало полей фактов")
    return Facts.model_validate(payload)


def load_json_object(raw: str) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        raise ParseError("Пустой ответ модели")
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ParseError("Ответ модели не JSON") from None
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ParseError("Ответ модели не JSON") from exc
    if not isinstance(payload, dict):
        raise ParseError("Ответ модели не объект JSON")
    return payload
