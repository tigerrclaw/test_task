from __future__ import annotations

import calendar
import json
import os
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path


class ConfigError(Exception):
    """Ошибка запуска: файл, регламент или переменные окружения."""


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def in_container() -> bool:
    return os.environ.get("HOTWORK_IN_CONTAINER") == "1" or Path("/.dockerenv").exists()


def resolve_data_dir(mount: Path, local_fallback: Path, container: bool) -> Path:
    """В контейнере /data — корень проекта, заявки лежат в /data/data. Если input.json сразу в /data, это контракт проверяющих."""
    if not container:
        return local_fallback
    if (mount / "input.json").is_file():
        return mount
    nested = mount / "data"
    if (nested / "input.json").is_file():
        return nested
    return mount


def data_dir() -> Path:
    return resolve_data_dir(Path("/data"), project_root() / "data", in_container())


def submission_root() -> Path:
    """Куда писать results.json и eval_report.md: корень проекта, рядом с каталогом data."""
    folder = data_dir()
    if folder.name == "data":
        return folder.parent
    return folder


def load_env_file(path: Path | None = None) -> None:
    """Подхватывает .env, не затирая переменные, которые уже заданы."""
    candidates = [path] if path is not None else [Path.cwd() / ".env", project_root() / ".env"]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen or not candidate.is_file():
            continue
        seen.add(resolved)
        for line in candidate.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            name, value = text.split("=", 1)
            name = name.strip()
            if not name or name in os.environ:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            os.environ[name] = value
        return


def shift_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


@dataclass(frozen=True)
class Regulation:
    version: str
    work_date: date
    instruction_max_age_months: int
    hazard_distance_m: float
    max_wind_ms: float
    height_threshold_m: float
    max_text_chars: int
    approximate_margin_ratio: float
    approximate_margin_abs: float

    def instruction_cutoff(self) -> date:
        return shift_months(self.work_date, -self.instruction_max_age_months)

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "work_date": self.work_date.isoformat(),
            "instruction_max_age_months": self.instruction_max_age_months,
            "hazard_distance_m": self.hazard_distance_m,
            "max_wind_ms": self.max_wind_ms,
            "height_threshold_m": self.height_threshold_m,
            "max_text_chars": self.max_text_chars,
            "approximate_margin_ratio": self.approximate_margin_ratio,
            "approximate_margin_abs": self.approximate_margin_abs,
        }


def with_overrides(regulation: Regulation, **overrides) -> Regulation:
    """Подмена порогов с запуска. None означает «оставить значение из файла»."""
    chosen = {key: value for key, value in overrides.items() if value is not None}
    if not chosen:
        return regulation
    if "work_date" in chosen and isinstance(chosen["work_date"], str):
        try:
            chosen["work_date"] = datetime.strptime(chosen["work_date"], "%Y-%m-%d").date()
        except ValueError as exc:
            raise ConfigError("Дата работ задаётся как YYYY-MM-DD") from exc
    positive = (
        "instruction_max_age_months",
        "hazard_distance_m",
        "max_wind_ms",
        "height_threshold_m",
        "max_text_chars",
        "approximate_margin_ratio",
        "approximate_margin_abs",
    )
    for key in positive:
        if key in chosen and chosen[key] <= 0:
            raise ConfigError(f"{key} должен быть больше нуля")
    version = regulation.version if regulation.version.endswith("+cli") else regulation.version + "+cli"
    return replace(regulation, version=version, **chosen)


def default_regulation_path() -> Path:
    return Path(__file__).with_name("regulation.json")


def load_regulation(path: str | Path | None = None) -> Regulation:
    source = Path(path) if path else default_regulation_path()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Не удалось прочитать регламент {source}: {exc}") from None
    try:
        work_date = datetime.strptime(payload["work_date"], "%Y-%m-%d").date()
        return Regulation(
            version=str(payload["version"]),
            work_date=work_date,
            instruction_max_age_months=int(payload["instruction_max_age_months"]),
            hazard_distance_m=float(payload["hazard_distance_m"]),
            max_wind_ms=float(payload["max_wind_ms"]),
            height_threshold_m=float(payload["height_threshold_m"]),
            max_text_chars=int(payload["max_text_chars"]),
            approximate_margin_ratio=float(payload["approximate_margin_ratio"]),
            approximate_margin_abs=float(payload["approximate_margin_abs"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"Регламент {source} заполнен не полностью: {exc}") from None


def llm_settings_from_env() -> tuple[str, str, str, float]:
    base_url = os.environ.get("LLM_BASE_URL", "").strip()
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    model = os.environ.get("LLM_MODEL", "").strip()
    timeout_raw = os.environ.get("LLM_TIMEOUT", "60").strip()
    missing = [
        name
        for name, value in (
            ("LLM_BASE_URL", base_url),
            ("LLM_API_KEY", api_key),
            ("LLM_MODEL", model),
        )
        if not value
    ]
    if missing:
        raise ConfigError("Не заданы переменные окружения: " + ", ".join(missing))
    try:
        timeout = float(timeout_raw)
    except ValueError as exc:
        raise ConfigError("LLM_TIMEOUT должен быть числом секунд") from exc
    return base_url, api_key, model, timeout


def fallback_settings_from_env() -> tuple[str, str, str] | None:
    names = ("LLM_FALLBACK_BASE_URL", "LLM_FALLBACK_API_KEY", "LLM_FALLBACK_MODEL")
    values = {name: os.environ.get(name, "").strip() for name in names}
    filled = [name for name, value in values.items() if value]
    if not filled:
        return None
    if len(filled) != len(names):
        missing = [name for name, value in values.items() if not value]
        raise ConfigError("Запасной провайдер задан не полностью, не хватает: " + ", ".join(missing))
    return values["LLM_FALLBACK_BASE_URL"], values["LLM_FALLBACK_API_KEY"], values["LLM_FALLBACK_MODEL"]
