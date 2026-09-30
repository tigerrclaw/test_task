from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from hotwork.config import (
    ConfigError,
    fallback_settings_from_env,
    llm_settings_from_env,
    data_dir,
    load_env_file,
    load_regulation,
    submission_root,
    with_overrides,
)
from hotwork.extract import LLMExtractor, PROMPT_VERSION
from hotwork.journal import JournalSession
from hotwork.llm import LLMClient
from hotwork.pipeline import display_path, needs_model, process_application


class LazyExtractor:
    prompt_version = PROMPT_VERSION

    def __init__(self) -> None:
        self._inner: LLMExtractor | None = None

    def extract(self, text: str):
        if self._inner is None:
            self._inner = build_extractor()
        return self._inner.extract(text)


def build_extractor() -> LLMExtractor:
    base_url, api_key, model, timeout = llm_settings_from_env()
    fallback_settings = fallback_settings_from_env()
    fallback = None
    if fallback_settings is not None:
        fallback_url, fallback_key, fallback_model = fallback_settings
        fallback = LLMClient(fallback_url, fallback_key, fallback_model, timeout)
    return LLMExtractor(LLMClient(base_url, api_key, model, timeout), fallback)


def load_applications(path: Path) -> list[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Не удалось прочитать {path}: {exc}") from None
    if not isinstance(payload, list):
        raise ConfigError("Входной файл должен быть JSON-списком объектов {id, text}")
    applications = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or "id" not in item:
            raise ConfigError(f"Элемент {index} должен быть объектом с полем id")
        text = item.get("text")
        if text is None:
            text = ""
        elif not isinstance(text, str):
            text = str(text)
        applications.append({"id": item["id"], "text": text})
    return applications


def run(
    input_path: Path,
    output_path: Path,
    journal_dir: Path,
    regulation_path: str | None,
    overrides: dict | None = None,
) -> None:
    regulation = with_overrides(load_regulation(regulation_path), **(overrides or {}))
    print(
        "регламент {version}: работы {work_date}, инструктаж {months} мес., "
        "склад {distance} м, ветер {wind} м/с, высота {height} м".format(
            version=regulation.version,
            work_date=regulation.work_date.isoformat(),
            months=regulation.instruction_max_age_months,
            distance=regulation.hazard_distance_m,
            wind=regulation.max_wind_ms,
            height=regulation.height_threshold_m,
        ),
        file=sys.stderr,
    )
    applications = load_applications(input_path)
    if any(needs_model(item["text"], regulation.max_text_chars) for item in applications):
        llm_settings_from_env()
        fallback_settings_from_env()
    extractor = LazyExtractor()
    journal = JournalSession(journal_dir, "request")
    results = [
        process_application(
            application,
            regulation=regulation,
            extractor=extractor,
            journal=journal,
        )
        for application in applications
    ]
    journal_path = journal.flush()
    print(f"журнал: {display_path(journal_path)}", file=sys.stderr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    output_path.write_text(payload, encoding="utf-8")
    print(f"{display_path(output_path)}", file=sys.stderr)
    for copy in result_copies(output_path):
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_text(payload, encoding="utf-8")
        print(f"{display_path(copy)}", file=sys.stderr)


def result_copies(output_path: Path) -> list[Path]:
    """Куда ещё положить тот же JSON, что и в output. В тестах копии не пишутся."""
    if "pytest" in sys.modules:
        return []
    root = submission_root()
    copies = [root / "results.json"]
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in copies:
        resolved = path.resolve()
        if resolved == output_path.resolve() or resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def main(argv: list[str] | None = None) -> int:
    if "pytest" not in sys.modules:
        load_env_file()
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["eval"]:
        from eval.run_eval import main as eval_main

        return eval_main()
    if args[:1] == ["menu"]:
        from hotwork.menu import run_menu

        return run_menu()
    parser = argparse.ArgumentParser(description="Проверка заявок на огневые работы")
    folder = data_dir()
    parser.add_argument("input", nargs="?", default=os.environ.get("INPUT_PATH", str(folder / "input.json")))
    parser.add_argument("-o", "--output", default=os.environ.get("OUTPUT_PATH", str(folder / "output.json")))
    parser.add_argument("--journal-dir", default=os.environ.get("JOURNAL_DIR", str(folder / "journal")))
    parser.add_argument("--regulation", default=os.environ.get("REGULATION_PATH"), help="JSON-файл регламента")
    parser.add_argument("--work-date", help="Дата работ, YYYY-MM-DD")
    parser.add_argument("--instruction-months", type=int, help="П2: инструктаж не старше N месяцев")
    parser.add_argument("--hazard-distance", type=float, help="П3: работы ближе N метров требуют письменного согласования")
    parser.add_argument("--max-wind", type=float, help="П5: на открытой площадке запрещён ветер строго выше N м/с")
    parser.add_argument("--height", type=float, help="П6: выше N метров нужен допуск на высоту")
    parser.add_argument("--max-text-chars", type=int, help="Текст длиннее N символов не отправляется в модель")
    args = parser.parse_args(argv)
    overrides = {
        "work_date": args.work_date,
        "instruction_max_age_months": args.instruction_months,
        "hazard_distance_m": args.hazard_distance,
        "max_wind_ms": args.max_wind,
        "height_threshold_m": args.height,
        "max_text_chars": args.max_text_chars,
    }
    try:
        run(Path(args.input), Path(args.output), Path(args.journal_dir), args.regulation, overrides)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
