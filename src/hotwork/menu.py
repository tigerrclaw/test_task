from __future__ import annotations

import os
from pathlib import Path

from hotwork.config import ConfigError, Regulation, data_dir, default_regulation_path, load_regulation
from hotwork.rules import fmt_date, fmt_num

MENU = """
Агент допуска к огневым работам
1. Текущий регламент
2. Справка по запуску
3. Проверить одну заявку
4. Выход
""".strip()


def regulation_text(regulation: Regulation) -> str:
    distance = fmt_num(regulation.hazard_distance_m)
    wind = fmt_num(regulation.max_wind_ms)
    height = fmt_num(regulation.height_threshold_m)
    margin_ratio = fmt_num(regulation.approximate_margin_ratio * 100)
    margin_abs = fmt_num(regulation.approximate_margin_abs)
    return "\n".join(
        [
            f"Регламент {regulation.version}",
            f"Файл: {default_regulation_path()}",
            f"Дата работ: {fmt_date(regulation.work_date)}",
            "",
            "П1. В заявке должен быть номер наряда. Нет номера — отказ.",
            f"П2. Инструктаж по пожарной безопасности не раньше чем за {regulation.instruction_max_age_months} месяцев до {fmt_date(regulation.work_date)}. Дата неясна — эскалация, дата старше порога — отказ.",
            f"П3. Ближе {distance} м от склада угля или ГСМ — только письменное согласование. Устное или отсутствие — отказ. «Согласование есть» без формы — эскалация.",
            "П4. Нужны огнетушитель и наблюдающий, второй человек, не исполнитель. Явно нет — отказ. Не сказано — эскалация.",
            f"П5. Открытая площадка и ветер строго выше {wind} м/с — отказ. Ровно {wind} м/с можно. Помещение — пункт не применяется. Место неясно — эскалация.",
            f"П6. Высота строго выше {height} м и отдельный допуск указан — проходит. Допуска в заявке нет — эскалация, не отказ. Ровно {height} м допуск не нужен.",
            "П7. Нельзя однозначно установить пункт — эскалация. Приоритет: отказ важнее эскалации, эскалация важнее допуска.",
            "",
            f"Текст длиннее {regulation.max_text_chars} символов в модель не отправляется.",
            f"«Около» у порога не решается автоматически: запас {margin_ratio}% или {margin_abs}, что больше.",
            "Флаги запуска подменяют эти числа только на время прогона. Здесь показан файл по умолчанию.",
        ]
    )


def help_text() -> str:
    return "\n".join(
        [
            "Проверка заявок. Контейнер без аргументов читает /data/input.json и пишет /data/output.json.",
            "Журнал: /data/journal/.",
            "",
            "Меню, одна заявка текстом:",
            "docker run -it -v <папка>:/data --env-file <файл> <образ> menu",
            "python -m hotwork menu",
            "В меню цифра 3. Вставьте текст и нажмите Enter на пустой строке.",
            "",
            "Пакет из файла, как у проверяющих:",
            "docker build .",
            "docker run -v <папка>:/data --env-file <файл> <образ>",
            "python -m hotwork data/applications.json --journal-dir build/journal",
            "Проверки качества:",
            "python -m hotwork eval",
            "docker run -v <папка>:/data --env-file <файл> <образ> eval",
            "",
            "Обязательные переменные: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL.",
            "Запасной провайдер, все три или ни одной: LLM_FALLBACK_BASE_URL, LLM_FALLBACK_API_KEY, LLM_FALLBACK_MODEL.",
            "Пороги на один запуск: --work-date, --instruction-months, --hazard-distance, --max-wind, --height, --max-text-chars.",
            "Свой файл регламента: --regulation или REGULATION_PATH.",
            "Ключ в журнал не пишется. В терминале и в журнале видны ошибки вызова, паузы и модель, которая ответила.",
        ]
    )


def manual_journal_dir() -> Path:
    configured = os.environ.get("JOURNAL_DIR", "").strip()
    if configured:
        return Path(configured)
    return data_dir() / "journal"


def read_pasted_application() -> str | None:
    print("Вставьте текст заявки. Пустая строка отправляет её на проверку.")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            print()
            return None
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def submit_pasted_application(regulation: Regulation) -> None:
    text = read_pasted_application()
    if text is None:
        return
    if not text:
        print("Текст пустой, проверка не запускалась.")
        return
    from hotwork.cli import LazyExtractor
    from hotwork.journal import JournalSession
    from hotwork.pipeline import display_path, process_application

    journal_dir = manual_journal_dir()
    journal = JournalSession(journal_dir, "manual_request")
    try:
        result = process_application(
            {"id": "manual", "text": text},
            regulation=regulation,
            extractor=LazyExtractor(),
            journal=journal,
        )
    except ConfigError as exc:
        print(str(exc))
        return
    journal_path = journal.flush()
    print()
    print(f"Решение: {result['decision']}")
    rules = ", ".join(result["rules"]) or "—"
    print(f"Пункты: {rules}")
    print(result["explanation"])
    if result["missing"]:
        print("Не хватает: " + ", ".join(result["missing"]))
    print(f"Журнал: {display_path(journal_path)}")


def run_menu() -> int:
    regulation = load_regulation()
    screens = {
        "1": lambda: regulation_text(regulation),
        "2": help_text,
        "3": lambda: submit_pasted_application(regulation),
    }
    while True:
        print(MENU)
        try:
            choice = input("> ").strip().lower()
        except EOFError:
            print()
            print(regulation_text(regulation))
            print()
            print(help_text())
            return 0
        if choice in {"4", "q", "й", "выход", "exit"}:
            return 0
        screen = screens.get(choice)
        if screen is None:
            print("Нужна цифра от 1 до 4.")
            continue
        shown = screen()
        if shown:
            print(shown)
        print()
