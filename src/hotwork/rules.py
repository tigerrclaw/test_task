from __future__ import annotations

from dataclasses import dataclass

from hotwork.config import Regulation
from hotwork.models import Approval, Decision, Facts, Location, Precision, RuleStatus


@dataclass(frozen=True)
class RuleOutcome:
    rule_id: str
    status: RuleStatus
    reason: str
    missing: tuple[str, ...] = ()


def fmt_date(value) -> str:
    return value.strftime("%d.%m.%Y")


def fmt_num(value: float) -> str:
    if float(value).is_integer():
        text = str(int(value))
    else:
        text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", ",")


def classify_strict(
    value: float | None,
    precision: Precision,
    threshold: float,
    mode: str,
    regulation: Regulation,
) -> str:
    """Строгое сравнение с порогом. Для «около» у границы возвращает unknown."""
    if value is None:
        return "unknown"
    if precision is Precision.APPROXIMATE:
        margin = max(threshold * regulation.approximate_margin_ratio, regulation.approximate_margin_abs)
        if abs(value - threshold) <= margin:
            return "unknown"
    if mode == "gt":
        return "yes" if value > threshold else "no"
    if mode == "lt":
        return "yes" if value < threshold else "no"
    raise ValueError(f"Неизвестный режим сравнения: {mode}")


def check_p1(facts: Facts, regulation: Regulation) -> RuleOutcome:
    del regulation
    if facts.permit_number:
        return RuleOutcome("П1", RuleStatus.PASS, f"Указан наряд {facts.permit_number}.")
    if facts.permit_explicitly_absent:
        return RuleOutcome(
            "П1",
            RuleStatus.FAIL,
            "Наряд-допуск отсутствует: в заявке прямо сказано, что номера наряда нет.",
        )
    return RuleOutcome("П1", RuleStatus.FAIL, "Номер наряда-допуска в заявке не указан.")


def check_p2(facts: Facts, regulation: Regulation) -> RuleOutcome:
    cutoff = regulation.instruction_cutoff()
    work = fmt_date(regulation.work_date)
    if facts.instruction_date is None:
        return RuleOutcome(
            "П2",
            RuleStatus.UNKNOWN,
            "Дата инструктажа по пожарной безопасности не указана однозначно.",
            ("дата инструктажа",),
        )
    stated = fmt_date(facts.instruction_date)
    if facts.instruction_date > regulation.work_date:
        return RuleOutcome(
            "П2",
            RuleStatus.UNKNOWN,
            f"Дата инструктажа {stated} позже даты работ {work}.",
            ("дата инструктажа",),
        )
    months = regulation.instruction_max_age_months
    if facts.instruction_date >= cutoff:
        return RuleOutcome(
            "П2",
            RuleStatus.PASS,
            f"Инструктаж {stated} не раньше порога {fmt_date(cutoff)} ({months} мес. до {work}).",
        )
    return RuleOutcome(
        "П2",
        RuleStatus.FAIL,
        f"Инструктаж {stated} раньше чем за {months} мес. до даты работ {work}: порог {fmt_date(cutoff)}.",
    )


def check_p3(facts: Facts, regulation: Regulation) -> RuleOutcome:
    threshold = fmt_num(regulation.hazard_distance_m)
    if facts.hazard_present is False:
        return RuleOutcome("П3", RuleStatus.PASS, "Рядом нет склада угля или ГСМ.")
    if facts.hazard_present is None and facts.distance_m is None:
        return RuleOutcome("П3", RuleStatus.PASS, "Склад угля или ГСМ в заявке не указан.")

    relation = classify_strict(
        facts.distance_m,
        facts.distance_precision,
        regulation.hazard_distance_m,
        "lt",
        regulation,
    )
    if relation == "no" and facts.distance_m is not None:
        return RuleOutcome(
            "П3",
            RuleStatus.PASS,
            f"Расстояние {fmt_num(facts.distance_m)} м не ближе {threshold} м, письменное согласование по П3 не требуется.",
        )
    if relation == "unknown":
        if facts.approval is Approval.WRITTEN:
            return RuleOutcome(
                "П3",
                RuleStatus.PASS,
                "Расстояние до склада угля или ГСМ задано неточно, но письменное согласование начальника участка есть.",
            )
        missing = ["расстояние до склада угля или ГСМ"]
        if facts.approval is Approval.UNKNOWN:
            missing.append("форма согласования начальника участка")
        return RuleOutcome(
            "П3",
            RuleStatus.UNKNOWN,
            f"Нельзя однозначно установить, ближе ли работы {threshold} м к складу угля или ГСМ.",
            tuple(missing),
        )
    return _approval_when_close(facts, threshold)


def _approval_when_close(facts: Facts, threshold: str) -> RuleOutcome:
    distance = fmt_num(facts.distance_m) if facts.distance_m is not None else "не указано"
    if facts.approval is Approval.WRITTEN:
        return RuleOutcome(
            "П3",
            RuleStatus.PASS,
            f"Работы ближе {threshold} м ({distance} м), есть письменное согласование начальника участка.",
        )
    if facts.approval is Approval.ORAL:
        return RuleOutcome(
            "П3",
            RuleStatus.FAIL,
            f"Работы ближе {threshold} м ({distance} м), согласование только устное, а П3 требует письменное.",
        )
    if facts.approval is Approval.ABSENT:
        return RuleOutcome(
            "П3",
            RuleStatus.FAIL,
            f"Работы ближе {threshold} м ({distance} м), письменного согласования начальника участка нет.",
        )
    return RuleOutcome(
        "П3",
        RuleStatus.UNKNOWN,
        f"Работы ближе {threshold} м ({distance} м), но из заявки неясно, письменное ли согласование.",
        ("форма согласования начальника участка",),
    )


def check_p4(facts: Facts, regulation: Regulation) -> RuleOutcome:
    del regulation
    failures: list[str] = []
    missing: list[str] = []

    if facts.extinguisher_present is False:
        failures.append("Огнетушителя на месте нет.")
    elif facts.extinguisher_present is None:
        missing.append("наличие огнетушителя")

    if facts.observer_is_worker is True:
        failures.append("Наблюдающий совпадает с исполнителем, второго человека нет.")
    elif facts.observer_present is False:
        failures.append("Наблюдающего нет: работу выполняет один человек.")
    elif facts.observer_present is None:
        missing.append("наблюдающий")

    if failures:
        return RuleOutcome("П4", RuleStatus.FAIL, " ".join(failures))
    if missing:
        return RuleOutcome(
            "П4",
            RuleStatus.UNKNOWN,
            "Не хватает данных по огнетушителю или наблюдающему.",
            tuple(missing),
        )
    return RuleOutcome(
        "П4",
        RuleStatus.PASS,
        "Огнетушитель есть, наблюдающий — второй человек, не исполнитель.",
    )


def check_p5(facts: Facts, regulation: Regulation) -> RuleOutcome:
    if facts.location is Location.INDOOR:
        return RuleOutcome("П5", RuleStatus.PASS, "Работы в помещении, ограничение по ветру не применяется.")
    if facts.location is not Location.OUTDOOR:
        return RuleOutcome(
            "П5",
            RuleStatus.UNKNOWN,
            "Нельзя однозначно установить, помещение это или открытая площадка.",
            ("тип места работ",),
        )
    relation = classify_strict(
        facts.wind_speed_ms,
        facts.wind_precision,
        regulation.max_wind_ms,
        "gt",
        regulation,
    )
    limit = fmt_num(regulation.max_wind_ms)
    if relation == "unknown":
        return RuleOutcome(
            "П5",
            RuleStatus.UNKNOWN,
            "Скорость ветра на открытой площадке не указана однозначно.",
            ("скорость ветра",),
        )
    speed = fmt_num(facts.wind_speed_ms) if facts.wind_speed_ms is not None else "?"
    if relation == "yes":
        return RuleOutcome(
            "П5",
            RuleStatus.FAIL,
            f"Открытая площадка, ветер {speed} м/с, это больше {limit} м/с.",
        )
    return RuleOutcome(
        "П5",
        RuleStatus.PASS,
        f"Открытая площадка, ветер {speed} м/с, не больше {limit} м/с.",
    )


def check_p6(facts: Facts, regulation: Regulation) -> RuleOutcome:
    mentioned = facts.height_m is not None or facts.height_is_unclear
    limit = fmt_num(regulation.height_threshold_m)
    if not mentioned:
        return RuleOutcome("П6", RuleStatus.PASS, f"Работы на высоте более {limit} м в заявке не заявлены.")
    if facts.height_m is None:
        return RuleOutcome(
            "П6",
            RuleStatus.UNKNOWN,
            "Высота работ упомянута, но число метров не установлено.",
            ("высота работ",),
        )
    relation = classify_strict(
        facts.height_m,
        facts.height_precision,
        regulation.height_threshold_m,
        "gt",
        regulation,
    )
    if relation == "unknown":
        return RuleOutcome(
            "П6",
            RuleStatus.UNKNOWN,
            f"Нельзя однозначно сравнить высоту с порогом {limit} м.",
            ("высота работ",),
        )
    height = fmt_num(facts.height_m)
    if relation == "no":
        return RuleOutcome(
            "П6",
            RuleStatus.PASS,
            f"Высота {height} м не больше {limit} м, отдельный допуск на высоту не требуется.",
        )
    if facts.height_permit_present is True:
        return RuleOutcome(
            "П6",
            RuleStatus.PASS,
            f"Высота {height} м больше {limit} м, отдельный допуск на высоту указан.",
        )
    return RuleOutcome(
        "П6",
        RuleStatus.UNKNOWN,
        f"Высота {height} м больше {limit} м, отдельный допуск на высоту в заявке не указан. По П6 это решает человек.",
        ("допуск на высоту",),
    )


RULES = (check_p1, check_p2, check_p3, check_p4, check_p5, check_p6)


def evaluate(facts: Facts, regulation: Regulation) -> list[RuleOutcome]:
    return [rule(facts, regulation) for rule in RULES]


def decide(outcomes: list[RuleOutcome]) -> tuple[Decision, list[str], str, list[str]]:
    failed = [item for item in outcomes if item.status is RuleStatus.FAIL]
    unknown = [item for item in outcomes if item.status is RuleStatus.UNKNOWN]
    passed = [item for item in outcomes if item.status is RuleStatus.PASS]
    if failed:
        return (
            Decision.REJECT,
            [item.rule_id for item in failed],
            " ".join(item.reason for item in failed),
            [],
        )
    if unknown:
        missing: list[str] = []
        for item in unknown:
            for name in item.missing:
                if name not in missing:
                    missing.append(name)
        explanation = " ".join(item.reason for item in unknown)
        explanation += " По П7 выполнение нельзя установить однозначно, поэтому решение эскалируется."
        return Decision.ESCALATE, [item.rule_id for item in unknown] + ["П7"], explanation, missing
    return (
        Decision.ALLOW,
        [item.rule_id for item in passed],
        " ".join(item.reason for item in passed),
        [],
    )
