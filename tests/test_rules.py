from datetime import date

import pytest

from hotwork.config import load_regulation, shift_months
from hotwork.models import Approval, Facts, Location, Precision
from hotwork.rules import classify_strict, decide, evaluate


@pytest.fixture
def regulation():
    return load_regulation()


def allow_facts(**overrides) -> Facts:
    payload = dict(
        permit_number="НД-1",
        permit_explicitly_absent=False,
        instruction_date=date(2026, 9, 1),
        hazard_present=False,
        approval=Approval.ABSENT,
        extinguisher_present=True,
        observer_present=True,
        observer_is_worker=False,
        location=Location.INDOOR,
        height_is_unclear=False,
    )
    payload.update(overrides)
    return Facts(**payload)


def statuses(facts: Facts, regulation) -> dict[str, str]:
    return {item.rule_id: item.status.value for item in evaluate(facts, regulation)}


def decision_of(facts: Facts, regulation):
    return decide(evaluate(facts, regulation))


def test_instruction_cutoff_is_calendar_six_months(regulation):
    assert regulation.work_date == date(2026, 10, 15)
    assert regulation.instruction_cutoff() == date(2026, 4, 15)
    assert shift_months(date(2026, 3, 31), -1) == date(2026, 2, 28)


@pytest.mark.parametrize(
    ("value", "precision", "mode", "expected"),
    [
        (10, Precision.EXACT, "gt", "no"),
        (10.01, Precision.EXACT, "gt", "yes"),
        (10, Precision.APPROXIMATE, "gt", "unknown"),
        (14, Precision.APPROXIMATE, "gt", "yes"),
        (9.99, Precision.EXACT, "lt", "yes"),
        (10, Precision.EXACT, "lt", "no"),
        (10, Precision.APPROXIMATE, "lt", "unknown"),
        (6, Precision.APPROXIMATE, "lt", "yes"),
        (1.8, Precision.EXACT, "gt", "no"),
        (1.81, Precision.EXACT, "gt", "yes"),
        (1.8, Precision.APPROXIMATE, "gt", "unknown"),
        (3, Precision.APPROXIMATE, "gt", "yes"),
        (None, Precision.EXACT, "gt", "unknown"),
    ],
)
def test_boundary_comparisons(regulation, value, precision, mode, expected):
    threshold = 1.8 if mode == "gt" and value is not None and value < 5 else 10
    if value in (1.8, 1.81, 3):
        threshold = 1.8
    assert classify_strict(value, precision, threshold, mode, regulation) == expected


def test_instruction_on_cutoff_passes_and_day_before_fails(regulation):
    on_cutoff, rules, _, missing = decision_of(
        allow_facts(instruction_date=date(2026, 4, 15)), regulation
    )
    assert on_cutoff.value == "ДОПУСТИТЬ"
    assert "П2" in rules
    assert missing == []

    before, rules, _, _ = decision_of(allow_facts(instruction_date=date(2026, 4, 14)), regulation)
    assert before.value == "ОТКАЗАТЬ"
    assert rules == ["П2"]


def test_missing_or_future_instruction_escalates(regulation):
    decision, rules, _, missing = decision_of(allow_facts(instruction_date=None), regulation)
    assert decision.value == "ЭСКАЛИРОВАТЬ"
    assert rules == ["П2", "П7"]
    assert missing == ["дата инструктажа"]

    future, rules, _, _ = decision_of(allow_facts(instruction_date=date(2026, 10, 16)), regulation)
    assert future.value == "ЭСКАЛИРОВАТЬ"
    assert "П2" in rules


def test_distance_boundaries(regulation):
    exact_ten, _, _, _ = decision_of(
        allow_facts(
            hazard_present=True,
            distance_m=10,
            distance_precision=Precision.EXACT,
            approval=Approval.ABSENT,
            location=Location.OUTDOOR,
            wind_speed_ms=3,
            wind_precision=Precision.EXACT,
        ),
        regulation,
    )
    assert exact_ten.value == "ДОПУСТИТЬ"

    closer, rules, _, _ = decision_of(
        allow_facts(
            hazard_present=True,
            distance_m=9.99,
            distance_precision=Precision.EXACT,
            approval=Approval.ABSENT,
            location=Location.OUTDOOR,
            wind_speed_ms=3,
            wind_precision=Precision.EXACT,
        ),
        regulation,
    )
    assert closer.value == "ОТКАЗАТЬ"
    assert rules == ["П3"]


def test_unspecified_approval_form_escalates_when_close(regulation):
    decision, rules, _, missing = decision_of(
        allow_facts(
            hazard_present=True,
            distance_m=6,
            distance_precision=Precision.EXACT,
            approval=Approval.UNKNOWN,
            location=Location.OUTDOOR,
            wind_speed_ms=3,
            wind_precision=Precision.EXACT,
        ),
        regulation,
    )
    assert decision.value == "ЭСКАЛИРОВАТЬ"
    assert rules == ["П3", "П7"]
    assert "форма согласования начальника участка" in missing


def test_oral_approval_is_a_violation(regulation):
    decision, rules, explanation, missing = decision_of(
        allow_facts(
            permit_number="НД-160",
            instruction_date=date(2026, 9, 18),
            hazard_present=True,
            distance_m=9,
            distance_precision=Precision.EXACT,
            approval=Approval.ORAL,
            location=Location.OUTDOOR,
            wind_speed_ms=2,
            wind_precision=Precision.EXACT,
        ),
        regulation,
    )
    assert decision.value == "ОТКАЗАТЬ"
    assert rules == ["П3"]
    assert missing == []
    assert "устное" in explanation


def test_written_approval_allows_work_inside_ten_meters(regulation):
    decision, _, _, _ = decision_of(
        allow_facts(
            hazard_present=True,
            distance_m=6,
            distance_precision=Precision.EXACT,
            approval=Approval.WRITTEN,
            location=Location.OUTDOOR,
            wind_speed_ms=3,
            wind_precision=Precision.EXACT,
        ),
        regulation,
    )
    assert decision.value == "ДОПУСТИТЬ"


def test_height_and_wind_boundaries(regulation):
    at_height, _, _, _ = decision_of(allow_facts(height_m=1.8, height_precision=Precision.EXACT), regulation)
    assert at_height.value == "ДОПУСТИТЬ"

    above, rules, _, missing = decision_of(
        allow_facts(height_m=1.81, height_precision=Precision.EXACT, height_permit_present=False),
        regulation,
    )
    assert above.value == "ЭСКАЛИРОВАТЬ"
    assert rules == ["П6", "П7"]
    assert missing == ["допуск на высоту"]

    about_three, rules, _, missing = decision_of(
        allow_facts(height_m=3, height_precision=Precision.APPROXIMATE),
        regulation,
    )
    assert about_three.value == "ЭСКАЛИРОВАТЬ"
    assert "П6" in rules
    assert missing == ["допуск на высоту"]

    outdoor = dict(location=Location.OUTDOOR, wind_precision=Precision.EXACT)
    at_wind, _, _, _ = decision_of(allow_facts(wind_speed_ms=10, **outdoor), regulation)
    assert at_wind.value == "ДОПУСТИТЬ"
    over_wind, rules, _, _ = decision_of(allow_facts(wind_speed_ms=10.01, **outdoor), regulation)
    assert over_wind.value == "ОТКАЗАТЬ"
    assert rules == ["П5"]

    indoor_gale, _, _, _ = decision_of(
        allow_facts(location=Location.INDOOR, wind_speed_ms=14, wind_precision=Precision.EXACT),
        regulation,
    )
    assert indoor_gale.value == "ДОПУСТИТЬ"


def test_same_observer_and_missing_people(regulation):
    same, rules, _, _ = decision_of(allow_facts(observer_is_worker=True), regulation)
    assert same.value == "ОТКАЗАТЬ"
    assert rules == ["П4"]

    alone, rules, _, _ = decision_of(allow_facts(observer_present=False), regulation)
    assert alone.value == "ОТКАЗАТЬ"
    assert rules == ["П4"]

    no_extinguisher, rules, _, _ = decision_of(allow_facts(extinguisher_present=False), regulation)
    assert no_extinguisher.value == "ОТКАЗАТЬ"
    assert rules == ["П4"]


def test_contradictory_location_escalates(regulation):
    decision, rules, _, missing = decision_of(
        allow_facts(location=Location.CONTRADICTORY, wind_speed_ms=3, wind_precision=Precision.EXACT),
        regulation,
    )
    assert decision.value == "ЭСКАЛИРОВАТЬ"
    assert rules == ["П5", "П7"]
    assert "тип места работ" in missing


def test_known_violation_beats_missing_data(regulation):
    decision, rules, _, missing = decision_of(
        allow_facts(permit_number=None, permit_explicitly_absent=True, instruction_date=None),
        regulation,
    )
    assert decision.value == "ОТКАЗАТЬ"
    assert rules == ["П1"]
    assert missing == []


def test_height_permit_never_rejects_by_itself(regulation):
    outcomes = evaluate(
        allow_facts(height_m=4, height_precision=Precision.EXACT, height_permit_present=False),
        regulation,
    )
    height = next(item for item in outcomes if item.rule_id == "П6")
    assert height.status.value == "unknown"


def test_positions_for_control_applications(regulation):
    ninth, _, _, _ = decision_of(
        allow_facts(
            permit_number="НД-155",
            instruction_date=date(2026, 6, 11),
            hazard_present=True,
            distance_m=12,
            distance_precision=Precision.EXACT,
            approval=Approval.WRITTEN,
            location=Location.OUTDOOR,
            wind_speed_ms=8,
            wind_precision=Precision.EXACT,
        ),
        regulation,
    )
    assert ninth.value == "ДОПУСТИТЬ"

    tenth, rules, _, missing = decision_of(
        allow_facts(
            permit_number="НД-158",
            instruction_date=None,
            location=Location.OUTDOOR,
            wind_speed_ms=None,
        ),
        regulation,
    )
    assert tenth.value == "ЭСКАЛИРОВАТЬ"
    assert set(rules) == {"П2", "П5", "П7"}
    assert "дата инструктажа" in missing
    assert "скорость ветра" in missing


def test_threshold_comes_from_regulation(tmp_path):
    source = tmp_path / "regulation.json"
    source.write_text(
        """
        {
          "version": "custom",
          "work_date": "2026-10-15",
          "instruction_max_age_months": 6,
          "hazard_distance_m": 5,
          "max_wind_ms": 10,
          "height_threshold_m": 1.8,
          "max_text_chars": 12000,
          "approximate_margin_ratio": 0.15,
          "approximate_margin_abs": 0.3
        }
        """,
        encoding="utf-8",
    )
    custom = load_regulation(source)
    decision, rules, _, _ = decision_of(
        allow_facts(
            hazard_present=True,
            distance_m=6,
            distance_precision=Precision.EXACT,
            approval=Approval.ABSENT,
            location=Location.OUTDOOR,
            wind_speed_ms=3,
            wind_precision=Precision.EXACT,
        ),
        custom,
    )
    assert decision.value == "ДОПУСТИТЬ"
    assert "П3" in rules
    assert statuses(
        allow_facts(
            hazard_present=True,
            distance_m=4,
            distance_precision=Precision.EXACT,
            approval=Approval.ABSENT,
        ),
        custom,
    )["П3"] == "fail"
