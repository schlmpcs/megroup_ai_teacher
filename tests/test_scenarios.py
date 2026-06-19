import pytest

from app.services import scenarios


def test_load_known_scenario():
    doc = scenarios.load_scenario("physics_lab_02_heating")
    assert doc["subject"] == "Физика"
    assert "спиртовка" in doc["objects"]


def test_load_unknown_scenario_raises():
    with pytest.raises(scenarios.ScenarioNotFoundError):
        scenarios.load_scenario("does_not_exist")


def test_path_traversal_blocked():
    # basename() strips the traversal; the resulting id has no file -> not found.
    with pytest.raises(scenarios.ScenarioNotFoundError):
        scenarios.load_scenario("../../etc/passwd")


def test_format_scenario_context_renders_labels():
    doc = scenarios.load_scenario("physics_lab_02_heating")
    text = scenarios.format_scenario_context(doc)
    assert "Сценарий:" in text
    assert "Последовательность действий:" in text
    # List fields render as bullet lines.
    assert "- спиртовка" in text


def test_format_skips_empty_fields():
    text = scenarios.format_scenario_context({"scenario_name": "X", "topic": ""})
    assert "Сценарий: X" in text
    assert "Тема" not in text


def test_format_scenario_state_renders_step_and_items():
    text = scenarios.format_scenario_state(
        current_step="Зажечь спиртовку", held_items=["спички", " ", "термометр"]
    )
    assert "Текущий шаг ученика: Зажечь спиртовку" in text
    # Blank items are dropped.
    assert "спички, термометр" in text


def test_format_scenario_state_empty_when_nothing_supplied():
    assert scenarios.format_scenario_state() == ""
    assert scenarios.format_scenario_state(current_step="   ", held_items=[]) == ""


def test_get_scenario_context_none_for_no_id():
    assert scenarios.get_scenario_context(None) is None


def test_list_scenarios_includes_example():
    ids = [s["scenario_id"] for s in scenarios.list_scenarios()]
    assert "physics_lab_02_heating" in ids
