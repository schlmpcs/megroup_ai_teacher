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
    assert "актуальный снимок сцены от симулятора" in text
    assert "имеют приоритет" in text
    assert "Текущий шаг ученика: Зажечь спиртовку" in text
    # Blank items are dropped.
    assert "спички, термометр" in text


def test_format_scenario_state_renders_full_authoritative_snapshot():
    text = scenarios.format_scenario_state(
        current_step_id="heat-water",
        current_step_index=3,
        current_step="Нагреть воду",
        next_step_id="record-temperature",
        next_step="Записать температуру",
        completed_steps=["prepare-stand", "measure-water"],
        held_items=[],
        visible_items=["термометр", "стакан"],
        allowed_actions=["включить нагрев", "поставить стакан"],
        last_action="Установил стакан на штатив",
        last_action_result="Успешно",
    )

    assert "ID текущего шага: heat-water" in text
    assert "Индекс текущего шага: 3" in text
    assert "ID следующего шага: record-temperature" in text
    assert "Следующий шаг, назначенный симулятором: Записать температуру" in text
    assert "Завершённые шаги: prepare-stand, measure-water" in text
    assert "Предметы в руках у ученика: нет" in text
    assert "Предметы, видимые ученику: термометр, стакан" in text
    assert "Разрешённые действия сейчас: включить нагрев, поставить стакан" in text
    assert "Последнее действие ученика: Установил стакан на штатив" in text
    assert "Результат последнего действия: Успешно" in text


def test_format_scenario_state_empty_when_nothing_supplied():
    assert scenarios.format_scenario_state() == ""
    assert scenarios.format_scenario_state(current_step="   ") == ""


def test_format_scenario_state_explicit_empty_list_means_none():
    text = scenarios.format_scenario_state(held_items=[])
    assert "Предметы в руках у ученика: нет" in text


def test_get_scenario_context_none_for_no_id():
    assert scenarios.get_scenario_context(None) is None


def test_list_scenarios_includes_example():
    ids = [s["scenario_id"] for s in scenarios.list_scenarios()]
    assert "physics_lab_02_heating" in ids
