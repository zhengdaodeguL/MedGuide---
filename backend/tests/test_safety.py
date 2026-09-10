from app.safety import SafetyEngine


def test_red_flag_is_high_priority() -> None:
    result = SafetyEngine().screen("突然胸痛并且呼吸困难")
    assert result.level == "high"
    assert "胸痛或胸部压榨感" in result.flags
    assert "呼吸困难或口唇发紫" in result.flags
    assert "急诊" in result.advice


def test_non_urgent_symptom_is_watch() -> None:
    result = SafetyEngine().screen("咳嗽持续三天")
    assert result.level == "watch"


def test_special_population_is_flagged() -> None:
    result = SafetyEngine().screen("发热", {"age": 1})
    assert result.level == "high"
    assert "婴幼儿或高龄人群" in result.flags


def test_negation_scope_does_not_hide_a_later_red_flag() -> None:
    engine = SafetyEngine()

    assert engine.screen("我不舒服，胸痛").level == "high"
    assert engine.screen("我很不舒服，呼吸困难").level == "high"
    assert engine.screen("没有胸痛但呼吸困难").flags == ("呼吸困难或口唇发紫",)
    assert engine.screen("胸痛，但是没有呼吸困难").flags == ("胸痛或胸部压榨感",)
    assert engine.screen("患者否认胸痛，但口唇发紫").flags == ("呼吸困难或口唇发紫",)


def test_negated_and_knowledge_mentions_are_not_treated_as_current_symptoms() -> None:
    engine = SafetyEngine()

    assert engine.screen("没有胸痛，也不喘").level != "high"
    assert engine.screen("胸痛和呼吸困难都没有").level != "high"
    assert engine.screen("胸痛、呼吸困难均无").level != "high"
    assert engine.screen("胸痛、呼吸困难均否认").level != "high"
    assert engine.screen("胸痛、呼吸困难均不存在").level != "high"
    assert engine.screen("胸痛和呼吸困难都未出现").level != "high"
    assert engine.screen("没有胸痛以及呼吸困难").level != "high"
    assert engine.screen("没有胸痛或呼吸困难").level != "high"
    assert engine.screen("胸痛或呼吸困难均无").level != "high"
    assert engine.screen("不伴有胸痛和呼吸困难").level != "high"
    assert engine.screen("我没有出现过胸痛，只有咳嗽").level != "high"
    assert engine.screen("胸痛有哪些表现？").level != "high"
    assert engine.screen("什么是胸痛").level != "high"
    assert engine.screen("我胸痛需要急诊吗？").level == "high"


def test_negation_scope_does_not_hide_a_measured_high_temperature() -> None:
    engine = SafetyEngine()

    assert engine.screen("体温没有39.5度").level != "high"
    assert engine.screen("体温是否39.5度？").level != "high"
    assert engine.screen("体温是否达到39.5度？").level != "high"
    assert engine.screen("体温没有到39.5度").level != "high"
    assert engine.screen("体温低于39.5度").level != "high"
    assert engine.screen("体温不超过39.5度").level != "high"
    assert engine.screen("体温39.5度以下").level != "high"
    assert engine.screen("体温39.5度怎么办？").level == "high"
    assert engine.screen("我不舒服，体温39.5度").level == "high"
    assert engine.screen("体温39.5摄氏度").level == "high"


def test_negated_or_unrelated_pregnancy_terms_are_not_special_population_flags() -> None:
    engine = SafetyEngine()

    for text in ("我没有怀孕", "否认怀孕", "未怀孕", "怀孕吗？", "孕酮偏低", "孕检多少钱"):
        assert "孕期或哺乳期特殊人群" not in engine.screen(text).flags

    assert "孕期或哺乳期特殊人群" in engine.screen("我已经怀孕12周").flags
