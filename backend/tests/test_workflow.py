from app.retrieval import HybridRetriever
from app.safety import SafetyEngine
from app.sql_guard import ReadOnlySQLGuard
from app.workflow import MedGuideWorkflow
import pytest


def make_workflow() -> MedGuideWorkflow:
    return MedGuideWorkflow(HybridRetriever(), ReadOnlySQLGuard(), SafetyEngine())


@pytest.mark.parametrize("text,sex", [
    ("32岁男性，咳嗽三天", "男"), ("28岁女性咳嗽", "女"),
    ("我32岁男，咳嗽", "男"), ("我男朋友32岁男性，咳嗽", None),
    ("不是32岁男性", None), ("32岁男朋友咳嗽", None),
])
def test_compact_age_gender_profile(text, sex) -> None:
    result = make_workflow().extract_profile({"normalized_message": text, "profile": {}})
    assert result["profile"].get("sex") == sex


@pytest.mark.parametrize("text", ["我突然剧烈头痛", "我突然剧烈腹痛", "呕吐物是绿色", "我有咖啡渣样呕吐"])
def test_new_topic_emergency_signals_skip_generation(text):
    result = make_workflow().run({"session_id": "synthetic", "user_message": text, "profile": {}})
    assert result["risk_level"] == "high"
    assert result["citations"] == []
    assert "急诊" in result["response"]


def test_cough_retrieval_does_not_expand_demographics_into_urinary_advice():
    result = make_workflow().run({"session_id": "synthetic", "user_message":
        "32岁男性，咳嗽三天，没有胸痛，没有呼吸困难", "profile": {}})
    ids = [item["id"] for item in result["citations"]]
    assert any(item.startswith("disease-cough-001") for item in ids)
    assert not any(item.startswith("disease-uti-001") for item in ids)


def test_workflow_collects_profile_and_citations() -> None:
    workflow = make_workflow()
    result = workflow.run({"session_id": "test", "user_message": "我32岁，咳嗽持续3天，想知道看什么科室", "profile": {"symptoms": []}})
    assert result["intent"] in ("department", "disease")
    assert result["profile"]["age"] == 32
    assert result["profile"]["duration"] == "3天"
    assert result["citations"]
    assert result["response"]
    assert "32岁" in result["summary"]
    assert "主要症状：咳嗽" in result["summary"]
    assert len(result["events"]) == 9


def test_workflow_engine_label_matches_actual_runner() -> None:
    workflow = make_workflow()
    result = workflow.run({"session_id": "engine", "user_message": "咳嗽", "profile": {"symptoms": []}})

    assert result["workflow_engine"] == ("langgraph" if workflow.graph is not None else "deterministic")


def test_workflow_never_drafts_diagnosis_for_red_flag() -> None:
    result = make_workflow().run({"session_id": "test", "user_message": "突然胸痛，呼吸困难", "profile": {"symptoms": []}})
    assert result["risk_level"] == "high"
    assert "急诊" in result["response"]
    assert "确诊" not in result["response"]
    assert result["citations"] == []


def test_workflow_executes_structured_query() -> None:
    result = make_workflow().run({"session_id": "test", "user_message": "查布洛芬库存", "profile": {"symptoms": []}})
    assert result["intent"] == "structured"
    assert result["structured_result"]["blocked"] is False
    assert result["structured_result"]["rows"]
