from app.retrieval import HybridRetriever
from app.safety import SafetyEngine
from app.sql_guard import ReadOnlySQLGuard
from app.workflow import MedGuideWorkflow


def make_workflow() -> MedGuideWorkflow:
    return MedGuideWorkflow(HybridRetriever(), ReadOnlySQLGuard(), SafetyEngine())


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
