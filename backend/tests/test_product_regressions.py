from __future__ import annotations

from pathlib import Path

from app.retrieval import HybridRetriever, IntentRouter
from app.safety import SafetyEngine
from app.sql_guard import ReadOnlySQLGuard
from app.workflow import MedGuideWorkflow, WorkflowExecutionError


def make_workflow() -> MedGuideWorkflow:
    return MedGuideWorkflow(HybridRetriever(), ReadOnlySQLGuard(), SafetyEngine())


def test_associated_symptoms_and_negative_history_survive_follow_up() -> None:
    workflow = make_workflow()
    first = workflow.run(
        {
            "session_id": "profile-regression",
            "user_message": "我32岁，咳嗽三天，还流鼻涕，没有哮喘等既往病史",
            "profile": {"symptoms": [], "associated_symptoms": []},
        }
    )

    assert first["profile"]["chief_complaint"] == "咳嗽"
    assert "流鼻涕" in first["profile"]["associated_symptoms"]
    assert first["profile"]["history"] == "否认哮喘"
    assert first["next_question"] is None

    first["user_message"] = "没有发热，也没有呼吸困难"
    second = workflow.run(first)

    assert "流鼻涕" in second["profile"]["associated_symptoms"]
    assert second["profile"]["history"] == "否认哮喘"
    assert second["next_question"] is None


def test_common_department_phrasing_routes_to_department_knowledge() -> None:
    router = IntentRouter()
    for query in (
        "咳嗽三天应该挂什么科？",
        "咳嗽应该去哪个科？",
        "皮疹看什么科？",
    ):
        assert router.classify(query) == "department"

    result = make_workflow().run(
        {
            "session_id": "department-regression",
            "user_message": "我32岁，咳嗽三天应该挂什么科？",
            "profile": {"symptoms": [], "associated_symptoms": []},
        }
    )

    assert result["intent"] == "department"
    assert result["citations"]
    assert result["citations"][0]["category"] == "department"


def test_exact_exam_entity_outranks_other_exam_documents() -> None:
    results = HybridRetriever().search("血常规检查前需要注意什么？", "exam", top_k=4)

    assert results
    assert results[0].id.startswith("exam-cbc-001")


def test_product_boundary_question_routes_to_faq_and_switches_topic() -> None:
    workflow = make_workflow()
    first = workflow.run(
        {
            "session_id": "faq-switch-regression",
            "user_message": "血常规检查前需要注意什么？",
            "profile": {"symptoms": [], "associated_symptoms": []},
        }
    )
    assert first["intent"] == "exam"

    first["user_message"] = "MedGuide 能替代医生诊断吗？"
    second = workflow.run(first)

    assert second["intent"] == "faq"
    assert second["citations"]
    assert all(item["category"] == "faq" for item in second["citations"])
    assert all("影像" not in item["title"] for item in second["citations"])


def test_clinical_reasoning_language_is_not_mistaken_for_a_diagnosis() -> None:
    workflow = make_workflow()
    for response in (
        "血常规可帮助医生结合症状判断是否需要进一步检查。",
        "这些信息用于判断是否需要就医，不能替代医生诊断。",
    ):
        reviewed = workflow.safety_review({"risk_level": "watch", "response": response})
        assert reviewed["safety_blocked"] is False
        assert "判断是否需要" in reviewed["response"]
        assert reviewed["response"] != MedGuideWorkflow.SAFE_REVIEW_FALLBACK


def test_benign_watch_query_keeps_its_grounded_answer() -> None:
    result = make_workflow().run(
        {
            "session_id": "watch-regression",
            "user_message": "28岁女性，头痛两天",
            "profile": {"symptoms": [], "associated_symptoms": []},
        }
    )

    assert result["risk_level"] != "high"
    assert result["safety_blocked"] is False
    assert result["response"] != MedGuideWorkflow.SAFE_REVIEW_FALLBACK


def test_production_generation_failure_never_silently_uses_a_template() -> None:
    class RequiredUnavailableAnswerer:
        required = True

        @staticmethod
        def generate(_query: str, _risk: str, _context: str) -> None:
            return None

    workflow = MedGuideWorkflow(
        HybridRetriever(),
        ReadOnlySQLGuard(),
        SafetyEngine(),
        RequiredUnavailableAnswerer(),
    )

    try:
        workflow.run(
            {
                "session_id": "provider-regression",
                "user_message": "我32岁，咳嗽三天，应该挂什么科？",
                "profile": {"symptoms": [], "associated_symptoms": []},
            }
        )
    except WorkflowExecutionError:
        pass
    else:
        raise AssertionError("required provider failure must not return a local answer template")


def test_production_gateway_has_auth_throttling_and_strict_browser_headers() -> None:
    root = Path(__file__).resolve().parents[2]
    nginx = (root / "frontend" / "nginx.conf").read_text(encoding="utf-8")
    styles = (root / "frontend" / "src" / "styles.css").read_text(encoding="utf-8")

    assert "zone=medguide_auth" in nginx
    assert "location ~ ^/api/auth/(login|register)$" in nginx
    assert "Content-Security-Policy" in nginx
    assert "frame-ancestors 'none'" in nginx
    assert "location = /api/health" in nginx
    assert "location = /api/ready" in nginx
    assert "fonts.googleapis.com" not in styles
    assert ".workspace { height: calc(100vh - 60px); }" in styles
