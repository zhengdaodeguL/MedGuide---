from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import app.main as main_module
from app.llm import OpenAIAnswerer
from app.main import AppServices, _principal_from_headers, app
from app.retrieval import HybridRetriever
from app.safety import SafetyEngine
from app.sql_guard import ReadOnlySQLGuard
from app.store import SessionAuthorizationError, SessionStore
from app.workflow import MedGuideWorkflow, WorkflowExecutionError


def _workflow() -> MedGuideWorkflow:
    return MedGuideWorkflow(HybridRetriever(), ReadOnlySQLGuard(), SafetyEngine())


def test_cors_defaults_to_same_origin_and_requires_explicit_allowlist(monkeypatch) -> None:
    monkeypatch.delenv("MEDGUIDE_CORS_ORIGINS", raising=False)
    assert main_module._cors_origins() == []

    monkeypatch.setenv(
        "MEDGUIDE_CORS_ORIGINS",
        "https://medguide.example, https://admin.medguide.example/",
    )
    assert main_module._cors_origins() == [
        "https://medguide.example",
        "https://admin.medguide.example",
    ]

    monkeypatch.setenv("MEDGUIDE_CORS_ORIGINS", "*")
    assert main_module._cors_origins() == []


def test_production_requires_secure_cookie_and_https_pair(monkeypatch) -> None:
    monkeypatch.setenv("MEDGUIDE_COOKIE_SECURE", "true")
    monkeypatch.setenv("MEDGUIDE_REQUIRE_HTTPS", "true")
    assert main_module._secure_transport_configuration() is True

    monkeypatch.setenv("MEDGUIDE_COOKIE_SECURE", "false")
    assert main_module._secure_transport_configuration() is False
    monkeypatch.setenv("MEDGUIDE_COOKIE_SECURE", "true")
    monkeypatch.setenv("MEDGUIDE_REQUIRE_HTTPS", "false")
    assert main_module._secure_transport_configuration() is False

    monkeypatch.setenv("MEDGUIDE_REQUIRE_HTTPS", "invalid")
    with pytest.raises(ValueError):
        main_module._secure_transport_configuration()


def test_api_proxy_cidr_prefers_compose_specific_setting(monkeypatch) -> None:
    monkeypatch.setenv("MEDGUIDE_TRUSTED_PROXY_CIDR", "127.0.0.1/32")
    monkeypatch.setenv("MEDGUIDE_API_TRUSTED_PROXY_CIDR", "172.30.0.0/16")
    assert str(main_module._trusted_proxy_networks()[0]) == "172.30.0.0/16"


def test_production_api_rejects_http_even_with_untrusted_forwarded_proto(monkeypatch) -> None:
    current = SimpleNamespace(
        production=True,
        api_tokens=("token-a",),
        production_missing=(),
        store=SessionStore(mode="production"),
    )
    monkeypatch.setattr(main_module, "get_services", lambda: current)

    with TestClient(app, base_url="http://testserver") as client:
        response = client.post(
            "/api/sessions",
            headers={
                "Authorization": "Bearer token-a",
                "X-Forwarded-Proto": "https",
            },
        )

    assert response.status_code == 426


def test_session_creation_is_idempotent_for_one_client_key() -> None:
    with TestClient(app) as client:
        headers = {"Idempotency-Key": "session-browser-load-001"}
        first = client.post("/api/sessions", headers=headers)
        second = client.post("/api/sessions", headers=headers)
        other = client.post("/api/sessions", headers={"Idempotency-Key": "session-browser-load-002"})

    assert first.status_code == second.status_code == other.status_code == 200
    assert first.json()["session_id"] == second.json()["session_id"]
    assert other.json()["session_id"] != first.json()["session_id"]


def test_chat_request_id_replays_result_without_advancing_turn() -> None:
    with TestClient(app) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        request = {
            "session_id": session_id,
            "message": "我32岁，咳嗽持续3天",
            "request_id": "turn-browser-0001",
        }
        first = client.post("/api/chat", json=request)
        replay = client.post("/api/chat", json=request)
        recovered = client.get(f"/api/sessions/{session_id}/requests/turn-browser-0001")
        state = client.get(f"/api/sessions/{session_id}")

    assert first.status_code == replay.status_code == recovered.status_code == state.status_code == 200
    assert replay.json()["response_id"] == first.json()["response_id"]
    assert recovered.json()["response_id"] == first.json()["response_id"]
    assert state.json()["turn_count"] == 1


def test_request_id_cannot_be_reused_for_different_message() -> None:
    with TestClient(app) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        first = client.post(
            "/api/chat",
            json={"session_id": session_id, "message": "我咳嗽", "request_id": "turn-collision-0001"},
        )
        conflict = client.post(
            "/api/chat",
            json={"session_id": session_id, "message": "我腹痛", "request_id": "turn-collision-0001"},
        )

    assert first.status_code == 200
    assert conflict.status_code == 409


def test_duplicate_feedback_is_an_upsert_not_a_second_metric() -> None:
    with TestClient(app) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        answer = client.post(
            "/api/chat",
            json={"session_id": session_id, "message": "我咳嗽", "request_id": "turn-feedback-0001"},
        ).json()
        payload = {"session_id": session_id, "message_id": answer["response_id"], "rating": "up"}
        first = client.post("/api/feedback", json=payload)
        duplicate = client.post("/api/feedback", json=payload)
        metrics = client.get("/api/metrics").json()

    assert first.json()["changed"] is True
    assert duplicate.json()["changed"] is False
    assert metrics["feedback"]["up"] == 1


def test_session_owner_is_enforced_without_exposing_existence() -> None:
    store = SessionStore()
    session = store.create(owner_id="principal-a")

    assert store.get(session["session_id"], owner_id="principal-a") is not None
    with pytest.raises(SessionAuthorizationError):
        store.get(session["session_id"], owner_id="principal-b")


def test_in_memory_session_and_creation_key_expire() -> None:
    now = [0.0]
    store = SessionStore(ttl_seconds=60, clock=lambda: now[0])
    first = store.create(owner_id="owner", idempotency_key="create-key")
    now[0] = 61.0

    assert store.get(first["session_id"], owner_id="owner") is None
    second = store.create(owner_id="owner", idempotency_key="create-key")
    assert second["session_id"] != first["session_id"]


def test_public_state_does_not_claim_low_risk_before_screening() -> None:
    store = SessionStore()
    state = store.create()
    assert store.public(state)["risk_level"] is None
    assert store.public(state)["risk_assessed"] is False

    state["risk_level"] = "low"
    state["risk_assessed"] = True
    assert store.public(state)["risk_level"] == "low"


def test_production_tokens_map_to_distinct_principals() -> None:
    current = SimpleNamespace(production=True, api_tokens=("token-a", "token-b"))
    first = _principal_from_headers(current, "Bearer token-a")
    second = _principal_from_headers(current, "Bearer token-b")

    assert first and second and first != second
    with pytest.raises(HTTPException) as exc_info:
        _principal_from_headers(current, "Bearer wrong")
    assert exc_info.value.status_code == 401


def test_production_http_authentication_and_session_ownership(monkeypatch) -> None:
    current = SimpleNamespace(
        production=True,
        api_tokens=("token-a", "token-b"),
        production_missing=(),
        store=SessionStore(mode="production"),
    )
    monkeypatch.setattr(main_module, "get_services", lambda: current)

    with TestClient(app, base_url="https://testserver") as client:
        created = client.post(
            "/api/sessions",
            headers={"Authorization": "Bearer token-a"},
        )
        session_id = created.json()["session_id"]
        missing = client.get(f"/api/sessions/{session_id}")
        invalid = client.get(
            f"/api/sessions/{session_id}",
            headers={"Authorization": "Bearer invalid"},
        )
        other_owner = client.get(
            f"/api/sessions/{session_id}",
            headers={"Authorization": "Bearer token-b"},
        )
        owner = client.get(
            f"/api/sessions/{session_id}",
            headers={"Authorization": "Bearer token-a"},
        )
        non_idempotent_chat = client.post(
            "/api/chat",
            headers={"Authorization": "Bearer token-a"},
            json={"session_id": session_id, "message": "我咳嗽"},
        )

    assert created.status_code == 200
    assert missing.status_code == invalid.status_code == 401
    assert other_owner.status_code == 404
    assert owner.status_code == 200
    assert non_idempotent_chat.status_code == 400
    assert "request_id" in non_idempotent_chat.json()["detail"]


def test_offline_mode_never_constructs_openai_resources(monkeypatch) -> None:
    monkeypatch.setenv("MEDGUIDE_MODE", " offline ")
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")

    def fail_if_called(_self):
        raise AssertionError("offline mode must not construct an OpenAI client")

    monkeypatch.setattr(OpenAIAnswerer, "_create_resources", fail_if_called)
    answerer = OpenAIAnswerer()

    assert answerer.available is False
    assert answerer.ensure_ready() is False


def test_production_requires_an_explicit_openai_model(monkeypatch) -> None:
    monkeypatch.setenv("MEDGUIDE_MODE", "production")
    monkeypatch.setenv("OPENAI_API_KEY", "configured-test-key")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    def fail_if_called(_self):
        raise AssertionError("production must not construct a client without OPENAI_MODEL")

    monkeypatch.setattr(OpenAIAnswerer, "_create_resources", fail_if_called)
    answerer = OpenAIAnswerer()

    assert answerer.model == ""
    assert answerer.available is False
    assert answerer.ensure_ready() is False
    assert answerer.last_error == "MissingModelConfiguration"


def test_production_readiness_retries_answerer_after_backoff() -> None:
    class RecoverableAnswerer:
        available = False

        def __init__(self) -> None:
            self.calls = 0

        def ensure_ready(self) -> bool:
            self.calls += 1
            self.available = True
            return True

    class ReadyEmbedding:
        available = True

        def ensure_ready(self) -> bool:
            return True

    class ReadyAdapter:
        ready = True

        def ensure_ready(self) -> bool:
            return True

    current = AppServices.__new__(AppServices)
    current.production = True
    current.answerer = RecoverableAnswerer()
    current.embedding_provider = ReadyEmbedding()
    current.milvus = ReadyAdapter()
    current.mysql = ReadyAdapter()
    current.redis = ReadyAdapter()
    current.retriever = SimpleNamespace(keyword_ready=True)
    current.api_tokens = ("configured",)

    assert current.production_missing == ()
    assert current.answerer.calls == 1


def test_high_risk_structured_request_never_calls_query_backend() -> None:
    class Guard:
        def from_natural_language(self, _text):
            raise AssertionError("high-risk flow must not query structured data")

    workflow = MedGuideWorkflow(HybridRetriever(), Guard(), SafetyEngine())
    result = workflow.run(
        {
            "session_id": "high-risk-query",
            "user_message": "突然胸痛，帮我查布洛芬库存",
            "profile": {"symptoms": [], "associated_symptoms": []},
        }
    )

    assert result["risk_level"] == "high"
    assert result["structured_result"]["blocked"] is True
    assert result["structured_result"]["rows"] == []
    assert "急诊" in result["response"]


def test_external_answerer_never_receives_explicit_identity_fields() -> None:
    captured: dict[str, str] = {}

    class CapturingAnswerer:
        available = True

        def generate(self, query: str, risk: str, context: str) -> str:
            captured.update(query=query, risk=risk, context=context)
            return "仅用于验证脱敏边界的资料整理。"

    workflow = MedGuideWorkflow(
        HybridRetriever(),
        ReadOnlySQLGuard(),
        SafetyEngine(),
        CapturingAnswerer(),
    )
    result = workflow.run(
        {
            "session_id": "phi-boundary",
            "user_message": "姓名：张三，邮箱 zhangsan@example.com，地址：北京市朝阳区建国路88号，我咳嗽3天",
            "profile": {"symptoms": [], "associated_symptoms": []},
        }
    )

    assert result["answer_source"] == "openai-grounded"
    outbound = f"{captured['query']}\n{captured['context']}"
    assert "张三" not in outbound
    assert "zhangsan@example.com" not in outbound
    assert "北京市朝阳区建国路88号" not in outbound
    assert "主要症状：咳嗽" in captured["query"]


@pytest.mark.parametrize(
    ("message", "excluded", "included"),
    (
        ("我没有出现过胸痛，只有咳嗽", "胸痛", "咳嗽"),
        ("胸痛的不是我，是我父亲，我只有咳嗽", "胸痛", "咳嗽"),
    ),
)
def test_profile_extraction_excludes_negated_or_third_party_symptoms(
    message: str,
    excluded: str,
    included: str,
) -> None:
    workflow = _workflow()
    state = workflow.normalize({"user_message": message, "profile": {"symptoms": [], "associated_symptoms": []}})
    result = workflow.extract_profile(state)
    symptoms = result["profile"]["symptoms"]

    assert excluded not in symptoms
    assert included in symptoms


def test_profile_extraction_applies_negation_to_coordinated_symptoms() -> None:
    workflow = _workflow()
    state = workflow.normalize(
        {
            "user_message": "咳嗽，已经三天，没有胸痛或呼吸困难",
            "profile": {"symptoms": [], "associated_symptoms": []},
        }
    )
    result = workflow.extract_profile(state)

    assert result["profile"]["symptoms"] == ["咳嗽"]
    assert result["profile"]["associated_symptoms"] == []


def test_external_answerer_receives_no_raw_unlabeled_identity_text() -> None:
    captured: dict[str, str] = {}

    class CapturingAnswerer:
        available = True

        def generate(self, query: str, risk: str, context: str) -> str:
            captured.update(query=query, risk=risk, context=context)
            return "仅用于验证最小化出站内容。"

    workflow = MedGuideWorkflow(
        HybridRetriever(),
        ReadOnlySQLGuard(),
        SafetyEngine(),
        CapturingAnswerer(),
    )
    result = workflow.run(
        {
            "session_id": "phi-unlabeled",
            "user_message": "患者张三现住北京市朝阳区建国路88号，我咳嗽3天",
            "profile": {"symptoms": [], "associated_symptoms": []},
        }
    )

    assert result["answer_source"] == "openai-grounded"
    outbound = f"{captured['query']}\n{captured['context']}"
    assert "张三" not in outbound
    assert "北京市朝阳区建国路88号" not in outbound
    assert "主要症状：咳嗽" in captured["query"]


def test_langgraph_runtime_failure_is_not_replayed() -> None:
    workflow = _workflow()
    calls = {"graph": 0, "normalize": 0}

    class BrokenGraph:
        def invoke(self, _state):
            calls["graph"] += 1
            raise RuntimeError("partial graph failure")

    original_normalize = workflow.normalize

    def counted_normalize(state):
        calls["normalize"] += 1
        return original_normalize(state)

    workflow.graph = BrokenGraph()
    workflow.normalize = counted_normalize
    with pytest.raises(WorkflowExecutionError):
        workflow.run({"session_id": "graph-failure", "user_message": "咳嗽", "profile": {"symptoms": []}})

    assert calls == {"graph": 1, "normalize": 0}
