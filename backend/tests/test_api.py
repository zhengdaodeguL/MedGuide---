from fastapi.testclient import TestClient
import pytest

from app.main import AppServices, app


def test_session_chat_and_search_api() -> None:
    with TestClient(app, base_url="https://testserver") as client:
        created = client.post("/api/sessions")
        assert created.status_code == 200
        session_id = created.json()["session_id"]
        response = client.post("/api/chat", json={"session_id": session_id, "message": "我32岁，咳嗽持续3天"})
        assert response.status_code == 200
        body = response.json()
        assert body["session_id"] == session_id
        assert body["response_id"] == f"{session_id}:turn:1"
        assert body["citations"]
        assert body["state"]["profile"]["age"] == 32
        assert body["state"]["mode"] == "test"
        assert body["state"]["workflow_engine"] in {"langgraph", "deterministic"}
        feedback = client.post(
            "/api/feedback",
            json={"session_id": session_id, "message_id": body["response_id"], "rating": "up", "comment": "清楚"},
        )
        assert feedback.status_code == 200
        unknown_feedback = client.post(
            "/api/feedback",
            json={"session_id": session_id, "message_id": "other-session:turn:1", "rating": "down"},
        )
        assert unknown_feedback.status_code == 422
        metrics = client.get("/api/metrics")
        assert metrics.json()["requests"] >= 1
        searched = client.post("/api/search", json={"query": "布洛芬副作用", "intent": "drug"})
        assert searched.status_code == 200
        assert searched.json()["count"] > 0


def test_sse_chat_has_node_and_final_events() -> None:
    with TestClient(app) as client:
        response = client.post("/api/chat?stream=true", json={"message": "查布洛芬库存"})
        assert response.status_code == 200
        assert "event: node" in response.text
        assert "event: final" in response.text


def test_production_mode_reports_and_blocks_missing_dependencies(monkeypatch) -> None:
    monkeypatch.setenv("MEDGUIDE_MODE", "production")
    monkeypatch.setenv("MEDGUIDE_API_TOKEN", "test-production-token")
    monkeypatch.delenv("MEDGUIDE_API_TOKENS", raising=False)
    for name in ("OPENAI_API_KEY", "MILVUS_URI", "MYSQL_DSN", "REDIS_URL"):
        monkeypatch.delenv(name, raising=False)

    with TestClient(app, base_url="https://testserver") as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["live"] is True
        assert health.json()["mode"] == "production"
        assert health.json()["auth_configured"] is True

        ready = client.get("/api/ready")
        assert ready.status_code == 503

        unauthenticated = client.post("/api/sessions")
        assert unauthenticated.status_code == 401

        session = client.post(
            "/api/sessions",
            headers={"Authorization": "Bearer test-production-token"},
        )
        assert session.status_code == 503
        assert "生产依赖未就绪" in session.json()["detail"]


def test_invalid_proxy_cidr_is_rejected_during_service_initialization(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MEDGUIDE_MODE", "offline")
    monkeypatch.setenv("MEDGUIDE_AUTH_DB_PATH", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setenv("MEDGUIDE_API_TRUSTED_PROXY_CIDR", "not-a-cidr")

    with pytest.raises(ValueError, match="MEDGUIDE_API_TRUSTED_PROXY_CIDR"):
        AppServices()


def test_missing_mode_defaults_to_fail_closed_production(tmp_path, monkeypatch) -> None:
    import app.main as main_module

    monkeypatch.delenv("MEDGUIDE_MODE", raising=False)
    monkeypatch.setenv("MEDGUIDE_AUTH_DB_PATH", str(tmp_path / "auth.sqlite3"))
    for name in (
        "MEDGUIDE_API_TOKEN",
        "MEDGUIDE_API_TOKENS",
        "MEDGUIDE_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "MILVUS_URI",
        "MYSQL_DSN",
        "REDIS_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    current = main_module.AppServices()
    try:
        assert current.mode == "production"
        assert current.production is True
        assert current.production_missing
    finally:
        current.close()


def test_api_responses_disable_shared_caching() -> None:
    with TestClient(app) as client:
        response = client.post("/api/sessions")
        assert response.headers["cache-control"] == "no-store, private"
        assert response.headers["pragma"] == "no-cache"
        assert "Cookie" in response.headers["vary"]


def test_production_health_is_minimal_and_requires_explicit_knowledge_backend(monkeypatch, tmp_path) -> None:
    import app.main as main_module

    monkeypatch.setenv("MEDGUIDE_MODE", "production")
    monkeypatch.setenv("MEDGUIDE_AUTH_DB_PATH", str(tmp_path / "auth.sqlite3"))
    for name in ("OPENAI_API_KEY", "MILVUS_URI", "MYSQL_DSN", "REDIS_URL", "MEDGUIDE_API_TOKENS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("MEDGUIDE_KNOWLEDGE_BACKEND", raising=False)

    current = main_module.AppServices()
    try:
        assert current.knowledge_backend == "milvus"
        assert "Milvus" in current.production_missing
    finally:
        current.close()

    with TestClient(main_module.app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert "generator" not in health.json()
