from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.embeddings import OpenAIEmbeddingProvider
from app.auth import AuthStore
from app.llm import OpenAIAnswerer
from app.main import (
    MAX_REQUESTS_PER_SESSION,
    AppServices,
    ChatRequest,
    _check_request_fingerprint,
    _message_fingerprint,
    _run_chat_transaction,
    _state_for_request,
    stream_chat,
)
from app.store import SessionAuthorizationError, SessionConflictError, SessionStore


class _ReadySessionBackend:
    ready = True

    def __init__(self, state: dict | None = None) -> None:
        self.state = deepcopy(state)

    def ensure_ready(self) -> bool:
        return True

    def get_state(self, session_id: str) -> dict | None:
        if self.state is None or self.state.get("session_id") != session_id:
            return None
        return deepcopy(self.state)

    def save_state(self, state: dict, expected_version: int | None = None) -> bool:
        self.state = deepcopy(state)
        return True


def test_redis_backed_local_session_metadata_expires_with_ttl() -> None:
    now = [0.0]
    backend = _ReadySessionBackend()
    store = SessionStore(backend=backend, mode="production", ttl_seconds=60, clock=lambda: now[0])
    state = store.create(owner_id="owner-a", idempotency_key="create-request-0001")
    session_id = state["session_id"]
    with store.session_lock(session_id):
        pass

    assert session_id in store._sessions
    assert session_id in store._expires_at
    assert session_id in store._session_locks
    assert ("owner-a", "create-request-0001") in store._create_idempotency

    now[0] = 61.0
    store._purge_expired()

    assert session_id not in store._sessions
    assert session_id not in store._expires_at
    assert session_id not in store._session_locks
    assert ("owner-a", "create-request-0001") not in store._create_idempotency


def test_redis_remote_read_cache_receives_and_honors_local_ttl() -> None:
    now = [10.0]
    backend = _ReadySessionBackend(
        {
            "session_id": "mg-remote-cache",
            "owner_id": "owner-a",
            "profile": {},
            "_version": 2,
        }
    )
    store = SessionStore(backend=backend, mode="production", ttl_seconds=60, clock=lambda: now[0])

    assert store.get("mg-remote-cache", owner_id="owner-a") is not None
    assert store._expires_at["mg-remote-cache"] == 70.0

    now[0] = 71.0
    store._purge_expired()

    assert "mg-remote-cache" not in store._sessions
    assert "mg-remote-cache" not in store._expires_at


def test_redis_owner_mismatch_remains_an_authorization_error() -> None:
    backend = _ReadySessionBackend(
        {
            "session_id": "mg-owner-check",
            "owner_id": "owner-a",
            "profile": {},
            "_version": 0,
        }
    )
    store = SessionStore(backend=backend, mode="production")

    with pytest.raises(SessionAuthorizationError):
        store.get("mg-owner-check", owner_id="owner-b")

    assert backend.ready is True


def test_evicted_idempotency_result_cannot_advance_session_again() -> None:
    class CountingWorkflow:
        def run(self, state: dict) -> dict:
            result = deepcopy(state)
            result["turn_count"] = int(result.get("turn_count", 0)) + 1
            result["response_id"] = f"response-{result['turn_count']}"
            result["response"] = "ok"
            return result

    current = SimpleNamespace(
        store=SessionStore(),
        workflow=CountingWorkflow(),
        metrics=SimpleNamespace(record=lambda _state: None),
    )
    session_id = current.store.create()["session_id"]

    for index in range(21):
        request = ChatRequest(
            session_id=session_id,
            message=f"message {index}",
            request_id=f"request-{index:04d}",
        )
        _run_chat_transaction(current, _state_for_request(current, request))

    retry = ChatRequest(session_id=session_id, message="message 0", request_id="request-0000")
    with pytest.raises(SessionConflictError, match="回放窗口"):
        _run_chat_transaction(current, _state_for_request(current, retry))

    assert current.store.get(session_id)["turn_count"] == 21


def test_missing_authoritative_session_never_runs_or_recreates_a_turn() -> None:
    class Workflow:
        calls = 0

        def run(self, state: dict) -> dict:
            self.calls += 1
            return state

    backend = _ReadySessionBackend()
    store = SessionStore(backend=backend, mode="production")
    workflow = Workflow()
    recorded: list[dict] = []
    current = SimpleNamespace(
        store=store,
        workflow=workflow,
        metrics=SimpleNamespace(record=recorded.append),
    )
    stale = {
        "session_id": "mg_expired",
        "owner_id": "owner",
        "_version": 0,
        "user_message": "我咳嗽",
    }

    with pytest.raises(SessionConflictError, match="会话已过期"):
        _run_chat_transaction(current, stale)

    assert workflow.calls == 0
    assert backend.state is None
    assert recorded == []


def test_session_deleted_before_commit_is_not_saved_or_recorded() -> None:
    backend = _ReadySessionBackend(
        {
            "session_id": "mg_deleted_during_turn",
            "owner_id": "owner",
            "profile": {},
            "turn_count": 0,
            "_version": 0,
            "request_results": {},
            "request_fingerprints": {},
        }
    )
    saves = [0]
    original_save = backend.save_state

    def save_state(state: dict, expected_version: int | None = None) -> bool:
        saves[0] += 1
        return original_save(state, expected_version)

    backend.save_state = save_state

    class DeletingWorkflow:
        def run(self, state: dict) -> dict:
            backend.state = None
            return {**state, "response": "must not commit", "turn_count": 1}

    recorded: list[dict] = []
    current = SimpleNamespace(
        store=SessionStore(backend=backend, mode="production"),
        workflow=DeletingWorkflow(),
        metrics=SimpleNamespace(record=recorded.append),
    )
    state = current.store.get("mg_deleted_during_turn")
    assert state is not None
    state["user_message"] = "我咳嗽"

    with pytest.raises(SessionConflictError, match="会话已过期"):
        _run_chat_transaction(current, state)

    assert saves == [0]
    assert recorded == []


def test_old_request_replay_keeps_original_answer_and_current_medical_state(monkeypatch) -> None:
    class Workflow:
        calls = 0

        def run(self, state: dict) -> dict:
            self.calls += 1
            turn = int(state.get("turn_count", 0)) + 1
            profile = deepcopy(state.get("profile", {}))
            if turn == 2:
                profile.update({"age": 32, "duration": "3天"})
            return {
                **state,
                "turn_count": turn,
                "profile": profile,
                "summary": f"current-summary-{turn}",
                "response": f"answer-{turn}",
                "response_id": f"response-{turn}",
                "citations": [{"id": f"citation-{turn}"}],
                "events": [],
                "confidence": 0.8,
                "latency_ms": 1,
            }

    store = SessionStore(mode="test")
    workflow = Workflow()
    current = SimpleNamespace(
        store=store,
        workflow=workflow,
        metrics=SimpleNamespace(record=lambda _state: None),
    )
    created = store.create()
    first_request = ChatRequest(
        session_id=created["session_id"],
        message="我咳嗽",
        request_id="request-0001",
    )
    second_request = ChatRequest(
        session_id=created["session_id"],
        message="我32岁，持续3天",
        request_id="request-0002",
    )
    _run_chat_transaction(current, _state_for_request(current, first_request))
    _run_chat_transaction(current, _state_for_request(current, second_request))

    replay = _run_chat_transaction(current, _state_for_request(current, first_request))
    recovered = store.request_result(created["session_id"], "request-0001")

    assert workflow.calls == 2
    for result in (replay, recovered):
        assert result is not None
        assert result["response"] == "answer-1"
        assert result["response_id"] == "response-1"
        assert result["citations"] == [{"id": "citation-1"}]
        assert result["turn_count"] == 2
        assert result["profile"] == {"symptoms": [], "associated_symptoms": [], "age": 32, "duration": "3天"}
        assert result["summary"] == "current-summary-2"
        assert result["_version"] == 2

    monkeypatch.setattr(main_module, "get_services", lambda: current)

    async def collect_stream() -> str:
        response = stream_chat(current, _state_for_request(current, first_request))
        return "".join([chunk async for chunk in response.body_iterator])

    stream_body = asyncio.run(collect_stream())
    final_frame = next(frame for frame in stream_body.split("\n\n") if frame.startswith("event: final"))
    payload = json.loads(next(line[6:] for line in final_frame.splitlines() if line.startswith("data: ")))
    assert payload["answer"] == "answer-1"
    assert payload["response_id"] == "response-1"
    assert payload["state"]["turn_count"] == 2
    assert payload["state"]["profile"]["age"] == 32
    assert payload["state"]["summary"] == "current-summary-2"
    assert workflow.calls == 2


def test_stream_rejects_an_expired_session_before_running_workflow(monkeypatch) -> None:
    class Workflow:
        calls = 0

        def run(self, state: dict, **_kwargs) -> dict:
            self.calls += 1
            return state

    workflow = Workflow()
    current = SimpleNamespace(
        store=SessionStore(backend=_ReadySessionBackend(), mode="production"),
        workflow=workflow,
        metrics=SimpleNamespace(record=lambda _state: None),
    )
    monkeypatch.setattr(main_module, "get_services", lambda: current)

    async def collect_stream() -> str:
        response = stream_chat(
            current,
            {"session_id": "mg_expired_stream", "_version": 0, "user_message": "我咳嗽"},
        )
        return "".join([chunk async for chunk in response.body_iterator])

    body = asyncio.run(collect_stream())
    assert "event: error" in body
    assert '"status": 409' in body
    assert "会话已过期" in body
    assert workflow.calls == 0


def test_app_services_close_includes_auth_is_idempotent_and_continues_after_failure(caplog) -> None:
    closed: list[str] = []

    class Resource:
        def __init__(self, name: str, *, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def close(self) -> None:
            closed.append(self.name)
            if self.fail:
                raise RuntimeError("close failed")

    current = AppServices.__new__(AppServices)
    current.answerer = Resource("answerer", fail=True)
    current.embedding_provider = Resource("embedding")
    current.milvus = Resource("milvus")
    current.mysql = Resource("mysql")
    current.redis = Resource("redis")
    current.auth = Resource("auth")

    current.close()
    current.close()

    assert closed == ["answerer", "embedding", "milvus", "mysql", "redis", "auth"]
    assert "Failed to close Resource: RuntimeError" in caplog.text


def test_production_auth_fallback_rejects_registration_and_login_without_writes(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "fallback-auth.sqlite3"
    auth = AuthStore(
        database_path,
        password_iterations=100_000,
        requires_shared_backend=True,
    )
    current = SimpleNamespace(production=True, auth=auth)
    monkeypatch.setattr(main_module, "get_services", lambda: current)

    client = TestClient(main_module.app, base_url="https://testserver")
    try:
        registration = client.post(
            "/api/auth/register",
            json={"username": "must_not_persist", "password": "Correct-Horse-42"},
        )
        login = client.post(
            "/api/auth/login",
            json={"username": "must_not_persist", "password": "Correct-Horse-42"},
        )
    finally:
        client.close()

    assert registration.status_code == 503
    assert login.status_code == 503
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM auth_sessions").fetchone()[0] == 0


def test_shared_auth_runtime_failure_returns_controlled_service_unavailable(monkeypatch) -> None:
    class BrokenSharedAuth:
        requires_shared_backend = True
        shared = True
        cookie_name = "medguide_session"

        def register_and_create_session(
            self, _username: str, _password: str, *, previous_token: str | None = None
        ) -> tuple[str, str, int]:
            raise OSError("database endpoint unavailable")

        def verify_credentials(self, _username: str, _password: str) -> str:
            raise OSError("database endpoint unavailable")

    current = SimpleNamespace(production=True, auth=BrokenSharedAuth())
    monkeypatch.setattr(main_module, "get_services", lambda: current)
    client = TestClient(main_module.app, base_url="https://testserver")
    try:
        registration = client.post(
            "/api/auth/register",
            json={"username": "shared_user", "password": "Correct-Horse-42"},
        )
        login = client.post(
            "/api/auth/login",
            json={"username": "shared_user", "password": "Correct-Horse-42"},
        )
    finally:
        client.close()

    assert registration.status_code == 503
    assert login.status_code == 503
    assert registration.json()["detail"] == "共享认证服务暂时不可用，请稍后重试"
    assert login.json()["detail"] == "共享认证服务暂时不可用，请稍后重试"


def test_request_tombstones_are_bounded_without_permitting_reexecution() -> None:
    fingerprints = {
        f"request-{index:04d}": "a" * 64
        for index in range(MAX_REQUESTS_PER_SESSION)
    }
    fingerprints["request-0000"] = _message_fingerprint("original message")
    state = {"request_fingerprints": fingerprints, "request_results": {}}

    with pytest.raises(SessionConflictError, match="请求上限"):
        _check_request_fingerprint(state, "request-new", "new message")

    with pytest.raises(SessionConflictError, match="回放窗口"):
        _check_request_fingerprint(state, "request-0000", "original message")

    with pytest.raises(SessionConflictError, match="其他消息"):
        _check_request_fingerprint(state, "request-0000", "different message")

    assert len(state["request_fingerprints"]) == MAX_REQUESTS_PER_SESSION


def test_api_token_sources_are_merged_and_deduplicated(monkeypatch) -> None:
    shared = SimpleNamespace(
        ready=True,
        available=True,
        spec=None,
        keyword_ready=True,
        ensure_ready=lambda **_kwargs: True,
        close=lambda: None,
    )
    monkeypatch.setenv("MEDGUIDE_MODE", "production")
    monkeypatch.setenv("MEDGUIDE_REQUIRE_PROXY_TOKEN", "true")
    monkeypatch.setenv("MEDGUIDE_AUTH_BACKEND", "sqlite")
    monkeypatch.setenv("MEDGUIDE_ALLOW_SQLITE_AUTH", "true")
    monkeypatch.setenv("MEDGUIDE_API_TOKENS", "token-a, token-b, proxy-token")
    monkeypatch.setenv("MEDGUIDE_API_TOKEN", "proxy-token")
    monkeypatch.setenv("MEDGUIDE_AUTH_TOKEN", "legacy-token")
    monkeypatch.setattr(main_module, "embedding_provider_from_env", lambda **_kwargs: shared)
    monkeypatch.setattr(main_module, "MilvusAdapter", lambda *_args, **_kwargs: shared)
    monkeypatch.setattr(main_module, "RedisSessionAdapter", lambda **_kwargs: shared)
    monkeypatch.setattr(main_module, "MySQLReadOnlyAdapter", lambda **_kwargs: shared)
    monkeypatch.setattr(main_module, "SessionStore", lambda *_args, **_kwargs: shared)
    monkeypatch.setattr(main_module, "HybridRetriever", lambda *_args, **_kwargs: shared)
    monkeypatch.setattr(main_module, "ReadOnlySQLGuard", lambda **_kwargs: shared)
    monkeypatch.setattr(main_module, "SafetyEngine", lambda: shared)
    monkeypatch.setattr(main_module, "OpenAIAnswerer", lambda: shared)
    monkeypatch.setattr(main_module, "MetricsStore", lambda **_kwargs: shared)
    monkeypatch.setattr(main_module, "MedGuideWorkflow", lambda *_args: shared)

    services = AppServices()

    assert services.api_tokens == ("token-a", "token-b", "proxy-token", "legacy-token")
    assert services.production_missing == ()

    monkeypatch.delenv("MEDGUIDE_API_TOKEN")
    plural_only = AppServices()
    assert "MEDGUIDE_API_TOKEN (UI proxy)" in plural_only.production_missing

    monkeypatch.delenv("MEDGUIDE_API_TOKENS")
    monkeypatch.delenv("MEDGUIDE_AUTH_TOKEN")
    monkeypatch.setenv("MEDGUIDE_API_TOKEN", "proxy-token")
    singular_only = AppServices()
    assert singular_only.api_tokens == ("proxy-token",)
    assert singular_only.production_missing == ()


@pytest.mark.parametrize(
    ("proxy_token", "api_tokens", "require_proxy", "expected_missing"),
    [
        ("proxy-token", ("proxy-token",), True, False),
        ("proxy-token", ("token-a", "proxy-token"), True, False),
        ("", ("token-a", "token-b"), True, True),
        ("proxy-token", ("token-a", "token-b"), True, True),
        ("", ("token-a", "token-b"), False, False),
    ],
)
def test_compose_proxy_token_contract_fails_readiness_for_invalid_topology(
    proxy_token: str,
    api_tokens: tuple[str, ...],
    require_proxy: bool,
    expected_missing: bool,
) -> None:
    class Dependency:
        available = True
        ready = True

        def ensure_ready(self, **_kwargs) -> bool:
            return True

    current = AppServices.__new__(AppServices)
    current.production = True
    current.proxy_api_token = proxy_token
    current.require_proxy_api_token = require_proxy
    current.api_tokens = api_tokens
    current.answerer = Dependency()
    current.embedding_provider = Dependency()
    current.milvus = Dependency()
    current.mysql = Dependency()
    current.redis = Dependency()
    current.retriever = SimpleNamespace(keyword_ready=True)

    missing = current.production_missing

    assert ("MEDGUIDE_API_TOKEN (UI proxy)" in missing) is expected_missing


def test_public_health_never_probes_dependencies_but_readiness_does(monkeypatch) -> None:
    probes = [0]

    class Services:
        production = True
        mode = "production"
        api_tokens = ()
        auth = object()

        @property
        def production_missing(self) -> tuple[str, ...]:
            probes[0] += 1
            return ("OpenAI",)

    monkeypatch.setattr(main_module, "get_services", lambda: Services())
    client = TestClient(main_module.app)
    try:
        health = client.get("/api/health")
        readiness = client.get("/api/ready")
    finally:
        client.close()

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["live"] is True
    assert probes == [1]
    assert readiness.status_code == 503


def test_readiness_cache_coalesces_concurrent_dependency_probes() -> None:
    class Services:
        production = True
        _readiness_lock = threading.Lock()
        _readiness_cache_until = 0.0
        _readiness_cache_value: tuple[str, ...] = ()
        _readiness_cache_seconds = 5.0
        calls = 0

        @property
        def production_missing(self) -> tuple[str, ...]:
            self.calls += 1
            time.sleep(0.03)
            return ("OpenAI",)

        readiness_missing = AppServices.readiness_missing

    current = Services()
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(lambda _index: current.readiness_missing(), range(40)))

    assert results == [("OpenAI",)] * 40
    assert current.calls == 1
    assert current.readiness_missing() == ("OpenAI",)
    assert current.calls == 1

    current._readiness_cache_until = 0.0
    assert current.readiness_missing() == ("OpenAI",)
    assert current.calls == 2


def test_answerer_is_not_ready_until_model_probe_succeeds(monkeypatch) -> None:
    class Completions:
        calls = 0

        def create(self, **_kwargs) -> None:
            self.calls += 1
            raise PermissionError("invalid credential")

    class Client:
        def __init__(self) -> None:
            self.completions = Completions()
            self.chat = SimpleNamespace(completions=self.completions)
            self.closed = False

        def close(self) -> None:
            self.closed = True

    client = Client()
    monkeypatch.setenv("MEDGUIDE_MODE", "production")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setattr(OpenAIAnswerer, "_create_resources", lambda _self: (client, object()))

    answerer = OpenAIAnswerer(api_key="invalid-test-key")

    assert answerer.available is False
    assert answerer.ensure_ready() is False
    assert client.completions.calls == 1
    assert client.closed is True
    assert answerer.last_error == "PermissionError"


def test_embedding_provider_is_not_ready_until_fixed_probe_succeeds(monkeypatch) -> None:
    class Embeddings:
        calls: list[dict] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            raise PermissionError("invalid credential")

    class Client:
        def __init__(self) -> None:
            self.embeddings = Embeddings()
            self.closed = False

        def close(self) -> None:
            self.closed = True

    client = Client()
    monkeypatch.setattr(OpenAIEmbeddingProvider, "_create_client", lambda _self: client)
    provider = OpenAIEmbeddingProvider(
        api_key="invalid-test-key",
        model="text-embedding-3-small",
        version="1",
        dimension=8,
    )

    assert provider.available is False
    assert provider.ensure_ready() is False
    assert len(client.embeddings.calls) == 1
    assert client.embeddings.calls[0]["input"] == ["MedGuide readiness probe"]
    assert client.closed is True
    assert provider.error == "PermissionError"


def test_injected_embedding_client_also_requires_a_successful_probe() -> None:
    class Embeddings:
        calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            raise PermissionError("invalid injected client")

    embeddings = Embeddings()
    provider = OpenAIEmbeddingProvider(
        api_key=None,
        model="text-embedding-3-small",
        version="1",
        dimension=8,
        client=SimpleNamespace(embeddings=embeddings),
    )

    assert provider.available is False
    assert provider.ensure_ready() is False
    assert embeddings.calls == 1
    assert provider.error == "PermissionError"
