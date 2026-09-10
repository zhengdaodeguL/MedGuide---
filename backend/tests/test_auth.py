from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.auth as auth_module
import app.main as main_module
from app.auth import (
    AuthStore,
    InvalidCredentialsError,
    MySQLAuthStore,
    UsernameUnavailableError,
)
from app.main import app
from app.store import SessionStore


def _runtime(tmp_path, *, clock=None, cookie_secure=True):
    kwargs = {"clock": clock} if clock is not None else {}
    auth = AuthStore(
        tmp_path / "runtime" / "auth.sqlite3",
        session_ttl_seconds=300,
        cookie_secure=cookie_secure,
        password_iterations=100_000,
        **kwargs,
    )
    return SimpleNamespace(
        production=True,
        mode="production",
        api_tokens=("service-client-token",),
        production_missing=(),
        auth=auth,
        store=SessionStore(mode="production"),
    )


def _client() -> TestClient:
    return TestClient(app, base_url="https://testserver")


@pytest.fixture(params=["sqlite", "mysql-adapter-on-sqlite"])
def atomic_store(request, tmp_path):
    database_path = tmp_path / "atomic-auth.sqlite3"
    kwargs = {
        "session_ttl_seconds": 300,
        "password_iterations": 100_000,
        "clock": lambda: 1_000.0,
    }
    if request.param == "sqlite":
        store = AuthStore(database_path, **kwargs)
        user_table, session_table = "users", "auth_sessions"
        integrity_error = sqlite3.IntegrityError
    else:
        from sqlalchemy import event
        from sqlalchemy.exc import IntegrityError

        class SQLiteSchemaMySQLAuthStore(MySQLAuthStore):
            def _initialize(self):
                # Exercise the adapter's real SQLAlchemy transactions with compatible DDL.
                event.listen(
                    self.engine,
                    "connect",
                    lambda connection, _record: connection.execute("PRAGMA foreign_keys = ON"),
                )
                with self.engine.begin() as connection:
                    connection.exec_driver_sql(
                        "CREATE TABLE medguide_users (username TEXT PRIMARY KEY, "
                        "password_salt BLOB NOT NULL, password_hash BLOB NOT NULL, "
                        "password_iterations INTEGER NOT NULL, created_at INTEGER NOT NULL)"
                    )
                    connection.exec_driver_sql(
                        "CREATE TABLE medguide_auth_sessions (token_hash BLOB PRIMARY KEY, "
                        "username TEXT NOT NULL REFERENCES medguide_users(username), "
                        "created_at INTEGER NOT NULL, expires_at INTEGER NOT NULL)"
                    )

        store = SQLiteSchemaMySQLAuthStore(f"sqlite:///{database_path.as_posix()}", **kwargs)
        user_table, session_table = "medguide_users", "medguide_auth_sessions"
        integrity_error = IntegrityError
    yield SimpleNamespace(
        store=store,
        database_path=database_path,
        user_table=user_table,
        session_table=session_table,
        integrity_error=integrity_error,
    )
    if isinstance(store, MySQLAuthStore):
        store.close()


def test_auth_store_uses_username_primary_key_and_never_stores_secrets(tmp_path) -> None:
    database_path = tmp_path / "auth.sqlite3"
    store = AuthStore(
        database_path,
        session_ttl_seconds=300,
        password_iterations=100_000,
    )

    username = store.register("medical_user", "Correct-Horse-42")
    token, _expires_at = store.create_session(username)

    with sqlite3.connect(database_path) as connection:
        columns = connection.execute("PRAGMA table_info(users)").fetchall()
        user = connection.execute(
            "SELECT username, password_salt, password_hash, password_iterations FROM users"
        ).fetchone()
        session = connection.execute("SELECT token_hash, username FROM auth_sessions").fetchone()

    username_column = next(column for column in columns if column[1] == "username")
    assert username_column[5] == 1
    assert user[0] == session[1] == "medical_user"
    assert len(user[1]) == 16
    assert len(user[2]) == 32
    assert user[3] == 100_000
    assert len(session[0]) == 32
    raw_database = database_path.read_bytes()
    assert b"Correct-Horse-42" not in raw_database
    assert token.encode("utf-8") not in raw_database


def test_registration_normalizes_username_and_rejects_duplicates(tmp_path) -> None:
    store = AuthStore(
        tmp_path / "auth.sqlite3",
        session_ttl_seconds=300,
        password_iterations=100_000,
    )

    assert store.register("  clinic_user  ", "Correct-Horse-42") == "clinic_user"
    with pytest.raises(UsernameUnavailableError):
        store.register("clinic_user", "Another-Correct-42")
    with pytest.raises(InvalidCredentialsError):
        store.verify_credentials("clinic_user", "Wrong-Password-42")
    assert store.verify_credentials("clinic_user", "Correct-Horse-42") == "clinic_user"


def test_atomic_registration_rolls_back_user_when_session_insert_fails(atomic_store, monkeypatch) -> None:
    store = atomic_store.store
    monkeypatch.setattr(auth_module.secrets, "token_urlsafe", lambda _size: "duplicate-token")
    existing = store.register("existing_user", "Correct-Horse-42")
    store.create_session(existing)
    monkeypatch.setattr(auth_module.secrets, "token_urlsafe", lambda _size: "expired-token")
    expired = store.register("expired_user", "Correct-Horse-42")
    store.create_session(expired)
    with sqlite3.connect(atomic_store.database_path) as connection:
        connection.execute(
            f"UPDATE {atomic_store.session_table} SET expires_at = 999 WHERE username = ?",
            (expired,),
        )
    monkeypatch.setattr(auth_module.secrets, "token_urlsafe", lambda _size: "duplicate-token")

    with pytest.raises(RuntimeError, match="unique authentication token"):
        store.register_and_create_session("rolled_back_user", "Correct-Horse-42")

    with sqlite3.connect(atomic_store.database_path) as connection:
        users = connection.execute(f"SELECT username FROM {atomic_store.user_table} ORDER BY username").fetchall()
        session_count = connection.execute(f"SELECT COUNT(*) FROM {atomic_store.session_table}").fetchone()[0]

    assert users == [("existing_user",), ("expired_user",)]
    assert session_count == 2
    monkeypatch.setattr(auth_module.secrets, "token_urlsafe", lambda _size: "retry-token")
    username, token, _ = store.register_and_create_session("rolled_back_user", "Correct-Horse-42")
    assert store.authenticate_session(token) == username == "rolled_back_user"
    with sqlite3.connect(atomic_store.database_path) as connection:
        assert connection.execute(f"SELECT COUNT(*) FROM {atomic_store.session_table}").fetchone()[0] == 2


def test_atomic_registration_returns_user_and_session(atomic_store) -> None:
    store = atomic_store.store
    username, token, expires_at = store.register_and_create_session(
        "  atomic_user  ",
        "Correct-Horse-42",
    )

    assert username == "atomic_user"
    assert expires_at == 1_300
    assert store.authenticate_session(token) == username


def test_atomic_duplicate_registration_preserves_account_and_session(atomic_store) -> None:
    store = atomic_store.store
    username, token, _ = store.register_and_create_session("  existing_user  ", "Correct-Horse-42")

    with pytest.raises(UsernameUnavailableError):
        store.register_and_create_session("existing_user", "Another-Correct-42", previous_token=token)

    assert store.authenticate_session(token) == username
    assert store.verify_credentials(username, "Correct-Horse-42") == username
    with sqlite3.connect(atomic_store.database_path) as connection:
        assert connection.execute(f"SELECT COUNT(*) FROM {atomic_store.user_table}").fetchone()[0] == 1
        assert connection.execute(f"SELECT COUNT(*) FROM {atomic_store.session_table}").fetchone()[0] == 1


def test_atomic_registration_retries_before_rotating_a_colliding_token(atomic_store, monkeypatch) -> None:
    store = atomic_store.store
    _, previous_token, _ = store.register_and_create_session("existing_user", "Correct-Horse-42")
    candidates = iter([previous_token, "new-authentication-token"])
    monkeypatch.setattr(auth_module.secrets, "token_urlsafe", lambda _size: next(candidates))

    username, token, _ = store.register_and_create_session(
        "new_user", "Correct-Horse-42", previous_token=previous_token
    )

    assert token == "new-authentication-token"
    assert store.authenticate_session(token) == username == "new_user"
    assert store.authenticate_session(previous_token) is None


@pytest.mark.parametrize("operation", ["user", "cleanup", "session", "previous"])
def test_atomic_registration_storage_failure_rolls_back_and_allows_retry(
    atomic_store, monkeypatch, operation
) -> None:
    store = atomic_store.store
    existing_user, previous_token, _ = store.register_and_create_session("existing_user", "Correct-Horse-42")
    table = atomic_store.user_table if operation == "user" else atomic_store.session_table
    action = "DELETE" if operation in {"cleanup", "previous"} else "INSERT"
    with sqlite3.connect(atomic_store.database_path) as connection:
        if operation != "previous":
            connection.execute(f"UPDATE {atomic_store.session_table} SET expires_at = 999")
        connection.execute(
            f"CREATE TRIGGER reject_registration BEFORE {action} ON {table} "
            "BEGIN SELECT RAISE(ABORT, 'forced storage failure'); END"
        )
    generated_tokens = []

    def next_token(_size):
        generated_tokens.append(f"retry-token-{len(generated_tokens)}")
        return generated_tokens[-1]

    monkeypatch.setattr(auth_module.secrets, "token_urlsafe", next_token)
    with pytest.raises(atomic_store.integrity_error, match="forced storage failure"):
        store.register_and_create_session("retry_user", "Correct-Horse-42", previous_token=previous_token)

    assert len(generated_tokens) == 1
    with sqlite3.connect(atomic_store.database_path) as connection:
        assert connection.execute(f"SELECT username FROM {atomic_store.user_table}").fetchall() == [(existing_user,)]
        assert connection.execute(f"SELECT COUNT(*) FROM {atomic_store.session_table}").fetchone()[0] == 1
        connection.execute("DROP TRIGGER reject_registration")

    username, token, _ = store.register_and_create_session(
        "retry_user", "Correct-Horse-42", previous_token=previous_token
    )
    assert store.authenticate_session(token) == username == "retry_user"
    assert store.authenticate_session(previous_token) is None


@pytest.mark.parametrize("atomic_store", ["mysql-adapter-on-sqlite"], indirect=True)
@pytest.mark.parametrize(
    ("statement_prefix", "error_code", "expected_error", "attempts"),
    [
        ("INSERT INTO medguide_users", 1062, UsernameUnavailableError, 1),
        ("INSERT INTO medguide_users", 1048, "integrity", 1),
        ("DELETE FROM medguide_auth_sessions", 1062, "integrity", 1),
        ("INSERT INTO medguide_auth_sessions", 1062, RuntimeError, 3),
        ("INSERT INTO medguide_auth_sessions", 1452, "integrity", 1),
    ],
)
def test_mysql_driver_integrity_errors_are_classified_without_hiding_failures(
    atomic_store, statement_prefix, error_code, expected_error, attempts
) -> None:
    from pymysql.err import IntegrityError as DriverIntegrityError
    from sqlalchemy import event
    from sqlalchemy.exc import IntegrityError

    store = atomic_store.store
    failure_count = []

    def fail_statement(_connection, _cursor, statement, parameters, _context, _executemany):
        if statement.startswith(statement_prefix):
            failure_count.append(statement)
            raise IntegrityError(statement, parameters, DriverIntegrityError(error_code, "forced driver failure"))

    event.listen(store.engine, "before_cursor_execute", fail_statement)
    try:
        expected_type = IntegrityError if expected_error == "integrity" else expected_error
        with pytest.raises(expected_type):
            store.register_and_create_session("retry_user", "Correct-Horse-42")
    finally:
        event.remove(store.engine, "before_cursor_execute", fail_statement)

    assert len(failure_count) == attempts
    with sqlite3.connect(atomic_store.database_path) as connection:
        assert connection.execute(f"SELECT COUNT(*) FROM {atomic_store.user_table}").fetchone()[0] == 0
        assert connection.execute(f"SELECT COUNT(*) FROM {atomic_store.session_table}").fetchone()[0] == 0
    username, token, _ = store.register_and_create_session("retry_user", "Correct-Horse-42")
    assert store.authenticate_session(token) == username == "retry_user"


def test_registration_cookie_rotation_failure_returns_503_and_can_retry(atomic_store, monkeypatch) -> None:
    store = atomic_store.store
    current = SimpleNamespace(production=True, auth=store)
    monkeypatch.setattr(main_module, "get_services", lambda: current)
    client = _client()
    try:
        assert client.post(
            "/api/auth/register", json={"username": "first_user", "password": "Correct-Horse-42"}
        ).status_code == 201
        previous_token = client.cookies.get(store.cookie_name)
        with sqlite3.connect(atomic_store.database_path) as connection:
            connection.execute(
                f"CREATE TRIGGER reject_rotation BEFORE DELETE ON {atomic_store.session_table} "
                "BEGIN SELECT RAISE(ABORT, 'forced rotation failure'); END"
            )

        response = client.post(
            "/api/auth/register", json={"username": "second_user", "password": "Correct-Horse-42"}
        )
        assert response.status_code == 503
        assert "set-cookie" not in response.headers
        assert client.cookies.get(store.cookie_name) == previous_token
        assert store.authenticate_session(previous_token) == "first_user"
        with sqlite3.connect(atomic_store.database_path) as connection:
            assert connection.execute(f"SELECT username FROM {atomic_store.user_table}").fetchall() == [("first_user",)]
            connection.execute("DROP TRIGGER reject_rotation")

        retried = client.post(
            "/api/auth/register", json={"username": "second_user", "password": "Correct-Horse-42"}
        )
        assert retried.status_code == 201
        assert client.get("/api/auth/me").json() == {"username": "second_user"}
        assert store.authenticate_session(previous_token) is None
    finally:
        client.close()


def test_expired_opaque_session_is_rejected_and_removed(tmp_path) -> None:
    now = [1_000.0]
    store = AuthStore(
        tmp_path / "auth.sqlite3",
        session_ttl_seconds=300,
        password_iterations=100_000,
        clock=lambda: now[0],
    )
    username = store.register("expiry_user", "Correct-Horse-42")
    token, expires_at = store.create_session(username)

    assert expires_at == 1_300
    assert store.authenticate_session(token) == username
    now[0] = 1_300.0
    assert store.authenticate_session(token) is None
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM auth_sessions").fetchone()[0] == 0


def test_register_me_logout_and_login_cookie_flow(tmp_path, monkeypatch) -> None:
    current = _runtime(tmp_path)
    monkeypatch.setattr(main_module, "get_services", lambda: current)
    client = _client()
    try:
        registered = client.post(
            "/api/auth/register",
            json={"username": "patient_001", "password": "Correct-Horse-42"},
        )
        assert registered.status_code == 201
        assert registered.json() == {"username": "patient_001"}
        cookie = registered.headers["set-cookie"].lower()
        assert "httponly" in cookie
        assert "samesite=strict" in cookie
        assert "secure" in cookie
        assert "path=/" in cookie

        assert client.get("/api/auth/me").json() == {"username": "patient_001"}
        duplicate = client.post(
            "/api/auth/register",
            json={"username": "patient_001", "password": "Another-Correct-42"},
        )
        assert duplicate.status_code == 409

        logged_out = client.post("/api/auth/logout")
        assert logged_out.status_code == 200
        assert logged_out.json() == {"ok": True}
        assert client.get("/api/auth/me").status_code == 401

        invalid = client.post(
            "/api/auth/login",
            json={"username": "patient_001", "password": "Wrong-Password-42"},
        )
        assert invalid.status_code == 401
        assert invalid.json()["detail"] == "用户名或密码错误"
        logged_in = client.post(
            "/api/auth/login",
            json={"username": "patient_001", "password": "Correct-Horse-42"},
        )
        assert logged_in.status_code == 200
        assert client.get("/api/auth/me").json() == {"username": "patient_001"}
    finally:
        client.close()


def test_business_sessions_are_owned_by_authenticated_username(tmp_path, monkeypatch) -> None:
    current = _runtime(tmp_path, cookie_secure=False)
    monkeypatch.setattr(main_module, "get_services", lambda: current)
    alice = _client()
    bob = _client()
    anonymous = _client()
    try:
        assert anonymous.post("/api/sessions").status_code == 401
        assert alice.post(
            "/api/auth/register",
            json={"username": "patient_alice", "password": "Correct-Horse-42"},
        ).status_code == 201
        assert bob.post(
            "/api/auth/register",
            json={"username": "patient_bob", "password": "Correct-Horse-43"},
        ).status_code == 201

        created = alice.post("/api/sessions", headers={"Idempotency-Key": "alice-session-001"})
        assert created.status_code == 200
        session_id = created.json()["session_id"]
        assert current.store.get(session_id)["owner_id"] == "user:patient_alice"
        assert alice.get(f"/api/sessions/{session_id}").status_code == 200
        assert bob.get(f"/api/sessions/{session_id}").status_code == 404
    finally:
        alice.close()
        bob.close()
        anonymous.close()


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("post", "/api/sessions", None),
        ("get", "/api/sessions/missing", None),
        ("get", "/api/sessions/missing/requests/request-0001", None),
        ("post", "/api/chat", {"message": "我咳嗽"}),
        ("post", "/api/search", {"query": "布洛芬", "intent": "drug"}),
        ("post", "/api/query", {"text": "查询检查价格"}),
        (
            "post",
            "/api/feedback",
            {"session_id": "missing", "message_id": "missing:turn:1", "rating": "up"},
        ),
        ("get", "/api/metrics", None),
    ],
)
def test_every_business_api_rejects_anonymous_production_requests(
    tmp_path,
    monkeypatch,
    method: str,
    path: str,
    payload: dict | None,
) -> None:
    current = _runtime(tmp_path, cookie_secure=False)
    monkeypatch.setattr(main_module, "get_services", lambda: current)
    client = _client()
    try:
        response = client.request(method, path, json=payload)
        assert response.status_code == 401
        expected_detail = "需要有效的 API 凭据" if path == "/api/metrics" else "请先登录"
        assert response.json()["detail"] == expected_detail
    finally:
        client.close()


def test_explicit_service_token_remains_supported_without_browser_cookie(tmp_path, monkeypatch) -> None:
    current = _runtime(tmp_path, cookie_secure=False)
    monkeypatch.setattr(main_module, "get_services", lambda: current)
    client = _client()
    try:
        created = client.post(
            "/api/sessions",
            headers={"Authorization": "Bearer service-client-token"},
        )
        assert created.status_code == 200
        state = current.store.get(created.json()["session_id"])
        assert state["owner_id"].startswith("service:")
        assert "service-client-token" not in state["owner_id"]
    finally:
        client.close()


def test_global_metrics_require_service_credentials_in_production(tmp_path, monkeypatch) -> None:
    current = _runtime(tmp_path, cookie_secure=False)
    current.metrics = SimpleNamespace(snapshot=lambda: {"requests": 7})
    monkeypatch.setattr(main_module, "get_services", lambda: current)
    browser = _client()
    service = _client()
    try:
        assert browser.post(
            "/api/auth/register",
            json={"username": "metrics_user", "password": "Correct-Horse-42"},
        ).status_code == 201
        assert browser.get("/api/metrics").status_code == 401

        allowed = service.get(
            "/api/metrics",
            headers={"Authorization": "Bearer service-client-token"},
        )
        assert allowed.status_code == 200
        assert allowed.json() == {"requests": 7}
    finally:
        browser.close()
        service.close()
