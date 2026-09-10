from __future__ import annotations

import threading
import time
import uuid
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from .models import WorkflowState


_REQUEST_RESPONSE_FIELDS = (
    "response",
    "response_id",
    "citations",
    "structured_result",
    "events",
    "confidence",
    "latency_ms",
)


def merge_request_result(current: WorkflowState, snapshot: WorkflowState) -> WorkflowState:
    """Return a historical answer without rolling back current session state."""
    result = deepcopy(current)
    for field in _REQUEST_RESPONSE_FIELDS:
        if field in snapshot:
            result[field] = deepcopy(snapshot[field])
        else:
            result.pop(field, None)
    return result


class SessionConflictError(RuntimeError):
    """Raised when a stale session snapshot tries to overwrite newer state."""


class SessionBackendError(RuntimeError):
    """Raised when the selected shared session backend cannot be reached."""


class SessionCancelledError(RuntimeError):
    """Raised when a streamed client disconnects before a turn is committed."""


class SessionAuthorizationError(RuntimeError):
    """Raised when a caller does not own the requested session."""


class SessionStore:
    """Session repository with an in-process test store and optional Redis backend.

    The in-process lock/CAS protects explicit test runs. A Redis adapter,
    when supplied and healthy, is used for persistence and performs its own
    compare-and-set operation so multiple workers cannot silently overwrite a
    newer turn.
    """

    def __init__(
        self,
        backend: Any | None = None,
        *,
        mode: str = "production",
        ttl_seconds: int = 86400,
        clock: Any = time.monotonic,
    ) -> None:
        self._sessions: dict[str, WorkflowState] = {}
        self._expires_at: dict[str, float] = {}
        self._lock = threading.RLock()
        self._session_locks: dict[str, threading.RLock] = {}
        self._session_lock_users: dict[str, int] = {}
        self._create_idempotency: dict[tuple[str, str], tuple[str, float]] = {}
        # A supplied backend is authoritative, even while unhealthy.  Keeping
        # it selected lets callers return an explicit 503 instead of silently
        # splitting a production session into worker-local memory.
        self.backend = backend
        self.mode = mode
        self.ttl_seconds = max(60, min(int(ttl_seconds), 365 * 24 * 60 * 60))
        self._clock = clock

    @property
    def backend_name(self) -> str:
        return "redis" if self.backend is not None else "in-memory-disabled"

    def _backend_ready(self) -> bool:
        if self.backend is None:
            return True
        ensure_ready = getattr(self.backend, "ensure_ready", None)
        if callable(ensure_ready):
            return bool(ensure_ready())
        return bool(getattr(self.backend, "ready", False))

    def _touch(self, session_id: str) -> None:
        self._expires_at[session_id] = float(self._clock()) + self.ttl_seconds

    def _purge_expired(self) -> None:
        with self._lock:
            now = float(self._clock())
            expired = [session_id for session_id, deadline in self._expires_at.items() if deadline <= now]
            for session_id in expired:
                self._sessions.pop(session_id, None)
                self._expires_at.pop(session_id, None)
                if self._session_lock_users.get(session_id, 0) == 0:
                    self._session_locks.pop(session_id, None)
                    self._session_lock_users.pop(session_id, None)
            expired_keys = [key for key, (_session_id, deadline) in self._create_idempotency.items() if deadline <= now]
            for key in expired_keys:
                self._create_idempotency.pop(key, None)

    @contextmanager
    def session_lock(self, session_id: str) -> Iterator[None]:
        """Serialize the complete read -> workflow -> save transaction."""
        with self._lock:
            self._purge_expired()
            lock = self._session_locks.setdefault(session_id, threading.RLock())
            self._session_lock_users[session_id] = self._session_lock_users.get(session_id, 0) + 1
        try:
            with lock:
                yield
        finally:
            with self._lock:
                remaining = self._session_lock_users.get(session_id, 1) - 1
                if remaining > 0:
                    self._session_lock_users[session_id] = remaining
                else:
                    self._session_lock_users.pop(session_id, None)
                    deadline = self._expires_at.get(session_id)
                    if session_id not in self._sessions or (
                        deadline is not None and deadline <= float(self._clock())
                    ):
                        self._session_locks.pop(session_id, None)

    def create(self, owner_id: str | None = None, idempotency_key: str | None = None) -> WorkflowState:
        owner = owner_id or ""
        if idempotency_key:
            key = (owner, idempotency_key)
            with self._lock:
                self._purge_expired()
                existing = self._create_idempotency.get(key)
                if existing and existing[1] > float(self._clock()):
                    state = self._sessions.get(existing[0])
                    if state is not None:
                        return deepcopy(state)
            # A Redis-backed store can share the creation key across workers.
            lookup = getattr(self.backend, "get_session_creation", None) if self.backend is not None else None
            if callable(lookup) and self._backend_ready():
                try:
                    existing_id = lookup(idempotency_key, owner)
                except Exception as exc:
                    raise SessionBackendError("共享会话后端暂时不可用，请稍后重试") from exc
                if existing_id:
                    existing_state = self.get(str(existing_id), owner_id=owner_id)
                    if existing_state is not None:
                        return existing_state
        session_id = f"mg_{uuid.uuid4().hex[:12]}"
        state: WorkflowState = {
            "session_id": session_id,
            "owner_id": owner,
            "profile": {"symptoms": [], "associated_symptoms": []},
            "turn_count": 0,
            "_version": 0,
            "mode": self.mode,
            "events": [],
            "citations": [],
            "risk_flags": [],
            "risk_assessed": False,
            "request_results": {},
            "request_fingerprints": {},
            "feedback_by_message": {},
        }
        if self.backend is not None:
            if not self._backend_ready():
                raise SessionBackendError("共享会话后端暂时不可用，请稍后重试")
            atomic_create = getattr(self.backend, "create_state", None)
            try:
                if idempotency_key and callable(atomic_create):
                    winner = atomic_create(state, idempotency_key, owner, self.ttl_seconds)
                    if str(winner) != session_id:
                        winner_state = self.get(str(winner), owner_id=owner_id)
                        if winner_state is None:
                            raise SessionBackendError("无法读取已创建的共享会话，请稍后重试")
                        state = winner_state
                        session_id = str(winner)
                elif not self.backend.save_state(state, expected_version=None):
                    raise SessionBackendError("无法创建共享会话，请稍后重试")
            except Exception as exc:
                if isinstance(exc, SessionBackendError):
                    raise
                raise SessionBackendError("共享会话后端暂时不可用，请稍后重试") from exc
            remember = getattr(self.backend, "remember_session_creation", None)
            if idempotency_key and not callable(atomic_create) and callable(remember):
                try:
                    winner = remember(idempotency_key, owner, session_id, self.ttl_seconds)
                except Exception as exc:
                    raise SessionBackendError("共享会话后端暂时不可用，请稍后重试") from exc
                if winner and str(winner) != session_id:
                    winner_state = self.get(str(winner), owner_id=owner_id)
                    if winner_state is not None:
                        state = winner_state
                        session_id = str(winner)
        with self._lock:
            self._purge_expired()
            self._sessions[session_id] = deepcopy(state)
            self._touch(session_id)
            if idempotency_key:
                self._create_idempotency[(owner, idempotency_key)] = (
                    session_id,
                    float(self._clock()) + self.ttl_seconds,
                )
        return deepcopy(state)

    def claim_request(
        self,
        session_id: str,
        request_id: str | None,
        fingerprint: str | None,
    ) -> tuple[str | None, WorkflowState | None]:
        """Claim one idempotent request when a shared Redis backend is active."""
        if not request_id or not fingerprint or self.backend is None:
            return None, None
        claim = getattr(self.backend, "claim_request", None)
        if not callable(claim):
            return None, None
        token = uuid.uuid4().hex
        try:
            status = str(claim(session_id, request_id, fingerprint, token))
        except Exception as exc:
            raise SessionBackendError("共享会话后端暂时不可用，请稍后重试") from exc
        if status == "acquired":
            return token, None
        if status == "completed":
            state = self.get(session_id)
            if state is None:
                raise SessionConflictError("会话已过期，请重新发起问诊")
            result = state.get("request_results", {}).get(request_id)
            if isinstance(result, dict):
                return None, merge_request_result(state, result)
            raise SessionConflictError("请求结果正在提交，请稍后重试")
        if status == "conflict":
            raise SessionConflictError("请求标识已用于其他消息，不能重复使用")
        if status == "pending":
            raise SessionConflictError("同一请求正在处理中，请稍后重试或查询请求结果")
        if status == "missing":
            raise SessionConflictError("会话已过期，请重新发起问诊")
        raise SessionBackendError("共享会话后端返回了无效的请求状态")

    def release_request_claim(
        self,
        session_id: str,
        request_id: str | None,
        fingerprint: str | None,
        token: str | None,
    ) -> None:
        if not request_id or not fingerprint or not token or self.backend is None:
            return
        release = getattr(self.backend, "release_request_claim", None)
        if callable(release):
            try:
                release(session_id, request_id, fingerprint, token)
            except Exception:
                pass

    def get(self, session_id: str, owner_id: str | None = None) -> WorkflowState | None:
        if self.backend is not None:
            if not self._backend_ready():
                raise SessionBackendError("共享会话后端暂时不可用，请稍后重试")
            try:
                remote = self.backend.get_state(session_id)
            except Exception as exc:
                raise SessionBackendError("共享会话后端暂时不可用，请稍后重试") from exc
            if remote is not None:
                if owner_id is not None and str(remote.get("owner_id", "")) != str(owner_id):
                    raise SessionAuthorizationError("会话不属于当前身份")
                with self._lock:
                    self._purge_expired()
                    self._sessions[session_id] = deepcopy(remote)
                    self._touch(session_id)
                return deepcopy(remote)
            # A selected shared backend is authoritative.  Falling through to
            # a worker-local snapshot would reintroduce split-brain sessions.
            with self._lock:
                self._sessions.pop(session_id, None)
                self._expires_at.pop(session_id, None)
            return None
        with self._lock:
            self._purge_expired()
            state = self._sessions.get(session_id)
            if state:
                if owner_id is not None and str(state.get("owner_id", "")) != str(owner_id):
                    raise SessionAuthorizationError("会话不属于当前身份")
                self._touch(session_id)
                return deepcopy(state)
            return None

    def save(
        self,
        state: WorkflowState,
        expected_version: int | None = None,
        cancel_event: threading.Event | None = None,
        request_claim: tuple[str, str, str] | None = None,
    ) -> WorkflowState:
        state = deepcopy(state)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        session_id = state["session_id"]
        # A state returned by get/create carries its version.  Callers may pass
        # an explicit expected version when constructing a state independently.
        if expected_version is None and "_version" in state:
            expected_version = int(state["_version"])
        with self.session_lock(session_id):
            if cancel_event is not None and cancel_event.is_set():
                raise SessionCancelledError()
            with self._lock:
                self._purge_expired()
                current = self._sessions.get(session_id)
                if current is None and expected_version is not None:
                    raise SessionConflictError("会话已过期，请重新发起问诊")
                current_version = int(current.get("_version", 0)) if current else 0
                if current is not None and expected_version is not None and current_version != expected_version:
                    raise SessionConflictError(
                        f"session {session_id} changed from version {expected_version} to {current_version}"
                    )
                next_version = current_version + 1
                state["_version"] = next_version
            if self.backend is not None:
                if cancel_event is not None and cancel_event.is_set():
                    raise SessionCancelledError()
                if not self._backend_ready():
                    raise SessionBackendError("共享会话后端暂时不可用，请稍后重试")
                try:
                    if request_claim is None:
                        accepted = self.backend.save_state(state, expected_version=expected_version)
                    else:
                        accepted = self.backend.save_state(
                            state,
                            expected_version=expected_version,
                            request_claim=request_claim,
                        )
                except Exception as exc:
                    raise SessionBackendError("共享会话后端暂时不可用，请稍后重试") from exc
                if not accepted:
                    raise SessionConflictError("会话已被其他请求更新，请重试")
            with self._lock:
                if self.backend is None and cancel_event is not None and cancel_event.is_set():
                    raise SessionCancelledError()
                self._sessions[session_id] = deepcopy(state)
                self._touch(session_id)
        return deepcopy(state)

    def request_result(self, session_id: str, request_id: str, owner_id: str | None = None) -> WorkflowState | None:
        state = self.get(session_id, owner_id=owner_id)
        if state is None:
            return None
        result = state.get("request_results", {}).get(request_id)
        return merge_request_result(state, result) if isinstance(result, dict) else None

    def public(self, state: WorkflowState) -> dict:
        return {
            "session_id": state.get("session_id"),
            "turn_count": state.get("turn_count", 0),
            "profile": state.get("profile", {}),
            "risk_level": state.get("risk_level") if state.get("risk_assessed") else None,
            "risk_flags": state.get("risk_flags", []),
            "risk_assessed": bool(state.get("risk_assessed", False)),
            "intent": state.get("intent", "unknown"),
            "next_question": state.get("next_question"),
            "summary": state.get("summary", ""),
            "mode": state.get("mode", "production"),
            "workflow_engine": state.get("workflow_engine", "deterministic"),
            "environment": self.mode if self.mode in {"offline", "test"} else "production",
            "network_enabled": self.mode not in {"offline", "test"},
            "updated_at": state.get("updated_at"),
        }
