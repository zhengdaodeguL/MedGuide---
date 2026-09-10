from __future__ import annotations

import json
import os
import asyncio
import ipaddress
import math
import threading
import hashlib
import hmac
import logging
import time
from copy import deepcopy
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, SecretStr

from .auth import (
    AuthStore,
    InvalidCredentialsError,
    InvalidPasswordError,
    InvalidUsernameError,
    UsernameUnavailableError,
)
from .embeddings import embedding_provider_from_env, keyword_embedding_provider_from_env
from .ingestion import MedicalDocumentCleaner
from .infra import AdapterStatus, MySQLReadOnlyAdapter, RedisSessionAdapter
from .llm import OpenAIAnswerer
from .metrics import MetricsStore
from .models import WorkflowState
from .retrieval import HybridRetriever, MilvusAdapter
from .safety import SafetyEngine
from .sql_guard import ReadOnlySQLGuard
from .store import (
    SessionAuthorizationError,
    SessionBackendError,
    SessionCancelledError,
    SessionConflictError,
    SessionStore,
    merge_request_result,
)
from .workflow import MedGuideWorkflow, WorkflowCancelled, WorkflowExecutionError
from .version import __version__


MAX_REQUESTS_PER_SESSION = 256
_PRODUCTION_KNOWLEDGE_BACKENDS = frozenset({"milvus", "bm25", "disabled"})
_CITATION_CATEGORIES = frozenset({"disease", "drug", "exam", "department", "faq"})
_CITATION_RETRIEVAL_METHODS = frozenset({"bm25", "vector", "hybrid"})
logger = logging.getLogger(__name__)
_citation_cleaner = MedicalDocumentCleaner()


def _knowledge_backend(*, production: bool) -> str:
    configured = (os.getenv("MEDGUIDE_KNOWLEDGE_BACKEND") or "").strip().lower()
    if not configured:
        return "milvus" if production else "memory"
    if configured not in _PRODUCTION_KNOWLEDGE_BACKENDS and configured != "memory":
        raise ValueError("MEDGUIDE_KNOWLEDGE_BACKEND must be milvus, bm25, disabled, or memory")
    if production and configured == "memory":
        raise ValueError("memory knowledge backend is only available outside production")
    return configured


def _path_from_environment(name: str, default: Path) -> Path:
    raw = (os.getenv(name) or "").strip()
    return Path(raw).expanduser() if raw else default


def _environment_bool(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _secure_transport_configuration() -> bool:
    return _environment_bool("MEDGUIDE_COOKIE_SECURE", default=True) and _environment_bool(
        "MEDGUIDE_REQUIRE_HTTPS", default=True
    )


def _trusted_proxy_networks() -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    api_value = os.getenv("MEDGUIDE_API_TRUSTED_PROXY_CIDR")
    raw = api_value
    if raw is None:
        # Keep direct API deployments compatible with the original setting;
        # Compose supplies the API-specific network explicitly.
        raw = os.getenv("MEDGUIDE_TRUSTED_PROXY_CIDR")
    raw = (raw or "").strip()
    if not raw:
        return ()
    networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for value in raw.split(","):
        item = value.strip()
        if not item:
            continue
        try:
            networks.append(ipaddress.ip_network(item, strict=False))
        except ValueError as exc:
            variable = "MEDGUIDE_API_TRUSTED_PROXY_CIDR" if api_value is not None else "MEDGUIDE_TRUSTED_PROXY_CIDR"
            raise ValueError(f"{variable} must contain valid CIDR values") from exc
    return tuple(networks)


def _request_uses_secure_transport(
    request: Request,
    trusted_proxy_networks: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] | None = None,
) -> bool:
    if request.url.scheme.lower() == "https":
        return True
    forwarded_scheme = request.headers.get("x-forwarded-proto", "").split(",", 1)[0].strip().lower()
    if forwarded_scheme != "https":
        return False
    client_host = request.client.host if request.client is not None else ""
    try:
        client_ip = ipaddress.ip_address(client_host)
    except ValueError:
        return False
    networks = _trusted_proxy_networks() if trusted_proxy_networks is None else trusted_proxy_networks
    return any(client_ip in network for network in networks)


def _require_secure_transport(current: AppServices, request: Request) -> None:
    if current.production and _environment_bool("MEDGUIDE_REQUIRE_HTTPS", default=True):
        trusted_proxy_networks = getattr(current, "trusted_proxy_networks", None)
        if trusted_proxy_networks is None:
            trusted_proxy_networks = _trusted_proxy_networks()
        if not _request_uses_secure_transport(request, trusted_proxy_networks):
            raise HTTPException(status_code=426, detail="请通过 HTTPS 受保护入口访问")


def _readiness_cache_seconds() -> float:
    try:
        return max(0.5, min(float(os.getenv("MEDGUIDE_READINESS_CACHE_SECONDS", "5")), 60.0))
    except ValueError:
        return 5.0


class SessionCreateResponse(BaseModel):
    session_id: str
    state: dict[str, Any]


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1, max_length=2000)
    request_id: str | None = Field(
        default=None,
        min_length=8,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]+$",
    )


class ChatResponse(BaseModel):
    session_id: str
    response_id: str
    answer: str
    state: dict[str, Any]
    citations: list[dict[str, Any]]
    structured_result: dict[str, Any] | None = None
    events: list[dict[str, Any]]
    confidence: float
    latency_ms: float
    summary: str = ""


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    intent: str = "unknown"
    top_k: int = Field(default=4, ge=1, le=8)


class QueryRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)


class FeedbackRequest(BaseModel):
    session_id: str
    rating: str = Field(pattern="^(up|down)$")
    message_id: str = Field(min_length=1, max_length=128)
    comment: str | None = Field(default=None, max_length=500)


class AuthCredentials(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: SecretStr = Field(min_length=8, max_length=128)


class AuthUserResponse(BaseModel):
    username: str


def _configured_api_tokens() -> tuple[str, ...]:
    tokens: list[str] = []
    seen: set[str] = set()
    for variable in ("MEDGUIDE_API_TOKENS", "MEDGUIDE_API_TOKEN", "MEDGUIDE_AUTH_TOKEN"):
        for raw_token in (os.getenv(variable) or "").split(","):
            token = raw_token.strip()
            if token and token not in seen:
                seen.add(token)
                tokens.append(token)
    return tuple(tokens)


class AppServices:
    def __init__(self) -> None:
        self.mode = os.getenv("MEDGUIDE_MODE", "production").strip().lower() or "production"
        production = self.mode not in {"offline", "test"}
        self.production = production
        # Parse once at startup so malformed proxy configuration cannot wait
        # until the first business request to fail, and reuse the exact same
        # trust boundary for every HTTPS gate.
        self.trusted_proxy_networks = _trusted_proxy_networks()
        if production and not _secure_transport_configuration():
            raise ValueError(
                "production requires MEDGUIDE_COOKIE_SECURE=true and MEDGUIDE_REQUIRE_HTTPS=true"
            )
        self._readiness_lock = threading.Lock()
        self._readiness_cache_until = 0.0
        self._readiness_cache_value: tuple[str, ...] = ()
        self._readiness_cache_seconds = _readiness_cache_seconds()
        self.knowledge_backend = _knowledge_backend(production=production)
        milvus_uri = (os.getenv("MILVUS_URI") or "").strip()
        mysql_dsn = (os.getenv("MYSQL_DSN") or "").strip()
        self.vector_search_enabled = production and self.knowledge_backend == "milvus"
        self.structured_data_enabled = production and bool(mysql_dsn)
        self.proxy_api_token = (os.getenv("MEDGUIDE_API_TOKEN") or "").strip()
        self.require_proxy_api_token = production and (
            (os.getenv("MEDGUIDE_REQUIRE_PROXY_TOKEN") or "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.api_tokens = _configured_api_tokens()
        self.auth = AuthStore.from_environment(production=production)
        self.embedding_provider = (
            keyword_embedding_provider_from_env()
            if self.knowledge_backend == "bm25"
            else embedding_provider_from_env(production=self.vector_search_enabled)
        )
        self.milvus = MilvusAdapter(
            milvus_uri if self.vector_search_enabled else None,
            embedding_spec=self.embedding_provider.spec if self.vector_search_enabled else None,
        )
        self.redis = RedisSessionAdapter(enabled=production)
        self.mysql = MySQLReadOnlyAdapter(dsn=mysql_dsn or None, enabled=self.structured_data_enabled)
        try:
            session_ttl = int(os.getenv("MEDGUIDE_SESSION_TTL_SECONDS", "86400"))
        except ValueError:
            session_ttl = 86400
        self.store = SessionStore(
            self.redis if production else None,
            mode=self.mode,
            ttl_seconds=session_ttl,
        )
        self.retriever = HybridRetriever(
            milvus=self.milvus if self.vector_search_enabled else None,
            embedding_provider=self.embedding_provider,
            bm25_index_path=(
                _path_from_environment(
                    "BM25_INDEX_PATH",
                    Path(__file__).resolve().parents[2] / "data" / "runtime" / "bm25-index.json",
                )
                if self.knowledge_backend in {"milvus", "bm25"}
                else None
            ),
            local_vector_search=not production and self.knowledge_backend == "memory",
        )
        # Supplying the disabled adapter in production prevents the bundled
        # query database from being mistaken for live operational data.
        self.sql_guard = ReadOnlySQLGuard(backend=self.mysql if production else None)
        self.safety = SafetyEngine()
        self.answerer = OpenAIAnswerer()
        self.metrics = MetricsStore(backend=self.redis if production else None)
        self.workflow = MedGuideWorkflow(self.retriever, self.sql_guard, self.safety, self.answerer)

    def close(self) -> None:
        """Release optional external resources in reverse dependency order."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        for resource in (self.answerer, self.embedding_provider, self.milvus, self.mysql, self.redis, self.auth):
            close = getattr(resource, "close", None) or getattr(resource, "shutdown", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    # Shutdown must continue so one failed client cannot leak
                    # every resource created before it.
                    logger.warning(
                        "Failed to close %s: %s",
                        type(resource).__name__,
                        type(exc).__name__,
                    )

    @property
    def production_missing(self) -> tuple[str, ...]:
        if not self.production:
            return ()
        knowledge_backend = getattr(self, "knowledge_backend", None)
        if knowledge_backend is None:
            knowledge_backend = "milvus" if getattr(self, "vector_search_enabled", True) else "bm25"
        required_adapters = [self.answerer, self.redis]
        if knowledge_backend == "milvus":
            required_adapters.extend((self.embedding_provider, self.milvus))
        if getattr(self, "structured_data_enabled", True):
            required_adapters.append(self.mysql)
        for adapter in required_adapters:
            ensure_ready = getattr(adapter, "ensure_ready", None)
            if callable(ensure_ready):
                try:
                    ensure_ready(revalidate=adapter is self.milvus)
                except TypeError:
                    ensure_ready()
        missing: list[str] = []
        auth_store = getattr(self, "auth", None)
        if getattr(auth_store, "requires_shared_backend", False) and not getattr(auth_store, "shared", False):
            missing.append("Shared auth database")
        if not self.answerer.available:
            missing.append("OPENAI_API_KEY/OpenAI")
        if knowledge_backend == "milvus" and not self.embedding_provider.available:
            missing.append("Embedding provider")
        if knowledge_backend == "milvus" and not self.milvus.ready:
            missing.append("Milvus")
        if knowledge_backend == "disabled":
            missing.append("Knowledge backend")
        if not self.retriever.keyword_ready:
            missing.append("BM25 index")
        if getattr(self, "structured_data_enabled", True) and not self.mysql.ready:
            missing.append("MySQL")
        if not self.redis.ready:
            missing.append("Redis")
        proxy_api_token = getattr(self, "proxy_api_token", "")
        if getattr(self, "require_proxy_api_token", False) and (
            not proxy_api_token or proxy_api_token not in self.api_tokens
        ):
            missing.append("MEDGUIDE_API_TOKEN (UI proxy)")
        return tuple(missing)

    @property
    def knowledge_missing(self) -> tuple[str, ...]:
        """Report a silently unavailable keyword tier outside production.

        Production already refuses to start without a usable index.  Outside
        production the same wiring mistake used to surface as ``ready: true``
        with an empty corpus, which let the deterministic fallback answer
        health questions with zero citations.  Deployment smoke tests need
        that state to be visible instead of green.
        """
        if self.knowledge_backend not in {"milvus", "bm25"}:
            return ()
        if self.retriever.keyword_ready:
            return ()
        detail = getattr(self.retriever, "bm25_error", None) or "not loaded"
        return (f"BM25 index ({detail})",)

    def readiness_missing(self) -> tuple[str, ...]:
        """Coalesce concurrent dependency probes and reuse their short-lived result."""
        if not self.production:
            return self.knowledge_missing
        now = time.monotonic()
        if now < self._readiness_cache_until:
            return self._readiness_cache_value
        with self._readiness_lock:
            now = time.monotonic()
            if now < self._readiness_cache_until:
                return self._readiness_cache_value
            missing = self.production_missing
            self._readiness_cache_value = missing
            self._readiness_cache_until = time.monotonic() + self._readiness_cache_seconds
            return missing


services: AppServices | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global services
    services = AppServices()
    try:
        yield
    finally:
        if services is not None:
            services.close()
        services = None


_IMPORT_MODE = os.getenv("MEDGUIDE_MODE", "production").strip().lower() or "production"
_IMPORT_PRODUCTION = _IMPORT_MODE not in {"offline", "test"}
app = FastAPI(
    title="MedGuide API",
    version=__version__,
    lifespan=lifespan,
    # API documentation is useful in explicit test modes, but exposing an
    # interactive schema in production creates an unnecessary attack surface.
    docs_url=None if _IMPORT_PRODUCTION else "/docs",
    redoc_url=None if _IMPORT_PRODUCTION else "/redoc",
    openapi_url=None if _IMPORT_PRODUCTION else "/openapi.json",
)


def _cors_origins() -> list[str]:
    # The browser deployment is same-origin. Cross-origin access must be an
    # explicit operator decision rather than a development-oriented default.
    raw = os.getenv("MEDGUIDE_CORS_ORIGINS", "")
    origins = [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]
    # Credentials and wildcard origins are mutually exclusive.  Treat a
    # wildcard as an explicit configuration error rather than silently
    # downgrading browser security.
    if "*" in origins:
        return []
    return origins


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "X-API-Key"],
)


def get_services() -> AppServices:
    global services
    if services is None:
        services = AppServices()
    return services


def _principal_from_headers(
    current: AppServices,
    authorization: str | None = None,
    api_key: str | None = None,
) -> str | None:
    """Authenticate production API calls and return a non-secret principal id."""
    if not current.production:
        return None
    configured = current.api_tokens
    if not configured:
        raise HTTPException(status_code=503, detail="服务鉴权未配置，请设置 MEDGUIDE_API_TOKENS")
    supplied = ""
    if authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer":
            supplied = value.strip()
    if not supplied and api_key:
        supplied = api_key.strip()
    matched = next((token for token in configured if supplied and hmac.compare_digest(supplied, token)), None)
    if matched is None:
        raise HTTPException(status_code=401, detail="需要有效的 API 凭据")
    digest = hashlib.sha256(matched.encode("utf-8")).hexdigest()[:24]
    return f"service:{digest}"


def _username_from_cookie(current: AppServices, session_token: str | None) -> str | None:
    auth_store = getattr(current, "auth", None)
    if auth_store is None:
        return None
    return auth_store.authenticate_session(session_token)


def _session_token_from_request(current: AppServices, http_request: Request) -> str | None:
    auth_store = getattr(current, "auth", None)
    cookie_name = getattr(auth_store, "cookie_name", "medguide_session")
    return http_request.cookies.get(cookie_name)


def _require_access(
    current: AppServices,
    request: Request,
    authorization: str | None = None,
    api_key: str | None = None,
    session_token: str | None = None,
) -> str | None:
    _require_secure_transport(current, request)
    if authorization or api_key:
        principal = _principal_from_headers(current, authorization, api_key)
    else:
        username = _username_from_cookie(current, session_token)
        principal = f"user:{username}" if username else None
    if principal is None and getattr(current, "production", False):
        raise HTTPException(status_code=401, detail="请先登录")
    _require_runtime(current)
    return principal


def _require_service_access(
    current: AppServices,
    request: Request,
    authorization: str | None = None,
    api_key: str | None = None,
) -> str | None:
    _require_secure_transport(current, request)
    if not current.production:
        return None
    principal = _principal_from_headers(current, authorization, api_key)
    _require_runtime(current)
    return principal


def _message_fingerprint(message: str) -> str:
    normalized = " ".join(str(message).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _require_runtime(current: AppServices) -> None:
    readiness = getattr(current, "readiness_missing", None)
    missing_dependencies = readiness() if callable(readiness) else current.production_missing
    if missing_dependencies:
        missing = "、".join(missing_dependencies)
        raise HTTPException(status_code=503, detail=f"生产依赖未就绪：{missing}；请检查配置和连通性")


def _require_auth_backend(current: AppServices) -> None:
    auth_store = current.auth
    if (
        current.production
        and getattr(auth_store, "requires_shared_backend", False)
        and not getattr(auth_store, "shared", False)
    ):
        raise HTTPException(status_code=503, detail="共享认证服务未就绪，请稍后重试")


def _sanitize_citation_payload(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    sanitized: list[dict[str, Any]] = []
    for citation in value:
        if not isinstance(citation, dict):
            continue
        category = citation.get("category")
        retrieval = citation.get("retrieval")
        score = citation.get("score")
        if (
            not isinstance(category, str)
            or category.strip() not in _CITATION_CATEGORIES
            or not isinstance(retrieval, str)
            or retrieval.strip() not in _CITATION_RETRIEVAL_METHODS
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
        ):
            continue
        try:
            score_value = float(score)
        except (OverflowError, ValueError):
            continue
        if not math.isfinite(score_value) or not 0 <= score_value <= 1:
            continue
        try:
            citation_id = _citation_cleaner.sanitize_opaque_id(citation.get("id"), "citation id")
            title = _citation_cleaner.sanitize_metadata(citation.get("title"), "title")
            source = _citation_cleaner.sanitize_metadata(citation.get("source"), "source")
            updated_at = _citation_cleaner.sanitize_metadata(citation.get("updated_at"), "updated_at")
            snippet = _citation_cleaner.clean(citation.get("snippet"))
            if not snippet:
                raise ValueError("citation snippet is empty after cleaning")
            source_url = _citation_cleaner.sanitize_source_url(citation.get("source_url", ""))
        except ValueError:
            continue
        sanitized.append(
            {
                "id": citation_id,
                "title": title,
                "category": category.strip(),
                "source": source,
                "updated_at": updated_at,
                "snippet": snippet,
                "score": score_value,
                "retrieval": retrieval.strip(),
                "source_url": source_url,
            }
        )
    return sanitized


def public_state(state: WorkflowState) -> dict[str, Any]:
    citations = _sanitize_citation_payload(state.get("citations", []))
    safe_state = deepcopy(state)
    safe_state["citations"] = citations
    public = get_services().store.public(safe_state)
    public["citations"] = citations
    return public


def _response_from_state(saved: WorkflowState) -> ChatResponse:
    state = public_state(saved)
    return ChatResponse(
        session_id=saved["session_id"],
        response_id=saved.get("response_id", ""),
        answer=saved.get("response", ""),
        state=state,
        citations=state["citations"],
        structured_result=saved.get("structured_result"),
        events=saved.get("events", []),
        confidence=saved.get("confidence", 0.0),
        latency_ms=saved.get("latency_ms", 0.0),
        summary=saved.get("summary", ""),
    )


def _state_for_request(
    current: AppServices,
    request: ChatRequest,
    owner_id: str | None = None,
) -> WorkflowState:
    if request.session_id:
        state = current.store.get(request.session_id, owner_id=owner_id)
        if state is None:
            raise HTTPException(status_code=404, detail="会话不存在或已过期")
    else:
        state = current.store.create(owner_id=owner_id)
    state["user_message"] = request.message
    if request.request_id:
        state["request_id"] = request.request_id
        state["_request_fingerprint"] = _message_fingerprint(request.message)
    return state


def _stored_request_result(state: WorkflowState, request_id: str | None) -> WorkflowState | None:
    if not request_id:
        return None
    stored = state.get("request_results", {}).get(request_id)
    return merge_request_result(state, stored) if isinstance(stored, dict) else None


def _check_request_fingerprint(state: WorkflowState, request_id: str | None, message: str) -> None:
    if not request_id:
        return
    fingerprints = state.get("request_fingerprints", {})
    previous = fingerprints.get(request_id) if isinstance(fingerprints, dict) else None
    if previous:
        if previous != _message_fingerprint(message):
            raise SessionConflictError("请求标识已用于其他消息，不能重复使用")
        if _stored_request_result(state, request_id) is None:
            raise SessionConflictError("请求结果已超过回放窗口，不能重新执行")
        return
    if isinstance(fingerprints, dict) and len(fingerprints) >= MAX_REQUESTS_PER_SESSION:
        raise SessionConflictError("当前会话已达到请求上限，请新建整理后继续")


def _prepare_persisted_result(
    result: WorkflowState,
    request_id: str | None,
    fingerprint: str | None = None,
) -> WorkflowState:
    # The workflow no longer needs raw/re-written input after it has produced a
    # response and structured summary.  Avoid retaining those fields in the
    # session payload by default.
    for key in ("user_message", "normalized_message", "rewritten_query"):
        result.pop(key, None)
    if not request_id:
        result.pop("request_id", None)
        result.pop("_request_fingerprint", None)
        return result
    result["request_id"] = request_id
    if fingerprint:
        fingerprints = dict(result.get("request_fingerprints", {}))
        fingerprints[request_id] = fingerprint
        # Fingerprints are compact tombstones retained for the session TTL.
        # Full response snapshots remain bounded below, while an old request
        # can never mutate the medical record a second time.
        result["request_fingerprints"] = fingerprints
    snapshot = deepcopy(result)
    snapshot.pop("request_results", None)
    snapshot.pop("request_fingerprints", None)
    snapshot.pop("_request_fingerprint", None)
    request_results = dict(result.get("request_results", {}))
    request_results[request_id] = snapshot
    # Limit both retention and payload growth. Dict insertion order keeps the
    # newest client turns at the end on supported Python versions.
    result["request_results"] = dict(list(request_results.items())[-20:])
    return result


def _run_chat_transaction(current: AppServices, state: WorkflowState) -> WorkflowState:
    session_id = state["session_id"]
    request_id = state.get("request_id")
    fingerprint = state.get("_request_fingerprint")
    claim_token: str | None = None
    with current.store.session_lock(session_id):
        try:
            # Re-read after acquiring the lock.  A request may have been waiting
            # while another turn committed a newer profile/version.
            latest = current.store.get(session_id)
            if latest is None:
                raise SessionConflictError("会话已过期，请重新发起整理")
            _check_request_fingerprint(latest, request_id, str(state.get("user_message", "")))
            replay = _stored_request_result(latest, request_id)
            if replay is not None:
                return replay
            claim_token, replay = current.store.claim_request(session_id, request_id, fingerprint)
            if replay is not None:
                return replay
            latest["user_message"] = state.get("user_message", "")
            if request_id:
                latest["request_id"] = request_id
            expected_version = latest.get("_version")
            result = _prepare_persisted_result(
                current.workflow.run(latest),
                request_id,
                fingerprint,
            )
            authoritative = current.store.get(session_id)
            if authoritative is None:
                raise SessionConflictError("会话已过期，请重新发起整理")
            if authoritative.get("_version") != expected_version:
                raise SessionConflictError("会话已被其他请求更新，请重试")
            request_claim = (
                (request_id, fingerprint, claim_token)
                if request_id and fingerprint and claim_token
                else None
            )
            saved = current.store.save(
                result,
                expected_version=expected_version,
                request_claim=request_claim,
            )
            current.metrics.record(saved)
            return saved
        finally:
            current.store.release_request_claim(
                session_id,
                request_id,
                fingerprint,
                claim_token,
            )


def _readiness_payload(current: AppServices, dependencies_ready: bool) -> dict[str, Any]:
    service_status = "ok" if dependencies_ready else "degraded"
    summary = {
        "status": service_status,
        "ready": bool(service_status == "ok"),
        "mode": current.mode,
        "environment": current.mode if not current.production else "production",
        "auth_configured": bool(getattr(current, "auth", None)),
        "service_auth_configured": bool(current.api_tokens) if current.production else False,
    }
    if current.production:
        # Direct API exposure must not disclose model names, provider topology,
        # corpus size, or adapter state. Internal operators can use an
        # authenticated service channel for detailed diagnostics.
        return summary
    return {
        **summary,
        "workflow": "langgraph-compatible" if current.workflow.langgraph_available else "deterministic",
        "knowledge": current.retriever.health(),
        "milvus": (
            AdapterStatus.describe(current.milvus, "milvus-unavailable")
            if current.milvus.ready or not current.production
            else "milvus-unavailable"
        ),
        "database": (
            current.sql_guard.backend_name
            if current.mysql.ready
            else (
                "mysql-unavailable"
                if current.production and getattr(current, "structured_data_enabled", True)
                else ("not-configured" if current.production else AdapterStatus.describe(current.mysql, "sqlite-test"))
            )
        ),
        "cache": (
            current.store.backend_name
            if current.redis.ready
            else ("redis-unavailable" if current.production else AdapterStatus.describe(current.redis, "in-memory-test"))
        ),
        "generator": (
            f"openai:{current.answerer.model}"
            if current.answerer.available
            else ("openai-unavailable" if current.production else "deterministic-grounded-test")
        ),
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Cheap process liveness probe; never contacts external dependencies."""
    current = get_services()
    return {
        "status": "ok",
        "live": True,
        "version": __version__,
        "mode": current.mode,
        "environment": current.mode if not current.production else "production",
        "auth_configured": bool(getattr(current, "auth", None)),
        "service_auth_configured": bool(current.api_tokens) if current.production else False,
    }


@app.middleware("http")
async def prevent_api_caching(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = (
            "no-cache, no-store, private"
            if request.url.path.endswith("/chat")
            else "no-store, private"
        )
        response.headers["Pragma"] = "no-cache"
        response.headers["Vary"] = "Cookie, Authorization"
    return response


@app.get("/api/ready")
def ready() -> dict[str, Any]:
    """Readiness probe used by orchestration; health is the liveness probe."""
    current = get_services()
    readiness = getattr(current, "readiness_missing", None)
    missing = readiness() if callable(readiness) else current.production_missing
    payload = _readiness_payload(current, not missing)
    if not payload["ready"]:
        raise HTTPException(status_code=503, detail={"message": "服务依赖尚未就绪", "missing": list(missing)})
    return payload


def _set_auth_cookie(response: Response, current: AppServices, token: str) -> None:
    response.set_cookie(
        key=current.auth.cookie_name,
        value=token,
        max_age=current.auth.session_ttl_seconds,
        path="/",
        secure=current.auth.cookie_secure,
        httponly=True,
        samesite="strict",
    )


def _clear_auth_cookie(response: Response, current: AppServices) -> None:
    response.delete_cookie(
        key=current.auth.cookie_name,
        path="/",
        secure=current.auth.cookie_secure,
        httponly=True,
        samesite="strict",
    )


@app.post("/api/auth/register", response_model=AuthUserResponse, status_code=201)
def register(credentials: AuthCredentials, response: Response, http_request: Request) -> AuthUserResponse:
    current = get_services()
    _require_secure_transport(current, http_request)
    _require_auth_backend(current)
    try:
        username, token, _expires_at = current.auth.register_and_create_session(
            credentials.username,
            credentials.password.get_secret_value(),
            previous_token=http_request.cookies.get(current.auth.cookie_name),
        )
    except UsernameUnavailableError as exc:
        raise HTTPException(status_code=409, detail="用户名已被使用") from exc
    except (InvalidUsernameError, InvalidPasswordError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.warning("Authentication registration failed: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="共享认证服务暂时不可用，请稍后重试") from exc
    _set_auth_cookie(response, current, token)
    return AuthUserResponse(username=username)


@app.post("/api/auth/login", response_model=AuthUserResponse)
def login(credentials: AuthCredentials, response: Response, http_request: Request) -> AuthUserResponse:
    current = get_services()
    _require_secure_transport(current, http_request)
    _require_auth_backend(current)
    try:
        username = current.auth.verify_credentials(
            credentials.username,
            credentials.password.get_secret_value(),
        )
        token, _expires_at = current.auth.create_session(username)
    except InvalidCredentialsError as exc:
        raise HTTPException(status_code=401, detail="用户名或密码错误") from exc
    except Exception as exc:
        logger.warning("Authentication login failed: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="共享认证服务暂时不可用，请稍后重试") from exc
    previous_token = http_request.cookies.get(current.auth.cookie_name)
    if previous_token:
        current.auth.revoke_session(previous_token)
    _set_auth_cookie(response, current, token)
    return AuthUserResponse(username=username)


@app.post("/api/auth/logout")
def logout(response: Response, http_request: Request) -> dict[str, bool]:
    current = get_services()
    _require_secure_transport(current, http_request)
    current.auth.revoke_session(http_request.cookies.get(current.auth.cookie_name))
    _clear_auth_cookie(response, current)
    return {"ok": True}


@app.get("/api/auth/me", response_model=AuthUserResponse)
def me(http_request: Request) -> AuthUserResponse:
    current = get_services()
    _require_secure_transport(current, http_request)
    username = _username_from_cookie(current, http_request.cookies.get(current.auth.cookie_name))
    if username is None:
        raise HTTPException(status_code=401, detail="请先登录")
    return AuthUserResponse(username=username)


@app.post("/api/sessions", response_model=SessionCreateResponse)
def create_session(
    http_request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    authorization: str | None = Header(default=None),
    api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> SessionCreateResponse:
    current = get_services()
    owner_id = _require_access(
        current,
        http_request,
        authorization,
        api_key,
        _session_token_from_request(current, http_request),
    )
    try:
        state = current.store.create(owner_id=owner_id, idempotency_key=idempotency_key)
    except SessionBackendError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return SessionCreateResponse(session_id=state["session_id"], state=public_state(state))


@app.get("/api/sessions/{session_id}")
def get_session(
    session_id: str,
    http_request: Request,
    authorization: str | None = Header(default=None),
    api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    current = get_services()
    owner_id = _require_access(
        current,
        http_request,
        authorization,
        api_key,
        _session_token_from_request(current, http_request),
    )
    try:
        state = current.store.get(session_id, owner_id=owner_id)
    except SessionAuthorizationError as exc:
        raise HTTPException(status_code=404, detail="会话不存在或已过期") from exc
    except SessionBackendError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if state is None:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    return public_state(state)


@app.get("/api/sessions/{session_id}/requests/{request_id}", response_model=ChatResponse)
def get_request_result(
    session_id: str,
    request_id: str,
    http_request: Request,
    authorization: str | None = Header(default=None),
    api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> ChatResponse:
    current = get_services()
    owner_id = _require_access(
        current,
        http_request,
        authorization,
        api_key,
        _session_token_from_request(current, http_request),
    )
    try:
        state = current.store.request_result(session_id, request_id, owner_id=owner_id)
    except SessionAuthorizationError as exc:
        raise HTTPException(status_code=404, detail="会话不存在或已过期") from exc
    except SessionBackendError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if state is None:
        raise HTTPException(status_code=404, detail="请求结果尚未提交或已过期")
    return _response_from_state(state)


@app.post("/api/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    http_request: Request,
    stream: bool = Query(default=False),
    authorization: str | None = Header(default=None),
    api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    current = get_services()
    owner_id = _require_access(
        current,
        http_request,
        authorization,
        api_key,
        _session_token_from_request(current, http_request),
    )
    if current.production and not request.request_id:
        raise HTTPException(status_code=400, detail="生产整理请求必须提供 request_id 以保证幂等")
    try:
        state = _state_for_request(current, request, owner_id=owner_id)
    except SessionAuthorizationError as exc:
        raise HTTPException(status_code=404, detail="会话不存在或已过期") from exc
    except SessionBackendError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if stream:
        return stream_chat(current, state)
    try:
        return _response_from_state(_run_chat_transaction(current, state))
    except WorkflowExecutionError as exc:
        raise HTTPException(status_code=503, detail="外部生成服务暂时不可用，请稍后重试") from exc
    except SessionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SessionBackendError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def stream_chat(current: AppServices, state: WorkflowState) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=64)
        stopped = threading.Event()

        def enqueue(item: tuple[str, Any]) -> None:
            if stopped.is_set():
                return
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                # There are only nine workflow nodes; dropping an event is
                # preferable to blocking the worker and the event loop.
                pass

        def schedule(item: tuple[str, Any]) -> None:
            if stopped.is_set():
                return
            try:
                loop.call_soon_threadsafe(enqueue, item)
            except RuntimeError:
                stopped.set()

        def on_event(event: dict[str, Any]) -> None:
            if stopped.is_set():
                raise WorkflowCancelled()
            schedule(("node", event))

        def run_sync() -> WorkflowState:
            session_id = state["session_id"]
            request_id = state.get("request_id")
            fingerprint = state.get("_request_fingerprint")
            claim_token: str | None = None
            with current.store.session_lock(session_id):
                try:
                    latest = current.store.get(session_id)
                    if latest is None:
                        raise SessionConflictError("会话已过期，请重新发起整理")
                    _check_request_fingerprint(latest, request_id, str(state.get("user_message", "")))
                    replay = _stored_request_result(latest, request_id)
                    if replay is not None:
                        schedule(("final", replay))
                        return replay
                    claim_token, replay = current.store.claim_request(session_id, request_id, fingerprint)
                    if replay is not None:
                        schedule(("final", replay))
                        return replay
                    latest["user_message"] = state.get("user_message", "")
                    if request_id:
                        latest["request_id"] = request_id
                    expected_version = latest.get("_version")
                    result = current.workflow.run(latest, on_event=on_event, cancel_event=stopped)
                    if stopped.is_set():
                        raise WorkflowCancelled()
                    result = _prepare_persisted_result(
                        result,
                        request_id,
                        fingerprint,
                    )
                    authoritative = current.store.get(session_id)
                    if authoritative is None:
                        raise SessionConflictError("会话已过期，请重新发起整理")
                    if authoritative.get("_version") != expected_version:
                        raise SessionConflictError("会话已被其他请求更新，请重试")
                    request_claim = (
                        (request_id, fingerprint, claim_token)
                        if request_id and fingerprint and claim_token
                        else None
                    )
                    saved = current.store.save(
                        result,
                        expected_version=expected_version,
                        cancel_event=stopped,
                        request_claim=request_claim,
                    )
                    current.metrics.record(saved)
                finally:
                    current.store.release_request_claim(
                        session_id,
                        request_id,
                        fingerprint,
                        claim_token,
                    )
            schedule(("final", saved))
            return saved

        async def worker() -> None:
            try:
                await asyncio.to_thread(run_sync)
            except (WorkflowCancelled, SessionCancelledError):
                return
            except SessionConflictError as exc:
                schedule(("error", {"status": 409, "message": str(exc)}))
            except SessionBackendError as exc:
                schedule(("error", {"status": 503, "message": str(exc)}))
            except WorkflowExecutionError:
                schedule(("error", {"status": 503, "message": "外部生成服务暂时不可用，请稍后重试"}))
            except Exception:
                # Do not expose provider/database internals through SSE.
                schedule(("error", {"status": 500, "message": "整理服务暂时不可用，请稍后重试"}))
            finally:
                schedule(("done", None))

        task = asyncio.create_task(worker())
        try:
            while True:
                event_type, payload = await queue.get()
                if event_type == "node":
                    yield f"event: node\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                elif event_type == "final":
                    saved = payload
                    public = public_state(saved)
                    final_payload = {
                        "session_id": saved["session_id"],
                        "response_id": saved.get("response_id", ""),
                        "answer": saved.get("response", ""),
                        "state": public,
                        "citations": public["citations"],
                        "structured_result": saved.get("structured_result"),
                        "confidence": saved.get("confidence", 0.0),
                        "latency_ms": saved.get("latency_ms", 0.0),
                        "summary": saved.get("summary", ""),
                    }
                    yield f"event: final\ndata: {json.dumps(final_payload, ensure_ascii=False)}\n\n"
                elif event_type == "error":
                    yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                elif event_type == "done":
                    break
        finally:
            stopped.set()
            if not task.done():
                task.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.post("/api/search")
def search(
    request: SearchRequest,
    http_request: Request,
    authorization: str | None = Header(default=None),
    api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    current = get_services()
    _require_access(
        current,
        http_request,
        authorization,
        api_key,
        _session_token_from_request(current, http_request),
    )
    intent = request.intent if request.intent in {"disease", "drug", "exam", "department", "faq", "unknown"} else "unknown"
    query_text = request.query.strip()
    if not query_text:
        return {"query": request.query, "intent": intent, "count": 0, "citations": []}
    citations = _sanitize_citation_payload(
        [citation.as_dict() for citation in current.retriever.search(query_text, intent, request.top_k)]
    )
    return {"query": request.query, "intent": intent, "count": len(citations), "citations": citations}


@app.post("/api/query")
def query(
    request: QueryRequest,
    http_request: Request,
    authorization: str | None = Header(default=None),
    api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    current = get_services()
    _require_access(
        current,
        http_request,
        authorization,
        api_key,
        _session_token_from_request(current, http_request),
    )
    result = current.sql_guard.from_natural_language(request.text)
    if result is None:
        return {"matched": False, "message": "未识别到可执行的结构化查询类型"}
    return {"matched": True, "result": result.as_dict()}


@app.post("/api/feedback")
def feedback(
    request: FeedbackRequest,
    http_request: Request,
    authorization: str | None = Header(default=None),
    api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    current = get_services()
    owner_id = _require_access(
        current,
        http_request,
        authorization,
        api_key,
        _session_token_from_request(current, http_request),
    )
    try:
        state = current.store.get(request.session_id, owner_id=owner_id)
    except SessionAuthorizationError as exc:
        raise HTTPException(status_code=404, detail="会话不存在或已过期") from exc
    except SessionBackendError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if state is None:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    if request.message_id not in state.get("response_ids", []):
        raise HTTPException(status_code=422, detail="反馈目标不属于当前会话")
    changed = current.metrics.add_feedback(
        request.rating,
        request.comment,
        request.message_id,
        session_id=request.session_id,
    )
    return {
        "accepted": True,
        "changed": changed,
        "session_id": request.session_id,
        "rating": request.rating,
        "message_id": request.message_id,
    }


@app.get("/api/metrics")
def metrics(
    http_request: Request,
    authorization: str | None = Header(default=None),
    api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    current = get_services()
    _require_service_access(
        current,
        http_request,
        authorization,
        api_key,
    )
    return current.metrics.snapshot()
