from __future__ import annotations

import json
import os
import hashlib
from collections.abc import Mapping
from datetime import date, datetime, time
from decimal import Decimal
from time import monotonic
from typing import Any


class _SessionPayloadError(RuntimeError):
    """Raised for invalid session data without marking Redis unavailable."""


def _json_scalar(value: Any) -> Any:
    """Normalize common SQL scalar types before API/session serialization."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _adapter_timeout() -> float:
    try:
        return max(0.1, min(float(os.getenv("MEDGUIDE_ADAPTER_TIMEOUT", "0.75")), 10.0))
    except ValueError:
        return 0.75


def _adapter_retry_delay() -> float:
    try:
        return max(0.1, min(float(os.getenv("MEDGUIDE_ADAPTER_RETRY_SECONDS", "5")), 60.0))
    except ValueError:
        return 5.0


def _session_ttl_seconds() -> int:
    """Return a bounded retention window for Redis-backed sessions."""
    default = 24 * 60 * 60
    try:
        value = int(os.getenv("MEDGUIDE_SESSION_TTL_SECONDS", str(default)))
    except (TypeError, ValueError):
        return default
    # A positive lower bound prevents accidental immortal records; cap the
    # upper bound so a malformed deployment setting cannot disable retention.
    return max(60, min(value, 365 * 24 * 60 * 60))


class RedisSessionAdapter:
    """Small Redis repository used only when a reachable Redis URL is configured."""

    prefix = "medguide:session:"
    creation_prefix = "medguide:session-create:"
    metrics_key = "medguide:metrics:v1"
    metrics_response_prefix = "medguide:metrics:response:"
    metrics_feedback_key = "medguide:metrics:feedback:v1"

    _CREATE_SESSION_SCRIPT = """
local mapped_session_id = redis.call('GET', KEYS[2])
if mapped_session_id then
    local mapped_session_key = ARGV[1] .. mapped_session_id
    if redis.call('EXISTS', mapped_session_key) == 1 then
        redis.call('EXPIRE', KEYS[2], ARGV[3])
        return mapped_session_id
    end
    redis.call('DEL', KEYS[2])
end
if redis.call('EXISTS', KEYS[1]) == 1 then
    return ''
end
redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3])
redis.call('SET', KEYS[2], ARGV[4], 'EX', ARGV[3])
return ARGV[4]
"""

    _CLAIM_REQUEST_SCRIPT = """
local raw = redis.call('GET', KEYS[1])
if not raw then
    return -2
end
local decoded, state = pcall(cjson.decode, raw)
if not decoded or type(state) ~= 'table' then
    return -3
end
local fingerprints = state['request_fingerprints']
local stored_fingerprint = nil
if type(fingerprints) == 'table' then
    stored_fingerprint = fingerprints[ARGV[1]]
end
if stored_fingerprint and stored_fingerprint ~= ARGV[2] then
    return -1
end
local results = state['request_results']
if type(results) == 'table' and results[ARGV[1]] then
    return 2
end
local existing = redis.call('GET', KEYS[2])
if existing then
    if string.sub(existing, 1, 64) ~= ARGV[2] then
        return -1
    end
    return 0
end
redis.call('SET', KEYS[2], ARGV[2] .. ':' .. ARGV[3], 'EX', ARGV[4], 'NX')
return 1
"""

    _SAVE_CLAIMED_SESSION_SCRIPT = """
local raw = redis.call('GET', KEYS[1])
if not raw then
    return -2
end
local decoded, state = pcall(cjson.decode, raw)
if not decoded or type(state) ~= 'table' then
    return -3
end
if tonumber(state['_version']) ~= tonumber(ARGV[1]) then
    return 0
end
if redis.call('GET', KEYS[2]) ~= ARGV[4] then
    return -1
end
redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3])
redis.call('DEL', KEYS[2])
return 1
"""

    _RELEASE_REQUEST_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""

    _UPSERT_FEEDBACK_SCRIPT = """
local previous = redis.call('HGET', KEYS[1], ARGV[1])
if previous == ARGV[2] then
    redis.call('EXPIRE', KEYS[1], ARGV[4])
    redis.call('EXPIRE', KEYS[2], ARGV[4])
    return 0
end
if previous == 'up' or previous == 'down' then
    redis.call('HINCRBY', KEYS[2], 'feedback_' .. previous, -1)
end
redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
redis.call('HINCRBY', KEYS[2], 'feedback_' .. ARGV[2], 1)
if ARGV[3] == '1' and not previous then
    redis.call('HINCRBY', KEYS[2], 'feedback_commented', 1)
end
redis.call('EXPIRE', KEYS[1], ARGV[4])
redis.call('EXPIRE', KEYS[2], ARGV[4])
return 1
"""

    _RECORD_METRICS_SCRIPT = """
if redis.call('EXISTS', KEYS[1]) == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
    return 0
end
redis.call('SET', KEYS[1], '1', 'EX', ARGV[1])
redis.call('HINCRBY', KEYS[2], 'requests', 1)
redis.call('HINCRBYFLOAT', KEYS[2], 'latency_total_ms', ARGV[2])
redis.call('HINCRBY', KEYS[2], 'citation_covered', ARGV[3])
redis.call('HINCRBY', KEYS[2], 'retrieval_hits', ARGV[4])
redis.call('HINCRBY', KEYS[2], 'blocked_queries', ARGV[5])
redis.call('HINCRBY', KEYS[2], 'structured_successes', ARGV[6])
redis.call('HINCRBY', KEYS[2], 'high_risk', ARGV[7])
redis.call('HINCRBY', KEYS[2], 'safety_reviewed', 1)
redis.call('HINCRBY', KEYS[2], 'safety_blocked', ARGV[8])
if ARGV[9] == '1' then
    redis.call('HINCRBY', KEYS[2], 'hallucination_labeled', 1)
    redis.call('HINCRBY', KEYS[2], 'hallucinations', ARGV[10])
end
redis.call('EXPIRE', KEYS[2], ARGV[1])
return 1
"""

    def __init__(self, url: str | None = None, client: Any | None = None, *, enabled: bool = True) -> None:
        self.url = (url or os.getenv("REDIS_URL")) if enabled else None
        self.client: Any | None = client
        self.configured = bool(self.url or client)
        self.available = False
        self.ready = False
        self.error: str | None = None
        self._retry_after = 0.0
        self._closed = False
        self.ttl_seconds = _session_ttl_seconds()
        self._owns_client = False
        if self.client is None and self.url:
            self._build_client()
        else:
            self.available = self.client is not None
        if self.available:
            self.ready = self._probe()

    def _create_client(self) -> Any:
        import redis

        timeout = _adapter_timeout()
        return redis.Redis.from_url(
            self.url,
            decode_responses=True,
            socket_connect_timeout=timeout,
            socket_timeout=timeout,
        )

    def _build_client(self) -> bool:
        if self._closed or not self.url:
            return False
        try:
            self.client = self._create_client()
            self._owns_client = True
        except Exception as exc:
            self.client = None
            self.available = False
            self.ready = False
            self.error = type(exc).__name__
            self._retry_after = monotonic() + _adapter_retry_delay()
            return False
        self.available = True
        return True

    def _discard_client(self) -> None:
        """Drop an owned client after a failed probe, best effort."""
        client = self.client
        self.client = None
        self.available = False
        self.ready = False
        self._owns_client = False
        if client is None:
            return
        close = getattr(client, "close", None)
        closed = False
        if callable(close):
            try:
                close()
                closed = True
            except Exception:
                pass
        if not closed:
            disconnect = getattr(getattr(client, "connection_pool", None), "disconnect", None)
            if callable(disconnect):
                try:
                    disconnect()
                except Exception:
                    pass

    def _probe(self) -> bool:
        if self.client is None:
            return False
        try:
            ping = getattr(self.client, "ping", None)
            ready = bool(ping()) if callable(ping) else True
            if ready:
                self.error = None
            return ready
        except Exception as exc:
            if self._owns_client:
                self._discard_client()
            self.error = type(exc).__name__
            self._retry_after = monotonic() + _adapter_retry_delay()
            return False

    def _mark_unavailable(self, exc: Exception) -> None:
        # Keep provider details out of API responses and make health reflect a
        # failure that happens after startup.
        self.ready = False
        if self._owns_client:
            self._discard_client()
        self.error = type(exc).__name__
        self._retry_after = monotonic() + _adapter_retry_delay()

    def ensure_ready(self) -> bool:
        if self._closed:
            return False
        if self.ready:
            return True
        if monotonic() < self._retry_after:
            return False
        if self.client is None and not self._build_client():
            return False
        self.ready = self._probe()
        if self.ready:
            self.error = None
        return self.ready

    @property
    def status(self) -> str:
        if self.ready:
            return "redis"
        if self.configured:
            return "redis-unavailable"
        return "in-memory-disabled"

    def _key(self, session_id: str) -> str:
        return f"{self.prefix}{session_id}"

    @classmethod
    def _request_key(cls, session_id: str, request_id: str) -> str:
        digest = hashlib.sha256(f"{session_id}\x00{request_id}".encode("utf-8")).hexdigest()
        return f"{cls.prefix}request:{digest}"

    @classmethod
    def _creation_key(cls, idempotency_key: str, owner_id: str = "") -> str:
        # Do not put a caller-controlled key (which may contain PHI) in Redis
        # key names or logs.
        digest = hashlib.sha256(f"{owner_id}\x00{idempotency_key}".encode("utf-8")).hexdigest()
        return f"{cls.creation_prefix}{digest}"

    def get_session_creation(self, idempotency_key: str, owner_id: str = "") -> str | None:
        if not idempotency_key or not self.ensure_ready() or self.client is None:
            return None
        try:
            value = self.client.get(self._creation_key(idempotency_key, owner_id))
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            return str(value) if value else None
        except Exception as exc:
            self._mark_unavailable(exc)
            raise RuntimeError("Redis idempotency read failed") from exc

    def remember_session_creation(
        self,
        idempotency_key: str,
        owner_id: str,
        session_id: str,
        ttl_seconds: int | None = None,
    ) -> str:
        if not self.ensure_ready() or self.client is None:
            raise RuntimeError("Redis adapter is not ready")
        key = self._creation_key(idempotency_key, owner_id)
        ttl = int(ttl_seconds or self.ttl_seconds)
        try:
            setter = getattr(self.client, "set", None)
            if not callable(setter):
                raise RuntimeError("Redis client does not support SET")
            try:
                accepted = setter(key, session_id, nx=True, ex=ttl)
            except TypeError:
                # Minimal Redis-compatible fakes may only support a plain SET;
                # retaining the read-before-write result is still preferable
                # to dropping the idempotency contract entirely.
                existing = self.client.get(key)
                if existing:
                    return existing.decode("utf-8") if isinstance(existing, bytes) else str(existing)
                setter(key, session_id)
                expire = getattr(self.client, "expire", None)
                if callable(expire):
                    expire(key, ttl)
                accepted = True
            if accepted:
                return session_id
            existing = self.client.get(key)
            if isinstance(existing, bytes):
                existing = existing.decode("utf-8")
            return str(existing or session_id)
        except Exception as exc:
            self._mark_unavailable(exc)
            raise RuntimeError("Redis idempotency write failed") from exc

    def create_state(
        self,
        state: dict[str, Any],
        idempotency_key: str,
        owner_id: str = "",
        ttl_seconds: int | None = None,
    ) -> str:
        """Atomically create a session and bind its client idempotency key."""
        if not idempotency_key or not self.ensure_ready() or self.client is None:
            raise RuntimeError("Redis adapter is not ready")
        evaluator = getattr(self.client, "eval", None)
        if not callable(evaluator):
            raise RuntimeError("Redis client does not support atomic session creation")
        session_id = str(state["session_id"])
        ttl = int(ttl_seconds or self.ttl_seconds)
        payload = self._serialize_state(state)
        try:
            winner = evaluator(
                self._CREATE_SESSION_SCRIPT,
                2,
                self._key(session_id),
                self._creation_key(idempotency_key, owner_id),
                self.prefix,
                payload,
                ttl,
                session_id,
            )
            if isinstance(winner, bytes):
                winner = winner.decode("utf-8")
            if not winner:
                raise RuntimeError("Redis session id collision")
            return str(winner)
        except (UnicodeDecodeError, _SessionPayloadError):
            raise
        except Exception as exc:
            self._mark_unavailable(exc)
            raise RuntimeError("Redis atomic session creation failed") from exc

    def claim_request(
        self,
        session_id: str,
        request_id: str,
        fingerprint: str,
        token: str,
    ) -> str:
        """Claim one request across workers or report its committed/in-flight state."""
        if not self.ensure_ready() or self.client is None:
            raise RuntimeError("Redis adapter is not ready")
        evaluator = getattr(self.client, "eval", None)
        if not callable(evaluator):
            raise RuntimeError("Redis client does not support atomic request claims")
        claim_ttl = min(self.ttl_seconds, 300)
        try:
            code = int(
                evaluator(
                    self._CLAIM_REQUEST_SCRIPT,
                    2,
                    self._key(session_id),
                    self._request_key(session_id, request_id),
                    request_id,
                    fingerprint,
                    token,
                    claim_ttl,
                )
            )
        except Exception as exc:
            self._mark_unavailable(exc)
            raise RuntimeError("Redis request claim failed") from exc
        if code == 1:
            return "acquired"
        if code == 2:
            return "completed"
        if code == 0:
            return "pending"
        if code == -1:
            return "conflict"
        if code == -2:
            return "missing"
        if code == -3:
            raise _SessionPayloadError("Redis session payload is invalid")
        raise RuntimeError("Redis request claim returned an invalid result")

    def release_request_claim(
        self,
        session_id: str,
        request_id: str,
        fingerprint: str,
        token: str,
    ) -> bool:
        """Release only the caller's own request claim."""
        if not self.ensure_ready() or self.client is None:
            return False
        evaluator = getattr(self.client, "eval", None)
        if not callable(evaluator):
            return False
        try:
            return bool(
                evaluator(
                    self._RELEASE_REQUEST_SCRIPT,
                    1,
                    self._request_key(session_id, request_id),
                    f"{fingerprint}:{token}",
                )
            )
        except Exception as exc:
            self._mark_unavailable(exc)
            return False

    @staticmethod
    def _metric_number(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def record_metrics(self, state: dict[str, Any]) -> bool:
        """Aggregate one response in Redis and deduplicate by response id."""
        if not self.ensure_ready() or self.client is None:
            return False
        response_id = str(state.get("response_id") or "")
        if not response_id:
            return False
        digest = hashlib.sha256(response_id.encode("utf-8")).hexdigest()
        dedupe_key = f"{self.metrics_response_prefix}{digest}"
        try:
            evaluator = getattr(self.client, "eval", None)
            if not callable(evaluator):
                return False
            latency = self._metric_number(state.get("latency_ms", 0.0))
            citations = state.get("citations") or []
            valid_citations = any(
                isinstance(citation, dict)
                and bool(citation.get("id"))
                and not citation.get("blocked", False)
                for citation in citations
            )
            structured = state.get("structured_result")
            blocked = bool(isinstance(structured, dict) and structured.get("blocked"))
            structured_success = bool(
                isinstance(structured, dict)
                and not blocked
                and structured.get("rows")
            )
            label = state.get("hallucination_label")
            evaluator(
                self._RECORD_METRICS_SCRIPT,
                2,
                dedupe_key,
                self.metrics_key,
                self.ttl_seconds,
                latency,
                int(valid_citations or structured_success),
                int(valid_citations),
                int(blocked),
                int(structured_success),
                int(state.get("risk_level") == "high"),
                int(bool(state.get("safety_blocked"))),
                int(isinstance(label, bool)),
                int(bool(label)) if isinstance(label, bool) else 0,
            )
            return True
        except Exception as exc:
            self._mark_unavailable(exc)
            return False

    def upsert_feedback(
        self,
        session_id: str,
        message_id: str,
        rating: str,
        comment: str | None = None,
    ) -> bool | None:
        if not self.ensure_ready() or self.client is None:
            return None
        try:
            evaluator = getattr(self.client, "eval", None)
            if not callable(evaluator):
                return None
            key = hashlib.sha256(f"{session_id}\x00{message_id}".encode("utf-8")).hexdigest()
            changed = evaluator(
                self._UPSERT_FEEDBACK_SCRIPT,
                2,
                self.metrics_feedback_key,
                self.metrics_key,
                key,
                rating,
                "1" if comment else "0",
                self.ttl_seconds,
            )
            return bool(int(changed))
        except Exception as exc:
            self._mark_unavailable(exc)
            return None

    def metrics_snapshot(self) -> dict[str, Any] | None:
        if not self.ensure_ready() or self.client is None:
            return None
        try:
            hgetall = getattr(self.client, "hgetall", None)
            if not callable(hgetall):
                return None
            raw = hgetall(self.metrics_key)
            if not raw:
                return None
            values: dict[str, Any] = {}
            for key, value in raw.items():
                if isinstance(key, bytes):
                    key = key.decode("utf-8")
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                values[str(key)] = value
            requests = int(float(values.get("requests", 0)))
            labeled = int(float(values.get("hallucination_labeled", 0)))
            feedback: dict[str, int] = {}
            for name in ("up", "down", "commented"):
                count = int(float(values.get(f"feedback_{name}", 0)))
                if count > 0:
                    feedback[name] = count
            return {
                "requests": requests,
                "avg_latency_ms": round(float(values.get("latency_total_ms", 0.0)) / requests, 2) if requests else 0.0,
                "citation_coverage": round(int(float(values.get("citation_covered", 0))) / requests, 3) if requests else 0.0,
                "retrieval_hit_rate": round(int(float(values.get("retrieval_hits", 0))) / requests, 3) if requests else 0.0,
                "high_risk_rate": round(int(float(values.get("high_risk", 0))) / requests, 3) if requests else 0.0,
                "blocked_queries": int(float(values.get("blocked_queries", 0))),
                "structured_successes": int(float(values.get("structured_successes", 0))),
                "safety_blocked": int(float(values.get("safety_blocked", 0))),
                "hallucination_rate": round(int(float(values.get("hallucinations", 0))) / labeled, 3) if labeled else None,
                "hallucination_labeled": labeled,
                "hallucination_label_status": "labeled" if labeled else "unlabeled; unavailable",
                "feedback": feedback,
                "feedback_by_message": {},
                "evaluation_note": "引用覆盖率仅统计有效引用或有行结构化结果；幻觉率需人工标注后计算。",
            }
        except Exception as exc:
            self._mark_unavailable(exc)
            return None

    @staticmethod
    def _reset_pipeline(pipe: Any) -> None:
        reset = getattr(pipe, "reset", None)
        if callable(reset):
            try:
                reset()
            except Exception:
                # Reset is cleanup only; a malformed record must still be
                # surfaced as a payload error rather than a provider outage.
                pass

    @staticmethod
    def _set_with_ttl(target: Any, key: str, payload: str, ttl_seconds: int) -> Any:
        """Set a value with expiry, retaining compatibility with tiny fakes."""
        setter = getattr(target, "set", None)
        if not callable(setter):
            setex = getattr(target, "setex", None)
            if callable(setex):
                return setex(key, ttl_seconds, payload)
            raise AttributeError("Redis client does not support SET")
        try:
            return setter(key, payload, ex=ttl_seconds)
        except TypeError as exc:
            # Older test doubles and a few Redis-compatible clients do not
            # expose the ``ex`` keyword.  Use SETEX when available, otherwise
            # issue an explicit EXPIRE after the plain SET.
            message = str(exc).lower()
            if "ex" not in message and "keyword" not in message and "argument" not in message:
                raise
            setex = getattr(target, "setex", None)
            if callable(setex):
                try:
                    return setex(key, ttl_seconds, payload)
                except TypeError:
                    pass
            result = setter(key, payload)
            expire = getattr(target, "expire", None)
            if callable(expire):
                expire(key, ttl_seconds)
            return result

    @classmethod
    def _validate_payload(cls, payload: Any, session_id: str) -> dict[str, Any]:
        if not isinstance(payload, dict) or payload.get("session_id") != session_id:
            raise _SessionPayloadError("Redis session payload is invalid")
        version = payload.get("_version")
        if type(version) is not int or version < 0:
            raise _SessionPayloadError("Redis session payload is invalid")
        turn_count = payload.get("turn_count")
        if turn_count is not None and (type(turn_count) is not int or turn_count < 0):
            raise _SessionPayloadError("Redis session payload is invalid")
        return payload

    @staticmethod
    def _serialize_state(state: dict[str, Any]) -> str:
        if type(state.get("_version")) is not int or state["_version"] < 0:
            raise _SessionPayloadError("Redis session payload is invalid")
        try:
            return json.dumps(
                state,
                ensure_ascii=False,
                separators=(",", ":"),
                default=_json_scalar,
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Session state is not JSON serializable") from exc

    def get_state(self, session_id: str) -> dict[str, Any] | None:
        if not self.ensure_ready() or self.client is None:
            return None
        try:
            raw = self.client.get(self._key(session_id))
        except Exception as exc:
            self._mark_unavailable(exc)
            raise RuntimeError("Redis read failed") from exc
        if raw is None:
            return None
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            # A corrupt session record is a data-integrity failure, not proof
            # that Redis itself is unavailable.
            raise _SessionPayloadError("Redis session payload is invalid") from exc
        return self._validate_payload(payload, session_id)

    def save_state(
        self,
        state: dict[str, Any],
        expected_version: int | None = None,
        *,
        request_claim: tuple[str, str, str] | None = None,
    ) -> bool:
        if not self.ensure_ready() or self.client is None:
            return False
        key = self._key(str(state["session_id"]))
        session_id = str(state["session_id"])
        if expected_version is not None and (type(expected_version) is not int or expected_version < 0):
            raise _SessionPayloadError("Redis session payload is invalid")
        payload = self._serialize_state(state)
        if request_claim is not None:
            if expected_version is None:
                raise _SessionPayloadError("Redis claimed session save requires a version")
            request_id, fingerprint, token = request_claim
            evaluator = getattr(self.client, "eval", None)
            if not callable(evaluator):
                raise RuntimeError("Redis client does not support atomic request commits")
            try:
                code = int(
                    evaluator(
                        self._SAVE_CLAIMED_SESSION_SCRIPT,
                        2,
                        key,
                        self._request_key(session_id, request_id),
                        expected_version,
                        payload,
                        self.ttl_seconds,
                        f"{fingerprint}:{token}",
                    )
                )
            except Exception as exc:
                self._mark_unavailable(exc)
                raise RuntimeError("Redis claimed session save failed") from exc
            if code == 1:
                return True
            if code in {0, -1, -2}:
                return False
            if code == -3:
                raise _SessionPayloadError("Redis session payload is invalid")
            raise RuntimeError("Redis claimed session save returned an invalid result")
        # WATCH/MULTI gives us a compare-and-set operation across workers.  A
        # simple SET is retained for tiny fakes used by offline tests.
        pipeline_factory = getattr(self.client, "pipeline", None)
        if callable(pipeline_factory):
            try:
                from redis.exceptions import WatchError

                pipe = pipeline_factory()
                pipe.watch(key)
                current_raw = pipe.get(key)
                current: dict[str, Any] | None = None
                if current_raw is not None:
                    try:
                        if isinstance(current_raw, bytes):
                            current_raw = current_raw.decode("utf-8")
                        current = json.loads(current_raw) if isinstance(current_raw, str) else current_raw
                    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
                        self._reset_pipeline(pipe)
                        raise _SessionPayloadError("Redis session payload is invalid") from exc
                    if current is not None:
                        try:
                            current = self._validate_payload(current, session_id)
                        except _SessionPayloadError:
                            self._reset_pipeline(pipe)
                            raise
                current_version = current.get("_version") if current is not None else None
                if expected_version is not None and current_version != expected_version:
                    self._reset_pipeline(pipe)
                    return False
                pipe.multi()
                self._set_with_ttl(pipe, key, payload, self.ttl_seconds)
                pipe.execute()
                return True
            except WatchError:
                return False
            except _SessionPayloadError:
                raise
            except Exception as exc:
                self._mark_unavailable(exc)
                raise RuntimeError("Redis compare-and-set failed") from exc
        try:
            if expected_version is not None:
                current = self.get_state(str(state["session_id"]))
                if current is not None and current.get("_version") != expected_version:
                    return False
            self._set_with_ttl(self.client, key, payload, self.ttl_seconds)
            return True
        except _SessionPayloadError:
            raise
        except Exception as exc:
            self._mark_unavailable(exc)
            raise RuntimeError("Redis write failed") from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        client = self.client
        self.client = None
        self.available = False
        self.ready = False
        self._owns_client = False
        if client is None:
            return
        close = getattr(client, "close", None)
        closed = False
        if callable(close):
            try:
                close()
                closed = True
            except Exception:
                pass
        if not closed:
            disconnect = getattr(getattr(client, "connection_pool", None), "disconnect", None)
            if callable(disconnect):
                try:
                    disconnect()
                except Exception:
                    pass

    def shutdown(self) -> None:
        self.close()

    async def aclose(self) -> None:
        self.close()


class MySQLReadOnlyAdapter:
    """Read-only SQL executor for production, with an explicit health state."""

    def __init__(self, dsn: str | None = None, engine: Any | None = None, *, enabled: bool = True) -> None:
        self.dsn = (dsn or os.getenv("MYSQL_DSN")) if enabled else None
        self.engine: Any | None = engine
        self.configured = bool(self.dsn or engine)
        self.available = False
        self.ready = False
        self.error: str | None = None
        self._retry_after = 0.0
        self._closed = False
        self._owns_engine = False
        if self.engine is None and self.dsn:
            self._build_engine()
        else:
            self.available = self.engine is not None
        if self.available:
            self.ready = self._probe()

    def _create_engine(self) -> Any:
        from sqlalchemy import create_engine

        kwargs: dict[str, Any] = {
            "pool_pre_ping": True,
            "pool_recycle": 1800,
            "execution_options": {"isolation_level": "AUTOCOMMIT"},
        }
        # Pool sizing and connect_timeout are driver-specific; avoid passing
        # them to SQLite (used only by local tests/tools).
        if self.dsn and self.dsn.lower().startswith(("mysql", "mariadb")):
            timeout = max(1, int(round(_adapter_timeout())))
            kwargs.update(
                pool_size=5,
                max_overflow=5,
                pool_timeout=timeout,
                connect_args={
                    "connect_timeout": timeout,
                    "read_timeout": timeout,
                    "write_timeout": timeout,
                },
            )
        return create_engine(self.dsn, **kwargs)

    def _build_engine(self) -> bool:
        if self._closed or not self.dsn:
            return False
        try:
            self.engine = self._create_engine()
            self._owns_engine = True
        except Exception as exc:
            self.engine = None
            self.available = False
            self.ready = False
            self.error = type(exc).__name__
            self._retry_after = monotonic() + _adapter_retry_delay()
            return False
        self.available = True
        return True

    def _discard_engine(self) -> None:
        """Dispose an owned SQLAlchemy engine after a failed operation."""
        engine = self.engine
        self.engine = None
        self.available = False
        self.ready = False
        self._owns_engine = False
        dispose = getattr(engine, "dispose", None)
        if callable(dispose):
            try:
                dispose()
            except Exception:
                pass

    def _probe(self) -> bool:
        if self.engine is None:
            return False
        try:
            with self.engine.connect() as connection:
                connection.exec_driver_sql("SELECT 1")
            self.error = None
            return True
        except Exception as exc:
            if self._owns_engine:
                self._discard_engine()
            self.error = type(exc).__name__
            self._retry_after = monotonic() + _adapter_retry_delay()
            return False

    def _mark_unavailable(self, exc: Exception) -> None:
        self.ready = False
        if self._owns_engine:
            self._discard_engine()
        self.error = type(exc).__name__
        self._retry_after = monotonic() + _adapter_retry_delay()

    def ensure_ready(self) -> bool:
        if self._closed:
            return False
        if self.ready:
            return True
        if monotonic() < self._retry_after:
            return False
        if self.engine is None and not self._build_engine():
            return False
        self.ready = self._probe()
        if self.ready:
            self.error = None
        return self.ready

    @property
    def status(self) -> str:
        if self.ready:
            return "mysql"
        if self.configured:
            return "mysql-unavailable"
        return "sqlite-disabled"

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...]]:
        if not self.ensure_ready() or self.engine is None:
            raise RuntimeError("MySQL adapter is not ready")
        try:
            with self.engine.connect() as connection:
                paramstyle = str(getattr(getattr(self.engine, "dialect", None), "paramstyle", "qmark"))
                statement = (
                    self._convert_qmark(sql, escape_percent=bool(params))
                    if paramstyle in {"format", "pyformat"}
                    else sql
                )
                result = connection.exec_driver_sql(statement, params)
                raw_columns = tuple(str(key) for key in result.keys())
                columns = self._unique_columns(raw_columns)
                try:
                    raw_rows = result.fetchall()
                except AttributeError:
                    raw_rows = result.mappings().all()
                rows: list[dict[str, Any]] = []
                for row in raw_rows:
                    if isinstance(row, Mapping):
                        values = [row.get(key) for key in raw_columns]
                    else:
                        values = list(row)
                    rows.append(
                        {
                            name: _json_scalar(values[index]) if index < len(values) else None
                            for index, name in enumerate(columns)
                        }
                    )
            return columns, tuple(rows)
        except Exception as exc:
            self._mark_unavailable(exc)
            raise RuntimeError("MySQL query failed") from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        engine = self.engine
        self.engine = None
        self.available = False
        self.ready = False
        self._owns_engine = False
        dispose = getattr(engine, "dispose", None)
        if callable(dispose):
            try:
                dispose()
            except Exception:
                pass

    def shutdown(self) -> None:
        self.close()

    async def aclose(self) -> None:
        self.close()

    @staticmethod
    def _unique_columns(columns: tuple[str, ...]) -> tuple[str, ...]:
        counts: dict[str, int] = {}
        unique: list[str] = []
        for column in columns:
            counts[column] = counts.get(column, 0) + 1
            unique.append(column if counts[column] == 1 else f"{column}_{counts[column]}")
        return tuple(unique)

    @staticmethod
    def _convert_qmark(sql: str, *, escape_percent: bool = False) -> str:
        """Convert qmark placeholders outside SQL strings/quoted identifiers."""
        output: list[str] = []
        index = 0
        quote: str | None = None
        while index < len(sql):
            char = sql[index]
            if quote is not None:
                output.append(char)
                if escape_percent and char == "%":
                    output.append("%")
                if char == "\\" and quote == "'" and index + 1 < len(sql):
                    # MySQL permits backslash-escaped quotes in string
                    # literals.  Keep the escaped character inside the quote.
                    output.append(sql[index + 1])
                    index += 2
                    continue
                if char == quote:
                    if index + 1 < len(sql) and sql[index + 1] == quote and quote in {"'", '"', "`"}:
                        output.append(sql[index + 1])
                        index += 2
                        continue
                    quote = None
            elif char in {"'", '"', "`"}:
                quote = char
                output.append(char)
            elif char == "[":
                quote = "]"
                output.append(char)
            elif char == "?":
                output.append("%s")
            elif escape_percent and char == "%":
                # Python format/pyformat DB-API drivers interpret every
                # percent sign, including signs inside SQL string literals.
                output.append("%%")
            else:
                output.append(char)
            index += 1
        return "".join(output)


class AdapterStatus:
    """Common status helper for health responses and dependency injection."""

    @staticmethod
    def describe(adapter: Any, fallback: str) -> str:
        return str(getattr(adapter, "status", fallback))
