from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from app.infra import RedisSessionAdapter
from app.main import _message_fingerprint, _run_chat_transaction
from app.metrics import MetricsStore
from app.store import SessionConflictError, SessionStore


class _AtomicRedis:
    """Small synchronized Redis model for exercising the adapter's Lua contracts."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.hashes: dict[str, dict[str, str | int]] = {}
        self._lock = threading.Lock()

    def ping(self) -> bool:
        return True

    def get(self, key: str):
        with self._lock:
            return self.values.get(key)

    def hgetall(self, key: str):
        with self._lock:
            return dict(self.hashes.get(key, {}))

    def eval(self, script: str, number_of_keys: int, *items):
        keys = list(items[:number_of_keys])
        args = list(items[number_of_keys:])
        with self._lock:
            if script == RedisSessionAdapter._CREATE_SESSION_SCRIPT:
                return self._create_session(keys, args)
            if script == RedisSessionAdapter._CLAIM_REQUEST_SCRIPT:
                return self._claim_request(keys, args)
            if script == RedisSessionAdapter._SAVE_CLAIMED_SESSION_SCRIPT:
                return self._save_claimed_session(keys, args)
            if script == RedisSessionAdapter._RELEASE_REQUEST_SCRIPT:
                return self._release_request(keys, args)
            if script == RedisSessionAdapter._UPSERT_FEEDBACK_SCRIPT:
                return self._upsert_feedback(keys, args)
            if script == RedisSessionAdapter._RECORD_METRICS_SCRIPT:
                return self._record_metrics(keys, args)
        raise AssertionError("unexpected Lua script")

    def _create_session(self, keys, args):
        session_key, creation_key = keys
        prefix, payload, _ttl, session_id = args
        mapped = self.values.get(creation_key)
        if mapped and f"{prefix}{mapped}" in self.values:
            return mapped
        if mapped:
            self.values.pop(creation_key, None)
        if session_key in self.values:
            return ""
        self.values[session_key] = payload
        self.values[creation_key] = session_id
        return session_id

    def _claim_request(self, keys, args):
        session_key, claim_key = keys
        request_id, fingerprint, token, _ttl = args
        raw = self.values.get(session_key)
        if raw is None:
            return -2
        try:
            state = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return -3
        stored_fingerprint = state.get("request_fingerprints", {}).get(request_id)
        if stored_fingerprint and stored_fingerprint != fingerprint:
            return -1
        if request_id in state.get("request_results", {}):
            return 2
        existing = self.values.get(claim_key)
        if existing:
            return 0 if existing[:64] == fingerprint else -1
        self.values[claim_key] = f"{fingerprint}:{token}"
        return 1

    def _save_claimed_session(self, keys, args):
        session_key, claim_key = keys
        expected_version, payload, _ttl, expected_claim = args
        raw = self.values.get(session_key)
        if raw is None:
            return -2
        try:
            current = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return -3
        if int(current.get("_version", -1)) != int(expected_version):
            return 0
        if self.values.get(claim_key) != expected_claim:
            return -1
        self.values[session_key] = payload
        self.values.pop(claim_key, None)
        return 1

    def _release_request(self, keys, args):
        claim_key = keys[0]
        if self.values.get(claim_key) != args[0]:
            return 0
        self.values.pop(claim_key, None)
        return 1

    def _upsert_feedback(self, keys, args):
        feedback_key, metrics_key = keys
        field, rating, commented, _ttl = args
        feedback = self.hashes.setdefault(feedback_key, {})
        metrics = self.hashes.setdefault(metrics_key, {})
        previous = feedback.get(field)
        if previous == rating:
            return 0
        if previous in {"up", "down"}:
            old_key = f"feedback_{previous}"
            metrics[old_key] = int(metrics.get(old_key, 0)) - 1
        feedback[field] = rating
        rating_key = f"feedback_{rating}"
        metrics[rating_key] = int(metrics.get(rating_key, 0)) + 1
        if commented == "1" and previous is None:
            metrics["feedback_commented"] = int(metrics.get("feedback_commented", 0)) + 1
        return 1

    def _record_metrics(self, keys, args):
        response_key, metrics_key = keys
        if response_key in self.values:
            return 0
        self.values[response_key] = "1"
        metrics = self.hashes.setdefault(metrics_key, {})
        fields = (
            ("requests", 1),
            ("latency_total_ms", float(args[1])),
            ("citation_covered", int(args[2])),
            ("retrieval_hits", int(args[3])),
            ("blocked_queries", int(args[4])),
            ("structured_successes", int(args[5])),
            ("high_risk", int(args[6])),
            ("safety_reviewed", 1),
            ("safety_blocked", int(args[7])),
        )
        for field, increment in fields:
            metrics[field] = float(metrics.get(field, 0)) + increment
        if str(args[8]) == "1":
            metrics["hallucination_labeled"] = int(metrics.get("hallucination_labeled", 0)) + 1
            metrics["hallucinations"] = int(metrics.get("hallucinations", 0)) + int(args[9])
        return 1


def test_atomic_session_creation_has_one_winner_and_no_orphan_sessions() -> None:
    redis = _AtomicRedis()
    adapter = RedisSessionAdapter(client=redis)

    def create(index: int) -> str:
        state = {"session_id": f"mg_{index:012d}", "owner_id": "owner", "_version": 0}
        return adapter.create_state(state, "browser-create", "owner", 3600)

    with ThreadPoolExecutor(max_workers=12) as pool:
        winners = list(pool.map(create, range(24)))

    assert len(set(winners)) == 1
    session_keys = [
        key
        for key in redis.values
        if key.startswith(adapter.prefix) and not key.startswith(f"{adapter.prefix}request:")
    ]
    assert session_keys == [adapter._key(winners[0])]


def test_session_store_uses_atomic_creation_across_worker_instances() -> None:
    redis = _AtomicRedis()
    stores = [SessionStore(RedisSessionAdapter(client=redis), mode="production") for _ in range(8)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        states = list(
            pool.map(
                lambda store: store.create(owner_id="owner", idempotency_key="shared-create"),
                stores,
            )
        )

    assert len({state["session_id"] for state in states}) == 1
    session_id = states[0]["session_id"]
    assert json.loads(redis.values[f"{RedisSessionAdapter.prefix}{session_id}"])["owner_id"] == "owner"


def test_request_claim_commit_and_result_check_are_consistent() -> None:
    redis = _AtomicRedis()
    adapter = RedisSessionAdapter(client=redis)
    state = {
        "session_id": "mg_request",
        "owner_id": "owner",
        "_version": 0,
        "request_results": {},
        "request_fingerprints": {},
    }
    adapter.create_state(state, "request-session", "owner", 3600)
    fingerprint = _message_fingerprint("我咳嗽")

    assert adapter.claim_request("mg_request", "request-0001", fingerprint, "token-a") == "acquired"
    assert adapter.claim_request("mg_request", "request-0001", fingerprint, "token-b") == "pending"
    assert adapter.claim_request("mg_request", "request-0001", "f" * 64, "token-c") == "conflict"
    assert adapter.release_request_claim("mg_request", "request-0001", fingerprint, "wrong") is False
    assert adapter.release_request_claim("mg_request", "request-0001", fingerprint, "token-a") is True
    assert adapter.claim_request("mg_request", "request-0001", fingerprint, "token-d") == "acquired"

    committed = {
        **state,
        "_version": 1,
        "request_fingerprints": {"request-0001": fingerprint},
        "request_results": {"request-0001": {"session_id": "mg_request", "response_id": "r1"}},
    }
    assert adapter.save_state(
        committed,
        expected_version=1,
        request_claim=("request-0001", fingerprint, "token-d"),
    ) is False
    assert adapter.claim_request("mg_request", "request-0001", fingerprint, "token-e") == "pending"
    assert adapter.save_state(
        committed,
        expected_version=0,
        request_claim=("request-0001", fingerprint, "token-d"),
    ) is True
    assert adapter.claim_request("mg_request", "request-0001", fingerprint, "token-f") == "completed"


def test_cross_worker_duplicate_request_runs_workflow_once_then_replays() -> None:
    redis = _AtomicRedis()
    first_store = SessionStore(RedisSessionAdapter(client=redis), mode="production")
    second_store = SessionStore(RedisSessionAdapter(client=redis), mode="production")
    created = first_store.create(owner_id="owner", idempotency_key="chat-session")
    second_store.create(owner_id="owner", idempotency_key="chat-session")
    started = threading.Event()
    finish = threading.Event()

    class Workflow:
        calls = 0

        def run(self, state):
            self.calls += 1
            started.set()
            assert finish.wait(timeout=2)
            return {
                **state,
                "turn_count": int(state.get("turn_count", 0)) + 1,
                "response": "请继续观察症状。",
                "response_id": "response-1",
                "response_ids": ["response-1"],
                "events": [],
                "citations": [],
            }

    workflow = Workflow()
    first = SimpleNamespace(store=first_store, workflow=workflow, metrics=MetricsStore())
    second = SimpleNamespace(store=second_store, workflow=workflow, metrics=MetricsStore())
    fingerprint = _message_fingerprint("我咳嗽")
    request_state = {
        **created,
        "user_message": "我咳嗽",
        "request_id": "request-0002",
        "_request_fingerprint": fingerprint,
    }

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(_run_chat_transaction, first, dict(request_state))
        assert started.wait(timeout=2)
        with pytest.raises(SessionConflictError, match="正在处理中"):
            _run_chat_transaction(second, dict(request_state))
        finish.set()
        committed = pending.result(timeout=2)

    replay = _run_chat_transaction(second, dict(request_state))
    assert workflow.calls == 1
    assert replay["response_id"] == committed["response_id"] == "response-1"


def test_feedback_upsert_is_atomic_under_concurrent_duplicates_and_changes() -> None:
    redis = _AtomicRedis()
    adapter = RedisSessionAdapter(client=redis)

    with ThreadPoolExecutor(max_workers=16) as pool:
        first_changes = list(
            pool.map(lambda _: adapter.upsert_feedback("session", "message", "up"), range(64))
        )
    with ThreadPoolExecutor(max_workers=16) as pool:
        second_changes = list(
            pool.map(lambda _: adapter.upsert_feedback("session", "message", "down"), range(64))
        )

    assert sum(bool(changed) for changed in first_changes) == 1
    assert sum(bool(changed) for changed in second_changes) == 1
    metrics = redis.hashes[adapter.metrics_key]
    assert metrics["feedback_up"] == 0
    assert metrics["feedback_down"] == 1


def test_shared_metrics_record_is_atomic_and_deduplicated_under_concurrency() -> None:
    redis = _AtomicRedis()
    adapter = RedisSessionAdapter(client=redis)
    state = {
        "response_id": "response-metrics-1",
        "latency_ms": 12.5,
        "citations": [{"id": "doc-1"}],
        "risk_level": "high",
        "safety_blocked": True,
        "hallucination_label": False,
    }

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda _: adapter.record_metrics(state), range(64)))

    assert all(results)
    snapshot = adapter.metrics_snapshot()
    assert snapshot is not None
    assert snapshot["requests"] == 1
    assert snapshot["avg_latency_ms"] == 12.5
    assert snapshot["citation_coverage"] == 1.0
    assert snapshot["high_risk_rate"] == 1.0
    assert snapshot["safety_blocked"] == 1
    assert snapshot["hallucination_labeled"] == 1
