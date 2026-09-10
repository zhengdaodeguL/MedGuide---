from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.llm import OpenAIAnswerer
from app.metrics import MetricsStore
from app.models import KnowledgeDocument
from app.retrieval import HybridRetriever
from app.safety import SafetyEngine
from app.store import SessionConflictError, SessionStore
from app.sql_guard import ReadOnlySQLGuard
from app.workflow import MedGuideWorkflow


def make_workflow(*, answerer: OpenAIAnswerer | None = None) -> MedGuideWorkflow:
    return MedGuideWorkflow(HybridRetriever(), ReadOnlySQLGuard(), SafetyEngine(), answerer)


def test_high_risk_is_sticky_across_follow_up_turns() -> None:
    workflow = make_workflow()
    first = workflow.run(
        {
            "session_id": "sticky",
            "user_message": "突然胸痛并且呼吸困难",
            "profile": {"symptoms": [], "associated_symptoms": []},
        }
    )
    first["user_message"] = "现在好多了"
    second = workflow.run(first)

    assert first["risk_level"] == "high"
    assert second["risk_level"] == "high"
    assert "胸痛或胸部压榨感" in second["risk_flags"]
    assert second["citations"] == []
    assert "急诊" in second["response"]


class _FakeCompletions:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.messages: list[dict] | None = None

    def create(self, **kwargs):
        if self.error:
            raise self.error
        self.messages = kwargs["messages"]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="  grounded answer  "))])


def _fake_answerer(completions: _FakeCompletions) -> OpenAIAnswerer:
    answerer = OpenAIAnswerer.__new__(OpenAIAnswerer)
    answerer.model = "test-model"
    answerer.max_tokens = 1600
    answerer._closed = False
    answerer.available = True
    answerer.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    answerer.prompt = SimpleNamespace(
        format_messages=lambda **_: [
            SimpleNamespace(type="system", content="system"),
            SimpleNamespace(type="human", content="user"),
        ]
    )
    return answerer


def test_openai_maps_langchain_human_role_to_user() -> None:
    completions = _FakeCompletions()
    answerer = _fake_answerer(completions)

    assert answerer.generate("query", "low", "context") == "grounded answer"
    assert completions.messages is not None
    assert [message["role"] for message in completions.messages] == ["system", "user"]


def test_openai_provider_error_returns_safe_fallback_signal() -> None:
    answerer = _fake_answerer(_FakeCompletions(error=TimeoutError("upstream timeout")))
    assert answerer.generate("query", "low", "context") is None


def test_retrieval_uses_clean_chunks_for_index_and_snippet(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    private_text = "咳嗽护理建议。" + ("补充说明。" * 80) + "目标症状词。联系电话 13800138000，身份证 11010519491231002X。"
    (knowledge_dir / "catalog.json").write_text(
        json.dumps(
            [
                {
                    "id": "private-doc",
                    "title": "清洗测试",
                    "category": "disease",
                    "source": "测试来源",
                    "updated_at": "2026-01-01",
                    "text": private_text,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    retriever = HybridRetriever(knowledge_dir)
    results = retriever.search("目标症状词", "disease", top_k=2)

    assert results
    assert all("13800138000" not in item.snippet for item in results)
    assert all("11010519491231002X" not in item.snippet for item in results)
    assert all("[已脱敏]" in item.snippet for item in results)
    assert len(retriever.chunks) > 1


def test_unrelated_or_blank_query_has_no_citations() -> None:
    retriever = HybridRetriever()
    assert retriever.search("   ", "unknown") == []
    assert retriever.search("xyzfoobar-not-medical", "unknown") == []


def test_safety_does_not_upgrade_negation_or_knowledge_question() -> None:
    engine = SafetyEngine()
    assert engine.screen("没有胸痛，也不喘").level != "high"
    assert engine.screen("胸痛有哪些表现？").level != "high"
    assert engine.screen("胸口压迫感，喘不上气").level == "high"
    assert engine.screen("体温39.5℃").level == "high"
    assert engine.screen("胸痛怎么办").level == "high"


def test_safety_review_fails_closed_for_diagnosis_and_medication_instructions() -> None:
    workflow = make_workflow()
    result = workflow.safety_review(
        {
            "risk_level": "low",
            "response": "你诊断为肺炎，可以停药并服用布洛芬 600mg，每日三次。",
        }
    )

    response = result["response"]
    assert "肺炎" not in response
    assert "600mg" not in response
    assert "不能提供诊断、处方或停药建议" in response


@pytest.mark.parametrize(
    "unsafe",
    [
        "你有肺炎",
        "请停用布洛芬",
        "停止服用布洛芬",
        "诊 断 为 肺炎",
        "布洛芬每次600毫克",
        "每天吃三片",
        "布洛芬每天吃三片",
        "建议停服布洛芬",
        "布洛芬每八小时服一片",
        "不要停药，应咨询医生。",
        "不要停用布洛芬",
        "不建议停药",
        "不要每天吃三片",
    ],
)
def test_safety_review_catches_common_instruction_variants(unsafe: str) -> None:
    workflow = make_workflow()
    result = workflow.safety_review({"risk_level": "low", "response": unsafe})
    assert result["safety_blocked"] is True


def test_safety_review_keeps_explicit_refusal_text() -> None:
    workflow = make_workflow()
    for response in (
        "不能自行停药，请咨询医生或药师。",
        "不要自行停药",
        "不要自行停服布洛芬",
        "不建议自行服用布洛芬",
    ):
        result = workflow.safety_review({"risk_level": "low", "response": response})
        assert result["safety_blocked"] is False, response


def test_session_store_rejects_stale_compare_and_swap() -> None:
    store = SessionStore()
    created = store.create()
    first = store.get(created["session_id"])
    second = store.get(created["session_id"])
    assert first is not None and second is not None

    first["user_message"] = "first"
    saved = store.save(first)
    assert saved["_version"] == 1

    second["user_message"] = "stale"
    with pytest.raises(SessionConflictError):
        store.save(second)


def test_metrics_do_not_count_blocked_or_empty_results_as_citation_coverage() -> None:
    metrics = MetricsStore()
    metrics.record({"latency_ms": 1, "citations": [], "structured_result": {"blocked": True, "rows": []}})
    metrics.record({"latency_ms": 1, "citations": [], "structured_result": {"blocked": False, "rows": []}})
    metrics.record({"latency_ms": 1, "citations": [{"id": "x", "title": "t", "source": "s", "snippet": "fact", "score": 0.4}]})

    snapshot = metrics.snapshot()
    assert snapshot["citation_coverage"] == pytest.approx(1 / 3, abs=0.001)
    assert snapshot["blocked_queries"] == 1
    assert snapshot["hallucination_rate"] is None
    assert snapshot["hallucination_label_status"].endswith("unavailable")


def test_stream_chat_emits_node_before_slow_workflow_finishes() -> None:
    """Regression guard for the endpoint's thread/queue handoff.

    This test is kept as a small executable probe rather than depending on a
    real LLM or external service.
    """
    from app.main import stream_chat

    class SlowWorkflow:
        def run(self, state, on_event=None, cancel_event=None):
            if on_event:
                on_event({"type": "node", "node": "normalize", "status": "completed"})
            time.sleep(0.15)
            return {**state, "response": "ok", "events": [], "citations": [], "confidence": 0.1, "latency_ms": 150, "summary": ""}

    store = SessionStore()
    state = store.create()
    state["user_message"] = "咳嗽"
    current = SimpleNamespace(
        workflow=SlowWorkflow(),
        metrics=MetricsStore(),
        store=store,
    )

    async def first_event() -> tuple[str, float]:
        response = stream_chat(current, state)
        started = time.perf_counter()
        async for chunk in response.body_iterator:
            return chunk, time.perf_counter() - started
        raise AssertionError("stream ended without an event")

    chunk, elapsed = asyncio.run(first_event())
    assert "event: node" in chunk
    assert elapsed < 0.12


def test_disconnected_stream_cancels_at_node_boundary_without_saving() -> None:
    from app.main import stream_chat
    from app.workflow import WorkflowCancelled

    finished = threading.Event()

    class CancellableWorkflow:
        def run(self, state, on_event=None, cancel_event=None):
            try:
                for index in range(20):
                    if cancel_event is not None and cancel_event.is_set():
                        raise WorkflowCancelled()
                    if on_event:
                        on_event({"type": "node", "node": f"node-{index}", "status": "completed"})
                    time.sleep(0.02)
                return {**state, "response": "should not save", "events": []}
            finally:
                finished.set()

    store = SessionStore()
    state = store.create()
    state["user_message"] = "咳嗽"
    current = SimpleNamespace(workflow=CancellableWorkflow(), metrics=MetricsStore(), store=store)

    async def disconnect() -> None:
        response = stream_chat(current, state)
        iterator = response.body_iterator
        first = await anext(iterator)
        assert "event: node" in first
        await iterator.aclose()
        for _ in range(50):
            if finished.is_set():
                break
            await asyncio.sleep(0.01)

    asyncio.run(disconnect())
    assert finished.is_set()
    persisted = store.get(state["session_id"])
    assert persisted is not None
    assert persisted["_version"] == 0
    assert persisted.get("response") is None


def test_stream_disconnect_after_last_node_still_prevents_commit() -> None:
    from app.main import stream_chat

    allow_return = threading.Event()
    finished = threading.Event()

    class LastNodeWorkflow:
        def run(self, state, on_event=None, cancel_event=None):
            try:
                if on_event:
                    on_event({"type": "node", "node": "finalize", "status": "completed"})
                assert allow_return.wait(timeout=1)
                return {**state, "response": "must not save", "events": [], "citations": []}
            finally:
                finished.set()

    store = SessionStore()
    state = store.create()
    state["user_message"] = "咳嗽"
    current = SimpleNamespace(workflow=LastNodeWorkflow(), metrics=MetricsStore(), store=store)

    async def disconnect() -> None:
        response = stream_chat(current, state)
        iterator = response.body_iterator
        first = await anext(iterator)
        assert "event: node" in first
        await iterator.aclose()
        allow_return.set()
        for _ in range(50):
            if finished.is_set():
                break
            await asyncio.sleep(0.01)

    asyncio.run(disconnect())
    assert finished.is_set()
    persisted = store.get(state["session_id"])
    assert persisted is not None
    assert persisted["_version"] == 0
    assert persisted.get("response") is None


def test_chat_transaction_serializes_same_session_updates() -> None:
    from app.main import _run_chat_transaction

    store = SessionStore()
    created = store.create()
    workflow = make_workflow()
    current = SimpleNamespace(store=store, workflow=workflow, metrics=MetricsStore())
    first = {**created, "user_message": "我32岁，咳嗽持续3天"}
    second = {**created, "user_message": "同时还有腹痛"}

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda item: _run_chat_transaction(current, item), (first, second)))

    persisted = store.get(created["session_id"])
    assert persisted is not None
    assert persisted["turn_count"] == 2
    assert persisted["_version"] == 2
    assert set(persisted["profile"]["symptoms"]) >= {"咳嗽", "腹痛"}
    assert sorted(result["turn_count"] for result in results) == [1, 2]


def test_mysql_backend_is_used_after_guard_validation() -> None:
    class FakeBackend:
        ready = True

        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple]] = []

        def execute(self, sql: str, params: tuple = ()):
            self.calls.append((sql, params))
            return ("name", "stock"), ({"name": "生产药品", "stock": 7},)

    backend = FakeBackend()
    result = ReadOnlySQLGuard(backend=backend).execute(
        "drug_inventory",
        "SELECT name, stock FROM drug_inventory WHERE name LIKE ? LIMIT 10",
        ("%药品%",),
    )

    assert result.blocked is False
    assert result.rows == ({"name": "生产药品", "stock": 7},)
    assert backend.calls
