from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import app.main as main_module
import pytest
from fastapi.testclient import TestClient
from app.ingestion import DocumentChunk, MedicalDocumentCleaner, load_catalog
from app.main import _response_from_state, _sanitize_citation_payload, app, public_state, stream_chat
from app.models import Citation
from app.store import SessionStore


def _citation(**overrides: object) -> dict[str, object]:
    citation: dict[str, object] = {
        "id": "guide-1#chunk-000",
        "title": "<b>咳嗽资料</b>",
        "category": "disease",
        "source": " NHS ",
        "updated_at": " 2026-09-08 ",
        "snippet": "联系电话：13800138000；<p>咳嗽参考。</p>",
        "score": 0.9,
        "retrieval": "bm25",
        "source_url": " https://www.nhs.uk/conditions/cough/ ",
        "internal": "must-not-leak",
    }
    citation.update(overrides)
    return citation


def _unsafe_citations() -> list[dict[str, object]]:
    return [
        _citation(id="bad-title", title="患者：张三"),
        _citation(id="bad-source", source="邮箱：patient@example.org"),
        _citation(id="bad-updated-at", updated_at="联系电话：13800138000"),
        _citation(id="bad-snippet", snippet="<script>secret</script>"),
        _citation(id="bad-source-url", source_url="https://www.nhs.uk/guide?api%255fkey=secret"),
        _citation(id="bad-category", category="patient@example.org"),
        _citation(id="bad-retrieval", retrieval="untrusted"),
        _citation(id="bad-score", score=1.1),
        _citation(id="患者：张三"),
    ]


def test_citation_payload_sanitizer_revalidates_visible_fields_and_drops_invalid_items() -> None:
    stored = [_citation(), *_unsafe_citations()]
    original = deepcopy(stored)

    sanitized = _sanitize_citation_payload(stored)

    assert stored == original
    assert sanitized == [
        {
            "id": "guide-1#chunk-000",
            "title": "咳嗽资料",
            "category": "disease",
            "source": "NHS",
            "updated_at": "2026-09-08",
            "snippet": "联系电话：[已脱敏]； 咳嗽参考。",
            "score": 0.9,
            "retrieval": "bm25",
            "source_url": "https://www.nhs.uk/conditions/cough/",
        }
    ]


@pytest.mark.parametrize(
    "query",
    [
        "token=secret",
        "api_key=secret",
        "api%255fkey=secret",
        "view=full&token=first&token=second",
    ],
)
def test_citation_payload_sanitizer_drops_source_url_query_credentials(query: str) -> None:
    assert _sanitize_citation_payload([_citation(source_url=f"https://example.org/guide?{query}")]) == []


@pytest.mark.parametrize(
    "value",
    ["", " leading-space", "patient@example.org", "患者-001", "a/b", "a" * 257, 123],
)
def test_opaque_citation_id_rejects_identity_and_non_ascii_values(value: object) -> None:
    assert _sanitize_citation_payload([_citation(id=value)]) == []


def test_catalog_and_stored_chunks_share_opaque_id_validation(tmp_path) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            [{
                "id": "patient@example.org",
                "title": "咳嗽资料",
                "category": "disease",
                "source": "NHS",
                "updated_at": "2026-09-08",
                "text": "咳嗽需要观察持续时间。",
                "source_url": "https://www.nhs.uk/conditions/cough/",
            }],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="knowledge document id must be .* ASCII opaque ID"):
        load_catalog(catalog)

    chunk = DocumentChunk(
        id="patient@example.org",
        document_id="guide-1",
        text="咳嗽需要观察持续时间。",
        metadata={
            "title": "咳嗽资料",
            "category": "disease",
            "source": "NHS",
            "updated_at": "2026-09-08",
            "source_url": "https://www.nhs.uk/conditions/cough/",
        },
    )
    with pytest.raises(ValueError, match="knowledge chunk id must be .* ASCII opaque ID"):
        MedicalDocumentCleaner().sanitize_chunk(chunk)


def test_search_sanitizes_citations_before_computing_count(monkeypatch) -> None:
    valid = Citation(
        id="guide-1#chunk-000",
        title="咳嗽资料",
        category="disease",
        source="NHS",
        updated_at="2026-09-08",
        snippet="咳嗽需要观察持续时间。",
        score=0.9,
        retrieval="bm25",
        source_url="https://www.nhs.uk/conditions/cough/",
    )
    invalid = Citation(**{**valid.__dict__, "id": "patient@example.org"})

    with TestClient(app, base_url="https://testserver") as client:
        monkeypatch.setattr(main_module.get_services().retriever, "search", lambda *_args, **_kwargs: [valid, invalid])
        response = client.post("/api/search", json={"query": "咳嗽", "intent": "disease"})

    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert [item["id"] for item in response.json()["citations"]] == [valid.id]


def test_response_and_public_state_share_sanitized_citations_without_mutating_state(monkeypatch) -> None:
    current = SimpleNamespace(store=SessionStore(mode="test"))
    monkeypatch.setattr(main_module, "get_services", lambda: current)
    stored = {
        "session_id": "mg-citation-response",
        "response_id": "response-1",
        "response": "请观察症状变化。",
        "citations": [_citation(), *_unsafe_citations()],
    }
    original = deepcopy(stored)

    response = _response_from_state(stored)
    exposed_state = public_state(stored)

    assert response.citations == response.state["citations"] == exposed_state["citations"]
    assert [item["id"] for item in response.citations] == ["guide-1#chunk-000"]
    assert stored == original


def test_sse_final_uses_one_sanitized_citation_view_and_preserves_stored_state(monkeypatch) -> None:
    class Workflow:
        @staticmethod
        def run(state: dict, **_kwargs: object) -> dict:
            return {
                **state,
                "response_id": "response-1",
                "response": "请观察症状变化。",
                "citations": [_citation(), *_unsafe_citations()],
                "events": [],
            }

    store = SessionStore(mode="test")
    current = SimpleNamespace(
        store=store,
        workflow=Workflow(),
        metrics=SimpleNamespace(record=lambda _state: None),
    )
    monkeypatch.setattr(main_module, "get_services", lambda: current)
    state = store.create()
    state["user_message"] = "我咳嗽"

    async def collect_stream() -> str:
        response = stream_chat(current, state)
        return "".join([chunk async for chunk in response.body_iterator])

    stream_body = asyncio.run(collect_stream())
    final_frame = next(frame for frame in stream_body.split("\n\n") if frame.startswith("event: final"))
    payload = json.loads(next(line[6:] for line in final_frame.splitlines() if line.startswith("data: ")))
    persisted = store.get(state["session_id"])

    assert payload["citations"] == payload["state"]["citations"]
    assert [item["id"] for item in payload["citations"]] == ["guide-1#chunk-000"]
    assert persisted is not None
    assert len(persisted["citations"]) == 10
    assert persisted["citations"][0]["title"] == "<b>咳嗽资料</b>"
    unsafe_url = next(item for item in persisted["citations"] if item["id"] == "bad-source-url")
    assert unsafe_url["source_url"].endswith("api%255fkey=secret")
