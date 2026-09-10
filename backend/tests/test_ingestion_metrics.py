from app.ingestion import MedicalDocumentCleaner
from app.metrics import MetricsStore
from app.models import KnowledgeDocument


def test_cleaner_redacts_identifiers_and_chunks_metadata() -> None:
    cleaner = MedicalDocumentCleaner()
    doc = KnowledgeDocument("d1", "测试", "faq", "测试源", "2026-01-01", "<p>联系电话 13800138000。</p>" * 12)
    chunks = cleaner.chunks(doc, chunk_size=80, overlap=10)
    assert chunks
    assert all("13800138000" not in chunk.text for chunk in chunks)
    assert chunks[0].metadata["source"] == "测试源"


def test_cleaner_redacts_inline_name_residence_and_landline() -> None:
    cleaned = MedicalDocumentCleaner().clean(
        "患者张三现住北京市朝阳区建国路88号，联系电话010-12345678，主诉咳嗽。"
    )
    assert "张三" not in cleaned
    assert "北京市朝阳区建国路88号" not in cleaned
    assert "010-12345678" not in cleaned
    assert "咳嗽" in cleaned


def test_metrics_snapshot_tracks_coverage_and_feedback() -> None:
    metrics = MetricsStore()
    metrics.record({"latency_ms": 20.0, "citations": [{"id": "x"}], "risk_level": "low"})
    metrics.record({"latency_ms": 40.0, "citations": [], "risk_level": "high"})
    metrics.add_feedback("up", "清楚")
    snapshot = metrics.snapshot()
    assert snapshot["requests"] == 2
    assert snapshot["avg_latency_ms"] == 30.0
    assert snapshot["citation_coverage"] == 0.5
    assert snapshot["retrieval_hit_rate"] == 0.5
    assert snapshot["hallucination_label_status"].startswith("unlabeled")
    assert snapshot["feedback"]["up"] == 1
