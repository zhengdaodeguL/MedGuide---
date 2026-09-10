from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.bm25 import BM25Index
from app.embeddings import EmbeddingSpec, OpenAIEmbeddingProvider
from app.ingestion import DocumentChunk, MedicalDocumentCleaner, load_catalog, main, upsert_catalog
from app.llm import _normalize_base_url
from app.main import AppServices
from app.models import KnowledgeDocument
from app.retrieval import HybridRetriever, MilvusAdapter


class RecordingEmbeddingProvider:
    def __init__(self) -> None:
        self.spec = EmbeddingSpec("test", "shared-model", "2026-08-30", 8)
        self.calls: list[list[str]] = []

    @property
    def available(self) -> bool:
        return True

    def embed(self, texts):
        values = list(texts)
        self.calls.append(values)
        return [[float(len(text) % 7), 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for text in values]


def test_provider_base_urls_are_normalized_and_endpoint_paths_are_stripped() -> None:
    assert _normalize_base_url("https://provider.example/v1/chat/completions") == "https://provider.example/v1"
    # Gateways have been observed advertising a ``chaat`` completion path in
    # their setup instructions.  A pasted endpoint is an endpoint either way,
    # so it normalizes to the same base URL instead of failing closed on a
    # typo the operator cannot see.
    assert _normalize_base_url("https://provider.example/v1/chaat/completions") == "https://provider.example/v1"
    assert _normalize_base_url("https://provider.example/v1/") == "https://provider.example/v1"


def test_keyword_tier_does_not_call_external_embedding_provider(tmp_path: Path) -> None:
    provider = RecordingEmbeddingProvider()
    chunk = DocumentChunk(
        "disease#chunk-000",
        "disease",
        "咳嗽可先观察症状变化。",
        {"title": "咳嗽资料", "category": "disease", "source": "知识库", "updated_at": "2026-08-30"},
    )
    index_path = tmp_path / "bm25.json"
    BM25Index([chunk], provider.spec).save(index_path)
    retriever = HybridRetriever(
        embedding_provider=provider,
        bm25_index_path=index_path,
        local_vector_search=False,
    )

    results = retriever.search("咳嗽", "disease")

    assert results
    assert provider.calls == []


def test_ingestion_rejects_identity_in_visible_metadata() -> None:
    document = KnowledgeDocument(
        id="phi-metadata",
        title="患者：张三",
        category="disease",
        source="MedGuide 医疗知识库",
        updated_at="2026-06-01",
        text="咳嗽通常需要结合持续时间和伴随症状观察。",
    )

    with pytest.raises(ValueError, match="metadata field title contains sensitive identity"):
        MedicalDocumentCleaner().chunks(document)


@pytest.mark.parametrize("field", ["title", "source"])
@pytest.mark.parametrize("value", [
    "患者教育", "患者安全规范", "病人转诊流程", "家属健康教育",
    "国家卫生健康委患者安全中心", "病历号管理规范", "住院号编码规则",
    "患者编号管理制度", "地址字段格式说明", "电子邮箱格式校验规则",
    "患者教育，安全用药", "家属健康教育；常见问题",
])
def test_ingestion_preserves_clinical_metadata(field: str, value: str) -> None:
    document = KnowledgeDocument(
        id="clinical-metadata", title="咳嗽资料", category="disease",
        source="MedGuide 医疗知识库", updated_at="2026-06-01", text="咳嗽需要观察持续时间。",
    )
    chunks = MedicalDocumentCleaner().chunks(replace(document, **{field: value}))

    assert chunks
    assert all(chunk.metadata[field] == value for chunk in chunks)


@pytest.mark.parametrize("value", [
    "患者：张三", "病人:李四", "家属：王五", "联系人姓名为张三",
    "病历号是MR12345", "病历号MR12345", "患者编号#PT98765", "住院号：20260001",
    "联系电话13800138000", "邮箱：patient@example.com", "微信号contact_123",
    "<b>患者</b>：张三", "患者&#xff1a;张三", "病历号：ＭＲ１２３４５",
])
def test_metadata_rejects_explicit_identifiers_after_normalization(value: str) -> None:
    with pytest.raises(ValueError, match="metadata field source contains sensitive identity"):
        MedicalDocumentCleaner().sanitize_metadata(value, "source")


def test_accepted_metadata_survives_index_and_citations(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    document = KnowledgeDocument(
        id="clinical-metadata", title="患者教育", category="disease",
        source="国家卫生健康委患者安全中心", updated_at="2026-06-01",
        text="咳嗽应结合持续时间和伴随症状观察。",
        source_url="https://www.nhs.uk/conditions/common-cold/",
    )
    provider = RecordingEmbeddingProvider()
    chunks = MedicalDocumentCleaner().chunks(document)
    path = tmp_path / "bm25.json"
    BM25Index(chunks, provider.spec).save(path)
    monkeypatch.setattr(HybridRetriever, "_load_documents", lambda _self: [document])
    retriever = HybridRetriever(embedding_provider=provider, bm25_index_path=path, local_vector_search=False)

    citations = retriever.search("咳嗽", "disease")

    assert citations
    assert citations[0].title == document.title
    assert citations[0].source == document.source
    assert citations[0].source_url == document.source_url
    assert citations[0].as_dict()["source_url"] == document.source_url
    assert provider.calls == []


@pytest.mark.parametrize("value", [
    "http://example.org/reference",
    "https://localhost/reference",
    "https://127.0.0.1/reference",
    "https://10.0.0.1/reference",
    "https://user:password@example.org/reference",
    "https://example.org/reference#fragment",
    "https://example.org:8443/reference",
    "https://single-label/reference",
])
def test_source_url_rejects_non_public_or_ambiguous_targets(value: str) -> None:
    with pytest.raises(ValueError, match="source_url must be a public HTTPS URL"):
        MedicalDocumentCleaner.sanitize_source_url(value)


def test_source_url_rejects_public_but_unapproved_host() -> None:
    with pytest.raises(ValueError, match="not an allowed source"):
        MedicalDocumentCleaner.sanitize_source_url("https://evil.example/phish")


@pytest.mark.parametrize("query", [
    "token=secret",
    "access_token=secret",
    "api_key=secret",
    "api%5fkey=secret",
    "api%255fkey=secret",
    "AWSAccessKeyId=secret",
    "secret=secret",
    "sig=secret",
    "signature=secret",
    "auth=secret",
    "authorization=Bearer%20secret",
    "credential=secret",
    "password=secret",
    "subscription-key=secret",
    "private_key=secret",
    "redirect=https%3A%2F%2Fother.test%2Fx%3Ftoken%3Dsecret",
    "redirect=https%253A%252F%252Fother.test%252Fx%253Fprivate_key%253Dsecret",
    "view=full&token=secret&token=second",
])
def test_source_url_rejects_sensitive_query_parameters(query: str) -> None:
    with pytest.raises(ValueError, match="sensitive query parameter"):
        MedicalDocumentCleaner.sanitize_source_url(f"https://www.nhs.uk/reference?{query}")


def test_source_url_preserves_non_sensitive_query_parameters() -> None:
    value = "https://www.nhs.uk/reference?language=zh-CN&view=full"
    assert MedicalDocumentCleaner.sanitize_source_url(value) == value


def test_bm25_load_rejects_tampered_metadata_and_old_format(tmp_path: Path) -> None:
    provider = RecordingEmbeddingProvider()
    chunk = DocumentChunk(
        "disease#chunk-000",
        "disease",
        "咳嗽可先观察症状变化。",
        {
            "title": "咳嗽资料",
            "category": "disease",
            "source": "知识库",
            "updated_at": "2026-08-30",
            "source_url": "https://www.nhs.uk/reference",
        },
    )
    path = tmp_path / "bm25.json"
    BM25Index([chunk], provider.spec).save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    payload["format_version"] = 1
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported BM25 index format"):
        BM25Index.load(path, provider.spec)

    payload["format_version"] = BM25Index.FORMAT_VERSION
    payload["chunks"][0]["metadata"]["source_url"] = "javascript:alert(1)"
    payload["corpus_generation"] = BM25Index.generation_for([
        DocumentChunk(
            payload["chunks"][0]["id"],
            payload["chunks"][0]["document_id"],
            payload["chunks"][0]["text"],
            payload["chunks"][0]["metadata"],
        )
    ])
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="source_url must be a public HTTPS URL"):
        BM25Index.load(path, provider.spec)


def test_retriever_drops_unsafe_milvus_citation_metadata() -> None:
    provider = RecordingEmbeddingProvider()

    class Remote:
        ready = True
        dimension = 8
        metric_type = "COSINE"

        @staticmethod
        def ensure_ready() -> bool:
            return True

        @staticmethod
        def search(_vector, top_k=4, **_kwargs):
            return [{
                "id": "remote#chunk-001",
                "score": 0.9,
                "document_id": "remote",
                "text": "远端咳嗽资料。",
                "title": "患者：张三",
                "category": "disease",
                "source": "远端来源",
                "updated_at": "2026-08-30",
                "source_url": "https://example.org/reference?token=secret",
            }]

    assert HybridRetriever(milvus=Remote(), embedding_provider=provider).search("咳嗽", "disease") == []


def test_catalog_external_medical_sources_are_traceable() -> None:
    root = Path(__file__).resolve().parents[2]
    documents = load_catalog(root / "data" / "knowledge" / "catalog.json")
    externally_sourced = [document for document in documents if document.id != "faq-boundary-001"]

    assert len(externally_sourced) == len(documents) - 1
    assert externally_sourced
    assert all(document.source_url.startswith("https://") for document in externally_sourced)
    assert {document.source_url.split("/", 3)[2] for document in externally_sourced} <= {
        "www.nhs.uk",
        "medlineplus.gov",
        "www.fda.gov",
        "telehealth.hhs.gov",
        "www.londonambulance.nhs.uk",
    }


class RecordingMilvusAdapter:
    ready = True
    vector_field = "embedding"
    primary_field = "id"
    output_fields = (
        "id",
        "chunk_id",
        "document_id",
        "text",
        "title",
        "category",
        "source",
        "updated_at",
        "source_url",
        "corpus_generation",
    )
    error = None

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.deleted_generations: list[str] = []
        self.closed = False

    def ensure_ready(self):
        return self.ready

    def upsert(self, rows):
        self.rows.extend(rows)
        return len(rows)

    def delete_stale_generations(self, corpus_generation):
        self.deleted_generations.append(corpus_generation)
        return 0

    def close(self):
        self.closed = True


def test_openai_embedding_provider_enforces_model_and_dimension_contract() -> None:
    calls: list[dict] = []

    class Embeddings:
        def create(self, **kwargs):
            calls.append(kwargs)
            if kwargs["input"] == [OpenAIEmbeddingProvider.READINESS_TEXT]:
                return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[0.5] * 8)])
            return SimpleNamespace(
                data=[
                    SimpleNamespace(index=1, embedding=[2.0] * 8),
                    SimpleNamespace(index=0, embedding=[1.0] * 8),
                ]
            )

    provider = OpenAIEmbeddingProvider(
        api_key=None,
        model="text-embedding-3-small",
        version="catalog-v1",
        dimension=8,
        client=SimpleNamespace(embeddings=Embeddings()),
    )

    assert provider.embed(["first", "second"]) == [[1.0] * 8, [2.0] * 8]
    assert calls == [
        {
            "input": [OpenAIEmbeddingProvider.READINESS_TEXT],
            "model": "text-embedding-3-small",
            "dimensions": 8,
        },
        {
            "input": ["first", "second"],
            "model": "text-embedding-3-small",
            "dimensions": 8,
        },
    ]


def test_openai_embedding_provider_recovers_after_runtime_failure(monkeypatch) -> None:
    class BrokenEmbeddings:
        def create(self, **kwargs):
            raise TimeoutError("upstream timeout")

    class HealthyEmbeddings:
        def create(self, **kwargs):
            return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0] * 8)])

    class Client:
        def __init__(self, embeddings) -> None:
            self.embeddings = embeddings
            self.closed = False

        def close(self) -> None:
            self.closed = True

    broken = Client(BrokenEmbeddings())
    healthy = Client(HealthyEmbeddings())
    clients = iter((broken, healthy))
    monkeypatch.setattr(OpenAIEmbeddingProvider, "_create_client", lambda self: next(clients))
    provider = OpenAIEmbeddingProvider(
        api_key="embedding-test-key",
        model="text-embedding-3-small",
        version="catalog-v1",
        dimension=8,
    )

    with pytest.raises(RuntimeError, match="embedding request failed"):
        provider.embed(["first"])
    assert provider.available is False
    assert broken.closed is True

    provider._retry_after = 0.0
    assert provider.ensure_ready() is True
    assert provider.embed(["second"]) == [[1.0] * 8]


def _write_catalog(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "id": "guide-1",
                    "title": "专用词护理",
                    "category": "disease",
                    "source": "测试来源",
                    "updated_at": "2026-08-30",
                    "source_url": "https://www.nhs.uk/guide-1",
                    "text": "蓝桉症状需要记录持续时间并咨询医生。",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_ingestion_and_query_use_one_embedding_contract(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    index_path = tmp_path / "bm25.json"
    _write_catalog(catalog)
    provider = RecordingEmbeddingProvider()

    class Client:
        def __init__(self):
            self.rows: list[dict] = []
            self.searches: list[dict] = []
            self.deletes: list[str] = []

        def has_collection(self, collection_name):
            return True

        def describe_collection(self, collection_name):
            return _collection_description(provider.spec)

        def list_indexes(self, collection_name):
            return ["embedding_idx"]

        def describe_index(self, collection_name, index_name):
            return {"field_name": "embedding", "metric_type": "COSINE"}

        def upsert(self, collection_name, data):
            self.rows.extend(data)
            return {"upsert_count": len(data)}

        def search(self, **kwargs):
            self.searches.append(kwargs)
            return [[]]

        def delete(self, collection_name, filter):
            self.deletes.append(filter)
            return {"delete_count": 0}

    client = Client()
    adapter = MilvusAdapter(client=client, embedding_spec=provider.spec)

    result = upsert_catalog(
        catalog_path=catalog,
        bm25_index_path=index_path,
        adapter=adapter,
        embedding_provider=provider,
    )
    index = BM25Index.load(index_path, provider.spec)
    citations = HybridRetriever(
        milvus=adapter,
        embedding_provider=provider,
        bm25_index=index,
    ).search("蓝桉症状", "disease")

    assert result["upserted"] == 1
    assert provider.calls[0] == ["蓝桉症状需要记录持续时间并咨询医生。"]
    assert client.rows[0]["embedding"][0] == float(len(provider.calls[0][0]) % 7)
    assert client.rows[0]["corpus_generation"] == result["corpus_generation"]
    assert client.rows[0]["source_url"] == "https://www.nhs.uk/guide-1"
    assert "蓝桉" not in provider.calls[-1][0]
    assert "症状" in provider.calls[-1][0]
    assert client.searches[0]["data"][0] == [
        float(len(provider.calls[-1][0]) % 7), 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    assert result["corpus_generation"] in client.searches[0]["filter"]
    assert client.deletes == [f'corpus_generation != "{result["corpus_generation"]}"']
    assert citations and citations[0].retrieval == "bm25"
    assert citations[0].source_url == "https://www.nhs.uk/guide-1"


def test_remote_score_keeps_verified_bm25_metadata_for_same_chunk() -> None:
    provider = RecordingEmbeddingProvider()
    chunk = DocumentChunk(
        "guide-1#chunk-000",
        "guide-1",
        "蓝桉症状需要记录持续时间并咨询医生。",
        {
            "title": "已审核护理资料",
            "category": "disease",
            "source": "NHS",
            "updated_at": "2026-08-30",
            "source_url": "https://www.nhs.uk/guide-1",
        },
    )
    index = BM25Index([chunk], provider.spec)

    class Remote:
        ready = True
        metric_type = "COSINE"
        embedding_spec = provider.spec

        @staticmethod
        def ensure_ready() -> bool:
            return True

        @staticmethod
        def search(_vector, top_k=4, **_kwargs):
            return [{
                "id": chunk.id,
                "document_id": chunk.document_id,
                "text": "旧向量库正文",
                "title": "旧向量库标题",
                "category": "disease",
                "source": "旧向量库来源",
                "updated_at": "2020-01-01",
                "source_url": "",
                "corpus_generation": index.corpus_generation,
                "score": 0.9,
            }]

    citations = HybridRetriever(
        milvus=Remote(),
        embedding_provider=provider,
        bm25_index=index,
    ).search("蓝桉症状", "disease")

    assert citations[0].retrieval == "hybrid"
    assert citations[0].title == "已审核护理资料"
    assert citations[0].source_url == "https://www.nhs.uk/guide-1"


def test_ingestion_probes_embedding_provider_before_availability_check(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    index_path = tmp_path / "bm25.json"
    _write_catalog(catalog)

    class InitiallyUnavailableProvider(RecordingEmbeddingProvider):
        def __init__(self) -> None:
            super().__init__()
            self.ready = False
            self.probes = 0

        @property
        def available(self) -> bool:
            return self.ready

        def ensure_ready(self) -> bool:
            self.probes += 1
            self.ready = True
            return True

    provider = InitiallyUnavailableProvider()
    result = upsert_catalog(
        catalog_path=catalog,
        bm25_index_path=index_path,
        adapter=RecordingMilvusAdapter(),
        embedding_provider=provider,
    )

    assert provider.probes == 1
    assert provider.calls
    assert result["upserted"] == 1


def test_bm25_recall_is_independent_of_milvus_hits() -> None:
    provider = RecordingEmbeddingProvider()
    chunk = DocumentChunk(
        "remote#chunk-000",
        "remote",
        "罕见专用词蓝桉的资料。",
        {"title": "蓝桉资料", "category": "disease", "source": "生产索引", "updated_at": "2026-08-30"},
    )

    class EmptyMilvus:
        ready = True
        dimension = 8
        metric_type = "COSINE"

        def ensure_ready(self):
            return True

        def search(self, vector, top_k):
            return []

    results = HybridRetriever(
        milvus=EmptyMilvus(),
        embedding_provider=provider,
        bm25_index=BM25Index([chunk], provider.spec),
    ).search("蓝桉", "disease")

    assert [result.id for result in results] == [chunk.id]
    assert results[0].retrieval == "bm25"


def _collection_description(spec: EmbeddingSpec) -> dict:
    return {
        "auto_id": False,
        "fields": [
            {"name": "id", "data_type": "Int64", "is_primary": True},
            {"name": "chunk_id", "data_type": "VarChar"},
            {"name": "embedding", "data_type": "FloatVector", "params": {"dim": spec.dimension}},
            {"name": "document_id", "data_type": "VarChar"},
            {"name": "text", "data_type": "VarChar"},
            {"name": "title", "data_type": "VarChar"},
            {"name": "category", "data_type": "VarChar"},
            {"name": "source", "data_type": "VarChar"},
            {"name": "updated_at", "data_type": "VarChar"},
            {"name": "source_url", "data_type": "VarChar"},
            {"name": "corpus_generation", "data_type": "VarChar"},
        ],
        "properties": spec.collection_properties(),
    }


def test_milvus_validates_collection_embedding_model_and_version() -> None:
    spec = EmbeddingSpec("openai", "text-embedding-3-small", "catalog-v4", 8)

    class Client:
        def __init__(self, description):
            self.description = description

        def has_collection(self, collection_name):
            return True

        def describe_collection(self, collection_name):
            return self.description

        def list_indexes(self, collection_name):
            return ["embedding_idx"]

        def describe_index(self, collection_name, index_name):
            return {"field_name": "embedding", "metric_type": "COSINE"}

    assert MilvusAdapter(client=Client(_collection_description(spec)), embedding_spec=spec).ready is True

    wrong = _collection_description(spec)
    wrong["properties"]["medguide.embedding_version"] = "catalog-v3"
    adapter = MilvusAdapter(client=Client(wrong), embedding_spec=spec)
    assert adapter.ready is False
    assert adapter.error == "MilvusEmbeddingContractMismatch"

    missing_join_key = _collection_description(spec)
    missing_join_key["fields"] = [
        field for field in missing_join_key["fields"] if field["name"] != "chunk_id"
    ]
    adapter = MilvusAdapter(client=Client(missing_join_key), embedding_spec=spec)
    assert adapter.ready is False
    assert adapter.error == "MilvusSchemaMismatch"

    missing_source_url = _collection_description(spec)
    missing_source_url["fields"] = [
        field for field in missing_source_url["fields"] if field["name"] != "source_url"
    ]
    adapter = MilvusAdapter(client=Client(missing_source_url), embedding_spec=spec)
    assert adapter.ready is False
    assert adapter.error == "MilvusSchemaMismatch"

    auto_id = _collection_description(spec)
    auto_id["auto_id"] = True
    adapter = MilvusAdapter(client=Client(auto_id), embedding_spec=spec)
    assert adapter.ready is False
    assert adapter.error == "MilvusSchemaMismatch"


def test_milvus_revalidates_embedding_contract_before_runtime_search() -> None:
    spec = EmbeddingSpec("openai", "text-embedding-3-small", "catalog-v4", 8)

    class Client:
        def __init__(self):
            self.description = _collection_description(spec)
            self.search_calls = 0

        def has_collection(self, collection_name):
            return True

        def describe_collection(self, collection_name):
            return self.description

        def list_indexes(self, collection_name):
            return ["embedding_idx"]

        def describe_index(self, collection_name, index_name):
            return {"field_name": "embedding", "metric_type": "COSINE"}

        def search(self, **kwargs):
            self.search_calls += 1
            return [[]]

    client = Client()
    adapter = MilvusAdapter(client=client, embedding_spec=spec)
    assert adapter.ready is True

    client.description["properties"]["medguide.embedding_version"] = "catalog-v5"

    assert adapter.search([0.0] * 8, corpus_generation="generation-v1") == []
    assert adapter.ready is False
    assert adapter.error == "MilvusEmbeddingContractMismatch"
    assert client.search_calls == 0


def test_production_readiness_forces_milvus_contract_revalidation() -> None:
    class Dependency:
        available = True
        ready = True

        def __init__(self):
            self.calls: list[dict] = []

        def ensure_ready(self, **kwargs):
            self.calls.append(kwargs)
            return self.ready

    current = AppServices.__new__(AppServices)
    current.production = True
    current.answerer = Dependency()
    current.embedding_provider = Dependency()
    current.milvus = Dependency()
    current.mysql = Dependency()
    current.redis = Dependency()
    current.retriever = SimpleNamespace(keyword_ready=True)
    current.api_tokens = ("configured",)

    assert current.production_missing == ()
    assert current.milvus.calls == [{"revalidate": True}]

def test_production_milvus_fails_closed_without_index_introspection() -> None:
    spec = EmbeddingSpec("openai", "text-embedding-3-small", "catalog-v4", 8)

    class Client:
        def has_collection(self, collection_name):
            return True

        def describe_collection(self, collection_name):
            return _collection_description(spec)

    adapter = MilvusAdapter(client=Client(), embedding_spec=spec)

    assert adapter.ready is False
    assert adapter.error == "MilvusMetricMismatch"


def test_external_query_embedding_receives_deidentified_text() -> None:
    provider = RecordingEmbeddingProvider()
    chunk = DocumentChunk(
        "remote#chunk-000",
        "remote",
        "咳嗽持续三天需要记录伴随症状。",
        {"title": "咳嗽资料", "category": "disease", "source": "生产索引", "updated_at": "2026-08-30"},
    )

    class EmptyMilvus:
        ready = True
        dimension = 8
        metric_type = "COSINE"

        def ensure_ready(self):
            return True

        def search(self, vector, top_k):
            return []

    retriever = HybridRetriever(
        milvus=EmptyMilvus(),
        embedding_provider=provider,
        bm25_index=BM25Index([chunk], provider.spec),
    )
    retriever.search(
        "姓名：张三，邮箱 zhangsan@example.com，地址：北京市朝阳区建国路88号，咳嗽持续三天",
        "disease",
    )

    outbound = provider.calls[-1][0]
    assert "张三" not in outbound
    assert "zhangsan@example.com" not in outbound
    assert "北京市朝阳区建国路88号" not in outbound
    assert "已脱敏" in outbound


def test_failed_vector_upsert_does_not_publish_bm25_snapshot(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    index_path = tmp_path / "bm25.json"
    _write_catalog(catalog)
    provider = RecordingEmbeddingProvider()

    class PartialAdapter(RecordingMilvusAdapter):
        def upsert(self, rows):
            return 0

    with pytest.raises(RuntimeError, match="acknowledged 0"):
        upsert_catalog(
            catalog_path=catalog,
            bm25_index_path=index_path,
            adapter=PartialAdapter(),
            embedding_provider=provider,
        )
    assert not index_path.exists()


def test_second_batch_failure_keeps_previous_generation_active(tmp_path: Path) -> None:
    provider = RecordingEmbeddingProvider()
    index_path = tmp_path / "bm25.json"
    old_chunk = DocumentChunk(
        "old#chunk-000",
        "old",
        "旧版资料",
        {"title": "旧版", "category": "faq", "source": "测试", "updated_at": "2026-08-01"},
    )
    old_index = BM25Index([old_chunk], provider.spec)
    old_index.save(index_path)
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            [
                {
                    "id": f"new-{index}",
                    "title": f"新版 {index}",
                    "category": "faq",
                    "source": "测试",
                    "updated_at": "2026-08-30",
                    "text": f"新版资料 {index}",
                }
                for index in (1, 2)
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    class FailsSecondBatch(RecordingMilvusAdapter):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def upsert(self, rows):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("second batch failed")
            return super().upsert(rows)

    adapter = FailsSecondBatch()
    with pytest.raises(RuntimeError, match="second batch failed"):
        upsert_catalog(
            catalog_path=catalog,
            bm25_index_path=index_path,
            adapter=adapter,
            embedding_provider=provider,
            batch_size=1,
        )

    still_active = BM25Index.load(index_path, provider.spec)
    assert still_active.corpus_generation == old_index.corpus_generation
    assert [chunk.id for chunk in still_active.chunks] == [old_chunk.id]
    assert len(adapter.rows) == 1
    assert adapter.rows[0]["corpus_generation"] != old_index.corpus_generation
    assert adapter.deleted_generations == []


def test_retriever_hot_loads_new_bm25_generation(tmp_path: Path) -> None:
    provider = RecordingEmbeddingProvider()
    index_path = tmp_path / "bm25.json"

    class Remote:
        ready = True
        dimension = 8
        metric_type = "COSINE"
        embedding_spec = provider.spec

        def ensure_ready(self):
            return True

        def search(self, vector, top_k, corpus_generation=None):
            return []

    retriever = HybridRetriever(
        milvus=Remote(),
        embedding_provider=provider,
        bm25_index_path=index_path,
    )
    assert retriever.keyword_ready is False

    first = DocumentChunk(
        "first#chunk-000",
        "first",
        "蓝桉第一版资料。",
        {"title": "第一版", "category": "disease", "source": "测试", "updated_at": "2026-08-29"},
    )
    BM25Index([first], provider.spec).save(index_path)
    assert retriever.keyword_ready is True
    assert [item.id for item in retriever.search("蓝桉第一版", "disease")] == [first.id]

    second = DocumentChunk(
        "second#chunk-000",
        "second",
        "蓝桉第二版资料，第一版已经撤回。",
        {"title": "第二版", "category": "disease", "source": "测试", "updated_at": "2026-08-30"},
    )
    BM25Index([second], provider.spec).save(index_path)
    assert [item.id for item in retriever.search("蓝桉第二版", "disease")] == [second.id]


def test_retriever_retries_when_generation_changes_during_search(tmp_path: Path) -> None:
    provider = RecordingEmbeddingProvider()
    index_path = tmp_path / "bm25.json"
    first = DocumentChunk(
        "first#chunk-000",
        "first",
        "蓝桉旧版资料。",
        {"title": "旧版", "category": "disease", "source": "测试", "updated_at": "2026-08-29"},
    )
    second = DocumentChunk(
        "second#chunk-000",
        "second",
        "蓝桉新版资料。",
        {"title": "新版", "category": "disease", "source": "测试", "updated_at": "2026-08-30"},
    )
    first_index = BM25Index([first], provider.spec)
    second_index = BM25Index([second], provider.spec)
    first_index.save(index_path)

    class Remote:
        ready = True
        dimension = 8
        metric_type = "COSINE"
        embedding_spec = provider.spec

        def __init__(self):
            self.generations: list[str] = []

        def ensure_ready(self):
            return True

        def search(self, vector, top_k, corpus_generation=None):
            self.generations.append(corpus_generation)
            if len(self.generations) == 1:
                second_index.save(index_path)
            return []

    remote = Remote()
    retriever = HybridRetriever(
        milvus=remote,
        embedding_provider=provider,
        bm25_index_path=index_path,
    )

    assert [item.id for item in retriever.search("蓝桉资料", "disease")] == [second.id]
    assert remote.generations == [first_index.corpus_generation, second_index.corpus_generation]


def test_ingestion_cli_is_executable_with_injected_adapters(tmp_path: Path, monkeypatch, capsys) -> None:
    catalog = tmp_path / "catalog.json"
    index_path = tmp_path / "bm25.json"
    _write_catalog(catalog)
    provider = RecordingEmbeddingProvider()
    adapter = RecordingMilvusAdapter()
    monkeypatch.setenv("MILVUS_URI", "http://milvus.test")
    monkeypatch.setattr("app.ingestion.embedding_provider_from_env", lambda production: provider)
    monkeypatch.setattr("app.retrieval.MilvusAdapter", lambda **kwargs: adapter)

    assert main(["--catalog", str(catalog), "--bm25-index", str(index_path)]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["upserted"] == 1
    assert index_path.exists()
    assert not index_path.with_name(f".{index_path.name}.lock").exists()
    assert adapter.closed is True


def test_bm25_snapshot_rejects_another_embedding_version(tmp_path: Path) -> None:
    first = EmbeddingSpec("test", "shared-model", "v1", 8)
    second = EmbeddingSpec("test", "shared-model", "v2", 8)
    chunk = DocumentChunk(
        "chunk-1",
        "doc-1",
        "测试内容",
        {"title": "测试", "category": "faq", "source": "测试", "updated_at": "2026-08-30"},
    )
    path = tmp_path / "bm25.json"
    BM25Index([chunk], first).save(path)

    with pytest.raises(ValueError, match="does not match"):
        BM25Index.load(path, second)
