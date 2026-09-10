from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from app.infra import MySQLReadOnlyAdapter, RedisSessionAdapter
from app.llm import OpenAIAnswerer
from app.retrieval import HybridRetriever, MilvusAdapter
from app.sql_guard import ReadOnlySQLGuard
from app.store import SessionStore


class _Connection:
    def __init__(self, operation):
        self.operation = operation

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def exec_driver_sql(self, sql, params=()):
        return self.operation(sql, params)


class _Engine:
    def __init__(self, operation, paramstyle="qmark"):
        self.dialect = SimpleNamespace(paramstyle=paramstyle)
        self.operation = operation

    def connect(self):
        return _Connection(self.operation)


class _RedisPipeline:
    def __init__(self, current=None):
        self.current = current
        self.set_calls = []
        self.reset_calls = 0
        self.execute_calls = 0

    def watch(self, key):
        self.watched = key

    def get(self, key):
        return self.current

    def multi(self):
        return None

    def set(self, key, value, **kwargs):
        self.set_calls.append((key, value, kwargs))

    def execute(self):
        self.execute_calls += 1

    def reset(self):
        self.reset_calls += 1


def test_mysql_driver_failure_is_converted_to_blocked_result_and_health_state() -> None:
    def fail(sql, params):
        raise RuntimeError("secret-dsn-password should not escape")

    adapter = MySQLReadOnlyAdapter(engine=_Engine(fail))
    adapter.ready = True
    result = ReadOnlySQLGuard(backend=adapter).execute(
        "drug_inventory",
        "SELECT name FROM drug_inventory LIMIT 1",
    )

    assert result.blocked is True
    assert "secret-dsn-password" not in (result.reason or "")
    assert "暂时不可用" in (result.reason or "")
    assert adapter.ready is False


def test_qmark_conversion_skips_all_quoted_contexts() -> None:
    sql = "SELECT * FROM t WHERE a = '?' AND b = \"?\" AND c = `?` AND d = ?"
    assert MySQLReadOnlyAdapter._convert_qmark(sql) == (
        "SELECT * FROM t WHERE a = '?' AND b = \"?\" AND c = `?` AND d = %s"
    )

    escaped = r"SELECT * FROM t WHERE a = 'it\'s ?' AND d = ?"
    assert MySQLReadOnlyAdapter._convert_qmark(escaped).endswith("AND d = %s")
    assert "it\\'s ?" in MySQLReadOnlyAdapter._convert_qmark(escaped)

    percent = "SELECT name FROM t WHERE ratio LIKE '100%' AND name = ?"
    assert MySQLReadOnlyAdapter._convert_qmark(percent, escape_percent=True) == (
        "SELECT name FROM t WHERE ratio LIKE '100%%' AND name = %s"
    )
    assert MySQLReadOnlyAdapter._convert_qmark(percent) == percent.replace("?", "%s")


def test_mysql_adapter_preserves_duplicate_driver_columns() -> None:
    class Result:
        def keys(self):
            return ("department", "department")

        def fetchall(self):
            return [("呼吸内科", "医学影像科")]

    adapter = MySQLReadOnlyAdapter(engine=_Engine(lambda *_: Result()))
    adapter.ready = True
    columns, rows = adapter.execute("SELECT department, department FROM doctor_schedules LIMIT 1")

    assert columns == ("department", "department_2")
    assert rows == ({"department": "呼吸内科", "department_2": "医学影像科"},)


def test_mysql_adapter_normalizes_json_incompatible_scalars() -> None:
    class Result:
        def keys(self):
            return ("price", "visit_date", "updated_at")

        def fetchall(self):
            return [(Decimal("19.90"), date(2026, 8, 30), datetime(2026, 8, 30, 9, 15, 0))]

    adapter = MySQLReadOnlyAdapter(engine=_Engine(lambda *_: Result()))
    adapter.ready = True
    _, rows = adapter.execute("SELECT price, visit_date, updated_at FROM exam_prices LIMIT 1")

    assert rows == ({
        "price": "19.90",
        "visit_date": "2026-08-30",
        "updated_at": "2026-08-30T09:15:00",
    },)
    json.dumps(rows)


class _MilvusClient:
    def __init__(self, *, dimension=64, hits=None):
        self.dimension = dimension
        self.hits = hits if hits is not None else [[]]
        self.requests = []

    def has_collection(self, collection_name):
        return True

    def describe_collection(self, collection_name):
        return {
            "fields": [
                {"name": "id", "data_type": "Int64"},
                {"name": "embedding", "data_type": "FloatVector", "params": {"dim": self.dimension}},
                {"name": "text", "data_type": "VarChar"},
                {"name": "title", "data_type": "VarChar"},
                {"name": "category", "data_type": "VarChar"},
                {"name": "source", "data_type": "VarChar"},
                {"name": "updated_at", "data_type": "VarChar"},
            ]
        }

    def search(self, **kwargs):
        self.requests.append(kwargs)
        return self.hits


def test_milvus_normalizes_entity_hits_and_l2_distance(monkeypatch) -> None:
    monkeypatch.setenv("MILVUS_METRIC_TYPE", "L2")
    client = _MilvusClient(
        hits=[[
            {
                "id": 7,
                "distance": 0.25,
                "entity": {
                    "chunk_id": "remote#chunk-001",
                    "document_id": "remote",
                    "text": "清洗后的远端资料",
                    "title": "远端标题",
                    "category": "disease",
                    "source": "远端来源",
                    "updated_at": "2026-01-01",
                },
            }
        ]]
    )
    adapter = MilvusAdapter(client=client)

    assert adapter.ready is True
    hits = adapter.search([0.0] * 64, top_k=1)
    assert hits[0]["id"] == "remote#chunk-001"
    assert hits[0]["text"] == "清洗后的远端资料"
    assert hits[0]["score"] == pytest.approx(0.8)
    assert client.requests[0]["anns_field"] == "embedding"


def test_milvus_schema_dimension_mismatch_is_not_ready(monkeypatch) -> None:
    monkeypatch.setenv("MILVUS_VECTOR_DIMENSION", "64")
    adapter = MilvusAdapter(client=_MilvusClient(dimension=1536))
    assert adapter.ready is False
    assert adapter.status == "milvus-unavailable"


def test_milvus_rejects_non_vector_schema_and_bounds_rpc_timeout() -> None:
    client = _MilvusClient()
    original_describe = client.describe_collection

    def bad_describe(collection_name):
        schema = original_describe(collection_name)
        schema["fields"][1]["data_type"] = "VarChar"
        return schema

    client.describe_collection = bad_describe
    adapter = MilvusAdapter(client=client)
    assert adapter.ready is False
    assert adapter.error == "MilvusSchemaMismatch"

    healthy = _MilvusClient(hits=[[{"id": 0, "distance": 0.5, "entity": {"text": "远端资料"}}]])
    adapter = MilvusAdapter(client=healthy)
    hits = adapter.search([0.0] * 64, top_k=1)
    assert hits[0]["id"] == "0"
    assert healthy.requests[0]["timeout"] == pytest.approx(adapter.timeout)


def test_milvus_upsert_failure_is_not_reported_as_zero_successes() -> None:
    class Client(_MilvusClient):
        def upsert(self, **kwargs):
            raise OSError("down")

    adapter = MilvusAdapter(client=Client())
    with pytest.raises(RuntimeError, match="upsert failed"):
        adapter.upsert([{"id": "chunk"}])
    assert adapter.ready is False


def test_milvus_discovers_default_vector_field_dimension_and_metric(monkeypatch) -> None:
    for name in ("MILVUS_VECTOR_FIELD", "MILVUS_VECTOR_DIMENSION", "MILVUS_METRIC_TYPE"):
        monkeypatch.delenv(name, raising=False)

    class Client(_MilvusClient):
        def describe_collection(self, collection_name):
            schema = super().describe_collection(collection_name)
            schema["fields"][1]["name"] = "vector"
            schema["fields"][1]["params"]["dim"] = 128
            return schema

        def list_indexes(self, collection_name, **kwargs):
            return ["vector_idx"]

        def describe_index(self, collection_name, index_name, **kwargs):
            return {"field_name": "vector", "metric_type": "L2"}

    adapter = MilvusAdapter(client=Client())
    assert adapter.ready is True
    assert adapter.vector_field == "vector"
    assert adapter.dimension == 128
    assert adapter.metric_type == "L2"


def test_milvus_accepts_real_pymilvus_schema_shape(monkeypatch) -> None:
    for name in ("MILVUS_VECTOR_FIELD", "MILVUS_VECTOR_DIMENSION"):
        monkeypatch.delenv(name, raising=False)

    class Client(_MilvusClient):
        def describe_collection(self, collection_name):
            return {
                "fields": [
                    {"name": "id", "type": 5},
                    {"name": "vector", "type": 101, "params": {"dim": 128}},
                    *[
                        {"name": name, "type": 21}
                        for name in ("text", "title", "category", "source", "updated_at")
                    ],
                ]
            }

    adapter = MilvusAdapter(client=Client())
    assert adapter.ready is True
    assert adapter.vector_field == "vector"
    assert adapter.dimension == 128


def test_milvus_rejects_explicit_metric_mismatch(monkeypatch) -> None:
    monkeypatch.setenv("MILVUS_METRIC_TYPE", "COSINE")

    class Client(_MilvusClient):
        def list_indexes(self, collection_name, **kwargs):
            return ["embedding_idx"]

        def describe_index(self, collection_name, index_name, **kwargs):
            return {"field_name": "embedding", "params": {"metric_type": "L2"}}

    adapter = MilvusAdapter(client=Client())
    assert adapter.ready is False
    assert adapter.error == "MilvusMetricMismatch"


def test_milvus_accepts_index_descriptions_returned_by_list_indexes(monkeypatch) -> None:
    monkeypatch.setenv("MILVUS_METRIC_TYPE", "COSINE")

    class Client(_MilvusClient):
        def list_indexes(self, collection_name, **kwargs):
            return [{"index_name": "embedding_idx", "field_name": "embedding", "metric_type": "COSINE"}]

        def describe_index(self, **kwargs):
            raise AssertionError("inline index metadata should not be described again")

    assert MilvusAdapter(client=Client()).ready is True


@pytest.mark.parametrize(
    "field_name",
    ("embedding", "text"),
)
def test_milvus_rejects_schema_fields_without_types(field_name) -> None:
    class Client(_MilvusClient):
        def describe_collection(self, collection_name):
            schema = super().describe_collection(collection_name)
            field = next(item for item in schema["fields"] if item["name"] == field_name)
            field.pop("data_type")
            return schema

    adapter = MilvusAdapter(client=Client())
    assert adapter.ready is False
    assert adapter.error == "MilvusSchemaMismatch"


def test_remote_retrieval_uses_cleaned_remote_text_and_fails_closed_on_outage() -> None:
    class Remote:
        ready = True
        dimension = 64

        def search(self, vector, top_k):
            return [{
                "id": "remote#chunk-001",
                "score": 0.9,
                "text": "资料 <b>内容</b>，联系电话 13800138000",
                "title": "远端资料",
                "category": "disease",
                "source": "远端",
                "updated_at": "2026-01-01",
            }]

    result = HybridRetriever(milvus=Remote()).search("资料内容", "disease")
    assert result
    assert "13800138000" not in result[0].snippet
    assert "[已脱敏]" in result[0].snippet

    class Broken(Remote):
        def search(self, vector, top_k):
            raise OSError("down")

        def mark_unavailable(self):
            self.ready = False

    broken = Broken()
    assert HybridRetriever(milvus=broken).search("咳嗽", "disease") == []
    assert broken.ready is False


def test_remote_retrieval_rejects_zero_score_hits() -> None:
    class Remote:
        ready = True
        dimension = 64

        def search(self, vector, top_k):
            return [{
                "id": "remote#chunk-001",
                "score": 0.0,
                "text": "与查询无关的资料",
                "title": "无关资料",
                "category": "disease",
            }]

    assert HybridRetriever(milvus=Remote()).search("xyzfoobar", "disease") == []


def test_remote_retrieval_does_not_fall_back_to_bundled_chunks() -> None:
    class Remote:
        ready = True
        dimension = 64
        metric_type = "COSINE"

        def search(self, vector, top_k):
            return []

    assert HybridRetriever(milvus=Remote()).search("咳嗽", "disease") == []


def test_remote_retrieval_uses_authoritative_text_for_matching_chunk_id() -> None:
    local = HybridRetriever()
    local_chunk_id = local.chunks[0].id

    class Remote:
        ready = True
        dimension = 64
        metric_type = "COSINE"

        def search(self, vector, top_k):
            return [{
                "id": local_chunk_id,
                "score": 0.9,
                "document_id": "remote-version",
                "text": "REMOTE UPDATED CONTENT",
                "title": "远端新版",
                "category": "disease",
                "source": "Milvus",
                "updated_at": "2026-08-30",
            }]

    result = HybridRetriever(milvus=Remote()).search("REMOTE UPDATED CONTENT", "disease")
    assert result[0].snippet == "REMOTE UPDATED CONTENT"
    assert result[0].title == "远端新版"
    assert result[0].source == "Milvus"


def test_inner_product_non_positive_hits_are_not_admitted(monkeypatch) -> None:
    monkeypatch.delenv("MILVUS_SCORE_THRESHOLD", raising=False)
    client = _MilvusClient(hits=[[{
        "id": "remote#negative",
        "distance": -1.0,
        "entity": {
            "text": "无关资料",
            "title": "无关",
            "category": "disease",
            "source": "Milvus",
            "updated_at": "2026-08-30",
        },
    }]])
    adapter = MilvusAdapter(client=client)
    adapter.metric_type = "IP"
    retriever = HybridRetriever(milvus=adapter)

    assert adapter._normalize_score(-1.0) == 0.0
    assert adapter._normalize_score(0.0) == 0.0
    assert retriever.remote_score_threshold == 0.5
    assert retriever.search("无关查询", "disease") == []


def test_redis_runtime_failure_marks_adapter_unavailable() -> None:
    class BrokenRedis:
        def ping(self):
            return True

        def get(self, key):
            raise OSError("down")

    adapter = RedisSessionAdapter(client=BrokenRedis())
    assert adapter.ready is True
    with pytest.raises(RuntimeError):
        adapter.get_state("session")
    assert adapter.ready is False


def test_redis_adapter_can_recover_after_a_transient_failure() -> None:
    class Redis:
        fail = True

        def ping(self):
            return True

        def get(self, key):
            if self.fail:
                raise OSError("down")
            return None

    client = Redis()
    adapter = RedisSessionAdapter(client=client)
    with pytest.raises(RuntimeError):
        adapter.get_state("session")

    client.fail = False
    adapter._retry_after = 0.0
    assert adapter.ensure_ready() is True
    assert adapter.get_state("session") is None


def test_redis_initialization_failure_rebuilds_after_backoff(monkeypatch) -> None:
    class Redis:
        def ping(self):
            return True

    calls = 0
    client = Redis()

    def create(self):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("startup race")
        return client

    monkeypatch.setattr(RedisSessionAdapter, "_create_client", create)
    adapter = RedisSessionAdapter(url="redis://redacted")

    assert adapter.ready is False
    assert adapter.client is None
    adapter._retry_after = 0.0
    assert adapter.ensure_ready() is True
    assert adapter.client is client
    assert adapter.error is None


def test_session_store_get_reprobes_a_recovered_redis_backend() -> None:
    class RecoveringBackend:
        ready = False
        probes = 0

        def ensure_ready(self):
            self.probes += 1
            self.ready = True
            return True

        def get_state(self, session_id):
            return {"session_id": session_id, "_version": 1, "turn_count": 1}

    backend = RecoveringBackend()
    store = SessionStore(backend=backend, mode="production")

    state = store.get("session")

    assert backend.probes == 1
    assert state is not None
    assert state["turn_count"] == 1


def test_redis_serializes_common_database_scalars() -> None:
    class Redis:
        value = None

        def ping(self):
            return True

        def set(self, key, value):
            self.value = value

    client = Redis()
    adapter = RedisSessionAdapter(client=client)
    state = {
        "session_id": "session",
        "_version": 0,
        "price": Decimal("19.90"),
        "visit_date": date(2026, 8, 30),
        "updated_at": datetime(2026, 8, 30, 9, 15, 0),
    }

    assert adapter.save_state(state) is True
    assert json.loads(client.value) == {
        "session_id": "session",
        "_version": 0,
        "price": "19.90",
        "visit_date": "2026-08-30",
        "updated_at": "2026-08-30T09:15:00",
    }


def test_redis_fallback_write_sets_bounded_ttl(monkeypatch) -> None:
    monkeypatch.setenv("MEDGUIDE_SESSION_TTL_SECONDS", "3600")

    class Redis:
        calls = []

        def ping(self):
            return True

        def set(self, key, value, **kwargs):
            self.calls.append((key, value, kwargs))

    client = Redis()
    adapter = RedisSessionAdapter(client=client)

    assert adapter.save_state({"session_id": "session", "_version": 0}) is True
    assert adapter.ttl_seconds == 3600
    assert client.calls[0][2] == {"ex": 3600}


def test_redis_pipeline_write_sets_ttl() -> None:
    pipeline = _RedisPipeline(current=json.dumps({"session_id": "session", "_version": 1}))

    class Redis:
        def ping(self):
            return True

        def pipeline(self):
            return pipeline

    adapter = RedisSessionAdapter(client=Redis())
    adapter.ttl_seconds = 7200

    assert adapter.save_state({"session_id": "session", "_version": 2}, expected_version=1) is True
    assert pipeline.set_calls[0][2] == {"ex": 7200}
    assert pipeline.execute_calls == 1


def test_redis_cas_invalid_utf8_is_payload_error_without_provider_outage() -> None:
    pipeline = _RedisPipeline(current=b"\xff")

    class Redis:
        def ping(self):
            return True

        def pipeline(self):
            return pipeline

    adapter = RedisSessionAdapter(client=Redis())

    with pytest.raises(RuntimeError, match="payload is invalid") as exc_info:
        adapter.save_state({"session_id": "session", "_version": 2}, expected_version=1)
    assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)
    assert adapter.ready is True
    assert adapter.status == "redis"
    assert adapter.error is None
    assert pipeline.reset_calls == 1
    assert pipeline.execute_calls == 0


def test_corrupt_redis_payload_does_not_fake_a_provider_outage() -> None:
    class Redis:
        def ping(self):
            return True

        def get(self, key):
            return "{not-json"

    adapter = RedisSessionAdapter(client=Redis())

    with pytest.raises(RuntimeError, match="payload is invalid"):
        adapter.get_state("session")
    assert adapter.ready is True
    assert adapter.status == "redis"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"session_id": "other", "_version": 0},
        {"session_id": "session", "_version": -1},
        {"session_id": "session", "_version": 1.5},
        {"session_id": "session", "_version": "1"},
    ],
)
def test_redis_rejects_semantically_invalid_session_payload(payload) -> None:
    class Redis:
        def ping(self):
            return True

        def get(self, key):
            return json.dumps(payload)

    adapter = RedisSessionAdapter(client=Redis())

    with pytest.raises(RuntimeError, match="payload is invalid"):
        adapter.get_state("session")
    assert adapter.ready is True


def test_mysql_adapter_can_recover_after_a_transient_failure() -> None:
    class Result:
        def keys(self):
            return ("one",)

        def fetchall(self):
            return [(1,)]

    adapter = MySQLReadOnlyAdapter(engine=_Engine(lambda *_: Result()))
    adapter._mark_unavailable(OSError("down"))
    adapter._retry_after = 0.0

    assert adapter.ensure_ready() is True
    assert adapter.execute("SELECT 1") == (("one",), ({"one": 1},))


def test_mysql_initialization_failure_rebuilds_after_backoff(monkeypatch) -> None:
    engine = _Engine(lambda *_: None)
    calls = 0

    def create(self):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("startup race")
        return engine

    monkeypatch.setattr(MySQLReadOnlyAdapter, "_create_engine", create)
    adapter = MySQLReadOnlyAdapter(dsn="mysql+pymysql://redacted")

    assert adapter.ready is False
    assert adapter.engine is None
    adapter._retry_after = 0.0
    assert adapter.ensure_ready() is True
    assert adapter.engine is engine
    assert adapter.error is None


def test_milvus_adapter_can_recover_after_a_transient_failure() -> None:
    adapter = MilvusAdapter(client=_MilvusClient())
    adapter.mark_unavailable(OSError("down"))
    adapter._retry_after = 0.0

    assert adapter.ensure_ready() is True
    assert adapter.status == "milvus"


def test_milvus_initialization_failure_rebuilds_after_backoff(monkeypatch) -> None:
    client = _MilvusClient()
    calls = 0

    def create(self):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("startup race")
        return client

    monkeypatch.setattr(MilvusAdapter, "_create_client", create)
    adapter = MilvusAdapter(uri="http://milvus.invalid")

    assert adapter.ready is False
    assert adapter.client is None
    adapter._retry_after = 0.0
    assert adapter.ensure_ready() is True
    assert adapter.client is client
    assert adapter.error is None


def test_openai_runtime_failure_closes_client_and_rebuilds_after_backoff(monkeypatch) -> None:
    class Client:
        def __init__(self, *, fail=False):
            self.fail = fail
            self.close_calls = 0
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            if self.fail:
                raise TimeoutError("provider timeout")
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="recovered"))])

        def close(self):
            self.close_calls += 1

    class Prompt:
        def format_messages(self, **kwargs):
            return [SimpleNamespace(type="human", content="query")]

    first = Client(fail=True)
    second = Client()
    resources = iter(((first, Prompt()), (second, Prompt())))
    monkeypatch.setenv("MEDGUIDE_MODE", "production")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setattr(OpenAIAnswerer, "_create_resources", lambda self: next(resources))
    answerer = OpenAIAnswerer(api_key="redacted")

    assert answerer.generate("query", "low", "context") is None
    assert answerer.available is False
    assert first.close_calls == 1
    answerer._retry_after = 0.0
    assert answerer.generate("query", "low", "context") == "recovered"
    assert answerer.available is True


def test_openai_readiness_probes_rebuilt_client_before_marking_recovered(monkeypatch) -> None:
    class Completions:
        def __init__(self, fail: bool) -> None:
            self.fail = fail
            self.calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            if self.fail:
                raise TimeoutError("provider still unavailable")
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))])

    class Client:
        def __init__(self, fail_probe: bool) -> None:
            self.completions = Completions(fail_probe)
            self.chat = SimpleNamespace(completions=self.completions)

        def close(self):
            return None

    class Prompt:
        def format_messages(self, **kwargs):
            return [SimpleNamespace(type="human", content="query")]

    first = Client(fail_probe=True)
    still_broken = Client(fail_probe=True)
    healthy = Client(fail_probe=False)
    resources = iter(((first, Prompt()), (still_broken, Prompt()), (healthy, Prompt())))
    monkeypatch.setenv("MEDGUIDE_MODE", "production")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setattr(OpenAIAnswerer, "_create_resources", lambda self: next(resources))
    answerer = OpenAIAnswerer(api_key="redacted")

    assert answerer.generate("query", "low", "context") is None
    answerer._retry_after = 0.0
    assert answerer.ensure_ready() is False
    assert still_broken.completions.calls == 1
    assert answerer.available is False
    answerer._retry_after = 0.0
    assert answerer.ensure_ready() is True
    assert healthy.completions.calls == 1
    assert answerer.available is True


def test_external_adapters_close_resources_idempotently() -> None:
    class Redis:
        close_calls = 0

        def ping(self):
            return True

        def close(self):
            self.close_calls += 1

    class Engine(_Engine):
        dispose_calls = 0

        def dispose(self):
            self.dispose_calls += 1

    class Milvus(_MilvusClient):
        close_calls = 0

        def close(self):
            self.close_calls += 1

    class OpenAI:
        close_calls = 0

        def close(self):
            self.close_calls += 1

    redis = Redis()
    engine = Engine(lambda *_: None)
    milvus = Milvus()
    openai = OpenAI()
    redis_adapter = RedisSessionAdapter(client=redis)
    mysql_adapter = MySQLReadOnlyAdapter(engine=engine)
    milvus_adapter = MilvusAdapter(client=milvus)
    answerer = OpenAIAnswerer.__new__(OpenAIAnswerer)
    answerer.client = openai
    answerer.prompt = object()
    answerer.available = True
    answerer._closed = False

    for adapter in (redis_adapter, mysql_adapter, milvus_adapter, answerer):
        adapter.close()
        adapter.close()
        assert adapter.ensure_ready() is False

    assert redis.close_calls == 1
    assert engine.dispose_calls == 1
    assert milvus.close_calls == 1
    assert openai.close_calls == 1
