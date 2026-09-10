from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from collections.abc import Mapping
from time import monotonic
from typing import Any

from .bm25 import BM25Index, meaningful_tokens, tokenize
from .embeddings import EmbeddingProvider, EmbeddingSpec, HashEmbeddingProvider
from .models import Citation, Intent, KnowledgeDocument, PatientProfile
from .ingestion import DocumentChunk, MedicalDocumentCleaner, load_catalog
from .infra import _adapter_retry_delay, _adapter_timeout


class IntentRouter:
    KEYWORDS: dict[Intent, tuple[str, ...]] = {
        "drug": ("药", "用药", "剂量", "副作用", "禁忌", "服用"),
        "exam": ("检查", "检验", "化验", "指标", "报告", "价格", "多少钱"),
        "department": (
            "科室", "挂什么号", "挂什么科", "挂哪个科", "看什么科", "去哪个科",
            "看哪个医生", "门诊", "排班", "挂号",
        ),
        "faq": (
            "medguide", "医保", "预约", "就诊", "空腹", "注意事项", "怎么做",
            "能做什么", "提供哪些帮助", "替代医生", "能否诊断", "隐私", "身份信息",
        ),
        "disease": ("症状", "可能是什么", "疾病", "发热", "咳嗽", "疼", "痛", "头晕", "皮疹"),
        "structured": ("库存", "有货", "排班", "号源", "价格", "多少钱", "费用"),
    }

    def classify(self, text: str) -> Intent:
        normalized = text.lower()
        structured = sum(1 for word in self.KEYWORDS["structured"] if word in normalized)
        if structured and any(word in normalized for word in ("库存", "有货", "排班", "号源", "价格", "多少钱", "费用")):
            return "structured"
        scores = {intent: sum(1 for word in words if word in normalized) for intent, words in self.KEYWORDS.items() if intent != "structured"}
        best = max(scores, key=scores.get)
        return best if scores[best] else "unknown"

    def rewrite(self, text: str, profile: PatientProfile | None = None) -> str:
        profile = profile or {}
        context: list[str] = []
        if profile.get("age"):
            context.append(f"{profile['age']}岁")
        if profile.get("duration"):
            context.append(f"持续{profile['duration']}")
        if profile.get("history"):
            context.append(f"既往史{profile['history']}")
        symptoms: list[str] = []
        complaint = profile.get("chief_complaint")
        if isinstance(complaint, str) and re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9-]{1,24}", complaint):
            symptoms.append(complaint)
        profile_symptoms = profile.get("symptoms")
        if isinstance(profile_symptoms, list):
            symptoms.extend(
                symptom
                for symptom in profile_symptoms
                if isinstance(symptom, str)
                and re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9-]{1,24}", symptom)
            )
        symptoms = list(dict.fromkeys(symptoms))
        if symptoms:
            context.append("主要症状" + "、".join(symptoms[:6]))
        return f"{'；'.join(context)}。{text}" if context else text


class HybridRetriever:
    def __init__(
        self,
        knowledge_dir: Path | None = None,
        milvus: Any | None = None,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        bm25_index: BM25Index | None = None,
        bm25_index_path: Path | None = None,
        local_vector_search: bool = True,
    ) -> None:
        self.knowledge_dir = knowledge_dir or Path(__file__).resolve().parents[2] / "data" / "knowledge"
        self.documents = self._load_documents()
        self.cleaner = MedicalDocumentCleaner()
        # Only the cleaner's curated clinical vocabulary may cross the
        # embedding boundary.  Document titles/tags can contain names,
        # addresses, or other identifiers from an imported corpus and must
        # never become an implicit allowlist.
        self._embedding_query_terms = tuple(sorted(MedicalDocumentCleaner._EMBEDDING_QUERY_TERMS))
        self.chunks = self.cleaner.ingest(self.documents)
        self.milvus = milvus
        self.remote_required = milvus is not None
        self.local_vector_search = bool(local_vector_search)
        try:
            fallback_dimension = max(8, int(getattr(milvus, "dimension", 64) or 64))
        except (TypeError, ValueError):
            fallback_dimension = 64
        self.embedding_provider = embedding_provider or HashEmbeddingProvider(fallback_dimension)
        self.vector_size = self.embedding_provider.spec.dimension
        remote_metric = str(getattr(milvus, "metric_type", "COSINE")).upper()
        self._remote_metric = remote_metric
        self.remote_score_threshold = self._remote_threshold(remote_metric)
        self.bm25_error: str | None = None
        self._bm25_index_path: Path | None = None
        self._bm25_signature: tuple[int, int] | None = None
        if (self.remote_required or bm25_index_path is not None) and bm25_index is None:
            raw_path = os.getenv("BM25_INDEX_PATH", "").strip()
            self._bm25_index_path = bm25_index_path or (
                Path(raw_path)
                if raw_path
                else Path(__file__).resolve().parents[2] / "data" / "runtime" / "bm25-index.json"
            )
        if bm25_index is not None:
            if bm25_index.embedding_spec != self.embedding_provider.spec:
                raise ValueError("BM25 index embedding contract does not match the query provider")
            self.bm25_index = bm25_index
        elif self.remote_required or bm25_index_path is not None:
            self.bm25_index = None
            self._refresh_bm25_index(force=True)
        else:
            self.bm25_index = BM25Index(self.chunks, self.embedding_provider.spec)
        self._local_vectors: dict[str, list[float]] = {}
        if self.local_vector_search and not self.remote_required and self.embedding_provider.available:
            texts = [BM25Index._chunk_text(chunk) for chunk in self.chunks]
            vectors = self.embedding_provider.embed(texts)
            self._local_vectors = {chunk.id: vector for chunk, vector in zip(self.chunks, vectors)}

    def _refresh_bm25_index(self, *, force: bool = False) -> None:
        path = self._bm25_index_path
        if path is None:
            return
        try:
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if not force and signature == self._bm25_signature:
                return
            self.bm25_index = BM25Index.load(path, self.embedding_provider.spec)
            self._bm25_signature = signature
            self.bm25_error = None
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.bm25_index = None
            self._bm25_signature = None
            self.bm25_error = type(exc).__name__

    @staticmethod
    def _remote_threshold(metric: str) -> float:
        normalized = str(metric).upper()
        default = "0.5" if normalized in {"IP", "INNER_PRODUCT", "L2", "EUCLIDEAN"} else "0.2"
        try:
            return min(1.0, max(0.0, float(os.getenv("MILVUS_SCORE_THRESHOLD", default))))
        except (TypeError, ValueError):
            return float(default)

    def _sync_remote_metric(self) -> None:
        if self.milvus is None:
            return
        metric = str(getattr(self.milvus, "metric_type", "COSINE")).upper()
        if metric != self._remote_metric:
            self._remote_metric = metric
            self.remote_score_threshold = self._remote_threshold(metric)

    def _load_documents(self) -> list[KnowledgeDocument]:
        catalog = self.knowledge_dir / "catalog.json"
        return load_catalog(catalog)

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right))

    def search(
        self,
        query: str,
        intent: Intent = "unknown",
        top_k: int = 4,
        _retry_on_generation_change: bool = True,
    ) -> list[Citation]:
        query = query.strip()
        if not query or top_k <= 0:
            return []
        self._refresh_bm25_index()
        active_generation = self.bm25_index.corpus_generation if self.bm25_index is not None else None
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        query_vector: list[float] | None = None
        embedding_query = self.cleaner.sanitize_query_for_embedding(
            query,
            intent=intent,
            allowed_terms=self._embedding_query_terms,
        )
        provider_name = str(getattr(self.embedding_provider.spec, "provider", ""))
        should_embed = self.remote_required or (
            self.local_vector_search and provider_name != "local"
        )
        if should_embed and self.embedding_provider.available:
            try:
                query_vector = self.embedding_provider.embed([embedding_query])[0]
            except (RuntimeError, IndexError):
                query_vector = None
        lexical_hits = self.bm25_index.search(query, intent, max(top_k * 2, 4)) if self.bm25_index else []
        remote_hits: dict[str, dict[str, Any]] = {}
        remote_ready = False
        if self.remote_required:
            ensure_ready = getattr(self.milvus, "ensure_ready", None)
            remote_ready = bool(ensure_ready()) if callable(ensure_ready) else bool(getattr(self.milvus, "ready", False))
            if remote_ready:
                self._sync_remote_metric()
        generation_required = getattr(self.milvus, "embedding_spec", None) is not None
        if remote_ready and query_vector is not None and (active_generation or not generation_required):
            try:
                try:
                    hits = self.milvus.search(
                        query_vector,
                        top_k=max(top_k * 2, 4),
                        corpus_generation=active_generation,
                    )
                except TypeError as exc:
                    if "corpus_generation" not in str(exc):
                        raise
                    hits = self.milvus.search(query_vector, top_k=max(top_k * 2, 4))
                for hit in hits:
                    if not isinstance(hit, dict):
                        continue
                    if generation_required and hit.get("corpus_generation") != active_generation:
                        continue
                    raw_id = hit.get("id")
                    if raw_id is None or raw_id == "":
                        raw_id = hit.get("chunk_id")
                    hit_id = str(raw_id) if raw_id is not None else ""
                    remote_score = self._remote_score(hit)
                    if hit_id and remote_score is not None and remote_score >= self.remote_score_threshold:
                        remote_hits[hit_id] = hit
            except Exception:
                # Do not silently present bundled data as production vector
                # results after a configured Milvus outage.
                mark_unavailable = getattr(self.milvus, "mark_unavailable", None)
                if callable(mark_unavailable):
                    mark_unavailable()

        candidates_by_id: dict[str, dict[str, Any]] = {
            hit.chunk.id: {"chunk": hit.chunk, "bm25": hit.score, "vector": 0.0}
            for hit in lexical_hits
        }
        if (
            self.local_vector_search
            and not self.remote_required
            and provider_name != "local"
            and query_vector is not None
        ):
            for candidate in candidates_by_id.values():
                local_vector = self._local_vectors.get(candidate["chunk"].id)
                if local_vector is not None:
                    candidate["vector"] = self._cosine(query_vector, local_vector)
        for hit_id, hit in remote_hits.items():
            text = self.cleaner.clean(str(hit.get("text", "")))
            if not text:
                continue
            category = str(hit.get("category", "unknown"))
            if intent not in ("unknown", "structured") and category != intent:
                continue
            try:
                remote_chunk = self.cleaner.sanitize_chunk(DocumentChunk(
                    id=hit_id,
                    document_id=str(hit.get("document_id") or hit_id.split("#", 1)[0]),
                    text=text,
                    metadata={
                        "title": str(hit.get("title") or hit_id),
                        "category": category,
                        "source": str(hit.get("source", "")),
                        "updated_at": str(hit.get("updated_at", "")),
                        "source_url": str(hit.get("source_url", "")),
                    },
                ))
            except ValueError:
                continue
            candidate = candidates_by_id.get(hit_id)
            if candidate is None:
                candidate = {"chunk": remote_chunk, "bm25": 0.0, "vector": 0.0}
                candidates_by_id[hit_id] = candidate
            candidate["vector"] = self._remote_score(hit) or 0.0

        candidates: list[tuple[float, float, float, DocumentChunk]] = []
        for candidate in candidates_by_id.values():
            chunk = candidate["chunk"]
            bm25 = float(candidate["bm25"])
            vector = float(candidate["vector"])
            lexical_score = bm25 / (bm25 + 5.0) if bm25 > 0 else 0.0
            intent_boost = 0.08 if chunk.metadata.get("category") == intent else 0.0
            if bm25 > 0 and vector > 0:
                fused = 0.72 * vector + 0.20 * lexical_score + intent_boost
            elif vector > 0:
                fused = 0.84 * vector + intent_boost
            else:
                fused = 0.82 * lexical_score + intent_boost
            candidates.append((min(1.0, max(0.0, fused)), bm25, vector, chunk))
        if not candidates:
            if _retry_on_generation_change and self._generation_changed(active_generation):
                return self.search(query, intent, top_k, False)
            return []
        candidates.sort(key=lambda item: item[0], reverse=True)
        seen_documents: set[str] = set()
        results: list[Citation] = []
        for fused, bm25, vector, chunk in candidates:
            if chunk.document_id in seen_documents:
                continue
            try:
                chunk = self.cleaner.sanitize_chunk(chunk)
            except ValueError:
                continue
            seen_documents.add(chunk.document_id)
            snippet = chunk.text[:360].rstrip()
            retrieval = "hybrid" if bm25 > 0 and vector > 0 else ("bm25" if bm25 > 0 else "vector")
            results.append(
                Citation(
                    id=chunk.id,
                    title=chunk.metadata.get("title", chunk.document_id),
                    category=chunk.metadata.get("category", "unknown"),
                    source=chunk.metadata.get("source", ""),
                    updated_at=chunk.metadata.get("updated_at", ""),
                    snippet=snippet,
                    score=fused,
                    retrieval=retrieval,
                    source_url=chunk.metadata.get("source_url", ""),
                )
            )
            if len(results) >= top_k:
                break
        if _retry_on_generation_change and self._generation_changed(active_generation):
            return self.search(query, intent, top_k, False)
        return results

    def _generation_changed(self, previous: str | None) -> bool:
        self._refresh_bm25_index()
        current = self.bm25_index.corpus_generation if self.bm25_index is not None else None
        return current != previous

    @staticmethod
    def _remote_score(hit: dict[str, Any] | None) -> float | None:
        if not hit:
            return None
        try:
            score = float(hit.get("score", hit.get("distance", 0.0)))
        except (TypeError, ValueError):
            return None
        return min(1.0, max(0.0, score))

    def health(self) -> dict[str, str | int]:
        self._refresh_bm25_index()
        vector_store = (
            "milvus"
            if self.milvus is not None and getattr(self.milvus, "ready", False)
            else ("milvus-unavailable" if self.remote_required else "disabled")
        )
        provider_name = str(getattr(self.embedding_provider.spec, "provider", ""))
        indexed_chunks = self.bm25_index.chunks if self.bm25_index is not None else []
        return {
            "documents": len({chunk.document_id for chunk in indexed_chunks}),
            "chunks": len(indexed_chunks),
            "vector_store": vector_store,
            "keyword_store": "bm25" if self.bm25_index is not None else "bm25-unavailable",
            "embedding_model": self.embedding_provider.spec.model if provider_name != "local" else "not-configured",
            "embedding_version": self.embedding_provider.spec.version,
        }

    @property
    def keyword_ready(self) -> bool:
        self._refresh_bm25_index()
        return self.bm25_index is not None


class MilvusAdapter:
    """Thin Milvus client with a health-checked, optional production path."""

    def __init__(
        self,
        uri: str | None = None,
        client: Any | None = None,
        collection: str | None = None,
        *,
        embedding_spec: EmbeddingSpec | None = None,
    ) -> None:
        self.uri = uri
        self.collection = collection or os.getenv("MILVUS_COLLECTION", "medguide_knowledge")
        self.embedding_spec = embedding_spec
        configured_vector_field = os.getenv("MILVUS_VECTOR_FIELD", "").strip()
        configured_metric = os.getenv("MILVUS_METRIC_TYPE", "").strip()
        configured_dimension = os.getenv("MILVUS_VECTOR_DIMENSION", "").strip()
        self.vector_field = configured_vector_field or "embedding"
        self.metric_type = (configured_metric or "COSINE").upper()
        self._vector_field_explicit = bool(configured_vector_field)
        self._metric_explicit = bool(configured_metric)
        self._dimension_explicit = bool(configured_dimension)
        self.timeout = _adapter_timeout()
        if embedding_spec is not None:
            self.dimension = embedding_spec.dimension
            self._dimension_explicit = True
        else:
            try:
                self.dimension = max(8, int(configured_dimension or "64"))
            except ValueError:
                self.dimension = 64
        self.primary_field = ""
        self.auto_id = False
        self.output_fields: tuple[str, ...] = (
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
        self.client: Any | None = client
        self.configured = bool(uri or client)
        self.available = False
        self.ready = False
        self.error: str | None = None
        self._retry_after = 0.0
        self._closed = False
        self._owns_client = False
        if self.client is None and uri:
            self._build_client()
        else:
            self.available = self.client is not None
        if self.available:
            self.ready = self._probe()

    def _create_client(self) -> Any:
        from pymilvus import MilvusClient

        try:
            return MilvusClient(uri=self.uri, timeout=self.timeout)
        except TypeError:
            return MilvusClient(uri=self.uri)

    def _build_client(self) -> bool:
        if self._closed or not self.uri:
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
        """Close and drop an owned client after a transient failure."""
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
            disconnect = getattr(client, "disconnect", None)
            if callable(disconnect):
                try:
                    disconnect()
                except Exception:
                    pass

    def _probe(self) -> bool:
        if self.client is None:
            return False
        try:
            has = getattr(self.client, "has_collection", None)
            if callable(has) and not bool(self._call_with_timeout(has, collection_name=self.collection)):
                return False
            describe = getattr(self.client, "describe_collection", None)
            description: Any = None
            if callable(describe):
                description = self._call_with_timeout(describe, collection_name=self.collection)
                if not self._schema_matches(description):
                    self.error = "MilvusSchemaMismatch"
                    return False
            elif self.embedding_spec is not None:
                self.error = "MilvusSchemaMismatch"
                return False
            if self.embedding_spec is not None and not self._embedding_contract_matches(description):
                self.error = "MilvusEmbeddingContractMismatch"
                return False
            if not self._index_matches():
                self.error = "MilvusMetricMismatch"
                return False
            self.error = None
            return True
        except Exception as exc:
            if self._owns_client:
                self._discard_client()
            self.error = type(exc).__name__
            self._retry_after = monotonic() + _adapter_retry_delay()
            return False

    def _schema_matches(self, description: Any) -> bool:
        if not isinstance(description, Mapping):
            return False
        fields = description.get("fields")
        schema = description.get("schema")
        if fields is None and isinstance(schema, Mapping):
            fields = schema.get("fields")
        if not isinstance(fields, list) or not fields:
            return False
        field_map = {
            str(field.get("name")): field
            for field in fields
            if isinstance(field, Mapping) and field.get("name")
        }
        schema_metadata = schema if isinstance(schema, Mapping) else {}
        self.auto_id = bool(description.get("auto_id", schema_metadata.get("auto_id", False)))
        primary_field = description.get("primary_field") or description.get("primary_field_name")
        if not primary_field:
            primary_field = next(
                (name for name, field in field_map.items() if bool(field.get("is_primary"))),
                "",
            )
        self.primary_field = str(primary_field or "")
        if self.vector_field not in field_map and not self._vector_field_explicit:
            vector_candidates = [
                name
                for name, field in field_map.items()
                if self._data_type(field.get("data_type", field.get("type"))) == "FLOATVECTOR"
            ]
            if len(vector_candidates) == 1:
                self.vector_field = vector_candidates[0]
        required = {"text", "title", "category", "source", "updated_at", self.vector_field}
        if self.embedding_spec is not None:
            required.update(("chunk_id", "document_id", "source_url", "corpus_generation"))
            if self.auto_id or self.primary_field != "id":
                return False
        if not required.issubset(field_map) or not ({"id", "chunk_id"} & field_map.keys()):
            return False
        if self.embedding_spec is not None and self.primary_field == "id":
            primary_type = self._data_type(field_map["id"].get("data_type", field_map["id"].get("type")))
            if primary_type != "INT64":
                return False
        self.output_fields = tuple(
            field
            for field in (
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
            if field in field_map
        )
        vector_field = field_map[self.vector_field]
        data_type = self._data_type(vector_field.get("data_type", vector_field.get("type")))
        if data_type != "FLOATVECTOR":
            return False
        scalar_fields = ["text", "title", "category", "source", "updated_at"]
        if "source_url" in field_map:
            scalar_fields.append("source_url")
        if self.embedding_spec is not None:
            scalar_fields.extend(("chunk_id", "document_id", "corpus_generation"))
        for name in scalar_fields:
            scalar_type = self._data_type(field_map[name].get("data_type", field_map[name].get("type")))
            if scalar_type != "VARCHAR":
                return False
        params = vector_field.get("params") if isinstance(vector_field, Mapping) else None
        raw_dimension = params.get("dim") if isinstance(params, Mapping) else vector_field.get("dim")
        try:
            if raw_dimension is None:
                return False
            actual_dimension = int(raw_dimension)
            if not self._dimension_explicit:
                self.dimension = actual_dimension
                return actual_dimension >= 8
            return actual_dimension == self.dimension
        except (TypeError, ValueError):
            return False

    def _embedding_contract_matches(self, description: Any) -> bool:
        if self.embedding_spec is None or not isinstance(description, Mapping):
            return self.embedding_spec is None
        properties: Any = description.get("properties")
        schema = description.get("schema")
        if properties is None and isinstance(schema, Mapping):
            properties = schema.get("properties")
        normalized: dict[str, str] = {}
        if isinstance(properties, Mapping):
            normalized = {str(key): str(value) for key, value in properties.items()}
        elif isinstance(properties, list):
            for item in properties:
                if not isinstance(item, Mapping):
                    continue
                key = item.get("key") or item.get("name") or item.get("property_name")
                value = item.get("value") or item.get("property_value")
                if key is not None and value is not None:
                    normalized[str(key)] = str(value)
        return all(
            normalized.get(key) == value
            for key, value in self.embedding_spec.collection_properties().items()
        )

    @staticmethod
    def _data_type(value: Any) -> str:
        normalized = re.sub(r"[^A-Z0-9]", "", str(value).upper()) if value is not None else ""
        return {
            "101": "FLOATVECTOR",
            "21": "VARCHAR",
            "5": "INT64",
        }.get(normalized, normalized)

    @staticmethod
    def _metric_from(description: Any) -> str | None:
        if not isinstance(description, Mapping):
            return None
        candidates = [description]
        for key in ("params", "index_param", "index_params"):
            nested = description.get(key)
            if isinstance(nested, Mapping):
                candidates.append(nested)
        for candidate in candidates:
            value = candidate.get("metric_type") or candidate.get("metric")
            if value:
                return str(value).strip().upper()
        return None

    @staticmethod
    def _canonical_metric(metric: str) -> str:
        return {"EUCLIDEAN": "L2", "INNER_PRODUCT": "IP"}.get(metric.upper(), metric.upper())

    def _index_matches(self) -> bool:
        if self.client is None:
            return False
        list_indexes = getattr(self.client, "list_indexes", None)
        describe_index = getattr(self.client, "describe_index", None)
        if not callable(list_indexes) or not callable(describe_index):
            # Legacy test doubles without an explicit production embedding
            # contract may omit index introspection. A production collection
            # must prove that its vector index exists and uses the same metric.
            return self.embedding_spec is None
        names = self._call_with_timeout(list_indexes, collection_name=self.collection)
        if isinstance(names, Mapping):
            names = names.get("index_names", names.get("data", []))
        if isinstance(names, (str, bytes)):
            names = [names]
        found_valid = False
        for item in names or []:
            if isinstance(item, Mapping):
                name = item.get("index_name") or item.get("name")
                description = item
                if self._metric_from(description) is None and name:
                    description = self._call_with_timeout(
                        describe_index,
                        collection_name=self.collection,
                        index_name=str(name),
                    )
            else:
                name = item
                description = self._call_with_timeout(
                    describe_index,
                    collection_name=self.collection,
                    index_name=str(name),
                )
            if not isinstance(description, Mapping):
                continue
            field_name = description.get("field_name") or description.get("field")
            if field_name and str(field_name) != self.vector_field:
                continue
            metric = self._metric_from(description)
            if metric is None:
                continue
            metric = self._canonical_metric(metric)
            if metric not in {"COSINE", "IP", "L2"}:
                continue
            if not self._metric_explicit:
                self.metric_type = metric
                return True
            if metric == self._canonical_metric(self.metric_type):
                found_valid = True
                return True
        return found_valid

    def mark_unavailable(self, exc: Exception | None = None) -> None:
        self.ready = False
        if self._owns_client:
            self._discard_client()
        self.error = type(exc).__name__ if exc is not None else "MilvusSearchError"
        self._retry_after = monotonic() + _adapter_retry_delay()

    def ensure_ready(self, *, revalidate: bool = False) -> bool:
        if self._closed:
            return False
        if self.ready and not revalidate:
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
            return "milvus"
        if self.configured:
            return "milvus-unavailable"
        return "vector-search-disabled"

    def upsert(self, chunks: list[dict]) -> int:
        if not self.ensure_ready(revalidate=self.embedding_spec is not None) or self.client is None:
            return 0
        try:
            insert = getattr(self.client, "upsert", None) or getattr(self.client, "insert", None)
            if not callable(insert):
                raise RuntimeError("Milvus client does not support upsert")
            result = self._call_with_timeout(insert, collection_name=self.collection, data=chunks)
            if isinstance(result, dict):
                return int(result.get("upsert_count", result.get("insert_count", len(chunks))))
            return len(chunks)
        except Exception as exc:
            self.mark_unavailable(exc)
            raise RuntimeError("Milvus upsert failed") from exc

    def delete_stale_generations(self, corpus_generation: str) -> int:
        if not self.ensure_ready(revalidate=self.embedding_spec is not None) or self.client is None:
            raise RuntimeError("Milvus collection is unavailable")
        delete = getattr(self.client, "delete", None)
        if not callable(delete):
            raise RuntimeError("Milvus client does not support generation cleanup")
        expression = f"corpus_generation != {json.dumps(corpus_generation)}"
        try:
            result = self._call_with_timeout(
                delete,
                collection_name=self.collection,
                filter=expression,
            )
            if isinstance(result, Mapping):
                return int(result.get("delete_count", 0))
            return 0
        except Exception as exc:
            raise RuntimeError("Milvus generation cleanup failed") from exc

    def search(
        self,
        vector: list[float],
        top_k: int = 4,
        *,
        corpus_generation: str | None = None,
    ) -> list[dict]:
        if not self.ensure_ready(revalidate=self.embedding_spec is not None) or self.client is None:
            return []
        try:
            search = getattr(self.client, "search", None)
            if not callable(search):
                return []
            request = {
                "collection_name": self.collection,
                "data": [vector],
                "limit": top_k,
                "anns_field": self.vector_field,
                "search_params": {"metric_type": self.metric_type, "params": {}},
                "output_fields": list(self.output_fields),
            }
            if corpus_generation:
                request["filter"] = f"corpus_generation == {json.dumps(corpus_generation)}"
            try:
                response = self._call_with_timeout(search, **request)
            except TypeError:
                # Keep compatibility with older MilvusClient releases whose
                # convenience method does not expose anns_field/search_params.
                request.pop("anns_field", None)
                request.pop("search_params", None)
                response = self._call_with_timeout(search, **request)
            if isinstance(response, Mapping) and "data" in response:
                response = response["data"]
            if isinstance(response, list) and response and not isinstance(response[0], Mapping):
                hits = response[0]
            else:
                hits = response
            normalized: list[dict] = []
            for hit in hits or []:
                if not isinstance(hit, Mapping):
                    continue
                entity = hit.get("entity")
                entity = entity if isinstance(entity, Mapping) else {}
                raw_id = next(
                    (
                        value
                        for value in (
                            entity.get("chunk_id"),
                            hit.get("chunk_id"),
                            entity.get("id"),
                            hit.get("id"),
                        )
                        if value is not None and value != ""
                    ),
                    "",
                )
                hit_id = str(raw_id)
                if "distance" not in hit and "score" not in hit:
                    continue
                raw_distance = hit.get("distance", hit.get("score"))
                normalized.append(
                    {
                        "id": hit_id,
                        "score": self._normalize_score(raw_distance),
                        "distance": raw_distance,
                        **{
                            key: entity.get(key, hit.get(key))
                            for key in (
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
                            if entity.get(key, hit.get(key)) is not None
                        },
                    }
                )
            return normalized
        except Exception as exc:
            self.mark_unavailable(exc)
            raise RuntimeError("Milvus search failed") from exc

    def _call_with_timeout(self, function: Any, **kwargs: Any) -> Any:
        """Pass the bounded timeout when the client method supports it."""
        try:
            return function(timeout=self.timeout, **kwargs)
        except TypeError as exc:
            # Test doubles and older clients may not expose ``timeout``.  Do
            # not hide genuine provider TypeErrors from the second call.
            if "timeout" not in str(exc).lower():
                raise
            return function(**kwargs)

    def _normalize_score(self, raw_score: Any) -> float:
        try:
            value = float(raw_score)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(value):
            return 0.0
        if self.metric_type in {"L2", "EUCLIDEAN"}:
            return 1.0 / (1.0 + max(0.0, value))
        if self.metric_type in {"IP", "INNER_PRODUCT"}:
            # Preserve the sign of inner-product similarity. Negative and zero
            # matches are not relevant evidence; positive scores remain
            # monotonic and are clipped only for the public 0..1 contract.
            return min(1.0, max(0.0, value))
        return min(1.0, max(0.0, value))

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
            disconnect = getattr(client, "disconnect", None)
            if callable(disconnect):
                try:
                    disconnect()
                except Exception:
                    pass

    def shutdown(self) -> None:
        self.close()

    async def aclose(self) -> None:
        self.close()
