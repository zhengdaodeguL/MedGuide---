from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from time import monotonic
from typing import Any, Protocol, Sequence
from urllib.parse import urlsplit, urlunsplit

from .infra import _adapter_retry_delay


_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")


@dataclass(frozen=True)
class EmbeddingSpec:
    provider: str
    model: str
    version: str
    dimension: int

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.model.strip() or not self.version.strip():
            raise ValueError("embedding provider, model, and version are required")
        if self.dimension < 8:
            raise ValueError("embedding dimension must be at least 8")

    def collection_properties(self) -> dict[str, str]:
        return {
            "medguide.embedding_provider": self.provider,
            "medguide.embedding_model": self.model,
            "medguide.embedding_version": self.version,
        }

    def as_dict(self) -> dict[str, str | int]:
        return {
            "provider": self.provider,
            "model": self.model,
            "version": self.version,
            "dimension": self.dimension,
        }


class EmbeddingProvider(Protocol):
    spec: EmbeddingSpec

    @property
    def available(self) -> bool: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def _validate_vectors(vectors: Sequence[Sequence[float]], expected: int, dimension: int) -> list[list[float]]:
    if len(vectors) != expected:
        raise RuntimeError(f"embedding provider returned {len(vectors)} vectors for {expected} texts")
    normalized: list[list[float]] = []
    for vector in vectors:
        values = [float(value) for value in vector]
        if len(values) != dimension or not all(math.isfinite(value) for value in values):
            raise RuntimeError("embedding provider returned an invalid vector")
        normalized.append(values)
    return normalized


class HashEmbeddingProvider:
    """Deterministic embedding used only by explicit offline and test modes."""

    def __init__(self, dimension: int = 64) -> None:
        self.spec = EmbeddingSpec("local", "sha256-token-hash", "1", dimension)

    @property
    def available(self) -> bool:
        return True

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.spec.dimension
            for token in _TOKEN_RE.findall(text.lower()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.spec.dimension
                vector[index] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


class KeywordOnlyEmbeddingProvider:
    """Explicit marker provider for a production BM25-only knowledge tier."""

    def __init__(self, *, dimension: int = 64, version: str = "1") -> None:
        self.spec = EmbeddingSpec("keyword", "bm25-only", version, max(8, int(dimension)))

    @property
    def available(self) -> bool:
        return True

    def embed(self, _texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("keyword-only knowledge backend does not embed text")

    def close(self) -> None:
        return None


class OpenAIEmbeddingProvider:
    READINESS_TEXT = "MedGuide readiness probe"

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        version: str,
        dimension: int,
        client: Any | None = None,
        base_url: str | None = None,
        api_key_override: str | None = None,
    ) -> None:
        self.spec = EmbeddingSpec("openai", model, version, dimension)
        self.client = client
        self.error: str | None = None
        self._api_key = api_key_override or api_key
        configured_base_url = base_url or os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        self.base_url = _normalize_embedding_base_url(configured_base_url)
        self._owns_client = client is None
        self._ready = False
        self._requires_probe = True
        self._retry_after = 0.0
        self._closed = False
        if self.client is None and api_key:
            self._build_client()

    @property
    def available(self) -> bool:
        return self._ready and self.client is not None and not self._closed

    def _create_client(self) -> Any:
        from openai import OpenAI

        return OpenAI(
            api_key=self._api_key,
            base_url=self.base_url,
            timeout=_embedding_timeout(),
            max_retries=0,
        )

    def _build_client(self) -> bool:
        if self._closed or not self._api_key:
            return False
        try:
            self.client = self._create_client()
        except Exception as exc:
            self.client = None
            self._ready = False
            self.error = type(exc).__name__
            self._retry_after = monotonic() + _adapter_retry_delay()
            return False
        self._owns_client = True
        self._ready = False
        self._requires_probe = True
        self.error = None
        return True

    def _probe_client(self) -> bool:
        if self.client is None:
            return False
        try:
            response = self.client.embeddings.create(
                input=[self.READINESS_TEXT],
                model=self.spec.model,
                dimensions=self.spec.dimension,
            )
            data = sorted(response.data, key=lambda item: int(getattr(item, "index", 0)))
            _validate_vectors(
                [getattr(item, "embedding", ()) for item in data],
                1,
                self.spec.dimension,
            )
        except Exception as exc:
            self._mark_unavailable(exc)
            return False
        self._requires_probe = False
        self._ready = True
        self.error = None
        return True

    def ensure_ready(self) -> bool:
        if self.available:
            return True
        if self._closed or monotonic() < self._retry_after:
            return False
        if self.client is None and not self._build_client():
            return False
        if self._requires_probe:
            return self._probe_client()
        return self.available

    def _mark_unavailable(self, exc: Exception) -> None:
        client = self.client
        self.client = None
        self._ready = False
        self._requires_probe = True
        self.error = type(exc).__name__
        self._retry_after = monotonic() + _adapter_retry_delay()
        if self._owns_client:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        values = [str(text) for text in texts]
        if not values:
            return []
        if not self.ensure_ready() or self.client is None:
            if self.error is not None:
                raise RuntimeError("embedding request failed")
            raise RuntimeError("embedding provider is unavailable")
        try:
            response = self.client.embeddings.create(
                input=values,
                model=self.spec.model,
                dimensions=self.spec.dimension,
            )
            data = sorted(response.data, key=lambda item: int(getattr(item, "index", 0)))
            return _validate_vectors(
                [getattr(item, "embedding", ()) for item in data],
                len(values),
                self.spec.dimension,
            )
        except Exception as exc:
            self._mark_unavailable(exc)
            raise RuntimeError("embedding request failed") from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        client = self.client
        self.client = None
        self._ready = False
        if self._owns_client:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass


def embedding_provider_from_env(*, production: bool) -> EmbeddingProvider:
    default_dimension = "1536" if production else "64"
    raw_dimension = os.getenv("EMBEDDING_DIMENSION") or os.getenv("MILVUS_VECTOR_DIMENSION", default_dimension)
    try:
        dimension = int(raw_dimension)
    except (TypeError, ValueError) as exc:
        raise ValueError("EMBEDDING_DIMENSION must be an integer") from exc
    if not production:
        return HashEmbeddingProvider(dimension)
    return OpenAIEmbeddingProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip() or "text-embedding-3-small",
        version=os.getenv("EMBEDDING_VERSION", "1").strip() or "1",
        dimension=dimension,
        base_url=os.getenv("EMBEDDING_BASE_URL"),
        api_key_override=os.getenv("EMBEDDING_API_KEY"),
    )


def keyword_embedding_provider_from_env() -> EmbeddingProvider:
    raw_dimension = os.getenv("EMBEDDING_DIMENSION") or os.getenv("MILVUS_VECTOR_DIMENSION", "64")
    try:
        dimension = int(raw_dimension)
    except (TypeError, ValueError) as exc:
        raise ValueError("EMBEDDING_DIMENSION must be an integer") from exc
    version = os.getenv("EMBEDDING_VERSION", "1").strip() or "1"
    return KeywordOnlyEmbeddingProvider(dimension=dimension, version=version)


def _embedding_timeout() -> float:
    try:
        return max(1.0, min(float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "15")), 120.0))
    except (TypeError, ValueError):
        return 15.0


_EMBEDDING_ENDPOINT_SUFFIXES = ("/embeddings", "/chat/completions", "/chaat/completions")


def _normalize_embedding_base_url(value: str | None) -> str | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    parsed = urlsplit(raw)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Embedding base URL must include an HTTP(S) host")
    path = parsed.path.rstrip("/")
    for suffix in _EMBEDDING_ENDPOINT_SUFFIXES:
        if path.lower().endswith(suffix):
            path = path[: -len(suffix)].rstrip("/")
            break
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment)).rstrip("/")
