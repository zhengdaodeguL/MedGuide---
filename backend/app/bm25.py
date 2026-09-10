from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .embeddings import EmbeddingSpec
from .ingestion import DocumentChunk, MedicalDocumentCleaner
from .models import Intent


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def meaningful_tokens(text: str) -> set[str]:
    terms: set[str] = set()
    for match in re.finditer(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+", text.lower()):
        value = match.group(0)
        if re.fullmatch(r"[a-z0-9]+", value):
            if len(value) >= 2:
                terms.add(value)
            continue
        if len(value) < 2:
            continue
        terms.add(value)
        for width in (2, 3, 4):
            if len(value) >= width:
                terms.update(value[index : index + width] for index in range(len(value) - width + 1))
    return terms


@dataclass(frozen=True)
class BM25Hit:
    chunk: DocumentChunk
    score: float


class BM25Index:
    FORMAT_VERSION = 2

    def __init__(
        self,
        chunks: Iterable[DocumentChunk],
        embedding_spec: EmbeddingSpec,
        corpus_generation: str | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.embedding_spec = embedding_spec
        calculated_generation = self.generation_for(self.chunks)
        if corpus_generation is not None and corpus_generation != calculated_generation:
            raise ValueError("BM25 corpus generation does not match its chunks")
        self.corpus_generation = calculated_generation
        self._tokens = [self._chunk_tokens(chunk) for chunk in self.chunks]
        self._meaningful = [self._chunk_meaningful_tokens(chunk) for chunk in self.chunks]
        self._average_length = sum(map(len, self._tokens)) / max(len(self._tokens), 1)
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            document_frequency.update(set(tokens))
        count = len(self._tokens)
        self._idf = {
            token: math.log((count + 1) / (frequency + 1)) + 1
            for token, frequency in document_frequency.items()
        }

    @staticmethod
    def generation_for(chunks: Iterable[DocumentChunk]) -> str:
        digest = hashlib.sha256()
        for chunk in chunks:
            payload = {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            digest.update(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    @staticmethod
    def _chunk_text(chunk: DocumentChunk) -> str:
        return " ".join((chunk.text, chunk.metadata.get("title", ""), chunk.metadata.get("category", "")))

    @classmethod
    def _chunk_tokens(cls, chunk: DocumentChunk) -> list[str]:
        return tokenize(cls._chunk_text(chunk))

    @classmethod
    def _chunk_meaningful_tokens(cls, chunk: DocumentChunk) -> set[str]:
        return meaningful_tokens(cls._chunk_text(chunk))

    def _score(self, query_tokens: list[str], document_tokens: list[str]) -> float:
        counts = Counter(document_tokens)
        score = 0.0
        for token in query_tokens:
            if token not in counts:
                continue
            term_frequency = counts[token]
            score += self._idf.get(token, 0.5) * (term_frequency * 2.2) / (
                term_frequency
                + 1.2 * (0.75 + 0.25 * len(document_tokens) / max(self._average_length, 1))
            )
        return score

    def search(self, query: str, intent: Intent = "unknown", top_k: int = 8) -> list[BM25Hit]:
        query_tokens = tokenize(query)
        query_meaningful = meaningful_tokens(query)
        if not query_tokens or not query_meaningful or top_k <= 0:
            return []
        hits: list[BM25Hit] = []
        for chunk, document_tokens, document_meaningful in zip(self.chunks, self._tokens, self._meaningful):
            category = chunk.metadata.get("category", "unknown")
            if intent not in ("unknown", "structured") and category != intent:
                continue
            if not query_meaningful.intersection(document_meaningful):
                continue
            score = self._score(query_tokens, document_tokens)
            if score > 0:
                hits.append(BM25Hit(chunk, score))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": self.FORMAT_VERSION,
            "corpus_generation": self.corpus_generation,
            "embedding": self.embedding_spec.as_dict(),
            "chunks": [
                {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }
                for chunk in self.chunks
            ],
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path, expected_spec: EmbeddingSpec) -> BM25Index:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("unsupported BM25 index format")
        embedding = payload.get("embedding")
        if not isinstance(embedding, dict):
            raise ValueError("BM25 index is missing its embedding contract")
        actual_spec = EmbeddingSpec(
            provider=str(embedding.get("provider", "")),
            model=str(embedding.get("model", "")),
            version=str(embedding.get("version", "")),
            dimension=int(embedding.get("dimension", 0)),
        )
        if actual_spec != expected_spec:
            raise ValueError("BM25 index embedding contract does not match the query provider")
        raw_chunks = payload.get("chunks")
        if not isinstance(raw_chunks, list) or not raw_chunks:
            raise ValueError("BM25 index chunks must be a non-empty list")
        chunks: list[DocumentChunk] = []
        cleaner = MedicalDocumentCleaner()
        for raw_chunk in raw_chunks:
            if not isinstance(raw_chunk, dict) or not isinstance(raw_chunk.get("metadata"), dict):
                raise ValueError("BM25 index contains an invalid chunk")
            chunks.append(
                cleaner.sanitize_chunk(DocumentChunk(
                    id=str(raw_chunk.get("id", "")),
                    document_id=str(raw_chunk.get("document_id", "")),
                    text=str(raw_chunk.get("text", "")),
                    metadata={str(key): str(value) for key, value in raw_chunk["metadata"].items()},
                ))
            )
        if any(not chunk.id or not chunk.document_id or not chunk.text for chunk in chunks):
            raise ValueError("BM25 index contains an incomplete chunk")
        generation = str(payload.get("corpus_generation", ""))
        if not generation:
            raise ValueError("BM25 index is missing its corpus generation")
        return cls(chunks, actual_spec, generation)
