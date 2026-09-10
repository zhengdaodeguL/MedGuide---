from __future__ import annotations

"""Build the immutable keyword index used by the explicit BM25 tier."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.bm25 import BM25Index  # noqa: E402
from app.embeddings import EmbeddingSpec, keyword_embedding_provider_from_env  # noqa: E402
from app.ingestion import MedicalDocumentCleaner, load_catalog  # noqa: E402


def embedding_spec_from_env() -> EmbeddingSpec:
    """Return the query-time embedding contract without contacting a provider."""
    backend = (os.getenv("MEDGUIDE_KNOWLEDGE_BACKEND") or "milvus").strip().lower()
    if backend == "bm25":
        return keyword_embedding_provider_from_env().spec
    if backend != "milvus":
        raise ValueError("BM25 bootstrap requires the milvus or bm25 knowledge backend")

    raw_dimension = os.getenv("EMBEDDING_DIMENSION") or os.getenv("MILVUS_VECTOR_DIMENSION", "1536")
    try:
        dimension = int(raw_dimension)
    except (TypeError, ValueError) as exc:
        raise ValueError("EMBEDDING_DIMENSION must be an integer") from exc
    return EmbeddingSpec(
        provider="openai",
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()
        or "text-embedding-3-small",
        version=os.getenv("EMBEDDING_VERSION", "1").strip() or "1",
        dimension=dimension,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the MedGuide BM25 knowledge index")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data" / "knowledge" / "catalog.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "runtime" / "bm25-index.json")
    parser.add_argument("--chunk-size", type=int, default=180)
    parser.add_argument(
        "--if-stale",
        action="store_true",
        help="reuse a valid current snapshot and atomically rebuild an outdated or invalid one",
    )
    args = parser.parse_args(argv)
    if args.chunk_size <= 30:
        parser.error("chunk size must be greater than 30")

    try:
        embedding_spec = embedding_spec_from_env()
        documents = load_catalog(args.catalog)
        chunks = MedicalDocumentCleaner().ingest(documents, chunk_size=args.chunk_size)
        if not chunks:
            raise ValueError("knowledge catalog produced no chunks")
        catalog_generation = BM25Index.generation_for(chunks)
        if args.if_stale and args.output.is_file() and args.output.stat().st_size > 0:
            try:
                existing = BM25Index.load(args.output, embedding_spec)
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
            else:
                if existing.corpus_generation == catalog_generation:
                    print(json.dumps({
                        "documents": len({chunk.document_id for chunk in existing.chunks}),
                        "chunks": len(existing.chunks),
                        "corpus_generation": existing.corpus_generation,
                        "output": str(args.output.resolve()),
                        "embedding": embedding_spec.as_dict(),
                        "reused": True,
                    }, ensure_ascii=False))
                    return 0
        index = BM25Index(chunks, embedding_spec)
        index.save(args.output)
        print(json.dumps({
            "documents": len({chunk.document_id for chunk in chunks}),
            "chunks": len(chunks),
            "corpus_generation": index.corpus_generation,
            "output": str(args.output.resolve()),
            "embedding": embedding_spec.as_dict(),
            "reused": False,
        }, ensure_ascii=False))
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"index build failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
