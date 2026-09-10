from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _build_index(
    output: Path,
    *,
    backend: str,
    if_stale: bool = False,
    catalog: Path | None = None,
) -> dict[str, object]:
    environment = os.environ.copy()
    environment.update(
        {
            "MEDGUIDE_KNOWLEDGE_BACKEND": backend,
            "EMBEDDING_MODEL": "embedding-contract-test",
            "EMBEDDING_VERSION": "2026-07",
            "EMBEDDING_DIMENSION": "24",
            "MILVUS_VECTOR_DIMENSION": "24",
        }
    )
    command = [
        sys.executable,
        str(ROOT / "scripts" / "build_bm25_index.py"),
        "--catalog",
        str(catalog or ROOT / "data" / "knowledge" / "catalog.json"),
        "--output",
        str(output),
    ]
    if if_stale:
        command.append("--if-stale")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.stderr == ""
    return json.loads(completed.stdout)


def test_bm25_bootstrap_is_deterministic_and_matches_milvus_embedding_contract(tmp_path: Path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    first = _build_index(first_path, backend="milvus")
    second = _build_index(second_path, backend="milvus")

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first["corpus_generation"] == second["corpus_generation"]
    assert first["embedding"] == {
        "provider": "openai",
        "model": "embedding-contract-test",
        "version": "2026-07",
        "dimension": 24,
    }


def test_bm25_only_bootstrap_uses_keyword_contract(tmp_path: Path) -> None:
    payload = _build_index(tmp_path / "keyword.json", backend="bm25")

    assert payload["embedding"] == {
        "provider": "keyword",
        "model": "bm25-only",
        "version": "2026-07",
        "dimension": 24,
    }


def test_bm25_bootstrap_reuses_current_snapshot_and_rebuilds_old_format(tmp_path: Path) -> None:
    path = tmp_path / "keyword.json"
    first = _build_index(path, backend="bm25", if_stale=True)
    unchanged = path.read_bytes()
    second = _build_index(path, backend="bm25", if_stale=True)

    assert first["reused"] is False
    assert second["reused"] is True
    assert path.read_bytes() == unchanged

    legacy = json.loads(path.read_text(encoding="utf-8"))
    legacy["format_version"] = 1
    path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
    rebuilt = _build_index(path, backend="bm25", if_stale=True)

    assert rebuilt["reused"] is False
    assert json.loads(path.read_text(encoding="utf-8"))["format_version"] == 2


def test_bm25_bootstrap_rebuilds_when_catalog_changes(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        (ROOT / "data" / "knowledge" / "catalog.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    path = tmp_path / "keyword.json"

    first = _build_index(path, backend="bm25", if_stale=True, catalog=catalog)
    documents = json.loads(catalog.read_text(encoding="utf-8"))
    documents[0]["text"] += " 咨询时请同时说明症状变化。"
    catalog.write_text(json.dumps(documents, ensure_ascii=False), encoding="utf-8")
    rebuilt = _build_index(path, backend="bm25", if_stale=True, catalog=catalog)

    assert first["reused"] is False
    assert rebuilt["reused"] is False
    assert rebuilt["corpus_generation"] != first["corpus_generation"]


def test_container_bootstraps_missing_index_before_api_process() -> None:
    dockerfile = (ROOT / "backend" / "Dockerfile").read_text(encoding="utf-8")
    entrypoint = (ROOT / "backend" / "docker-entrypoint.sh").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "COPY scripts/build_bm25_index.py" in dockerfile
    assert 'ENTRYPOINT ["/usr/local/bin/medguide-entrypoint"]' in dockerfile
    milvus_branch = entrypoint.split("  milvus)", 1)[1].split("  bm25)", 1)[0]
    bm25_branch = entrypoint.split("  bm25)", 1)[1].split("esac", 1)[0]
    assert "python -m app.ingestion" in milvus_branch
    assert '--bm25-index "$index_path"' in milvus_branch
    assert "build_bm25_index.py" not in milvus_branch
    assert "python /app/scripts/build_bm25_index.py" in bm25_branch
    assert "--if-stale" in bm25_branch
    assert "app.ingestion" not in bm25_branch
    assert "milvus|bm25)" not in entrypoint
    assert 'exec "$@"' in entrypoint
    assert "medguide_runtime:/app/data/runtime" in compose
    assert "condition: service_healthy" in compose
    assert 'if [ "${MEDGUIDE_MODE:-production}" != "offline" ] && [ "${MEDGUIDE_MODE:-production}" != "test" ]; then' in entrypoint
    assert "MEDGUIDE_NETWORK_SUBNET must be set in production" in entrypoint
    assert "must be set when MEDGUIDE_NETWORK_SUBNET is set" in entrypoint
    assert "if len(trusted) != 1 or trusted[0] != expected:" in entrypoint


def test_docker_context_excludes_local_dependencies_and_runtime_artifacts() -> None:
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for pattern in (".venv/", "node_modules/", "dist/", "__pycache__/", ".pytest_cache/", "output/", "data/runtime/"):
        assert pattern in dockerignore
    assert ".env.*" in dockerignore and "!.env.template" in dockerignore


def test_browser_deployment_is_same_origin_and_cookie_authenticated() -> None:
    app = (ROOT / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")
    index = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    nginx = (ROOT / "frontend" / "nginx.conf").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    template = (ROOT / ".env.template").read_text(encoding="utf-8")
    gateway = (ROOT / "deploy" / "gateway" / "Caddyfile").read_text(encoding="utf-8")
    gateway_entrypoint = (ROOT / "deploy" / "gateway" / "entrypoint.sh").read_text(encoding="utf-8")

    assert 'fetch(path, { ...init, credentials: "include" })' in app
    assert "VITE_API_BASE_URL" not in app
    assert "__MEDGUIDE_RUNTIME_CONFIG__" not in app
    assert "runtime-config.js" not in index
    assert "connect-src 'self'" in nginx
    assert "MEDGUIDE_API_ORIGIN" not in nginx
    assert "MEDGUIDE_API_ORIGIN" not in compose
    assert "MEDGUIDE_API_TOKEN:" not in compose
    assert "MEDGUIDE_AUTH_TOKEN" not in compose
    assert "MEDGUIDE_REQUIRE_PROXY_TOKEN" not in compose
    assert "MEDGUIDE_API_TOKEN=" not in template
    assert "MEDGUIDE_AUTH_TOKEN" not in template
    assert "MEDGUIDE_REQUIRE_PROXY_TOKEN" not in template
    assert "MEDGUIDE_API_PORT" not in template
    assert "MEDGUIDE_UI_PORT" not in template
    assert "OPENAI_BASE_URL: ${OPENAI_BASE_URL:-}" in compose
    assert "OPENAI_CHAT_BASE_URL: ${OPENAI_CHAT_BASE_URL:-}" in compose
    assert "EMBEDDING_API_KEY: ${EMBEDDING_API_KEY:-}" in compose
    assert "EMBEDDING_BASE_URL: ${EMBEDDING_BASE_URL:-}" in compose
    assert "MEDGUIDE_CORS_ORIGINS: ${MEDGUIDE_CORS_ORIGINS:-}" in compose
    assert "OPENAI_BASE_URL=\n" in template
    assert "OPENAI_MODEL=\n" in template
    assert "MEDGUIDE_COOKIE_SECURE: ${MEDGUIDE_COOKIE_SECURE:-true}" in compose
    assert "MEDGUIDE_COOKIE_SECURE=true" in template
    assert "MEDGUIDE_REQUIRE_HTTPS: ${MEDGUIDE_REQUIRE_HTTPS:-true}" in compose
    assert "MEDGUIDE_REQUIRE_HTTPS=true" in template
    assert "MEDGUIDE_API_TRUSTED_PROXY_CIDR: ${MEDGUIDE_NETWORK_SUBNET-172.30.0.0/16}" in compose
    assert "MEDGUIDE_TRUSTED_PROXY_CIDR: ${MEDGUIDE_NETWORK_SUBNET-172.30.0.0/16}" in compose
    assert compose.count("MEDGUIDE_NETWORK_SUBNET: ${MEDGUIDE_NETWORK_SUBNET-172.30.0.0/16}") == 2
    assert "MEDGUIDE_TRUSTED_PROXY_CIDR=" not in template
    assert "MEDGUIDE_API_TRUSTED_PROXY_CIDR=" not in template
    assert "MEDGUIDE_NETWORK_SUBNET=172.30.0.0/16" in template
    assert 'NGINX_ENVSUBST_FILTER: "^MEDGUIDE_(TRUSTED_PROXY_CIDR|REQUIRE_HTTPS)"' in compose
    assert "return 426;" in nginx
    assert "location = /_health" in nginx
    assert "location = /_api-ready" in nginx
    assert "http://127.0.0.1/_health" in (ROOT / "frontend" / "Dockerfile").read_text(encoding="utf-8")
    assert "http://127.0.0.1/_api-ready" in (ROOT / "frontend" / "Dockerfile").read_text(encoding="utf-8")
    assert "05-medguide-validate.sh" in (ROOT / "frontend" / "Dockerfile").read_text(encoding="utf-8")
    validator = (ROOT / "frontend" / "docker-entrypoint.d" / "05-medguide-validate.sh").read_text(encoding="utf-8")
    assert "1|true|yes|on|0|false|no|off" in validator
    assert "exit 1" in validator
    assert 'map "$medguide_trusted_proxy:$http_x_forwarded_proto"' in nginx
    assert 'map "${MEDGUIDE_REQUIRE_HTTPS}" $medguide_https_required' in nginx
    assert 'geo $realip_remote_addr $medguide_trusted_proxy' in nginx
    assert 'map "$medguide_https_required:$medguide_client_scheme" $medguide_https_block' in nginx
    assert 'if ($medguide_https_block = 1)' in nginx
    assert "proxy_set_header X-Forwarded-Proto $medguide_client_scheme;" in nginx
    dockerfile = (ROOT / "backend" / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY data/knowledge /app/data/knowledge" in dockerfile
    assert "COPY data /app/data" not in dockerfile
    assert '"--no-proxy-headers"' in dockerfile
    assert "MEDGUIDE_NETWORK_SUBNET: ${MEDGUIDE_NETWORK_SUBNET-172.30.0.0/16}" in compose
    assert "MEDGUIDE_API_TRUSTED_PROXY_CIDR: ${MEDGUIDE_NETWORK_SUBNET-172.30.0.0/16}" in compose
    assert "without commas or whitespace" in validator
    assert 'network_subnet="${MEDGUIDE_NETWORK_SUBNET:-}"' in validator
    assert 'if [ "$trusted_proxy_cidr" != "$network_subnet" ]; then' in validator
    assert "must exactly match MEDGUIDE_NETWORK_SUBNET" in validator
    assert "medguide-gateway:" in compose
    assert '"${MEDGUIDE_HTTP_PORT:-80}:80"' in compose
    assert '"${MEDGUIDE_HTTPS_PORT:-443}:443"' in compose
    assert 'reverse_proxy medguide-ui:80' in gateway
    assert 'header Strict-Transport-Security' in gateway
    assert 'MEDGUIDE_DOMAIN is required' in gateway_entrypoint
    assert "ports:" not in compose.split("medguide-gateway:", 1)[0]
