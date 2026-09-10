#!/bin/sh
set -eu

knowledge_backend="${MEDGUIDE_KNOWLEDGE_BACKEND:-milvus}"
index_path="${BM25_INDEX_PATH:-/app/data/runtime/bm25-index.json}"

# The API receives forwarded HTTPS headers from the UI container. In the
# Compose topology that container lives inside MEDGUIDE_NETWORK_SUBNET, so a
# missing, empty, malformed, or drifting trust boundary must fail before the
# server starts rather than turning into a runtime 426/500 surprise.
if [ "${MEDGUIDE_MODE:-production}" != "offline" ] && [ "${MEDGUIDE_MODE:-production}" != "test" ]; then
  python - <<'PY'
import ipaddress
import os

expected_raw = os.environ.get("MEDGUIDE_NETWORK_SUBNET", "").strip()
trusted_raw = os.environ.get("MEDGUIDE_API_TRUSTED_PROXY_CIDR", "")
if not expected_raw:
    raise SystemExit("MEDGUIDE_NETWORK_SUBNET must be set in production")
if not trusted_raw.strip():
    raise SystemExit("MEDGUIDE_API_TRUSTED_PROXY_CIDR must be set when MEDGUIDE_NETWORK_SUBNET is set")

try:
    expected = ipaddress.ip_network(expected_raw, strict=False)
    trusted = tuple(
        ipaddress.ip_network(item.strip(), strict=False)
        for item in trusted_raw.split(",")
        if item.strip()
    )
except ValueError as exc:
    raise SystemExit(f"invalid trusted proxy/network CIDR: {exc}") from exc

if len(trusted) != 1 or trusted[0] != expected:
    raise SystemExit(
        "MEDGUIDE_API_TRUSTED_PROXY_CIDR must contain the exact "
        "MEDGUIDE_NETWORK_SUBNET for the Compose UI-to-API proxy"
    )
PY
fi

case "$knowledge_backend" in
  milvus)
    mkdir -p "$(dirname "$index_path")"
    python -m app.ingestion \
      --catalog /app/data/knowledge/catalog.json \
      --bm25-index "$index_path"
    ;;
  bm25)
    mkdir -p "$(dirname "$index_path")"
    python /app/scripts/build_bm25_index.py \
      --catalog /app/data/knowledge/catalog.json \
      --output "$index_path" \
      --if-stale
    ;;
esac

exec "$@"
