#!/bin/sh
set -eu

domain="${MEDGUIDE_DOMAIN:-}"
case "$domain" in
  "")
    echo "MEDGUIDE_DOMAIN is required" >&2
    exit 1
    ;;
  *://*|*/*|*:*|.*|*.|*..*|*[!A-Za-z0-9.-]*)
    echo "MEDGUIDE_DOMAIN must be a valid hostname without scheme, path, or port" >&2
    exit 1
    ;;
esac

case "$domain" in
  *.*) ;;
  *)
    echo "MEDGUIDE_DOMAIN must be a fully qualified hostname" >&2
    exit 1
    ;;
esac

exec caddy run --config /etc/caddy/Caddyfile --adapter caddyfile
