#!/bin/sh
set -eu

normalized_require_https=$(printf '%s' "${MEDGUIDE_REQUIRE_HTTPS:-true}" | tr '[:upper:]' '[:lower:]')
case "$normalized_require_https" in
  1|true|yes|on|0|false|no|off)
    ;;
  *)
    echo "MEDGUIDE_REQUIRE_HTTPS must be a boolean value" >&2
    exit 1
    ;;
esac

trusted_proxy_cidr="${MEDGUIDE_TRUSTED_PROXY_CIDR:-}"
network_subnet="${MEDGUIDE_NETWORK_SUBNET:-}"
if [ -z "$network_subnet" ] || printf '%s' "$network_subnet" | grep -Eq '[,[:space:]]'; then
  echo "MEDGUIDE_NETWORK_SUBNET must be exactly one CIDR without commas or whitespace" >&2
  exit 1
fi
if [ -z "$trusted_proxy_cidr" ] || printf '%s' "$trusted_proxy_cidr" | grep -Eq '[,[:space:]]'; then
  echo "MEDGUIDE_TRUSTED_PROXY_CIDR must be exactly one CIDR without commas or whitespace" >&2
  exit 1
fi
if [ "$trusted_proxy_cidr" != "$network_subnet" ]; then
  echo "MEDGUIDE_TRUSTED_PROXY_CIDR must exactly match MEDGUIDE_NETWORK_SUBNET" >&2
  exit 1
fi
