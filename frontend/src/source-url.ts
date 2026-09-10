const SENSITIVE_QUERY_KEYS = new Set([
  "accesskey",
  "accesstoken",
  "apikey",
  "auth",
  "authorization",
  "credential",
  "credentials",
  "jwt",
  "key",
  "password",
  "passwd",
  "privatekey",
  "secret",
  "session",
  "sessionid",
  "sig",
  "signature",
  "subscriptionkey",
  "token",
]);

const ALLOWED_SOURCE_HOSTS = new Set([
  "www.nhs.uk",
  "medlineplus.gov",
  "www.fda.gov",
  "telehealth.hhs.gov",
  "www.londonambulance.nhs.uk",
]);

function normalizedQueryKey(value: string): string {
  let decoded = value;
  for (let index = 0; index < 3; index += 1) {
    try {
      const next = decodeURIComponent(decoded);
      if (next === decoded) break;
      decoded = next;
    } catch {
      return "invalidqueryencoding";
    }
  }
  return decoded.normalize("NFKC").toLowerCase().replace(/[^a-z0-9]/g, "");
}

function isSensitiveQueryKey(value: string): boolean {
  const normalized = normalizedQueryKey(value);
  return normalized === "invalidqueryencoding"
    || SENSITIVE_QUERY_KEYS.has(normalized)
    || [
      "accesskey",
      "apikey",
      "privatekey",
      "subscriptionkey",
      "token",
      "secret",
      "password",
      "credential",
      "signature",
    ].some((marker) => normalized.includes(marker));
}

function containsNestedCredentials(value: string): boolean {
  let decoded = value;
  for (let index = 0; index < 4; index += 1) {
    try {
      const next = decodeURIComponent(decoded);
      if (next === decoded) break;
      decoded = next;
    } catch {
      return true;
    }
  }
  const matches = decoded.normalize("NFKC").matchAll(/(?:^|[?&;\s])([^=&?#\s]+)\s*=/g);
  return Array.from(matches).some((match) => isSensitiveQueryKey(match[1]));
}

function isPrivateIpv4(hostname: string): boolean {
  const parts = hostname.split(".").map(Number);
  if (parts.length !== 4 || parts.some((part) => !Number.isInteger(part) || part < 0 || part > 255)) {
    return false;
  }
  return parts[0] === 10
    || parts[0] === 127
    || (parts[0] === 169 && parts[1] === 254)
    || (parts[0] === 172 && parts[1] >= 16 && parts[1] <= 31)
    || (parts[0] === 192 && parts[1] === 168);
}

export function safeCitationSourceUrl(value?: string): string | null {
  if (!value) return null;
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    return null;
  }
  const hostname = parsed.hostname.toLowerCase().replace(/\.$/, "");
  if (
    parsed.protocol !== "https:"
    || Boolean(parsed.username || parsed.password || parsed.hash)
    || (parsed.port !== "" && parsed.port !== "443")
    || !hostname.includes(".")
    || !ALLOWED_SOURCE_HOSTS.has(hostname)
    || hostname === "localhost"
    || hostname.endsWith(".localhost")
    || hostname === "[::1]"
    || isPrivateIpv4(hostname)
  ) {
    return null;
  }
  if (/%(?![0-9a-f]{2})/i.test(parsed.search)) return null;
  for (const [key, queryValue] of parsed.searchParams.entries()) {
    if (isSensitiveQueryKey(key) || containsNestedCredentials(queryValue)) {
      return null;
    }
  }
  return parsed.href;
}
