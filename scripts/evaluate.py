from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

# The published evaluation suite is deterministic and must never inherit a
# production key or endpoint from the operator's shell.  Live-provider
# benchmarking belongs in a separately reviewed command and dataset.
os.environ["MEDGUIDE_MODE"] = "offline"
for _secret_name in (
    "OPENAI_API_KEY",
    "OPENAI_CHAT_BASE_URL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_BASE_URL",
):
    os.environ.pop(_secret_name, None)

from app.retrieval import HybridRetriever  # noqa: E402
from app.safety import SafetyEngine  # noqa: E402
from app.sql_guard import ReadOnlySQLGuard  # noqa: E402
from app.workflow import MedGuideWorkflow  # noqa: E402


def structured_query_succeeded(result: object) -> bool:
    """Treat an explicitly blocked query as not executed, not as a result."""
    return isinstance(result, dict) and not result.get("blocked", False) and "rows" in result


def main() -> int:
    workflow = MedGuideWorkflow(HybridRetriever(), ReadOnlySQLGuard(), SafetyEngine())
    cases = [json.loads(line) for line in (ROOT / "data" / "eval_cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for case in cases:
        state = workflow.run({"session_id": f"eval-{case['id']}", "user_message": case["message"], "profile": {"symptoms": []}})
        citation_ok = bool(state.get("citations")) == bool(case.get("requires_citation"))
        structured_ok = structured_query_succeeded(state.get("structured_result")) == bool(
            case.get("requires_structured")
        )
        rows.append({
            "id": case["id"],
            "intent_ok": state.get("intent") == case["expected_intent"],
            "risk_ok": state.get("risk_level") == case["expected_risk"],
            "citation_ok": citation_ok,
            "structured_ok": structured_ok,
            "latency_ms": state.get("latency_ms", 0),
        })
    checks = ["intent_ok", "risk_ok", "citation_ok", "structured_ok"]
    summary = {key: round(sum(row[key] for row in rows) / len(rows), 3) for key in checks}
    summary["cases"] = len(rows)
    summary["all_passed"] = all(all(row[key] for key in checks) for row in rows)
    print(json.dumps({"summary": summary, "cases": rows}, ensure_ascii=False, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
