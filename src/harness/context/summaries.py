"""Structured summaries of older state (context compression without embeddings).

Instead of re-sending raw outputs, agents get compact summaries such as:

    Test summary:
    Attempt 1: FAIL - python3 -m pytest (1 passed, 2 failed)
    Attempt 2: FAIL - python3 -m pytest (2 passed, 1 failed)
    Attempt 3: PASS - python3 -m pytest (3 passed)

Only the most recent failure keeps its detail; earlier ones become one line each.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from harness.tools.base import truncate_lines, truncate_text

LATEST_FAILURE_CHARS = 2_500
LATEST_FAILURE_LINES = 40


def _counts(result: dict[str, Any]) -> str:
    parts = [
        f"{result[k]} {label}"
        for k, label in (("passed", "passed"), ("failed", "failed"), ("errors", "errors"),
                         ("skipped", "skipped"))
        if result.get(k)
    ]
    return ", ".join(parts) or "no counts"


def summarize_tests(test_results: Sequence[dict[str, Any]]) -> str:
    """One line per verification attempt, based on that attempt's decisive gating result."""
    if not test_results:
        return ""
    by_attempt: dict[int, list[dict[str, Any]]] = {}
    for result in test_results:
        by_attempt.setdefault(int(result.get("attempt", 0)), []).append(result)
    lines = ["Test summary:"]
    for attempt in sorted(by_attempt):
        gating = [r for r in by_attempt[attempt] if r.get("gating", True)] or by_attempt[attempt]
        failing = next((r for r in gating if r.get("status") not in ("PASS", "NOT_AVAILABLE")), None)
        decisive = failing or gating[-1]
        command = decisive.get("command", "").split(" -rfE")[0]
        lines.append(
            f"Attempt {attempt}: {decisive.get('status')} - {command} ({_counts(decisive)})"
        )
    return "\n".join(lines)


def summarize_failures(failure_history: Sequence[dict[str, Any]], *, detail_last: int = 1) -> str:
    """Earlier failures as one-liners; the latest ``detail_last`` with their evidence."""
    if not failure_history:
        return ""
    lines = ["Failure history:"]
    cutoff = len(failure_history) - detail_last
    for index, failure in enumerate(failure_history):
        head = (
            f"Attempt {failure.get('attempt')}: {failure.get('category')} - "
            f"{failure.get('counts') or failure.get('reason', '')[:150]}"
        )
        lines.append(head)
        if index >= cutoff:
            tests = failure.get("failed_tests") or []
            if tests:
                lines.append("  failing tests: " + ", ".join(tests[:10]))
            detail = failure.get("failure_summary") or failure.get("reason") or ""
            detail, _ = truncate_lines(detail, LATEST_FAILURE_LINES)
            detail, _ = truncate_text(detail, LATEST_FAILURE_CHARS)
            if detail:
                lines.append("  latest failure details:\n" + "\n".join("    " + ln for ln in detail.splitlines()))
    return "\n".join(lines)


def summarize_code_changes(code_changes: Sequence[dict[str, Any]]) -> str:
    if not code_changes:
        return ""
    files: dict[str, None] = {}
    lines = ["Code changes so far:"]
    for change in code_changes:
        for f in change.get("files_changed", []):
            files[f] = None
    diff = code_changes[-1].get("diff_summary") or {}
    lines.append("files: " + ", ".join(files))
    if diff.get("summary"):
        lines.append(f"latest diff: {diff['summary']}")
    return "\n".join(lines)


def summarize_research(findings: Sequence[dict[str, Any]], *, limit: int = 8) -> str:
    """Findings as one-liners with kind/confidence/source, plus technical details and the
    recommended action for usable ones."""
    if not findings:
        return ""
    lines = []
    for f in findings[:limit]:
        src = f" (source: {f['source']})" if f.get("source") else ""
        lines.append(f"- [{f.get('kind')}/{f.get('confidence')}] {f.get('finding')}{src}")
        if f.get("kind") != "UNCERTAINTY":
            if f.get("technical_details"):
                lines.append(f"    details: {truncate_text(str(f['technical_details']), 600)[0]}")
            if f.get("recommended_action"):
                lines.append(f"    action: {truncate_text(str(f['recommended_action']), 300)[0]}")
    if len(findings) > limit:
        lines.append(f"[OUTPUT TRUNCATED: {len(findings) - limit} more findings]")
    return "\n".join(lines)


def summarize_plan(plan: Sequence[Any]) -> str:
    return "\n".join(
        f"- {getattr(s, 'id', '?')} [{getattr(s, 'agent', '?')}] {getattr(s, 'description', '')} "
        f"({getattr(getattr(s, 'status', None), 'value', getattr(s, 'status', ''))})"
        for s in plan
    )
