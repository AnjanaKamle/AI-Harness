"""Parse test-runner output into counts and a focused failure summary.

The summary keeps the parts that matter for repair (assertion/error lines, file:line
locations, short test summary) and drops noise, instead of blindly cutting the output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_SUMMARY_CHARS = 6_000
MAX_BLOCK_LINES = 40
MAX_BLOCKS = 10


@dataclass
class ParsedOutput:
    passed: int | None = None
    failed: int | None = None
    skipped: int | None = None
    errors: int | None = None
    failed_tests: list[str] = field(default_factory=list)
    failure_summary: str = ""
    no_tests: bool = False
    recognized: bool = False


def _cap(text: str, limit: int = MAX_SUMMARY_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[failure summary truncated, {len(text) - limit} more chars]"


def tail(text: str, lines: int = 40) -> str:
    return "\n".join(text.rstrip().splitlines()[-lines:])


# --- pytest --------------------------------------------------------------------------------

_PYTEST_COUNT = re.compile(
    r"(\d+) (passed|failed|skipped|errors?|xfailed|xpassed|deselected|warnings?)"
)
_PYTEST_SUMMARY_LINE = re.compile(r"^=*\s*(?:\d+ \w+(?:, )?)+.* in [\d.]+s", re.MULTILINE)
_SECTION = re.compile(r"^={3,} (.+?) ={3,}$", re.MULTILINE)
_BLOCK_HEADER = re.compile(r"^_{3,} (.+?) _{3,}$", re.MULTILINE)
_LOCATION = re.compile(r"^\S[^\s:]*\.\w+:\d+(?::|\b)")
_SHORT_SUMMARY_ITEM = re.compile(r"^(FAILED|ERROR) (\S+)(?: - (.*))?$", re.MULTILINE)


def _sections(text: str) -> dict[str, str]:
    """Split pytest output on '==== NAME ====' banners."""
    matches = list(_SECTION.finditer(text))
    out: dict[str, str] = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[match.group(1).strip().lower()] = text[match.end() : end]
    return out


def _important_lines(block: str) -> list[str]:
    """Error lines, source locations and the code line that raised - the repair signal."""
    lines = block.strip("\n").splitlines()
    keep: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if (
            stripped.startswith(("E ", ">"))
            or _LOCATION.match(stripped)
            or re.match(r"^\w*(Error|Exception|Warning)\b", stripped)
        ):
            keep.append(line)
        elif i + 1 < len(lines) and lines[i + 1].lstrip().startswith("E "):
            keep.append(line)
    if not keep:
        keep = lines[-15:]
    if len(keep) > MAX_BLOCK_LINES:
        keep = keep[: MAX_BLOCK_LINES // 2] + ["    ..."] + keep[-MAX_BLOCK_LINES // 2 :]
    return keep


def parse_pytest(stdout: str, stderr: str, exit_code: int | None) -> ParsedOutput:
    text = stdout + ("\n" + stderr if stderr else "")
    parsed = ParsedOutput()

    summaries = _PYTEST_SUMMARY_LINE.findall(text)
    counts: dict[str, int] = {}
    if summaries:
        parsed.recognized = True
        for number, label in _PYTEST_COUNT.findall(summaries[-1]):
            key = "errors" if label.startswith("error") else label
            counts[key] = counts.get(key, 0) + int(number)
    parsed.passed = counts.get("passed", 0) + counts.get("xpassed", 0) if parsed.recognized else None
    parsed.failed = counts.get("failed", 0) if parsed.recognized else None
    parsed.skipped = counts.get("skipped", 0) + counts.get("xfailed", 0) if parsed.recognized else None
    parsed.errors = counts.get("errors", 0) if parsed.recognized else None
    if exit_code == 5 or "no tests ran" in text or "collected 0 items" in text:
        parsed.no_tests = True
        parsed.recognized = True

    sections = _sections(text)
    blocks: list[str] = []
    for name in ("errors", "failures"):
        body = sections.get(name)
        if not body:
            continue
        headers = list(_BLOCK_HEADER.finditer(body))
        for i, header in enumerate(headers[:MAX_BLOCKS]):
            end = headers[i + 1].start() if i + 1 < len(headers) else len(body)
            chunk = body[header.end() : end]
            blocks.append(f"--- {header.group(1)} ---\n" + "\n".join(_important_lines(chunk)))
        if len(headers) > MAX_BLOCKS:
            blocks.append(f"[{len(headers) - MAX_BLOCKS} more {name} not shown]")

    short = [m.group(0) for m in _SHORT_SUMMARY_ITEM.finditer(text)]
    parsed.failed_tests = [m.group(2) for m in _SHORT_SUMMARY_ITEM.finditer(text)]
    parts = []
    if short:
        parts.append("Short test summary:\n" + "\n".join(short))
    if blocks:
        parts.append("Details:\n" + "\n\n".join(blocks))
    if not parts and exit_code not in (0, None):
        parts.append("Output tail:\n" + tail(text))
    parsed.failure_summary = _cap("\n\n".join(parts))
    return parsed


# --- unittest ------------------------------------------------------------------------------

_UNITTEST_RAN = re.compile(r"^Ran (\d+) tests? in", re.MULTILINE)
_UNITTEST_RESULT = re.compile(r"^(OK|FAILED)(?: \((.*)\))?\s*$", re.MULTILINE)
_UNITTEST_ITEM = re.compile(r"^(FAIL|ERROR): (\S+) \(([\w.]+)\)", re.MULTILINE)


def parse_unittest(stdout: str, stderr: str, exit_code: int | None) -> ParsedOutput:
    text = stdout + "\n" + stderr
    parsed = ParsedOutput()
    ran = _UNITTEST_RAN.findall(text)
    result = _UNITTEST_RESULT.findall(text)
    if ran:
        parsed.recognized = True
        total = int(ran[-1])
        details = dict(
            kv.split("=") for kv in (result[-1][1].split(", ") if result and result[-1][1] else [])
        )
        parsed.failed = int(details.get("failures", 0))
        parsed.errors = int(details.get("errors", 0))
        parsed.skipped = int(details.get("skipped", 0))
        parsed.passed = max(total - parsed.failed - parsed.errors - parsed.skipped, 0)
        parsed.no_tests = total == 0
    parsed.failed_tests = [f"{m.group(3)}.{m.group(2)}" for m in _UNITTEST_ITEM.finditer(text)]
    if exit_code not in (0, None):
        chunks = re.split(r"^={50,}$", text, flags=re.MULTILINE)[1:]
        blocks = ["\n".join(_important_lines(c)) for c in chunks[:MAX_BLOCKS]]
        parsed.failure_summary = _cap("\n\n".join(blocks) or "Output tail:\n" + tail(text))
    return parsed


# --- JavaScript (jest / vitest / mocha / node:test) ----------------------------------------

_JEST_TESTS = re.compile(r"^Tests:\s+(.*)$", re.MULTILINE)
_VITEST_TESTS = re.compile(r"^\s*Tests\s+(.*\(\d+\))", re.MULTILINE)
_MOCHA = re.compile(r"^\s*(\d+) (passing|failing|pending)", re.MULTILINE)
_NODE_TEST = re.compile(r"^# (pass|fail|skipped|todo|cancelled) (\d+)", re.MULTILINE)
_JS_COUNT = re.compile(r"(\d+) (passed|failed|skipped|todo|pending)")
_JS_FAIL_LINE = re.compile(r"^\s*(●|✕|×|not ok \d+|\d+\) )\s*(.+)$", re.MULTILINE)
_JS_ERROR_HINT = re.compile(
    r"(Error|expect\(|Expected|Received|assert|AssertionError|error TS\d+|\bat \S+:\d+:\d+)"
)


def parse_javascript(stdout: str, stderr: str, exit_code: int | None) -> ParsedOutput:
    text = stdout + "\n" + stderr
    parsed = ParsedOutput()
    counts: dict[str, int] = {}
    if jest := _JEST_TESTS.findall(text) or _VITEST_TESTS.findall(text):
        for number, label in _JS_COUNT.findall(jest[-1]):
            counts[label] = int(number)
    elif mocha := _MOCHA.findall(text):
        for number, label in mocha:
            counts[{"passing": "passed", "failing": "failed", "pending": "skipped"}[label]] = int(
                number
            )
    elif node := _NODE_TEST.findall(text):
        for label, number in node:
            key = {"pass": "passed", "fail": "failed"}.get(label, "skipped")
            counts[key] = counts.get(key, 0) + int(number)
    if counts:
        parsed.recognized = True
        parsed.passed = counts.get("passed", 0)
        parsed.failed = counts.get("failed", 0)
        parsed.skipped = counts.get("skipped", 0) + counts.get("todo", 0) + counts.get("pending", 0)
        parsed.no_tests = sum(counts.values()) == 0

    parsed.failed_tests = [m.group(2).strip() for m in _JS_FAIL_LINE.finditer(text)][:50]
    if exit_code not in (0, None):
        lines = text.splitlines()
        keep_idx: set[int] = set()
        for i, line in enumerate(lines):
            if _JS_FAIL_LINE.match(line) or _JS_ERROR_HINT.search(line):
                keep_idx.update(range(max(0, i - 1), min(len(lines), i + 3)))
        focused = [lines[i] for i in sorted(keep_idx)][:200]
        parsed.failure_summary = _cap(
            "\n".join(focused) if focused else "Output tail:\n" + tail(text)
        )
    return parsed


# --- anything else (build, lint, typecheck, make) ------------------------------------------


def parse_generic(stdout: str, stderr: str, exit_code: int | None) -> ParsedOutput:
    parsed = ParsedOutput()
    if exit_code not in (0, None):
        text = (stdout + "\n" + stderr).strip()
        error_lines = [
            ln for ln in text.splitlines()
            if re.search(r"error|fail|warning|\.\w+:\d+", ln, re.IGNORECASE)
        ]
        body = "\n".join(error_lines[:150]) if error_lines else ""
        parsed.failure_summary = _cap(
            (body + "\n\nOutput tail:\n" + tail(text, 25)) if body else "Output tail:\n" + tail(text)
        )
    return parsed
