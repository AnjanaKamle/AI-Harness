"""BaseTool: the contract every harness tool (filesystem, terminal, search, git, web, music)
implements."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar

from harness.llm.models import ToolDefinition

log = logging.getLogger("harness.tools")


class ToolStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_name: str
    status: ToolStatus
    output: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is ToolStatus.SUCCESS

    @classmethod
    def success(cls, tool_name: str, data: dict[str, Any], output: str = "") -> ToolExecutionResult:
        return cls(tool_name, ToolStatus.SUCCESS, output=output, data=data)

    @classmethod
    def failure(
        cls, tool_name: str, error: str, data: dict[str, Any] | None = None
    ) -> ToolExecutionResult:
        return cls(tool_name, ToolStatus.FAILURE, error=error, data=data or {})

    def to_llm_content(self, max_chars: int, max_lines: int | None = None) -> str:
        """JSON payload for the model. Each text field is capped (chars and lines) with a
        visible [OUTPUT TRUNCATED] marker, then the whole payload is hard-capped."""
        payload: dict[str, Any] = {"ok": self.ok}
        if self.data:
            payload["data"] = (
                bound_value(self.data, max_chars=max_chars, max_lines=max_lines)
                if max_lines is not None
                else self.data
            )
        if self.output:
            payload["output"] = self.output
        if self.error:
            payload["error"] = self.error
        text = json.dumps(payload, ensure_ascii=False, default=str)
        return truncate_text(text, max_chars)[0]


class ToolInputError(ValueError):
    """Tool arguments do not satisfy the tool's input schema."""


class ToolError(Exception):
    """An expected, reportable tool failure (bad path, missing file, policy violation...).

    ``retryable`` marks transient failures (network timeouts, 5xx) that may succeed if the
    same call is repeated; deterministic failures must leave it False.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


TRUNCATION_MARKER = "[OUTPUT TRUNCATED"
DEFAULT_MAX_LIST_ITEMS = 200


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Keep the head and tail of ``text`` within ``max_chars``; mark what was dropped."""
    if len(text) <= max_chars:
        return text, False
    marker = f"\n{TRUNCATION_MARKER}: {len(text) - max_chars} chars omitted]\n"
    keep = max(max_chars - len(marker), 0)
    head = keep * 2 // 3
    tail = keep - head
    return text[:head] + marker + (text[-tail:] if tail else ""), True


def truncate_lines(text: str, max_lines: int) -> tuple[str, bool]:
    """Keep the first 2/3 and last 1/3 of ``max_lines`` lines; mark what was dropped."""
    lines = text.splitlines(keepends=True)
    if len(lines) <= max_lines:
        return text, False
    head = max(max_lines * 2 // 3, 1)
    tail = max(max_lines - head, 0)
    marker = f"{TRUNCATION_MARKER}: {len(lines) - head - tail} lines omitted]\n"
    kept_tail = lines[-tail:] if tail else []
    return "".join(lines[:head]) + marker + "".join(kept_tail), True


def bound_value(
    value: Any, *, max_chars: int, max_lines: int, max_items: int = DEFAULT_MAX_LIST_ITEMS
) -> Any:
    """Recursively cap strings (chars and lines) and lists (items) inside a tool payload."""
    if isinstance(value, str):
        text, _ = truncate_lines(value, max_lines)
        return truncate_text(text, max_chars)[0]
    if isinstance(value, dict):
        return {
            k: bound_value(v, max_chars=max_chars, max_lines=max_lines, max_items=max_items)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        items = [
            bound_value(v, max_chars=max_chars, max_lines=max_lines, max_items=max_items)
            for v in list(value)[:max_items]
        ]
        if len(value) > max_items:
            items.append(f"{TRUNCATION_MARKER}: {len(value) - max_items} more items]")
        return items
    return value


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
}


def _type_ok(value: Any, expected: str) -> bool:
    if expected in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, _JSON_TYPES.get(expected, (object,)))


class BaseTool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    input_schema: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

    @abstractmethod
    def execute(self, **arguments: Any) -> ToolExecutionResult:
        """Perform the tool's action. Called only after :meth:`validate_input` passes.

        Raise :class:`ToolError` for expected failures; :meth:`run` turns it into a result.
        """

    def definition(self) -> ToolDefinition:
        """The description handed to the LLM."""
        return ToolDefinition(self.name, self.description, self.input_schema)

    def validate_input(self, arguments: dict[str, Any]) -> None:
        """Check required keys, unknown keys (when disallowed), JSON types, enums and minimums."""
        if not isinstance(arguments, dict):
            raise ToolInputError(f"{self.name}: arguments must be a JSON object")
        properties: dict[str, Any] = self.input_schema.get("properties", {})
        missing = [k for k in self.input_schema.get("required", []) if k not in arguments]
        if missing:
            raise ToolInputError(f"{self.name}: missing required argument(s) {missing}")
        if self.input_schema.get("additionalProperties") is False:
            unknown = sorted(set(arguments) - set(properties))
            if unknown:
                raise ToolInputError(f"{self.name}: unknown argument(s) {unknown}")
        for key, value in arguments.items():
            spec = properties.get(key)
            if spec is None:
                continue
            expected = spec.get("type")
            if expected and not _type_ok(value, expected):
                raise ToolInputError(f"{self.name}: argument {key!r} must be of type {expected}")
            if "enum" in spec and value not in spec["enum"]:
                raise ToolInputError(f"{self.name}: argument {key!r} must be one of {spec['enum']}")
            if "minimum" in spec and value < spec["minimum"]:
                raise ToolInputError(f"{self.name}: argument {key!r} must be >= {spec['minimum']}")
            if "maximum" in spec and value > spec["maximum"]:
                raise ToolInputError(f"{self.name}: argument {key!r} must be <= {spec['maximum']}")

    def run(self, arguments: dict[str, Any]) -> ToolExecutionResult:
        """Validate, execute, and always return a structured result (never raises)."""
        started = time.perf_counter()
        try:
            if isinstance(arguments, dict):
                # A null optional argument means "use the default".
                required = set(self.input_schema.get("required", []))
                arguments = {k: v for k, v in arguments.items() if v is not None or k in required}
            self.validate_input(arguments)
            result = self.execute(**arguments)
        except ToolInputError as exc:
            result = ToolExecutionResult.failure(self.name, f"ToolInputError: {exc}")
        except ToolError as exc:
            result = ToolExecutionResult.failure(self.name, str(exc))
            result.metadata["retryable"] = exc.retryable
        except Exception as exc:  # boundary: unexpected tool bugs become data, and are logged
            log.exception("tool %s raised unexpectedly", self.name)
            result = ToolExecutionResult.failure(self.name, f"{type(exc).__name__}: {exc}")
        result.metadata.setdefault("duration_ms", round((time.perf_counter() - started) * 1000, 2))
        if not result.ok:
            log.warning("tool %s failed: %s", self.name, result.error)
        return result
