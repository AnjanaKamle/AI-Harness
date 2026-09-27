"""Verification data model: what was run, what happened, and why it failed."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class TestStatus(StrEnum):
    __test__ = False  # not a pytest test class

    PASS = "PASS"
    FAIL = "FAIL"  # the command ran and reported failing tests/checks
    ERROR = "ERROR"  # the command could not complete normally (collection/infra error)
    TIMEOUT = "TIMEOUT"
    NOT_AVAILABLE = "NOT_AVAILABLE"  # command/tool/script/tests not present


class CheckKind(StrEnum):
    TEST = "TEST"
    BUILD = "BUILD"
    LINT = "LINT"
    TYPECHECK = "TYPECHECK"


class FailureCategory(StrEnum):
    CODE_FAILURE = "CODE_FAILURE"
    TEST_FAILURE = "TEST_FAILURE"
    DEPENDENCY_FAILURE = "DEPENDENCY_FAILURE"
    ENVIRONMENT_FAILURE = "ENVIRONMENT_FAILURE"
    TIMEOUT = "TIMEOUT"
    UNKNOWN_FAILURE = "UNKNOWN_FAILURE"


# Categories the Coder can plausibly fix by changing code. The others stop the repair loop.
REPAIRABLE_CATEGORIES: frozenset[FailureCategory] = frozenset(
    {
        FailureCategory.CODE_FAILURE,
        FailureCategory.TEST_FAILURE,
        FailureCategory.TIMEOUT,
        FailureCategory.UNKNOWN_FAILURE,
    }
)


class VerificationStatus(StrEnum):
    NOT_VERIFIED = "NOT_VERIFIED"
    PASSED = "PASSED"
    FAILED = "FAILED"
    NOT_AVAILABLE = "NOT_AVAILABLE"


@dataclass(frozen=True)
class TestCommand:
    """A verification command chosen from repository evidence (never invented)."""

    __test__ = False

    command: str
    kind: CheckKind
    ecosystem: str  # python | node | make
    evidence: tuple[str, ...]
    gating: bool = True  # False: advisory (reported, but does not block success)
    targeted: bool = False  # runs only tests related to the changed files

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "evidence": list(self.evidence)}


@dataclass
class TestResult:
    __test__ = False

    command: str
    status: TestStatus
    kind: CheckKind = CheckKind.TEST
    exit_code: int | None = None
    passed: int | None = None
    failed: int | None = None
    skipped: int | None = None
    errors: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0
    failure_summary: str = ""
    failed_tests: list[str] = field(default_factory=list)
    gating: bool = True
    targeted: bool = False
    output_truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.status is TestStatus.PASS

    def counts_text(self) -> str:
        parts = [
            f"{n} {label}"
            for n, label in ((self.passed, "passed"), (self.failed, "failed"),
                             (self.errors, "errors"), (self.skipped, "skipped"))
            if n
        ]
        return ", ".join(parts) or "no counts parsed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestResult:
        data = dict(data)
        data["status"] = TestStatus(data["status"])
        data["kind"] = CheckKind(data.get("kind", CheckKind.TEST))
        return cls(**data)


@dataclass(frozen=True)
class FailureClassification:
    category: FailureCategory
    evidence: tuple[str, ...]
    repairable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "evidence": list(self.evidence),
            "repairable": self.repairable,
        }
