"""Verification contract: decides PASS (verified result) vs FAIL (recovery / retry)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from harness.core.state import SharedState


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    summary: str = ""
    failures: list[str] = field(default_factory=list)

    @classmethod
    def ok(cls, summary: str = "") -> VerificationResult:
        return cls(passed=True, summary=summary)

    @classmethod
    def fail(cls, *failures: str, summary: str = "") -> VerificationResult:
        return cls(passed=False, summary=summary, failures=list(failures))


class Verifier(ABC):
    @abstractmethod
    def verify(self, state: SharedState) -> VerificationResult:
        """Inspect ``state`` and report whether the task's output is acceptable."""
