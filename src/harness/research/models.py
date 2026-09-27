"""Structured research findings and source verification.

A finding is one of:
  FACT         established by a source the Researcher actually retrieved this session
  INFERENCE    reasoned from evidence; may cite a retrieved source, never an unseen one
  UNCERTAINTY  something that could not be established (no source required)

Sources are verified by the harness against what tools really returned - a model cannot
cite a URL, file or API it never retrieved.
"""

from __future__ import annotations

import urllib.parse
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class FindingKind(StrEnum):
    FACT = "FACT"
    INFERENCE = "INFERENCE"
    UNCERTAINTY = "UNCERTAINTY"


class Confidence(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class FindingError(ValueError):
    """A finding is malformed or cites an unverifiable source."""


@dataclass(frozen=True)
class ResearchFinding:
    question: str
    finding: str
    kind: FindingKind
    source: str | None
    relevance: str
    confidence: Confidence
    technical_details: str = ""
    recommended_action: str = ""
    source_verified: bool = False

    def __post_init__(self) -> None:
        if not self.question.strip() or not self.finding.strip():
            raise FindingError("finding requires non-empty 'question' and 'finding'")
        if self.kind is FindingKind.FACT and not self.source:
            raise FindingError("a FACT must cite the source it was established from")
        if self.kind is FindingKind.UNCERTAINTY and self.confidence is Confidence.HIGH:
            object.__setattr__(self, "confidence", Confidence.LOW)

    @property
    def usable(self) -> bool:
        """Can the Coder act on this? (facts, and inferences not marked low-confidence)"""
        return self.kind is FindingKind.FACT or (
            self.kind is FindingKind.INFERENCE and self.confidence is not Confidence.LOW
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchFinding:
        """Parse model/stored output; raises FindingError with a precise reason."""
        if not isinstance(data, dict):
            raise FindingError("finding must be a JSON object")
        try:
            kind = FindingKind(str(data.get("kind", "")).upper())
        except ValueError:
            raise FindingError(
                f"'kind' must be one of {[k.value for k in FindingKind]}, got {data.get('kind')!r}"
            ) from None
        try:
            confidence = Confidence(str(data.get("confidence", "LOW")).upper())
        except ValueError:
            raise FindingError(
                f"'confidence' must be one of {[c.value for c in Confidence]}"
            ) from None
        source = data.get("source")
        return cls(
            question=str(data.get("question", "")),
            finding=str(data.get("finding", "")),
            kind=kind,
            source=str(source).strip() if source else None,
            relevance=str(data.get("relevance", "")),
            confidence=confidence,
            technical_details=str(data.get("technical_details", "") or ""),
            recommended_action=str(data.get("recommended_action", "") or ""),
            source_verified=bool(data.get("source_verified", False)),
        )

    def one_line(self) -> str:
        src = f" (source: {self.source})" if self.source else ""
        return f"[{self.kind.value}/{self.confidence.value}] {self.finding}{src}"


# --- source verification --------------------------------------------------------------


def normalize_source(source: str) -> str:
    s = source.strip()
    lowered = s.lower()
    if lowered.startswith(("http://", "https://")):
        parts = urllib.parse.urlsplit(s)
        path = parts.path.rstrip("/") or "/"
        return urllib.parse.urlunsplit(
            (parts.scheme.lower(), parts.netloc.lower(), path, parts.query, "")
        )
    for prefix in ("repo:", "file:", "./"):
        if lowered.startswith(prefix):
            s = s[len(prefix):]
            break
    return s


@dataclass
class SourceLedger:
    """Every source the Researcher's tools actually returned this session."""

    evidence: set[str]  # retrieved content: fetched pages, registry data, API lookups, files
    leads: set[str]  # seen only as search-result links (not read)

    @classmethod
    def empty(cls) -> SourceLedger:
        return cls(set(), set())

    def add_evidence(self, sources: Iterable[str]) -> None:
        self.evidence.update(normalize_source(s) for s in sources if s)

    def add_leads(self, sources: Iterable[str]) -> None:
        self.leads.update(normalize_source(s) for s in sources if s)

    def status(self, source: str) -> str:
        """'evidence', 'lead' or 'unknown'."""
        norm = normalize_source(source)
        if norm in self.evidence:
            return "evidence"
        # python:<target> may be cited without the @version suffix
        if norm.startswith("python:") and any(
            e.split("@")[0] == norm.split("@")[0] for e in self.evidence if e.startswith("python:")
        ):
            return "evidence"
        if norm in self.leads:
            return "lead"
        return "unknown"

    def sorted_sources(self) -> list[str]:
        return sorted(self.evidence)


def verify_finding(finding: ResearchFinding, ledger: SourceLedger) -> ResearchFinding:
    """Return the finding with source_verified set, or raise FindingError if it cites a
    source that was never retrieved (fabricated). A FACT backed only by a search snippet
    is downgraded to INFERENCE."""
    if not finding.source:
        return finding
    status = ledger.status(finding.source)
    if status == "unknown":
        raise FindingError(f"cites a source that was never retrieved: {finding.source!r}")
    if status == "lead":
        return ResearchFinding(
            **{
                **finding.to_dict(),
                "kind": FindingKind.INFERENCE if finding.kind is FindingKind.FACT else finding.kind,
                "confidence": Confidence.LOW,
                "source_verified": False,
                "technical_details": (
                    finding.technical_details + " [source seen only as a search snippet]"
                ).strip(),
            }
        )
    return ResearchFinding(**{**finding.to_dict(), "source_verified": True})
