"""ContextManager: shared, categorized storage of everything agents may need.

It holds the original task, plan, repository facts, relevant files, research findings,
code changes, test results, failure history and agent outputs. Agents never receive this
store wholesale - :class:`harness.context.builder.ContextBuilder` packages a small,
relevant slice for each agent.

Old entries can be compressed into one structured summary entry per category (no
embeddings, vector databases or RAG).
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class ContextCategory(StrEnum):
    TASK = "TASK"  # original user task
    PLAN = "PLAN"
    REPOSITORY = "REPOSITORY"  # repository facts (profile, detected commands)
    RELEVANT_FILE = "RELEVANT_FILE"
    RESEARCH = "RESEARCH"
    CODE_CHANGE = "CODE_CHANGE"
    TEST_RESULT = "TEST_RESULT"
    FAILURE = "FAILURE"
    AGENT_OUTPUT = "AGENT_OUTPUT"
    SUMMARY = "SUMMARY"  # compressed older entries


@dataclass(frozen=True)
class ContextEntry:
    category: ContextCategory
    key: str
    content: str
    source: str = "system"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    task_id: str | None = None
    tags: tuple[str, ...] = ()


Summarizer = Callable[[list[ContextEntry]], str]


def default_summarizer(entries: list[ContextEntry]) -> str:
    """One line per entry: '<key>: <first line of content>'."""
    lines = []
    for e in entries:
        first = e.content.strip().splitlines()[0] if e.content.strip() else ""
        lines.append(f"- {e.key}: {first[:160]}")
    return "\n".join(lines)


class ContextManager(ABC):
    @abstractmethod
    def put(self, entry: ContextEntry) -> None:
        """Store ``entry``, replacing any existing entry with the same category and key."""

    @abstractmethod
    def get(self, category: ContextCategory, key: str) -> ContextEntry | None:
        """Return one entry, or None."""

    @abstractmethod
    def entries(self, category: ContextCategory | None = None) -> list[ContextEntry]:
        """Entries in insertion order, optionally filtered by category."""

    @abstractmethod
    def remove(self, category: ContextCategory, key: str) -> bool:
        """Delete one entry. Returns True if it existed."""

    @abstractmethod
    def clear(self) -> None:
        """Delete all entries."""

    # --- conveniences built on the abstract API ---------------------------------------

    def for_task(
        self, task_id: str, category: ContextCategory | None = None
    ) -> list[ContextEntry]:
        """Entries belonging to ``task_id`` (or not scoped to any task)."""
        return [e for e in self.entries(category) if e.task_id in (task_id, None)]

    def latest(
        self, category: ContextCategory, task_id: str | None = None
    ) -> ContextEntry | None:
        items = self.for_task(task_id, category) if task_id else self.entries(category)
        return items[-1] if items else None

    def compress(
        self,
        category: ContextCategory,
        *,
        keep_last: int,
        task_id: str | None = None,
        summarizer: Summarizer = default_summarizer,
    ) -> ContextEntry | None:
        """Replace all but the newest ``keep_last`` entries of ``category`` with one SUMMARY
        entry. Returns the summary entry, or None if nothing needed compressing."""
        items = self.for_task(task_id, category) if task_id else self.entries(category)
        old = items[: max(len(items) - keep_last, 0)]
        if not old:
            return None
        key = f"{task_id or 'global'}:{category.value}"
        previous = self.get(ContextCategory.SUMMARY, key)
        body = summarizer(old)
        if previous is not None:
            body = previous.content + "\n" + body
        summary = ContextEntry(
            ContextCategory.SUMMARY,
            key,
            body,
            source="context_manager",
            metadata={
                "category": category.value,
                "compressed_entries": len(old) + int(previous.metadata.get("compressed_entries", 0) if previous else 0),
            },
            task_id=task_id,
        )
        for entry in old:
            self.remove(entry.category, entry.key)
        self.put(summary)
        return summary


class InMemoryContextManager(ContextManager):
    def __init__(self) -> None:
        self._entries: dict[tuple[ContextCategory, str], ContextEntry] = {}
        self._lock = threading.RLock()

    def put(self, entry: ContextEntry) -> None:
        with self._lock:
            self._entries.pop((entry.category, entry.key), None)
            self._entries[(entry.category, entry.key)] = entry

    def get(self, category: ContextCategory, key: str) -> ContextEntry | None:
        with self._lock:
            return self._entries.get((category, key))

    def entries(self, category: ContextCategory | None = None) -> list[ContextEntry]:
        with self._lock:
            return [e for e in self._entries.values() if category is None or e.category == category]

    def remove(self, category: ContextCategory, key: str) -> bool:
        with self._lock:
            return self._entries.pop((category, key), None) is not None

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
