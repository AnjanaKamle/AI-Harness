"""Scheduler: runs READY graph nodes concurrently, respecting dependencies and conflicts.

- asyncio drives scheduling; agent work (blocking I/O, subprocesses, model calls) runs in a
  bounded ThreadPoolExecutor with ``max_concurrent`` workers - no uncontrolled threads.
- repository access is a readers/writer lock: a node that may write the repository (only
  the Coder implementing/repairing) never runs alongside any other repository reader or
  writer; read-only nodes may share; music (no repository access) runs alongside anything.
- one node per agent instance at a time (agents are not re-entrant).
- per-node time limit: on expiry the node's CancellationToken is cancelled (no further tool
  calls) and the scheduler waits for the operation to stop. If it does not stop within the
  grace period it is abandoned and keeps its repository lock, so no writer can start while it
  might still be running.
- results are handed to ``on_complete`` on the event-loop thread, one at a time - the only
  place shared state is mutated.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from harness.orchestrator.events import EventLog, EventType
from harness.orchestrator.graph import NodeStatus, TaskGraph, TaskNode
from harness.tools.registry import CancellationToken

log = logging.getLogger("harness.orchestrator.scheduler")

DEFAULT_GRACE_SECONDS = 30.0


@dataclass
class NodeOutcome:
    result: Any = None
    snapshot: Any = None  # the state copy the agent worked on
    baseline: Any = None  # the state copy at dispatch time (for commit validation)
    error: str | None = None
    traceback: str | None = None
    timed_out: bool = False
    abandoned: bool = False
    duration: float = 0.0


@dataclass
class _Running:
    node: TaskNode
    token: CancellationToken
    access: str
    agent_key: str
    deadline: float
    prepared: Any
    started: float = field(default_factory=time.monotonic)
    cancelled_at: float | None = None


Prepare = Callable[[TaskNode], Any]
Execute = Callable[[TaskNode, CancellationToken, Any], NodeOutcome]
Complete = Callable[[TaskNode, NodeOutcome], None]
Access = Callable[[TaskNode], str]


class Scheduler:
    def __init__(
        self,
        graph: TaskGraph,
        *,
        prepare: Prepare,
        execute: Execute,
        on_complete: Complete,
        access: Access,
        agent_key: Callable[[TaskNode], str] = lambda n: n.agent,
        max_concurrent: int = 3,
        node_timeout: float = 900.0,
        grace_seconds: float = DEFAULT_GRACE_SECONDS,
        events: EventLog | None = None,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self.graph = graph
        self.prepare = prepare
        self.execute = execute
        self.on_complete = on_complete
        self.access = access
        self.agent_key = agent_key
        self.max_concurrent = max_concurrent
        self.node_timeout = node_timeout
        self.grace_seconds = grace_seconds
        self.events = events or EventLog()
        self.max_observed_concurrency = 0
        self.concurrent_pairs: set[tuple[str, str]] = set()
        self._abandoned: list[_Running] = []

    # --- admission -----------------------------------------------------------------------

    def _can_start(self, node: TaskNode, active: list[_Running]) -> bool:
        access = self.access(node)
        key = self.agent_key(node)
        for other in active + self._abandoned:
            if other.agent_key == key:
                return False
            if access == "write" and other.access in ("read", "write"):
                return False
            if access == "read" and other.access == "write":
                return False
        return True

    def _guarded(self, node: TaskNode, token: CancellationToken, prepared: Any) -> NodeOutcome:
        started = time.monotonic()
        try:
            outcome = self.execute(node, token, prepared)
        except Exception as exc:  # boundary: a crash becomes a structured failure
            log.debug("node %s crashed", node.id, exc_info=True)
            outcome = NodeOutcome(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        outcome.duration = round(time.monotonic() - started, 3)
        return outcome

    # --- main loop -----------------------------------------------------------------------

    def run_sync(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        pool = ThreadPoolExecutor(max_workers=self.max_concurrent, thread_name_prefix="harness-agent")
        running: dict[asyncio.Future[NodeOutcome], _Running] = {}
        try:
            while True:
                self._refresh()
                for node in self.graph.ready():
                    if len(running) >= self.max_concurrent:
                        break
                    if not self._can_start(node, list(running.values())):
                        continue
                    prepared = self.prepare(node)
                    token = CancellationToken()
                    node.mark(NodeStatus.RUNNING)
                    entry = _Running(node, token, self.access(node), self.agent_key(node),
                                     time.monotonic() + self.node_timeout, prepared)
                    for other in running.values():
                        self.concurrent_pairs.add(tuple(sorted((node.id, other.node.id))))  # type: ignore[arg-type]
                    future = loop.run_in_executor(pool, self._guarded, node, token, prepared)
                    running[future] = entry
                    self.events.emit(EventType.AGENT_STARTED, node.description, agent=node.agent,
                                     node=node.id, attempt=node.retry_count + 1)
                self.max_observed_concurrency = max(self.max_observed_concurrency, len(running))

                if not running:
                    stuck = self.graph.ready()
                    for node in stuck:  # cannot happen with a valid graph; never spin forever
                        node.mark(NodeStatus.BLOCKED, error="could not be scheduled")
                    if not stuck:
                        break
                    continue

                now = time.monotonic()
                wake = min(
                    (r.cancelled_at + self.grace_seconds) if r.cancelled_at else r.deadline
                    for r in running.values()
                )
                done, _ = await asyncio.wait(
                    running.keys(), timeout=max(wake - now, 0.01),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                now = time.monotonic()
                for future, entry in list(running.items()):
                    if future in done:
                        continue
                    if entry.cancelled_at is None and now >= entry.deadline:
                        entry.cancelled_at = now
                        entry.token.cancel(f"time limit of {self.node_timeout:g}s exceeded")
                        self.events.emit(EventType.AGENT_TIMEOUT,
                                         f"exceeded {self.node_timeout:g}s; terminating",
                                         agent=entry.node.agent, node=entry.node.id)
                    elif entry.cancelled_at is not None and now >= entry.cancelled_at + self.grace_seconds:
                        running.pop(future)
                        self._abandoned.append(entry)
                        self.on_complete(entry.node, NodeOutcome(
                            error="operation did not stop after cancellation; abandoned",
                            timed_out=True, abandoned=True,
                            duration=round(now - entry.started, 3),
                        ))
                for future in done:
                    entry = running.pop(future)
                    outcome = future.result()
                    outcome.timed_out = outcome.timed_out or entry.cancelled_at is not None
                    self.on_complete(entry.node, outcome)
        finally:
            pool.shutdown(wait=not self._abandoned, cancel_futures=True)

    def _refresh(self) -> None:
        for node in self.graph.refresh():
            if node.status is NodeStatus.READY:
                self.events.emit(EventType.NODE_READY, node.description, agent=node.agent, node=node.id)
            elif node.status is NodeStatus.BLOCKED:
                self.events.emit(EventType.NODE_BLOCKED, node.error or "", agent=node.agent, node=node.id)
            elif node.status is NodeStatus.SKIPPED:
                self.events.emit(EventType.NODE_SKIPPED, node.error or "", agent=node.agent, node=node.id)
