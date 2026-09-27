"""TaskGraph: the plan as a DAG of agent tasks.

A node becomes READY when every dependency has succeeded (an *optional* dependency that
failed does not hold its dependents back). A node whose required dependency failed or was
blocked becomes BLOCKED (or SKIPPED if the node itself is optional). A failed node that
was superseded by a repair no longer counts against completion.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class NodeStatus(StrEnum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    SKIPPED = "SKIPPED"

    @property
    def is_terminal(self) -> bool:
        return self in (NodeStatus.SUCCESS, NodeStatus.FAILED, NodeStatus.BLOCKED, NodeStatus.SKIPPED)


class NodeKind(StrEnum):
    MUSIC = "MUSIC"
    RESEARCH = "RESEARCH"
    INSPECT = "INSPECT"
    IMPLEMENT = "IMPLEMENT"
    REPAIR = "REPAIR"
    TEST = "TEST"
    BASELINE = "BASELINE"  # tests run before any change, for before/after comparison
    VERIFY = "VERIFY"


class GraphError(ValueError):
    """The graph is not a valid DAG (unknown dependency, cycle, duplicate id)."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class TaskNode:
    id: str
    description: str
    agent: str  # coder | researcher | tester | music | orchestrator
    kind: NodeKind
    dependencies: list[str] = field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    result: dict[str, Any] | None = None
    retry_count: int = 0
    priority: int = 0  # higher runs first among READY nodes
    required: bool = True  # optional nodes (e.g. music) never block the task
    payload: dict[str, Any] = field(default_factory=dict)
    superseded_by: str | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def mark(self, status: NodeStatus, *, error: str | None = None) -> None:
        self.status = status
        if error is not None:
            self.error = error
        if status is NodeStatus.RUNNING:
            self.started_at = _now()
        elif status.is_terminal:
            self.finished_at = _now()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TaskGraph:
    def __init__(self, nodes: Iterable[TaskNode] = ()) -> None:
        self.nodes: dict[str, TaskNode] = {}
        for node in nodes:
            self.add(node)

    def add(self, node: TaskNode) -> TaskNode:
        if node.id in self.nodes:
            raise GraphError(f"duplicate node id {node.id!r}")
        self.nodes[node.id] = node
        return node

    def __getitem__(self, node_id: str) -> TaskNode:
        return self.nodes[node_id]

    def __contains__(self, node_id: object) -> bool:
        return node_id in self.nodes

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.nodes.values())

    def __len__(self) -> int:
        return len(self.nodes)

    # --- structure -----------------------------------------------------------------------

    def validate(self) -> None:
        for node in self.nodes.values():
            for dep in node.dependencies:
                if dep not in self.nodes:
                    raise GraphError(f"{node.id} depends on unknown node {dep!r}")
                if dep == node.id:
                    raise GraphError(f"{node.id} depends on itself")
        self.topological_order()

    def topological_order(self) -> list[str]:
        indegree = {nid: 0 for nid in self.nodes}
        for node in self.nodes.values():
            for _ in node.dependencies:
                indegree[node.id] += 1
        order: list[str] = []
        queue = sorted((nid for nid, d in indegree.items() if d == 0), key=self._sort_key)
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for other in self.nodes.values():
                if nid in other.dependencies:
                    indegree[other.id] -= 1
                    if indegree[other.id] == 0:
                        queue.append(other.id)
                        queue.sort(key=self._sort_key)
        if len(order) != len(self.nodes):
            cyclic = sorted(set(self.nodes) - set(order))
            raise GraphError(f"dependency cycle among {cyclic}")
        return order

    def _sort_key(self, node_id: str) -> tuple[int, str]:
        return (-self.nodes[node_id].priority, node_id)

    def dependents(self, node_id: str) -> list[TaskNode]:
        return [n for n in self.nodes.values() if node_id in n.dependencies]

    def independent(self, a: str, b: str) -> bool:
        """True if neither node (transitively) depends on the other."""
        return b not in self.ancestors(a) and a not in self.ancestors(b)

    def ancestors(self, node_id: str) -> set[str]:
        seen: set[str] = set()
        stack = list(self.nodes[node_id].dependencies)
        while stack:
            dep = stack.pop()
            if dep not in seen:
                seen.add(dep)
                stack.extend(self.nodes[dep].dependencies)
        return seen

    # --- state transitions ---------------------------------------------------------------

    def _dependency_state(self, dep: TaskNode) -> str:
        """'ok', 'wait' or 'broken' for a dependency."""
        if dep.status is NodeStatus.SUCCESS:
            return "ok"
        if dep.status.is_terminal:
            return "ok" if not dep.required else "broken"
        return "wait"

    def refresh(self) -> list[TaskNode]:
        """Promote PENDING nodes to READY / BLOCKED / SKIPPED. Returns nodes that changed."""
        changed: list[TaskNode] = []
        progress = True
        while progress:
            progress = False
            for node in self.nodes.values():
                if node.status not in (NodeStatus.PENDING, NodeStatus.READY):
                    continue
                states = [self._dependency_state(self.nodes[d]) for d in node.dependencies]
                if "broken" in states:
                    broken = [d for d in node.dependencies
                              if self._dependency_state(self.nodes[d]) == "broken"]
                    node.mark(
                        NodeStatus.BLOCKED if node.required else NodeStatus.SKIPPED,
                        error=f"dependency did not succeed: {', '.join(broken)}",
                    )
                    changed.append(node)
                    progress = True
                elif node.status is NodeStatus.PENDING and all(s == "ok" for s in states):
                    node.status = NodeStatus.READY
                    changed.append(node)
                    progress = True
        return changed

    def ready(self) -> list[TaskNode]:
        return sorted(
            (n for n in self.nodes.values() if n.status is NodeStatus.READY),
            key=lambda n: self._sort_key(n.id),
        )

    def running(self) -> list[TaskNode]:
        return [n for n in self.nodes.values() if n.status is NodeStatus.RUNNING]

    @property
    def finished(self) -> bool:
        return all(n.status.is_terminal for n in self.nodes.values())

    def unfinished(self) -> list[TaskNode]:
        return [n for n in self.nodes.values() if not n.status.is_terminal]

    def by_kind(self, kind: NodeKind) -> list[TaskNode]:
        return [n for n in self.nodes.values() if n.kind is kind]

    def next_id(self, prefix: str) -> str:
        n = 1
        while f"{prefix}-{n}" in self.nodes:
            n += 1
        return f"{prefix}-{n}"

    # --- presentation --------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [self.nodes[nid].to_dict() for nid in self.topological_order()],
            "order": self.topological_order(),
        }

    def describe(self) -> str:
        """Human-readable plan, e.g. 'research-1 [researcher] <- (none)'."""
        lines = []
        for nid in self.topological_order():
            node = self.nodes[nid]
            deps = ", ".join(node.dependencies) or "none"
            flag = "" if node.required else " (optional)"
            lines.append(f"{nid} [{node.agent}] {node.status.value}{flag} <- {deps}: {node.description}")
        return "\n".join(lines)
