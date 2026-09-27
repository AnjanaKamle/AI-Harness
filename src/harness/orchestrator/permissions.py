"""Formal agent capabilities, enforced by the Orchestrator.

Each tool requires one capability. An agent may only hold tools its capabilities allow, and
may only be assigned graph nodes whose kind it is permitted to perform. Only agents with
``repository_write`` may change the repository, which the scheduler also uses to avoid
conflicting concurrent writes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

from harness.orchestrator.graph import NodeKind


@dataclass(frozen=True)
class AgentCapabilities:
    repository_read: bool = False
    repository_write: bool = False
    terminal: bool = False
    web: bool = False
    music: bool = False

    def allows(self, capability: str) -> bool:
        return bool(getattr(self, capability, False))

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


CODER = AgentCapabilities(repository_read=True, repository_write=True, terminal=True)
RESEARCHER = AgentCapabilities(repository_read=True, web=True)
TESTER = AgentCapabilities(repository_read=True, terminal=True)
MUSIC = AgentCapabilities(music=True)

AGENT_CAPABILITIES: dict[str, AgentCapabilities] = {
    "coder": CODER,
    "researcher": RESEARCHER,
    "tester": TESTER,
    "music": MUSIC,
}

# Which capability each tool requires. Unknown tools are denied.
TOOL_CAPABILITY: dict[str, str] = {
    "list_files": "repository_read",
    "read_file": "repository_read",
    "search_code": "repository_read",
    "git_status": "repository_read",
    "git_diff": "repository_read",
    "git_log": "repository_read",
    "detect_tests": "repository_read",
    "lookup_python_api": "repository_read",
    "write_file": "repository_write",
    "edit_file": "repository_write",
    "terminal": "terminal",
    "run_tests": "terminal",
    "web_search": "web",
    "fetch_documentation": "web",
    "package_info": "web",
    "play_music": "music",
    "pause_music": "music",
    "resume_music": "music",
    "stop_music": "music",
    "set_volume": "music",
}

# Which capability a node kind needs from the agent that runs it.
KIND_CAPABILITY: dict[NodeKind, str | None] = {
    NodeKind.RESEARCH: "web",
    NodeKind.INSPECT: "repository_read",
    NodeKind.IMPLEMENT: "repository_write",
    NodeKind.REPAIR: "repository_write",
    NodeKind.TEST: "terminal",
    NodeKind.BASELINE: "terminal",
    NodeKind.MUSIC: "music",
    NodeKind.VERIFY: None,  # performed by the Orchestrator itself
}


class PermissionViolation(PermissionError):
    pass


def capabilities_for(agent_name: str) -> AgentCapabilities:
    try:
        return AGENT_CAPABILITIES[agent_name]
    except KeyError:
        raise PermissionViolation(f"no capabilities are defined for agent {agent_name!r}") from None


def check_tools(agent_name: str, tool_names: Iterable[str]) -> None:
    """Raise if the agent holds any tool its capabilities do not allow."""
    caps = capabilities_for(agent_name)
    violations = []
    for tool in sorted(tool_names):
        needed = TOOL_CAPABILITY.get(tool)
        if needed is None:
            violations.append(f"{tool} (unknown tool)")
        elif not caps.allows(needed):
            violations.append(f"{tool} (requires {needed})")
    if violations:
        raise PermissionViolation(f"agent {agent_name!r} holds disallowed tools: {violations}")


def check_assignment(agent_name: str, kind: NodeKind) -> None:
    needed = KIND_CAPABILITY.get(kind)
    if needed is not None and not capabilities_for(agent_name).allows(needed):
        raise PermissionViolation(f"agent {agent_name!r} may not perform {kind.value} ({needed} required)")


def repository_access(agent_name: str, kind: NodeKind) -> str:
    """'write', 'read' or 'none' - used by the scheduler to prevent conflicting access."""
    if agent_name not in AGENT_CAPABILITIES:
        return "none" if kind is NodeKind.VERIFY else "read"
    caps = AGENT_CAPABILITIES[agent_name]
    if kind in (NodeKind.IMPLEMENT, NodeKind.REPAIR) and caps.repository_write:
        return "write"
    if caps.repository_read or kind is NodeKind.VERIFY:
        return "read"
    return "none"
