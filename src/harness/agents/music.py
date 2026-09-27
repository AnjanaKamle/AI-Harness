"""MusicAgent: a capability controller, not an LLM agent.

It exposes play / pause / resume / stop / volume, sees only the playback command and its
parameters (ContextBuilder.for_music) - never the coding task, repository or history - and
has no repository access. When the environment cannot play audio the agent is DISABLED:
every request fails cleanly and the rest of the harness continues normally.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from harness.agents.base import AgentResult, AgentStatus, BaseAgent
from harness.config.settings import Settings
from harness.context.builder import ContextBuilder
from harness.context.manager import ContextManager
from harness.llm.client import LLMClient
from harness.orchestrator.state import AgentState
from harness.tools.base import BaseTool
from harness.tools.music import MusicPlayer, UnavailableMusicPlayer, build_music_player, build_music_tools
from harness.tools.registry import ToolRegistry

ACTIONS = ("play", "pause", "resume", "stop", "volume")

_STOP = re.compile(r"^\s*(?:stop|silence|turn off)\b", re.IGNORECASE)
_PAUSE = re.compile(r"^\s*pause\b", re.IGNORECASE)
_RESUME = re.compile(r"^\s*(?:resume|unpause|continue)\b", re.IGNORECASE)
_PLAY = re.compile(r"^\s*(?:play|put on|queue up|queue|listen to)\s+(?:some\s+)?(.+?)\s*$", re.IGNORECASE)
_VOLUME_ONLY = re.compile(
    r"^\s*(?:set\s+(?:the\s+)?)?volume\s+(?:to\s+)?(\d{1,3})\s*%?\s*$|"
    r"^\s*(?:turn|set)\s+(?:the\s+)?(?:music|volume)\s+(?:up|down\s+)?to\s+(\d{1,3})\s*%?\s*$",
    re.IGNORECASE,
)
_VOLUME = re.compile(r"\bat\s+(\d{1,3})\s*%?\s*volume\b|\bvolume\s+(\d{1,3})\b", re.IGNORECASE)


@dataclass
class MusicResult(AgentResult):
    action: str = ""
    query: str | None = None
    playback: dict[str, Any] = field(default_factory=dict)
    context_chars: int = 0
    disabled: bool = False


def parse_music_command(command: str) -> tuple[str, str | None, dict[str, Any]]:
    """('play', 'Beethoven', {'volume': 30}) / ('pause', None, {}) / ('volume', None,
    {'volume': 40}) / ('unknown', None, {})."""
    only = _VOLUME_ONLY.match(command)
    if only:
        return "volume", None, {"volume": min(int(only.group(1) or only.group(2)), 100)}
    params: dict[str, Any] = {}
    volume = _VOLUME.search(command)
    if volume:
        params["volume"] = min(int(volume.group(1) or volume.group(2)), 100)
        command = _VOLUME.sub("", command)
    if _STOP.match(command):
        return "stop", None, params
    if _PAUSE.match(command):
        return "pause", None, params
    if _RESUME.match(command):
        return "resume", None, params
    play = _PLAY.match(command)
    if play:
        return "play", play.group(1).strip(" .,!"), params
    return "unknown", None, params


class MusicAgent(BaseAgent):
    name = "music"
    description = "Controls music playback (play/pause/resume/stop/volume). No repository access."

    def __init__(
        self,
        llm: LLMClient | None,
        context: ContextManager,
        tools: Sequence[BaseTool] | None = None,
        *,
        player: MusicPlayer | None = None,
        settings: Settings | None = None,
    ) -> None:
        if tools is None:
            self.player = player or build_music_player(
                settings.music_backend if settings else "none",
                music_dir=settings.music_dir if settings else None,
            )
            tools = build_music_tools(self.player)
        else:
            self.player = player or UnavailableMusicPlayer("custom tools without a player")
        super().__init__(llm, context, tools)  # type: ignore[arg-type]
        self.registry = ToolRegistry(self.tools.values())

    @property
    def disabled(self) -> bool:
        return not self.player.available

    @property
    def status_text(self) -> str:
        if self.disabled:
            return f"DISABLED ({self.player.unavailable_reason})"
        return f"available ({self.player.name})"

    def execute(self, task: str, state: AgentState) -> MusicResult:
        action, query, params = parse_music_command(task)
        package = ContextBuilder.for_music(task, {"action": action, "query": query, **params})
        base: dict[str, Any] = {"action": action, "query": query, "disabled": self.disabled,
                                "context_chars": package.stats["context_chars"]}
        if action == "unknown":
            return MusicResult(self.name, AgentStatus.FAILURE,
                               f"Could not understand the music command: {task!r}",
                               errors=["unrecognized music command"], **base)
        if action == "play":
            outcome = self.registry.execute("play_music", {"query": query, **params})
        elif action == "volume":
            outcome = self.registry.execute("set_volume", {"level": params["volume"]})
        else:
            outcome = self.registry.execute(f"{action}_music", {})
        if not outcome.ok:
            return MusicResult(self.name, AgentStatus.FAILURE, f"Music {action} failed: {outcome.error}",
                               errors=[outcome.error or "music tool failed"], **base)
        title = outcome.data.get("title") or query or ""
        summary = {"play": f"Music: playing {title}", "pause": "Music: paused",
                   "resume": "Music: resumed", "stop": "Music: stopped",
                   "volume": f"Music: volume {outcome.data.get('volume')}%"}[action]
        return MusicResult(self.name, AgentStatus.SUCCESS, summary, playback=outcome.data, **base)

    def shutdown(self) -> None:
        """Stop playback when the session ends (best effort)."""
        if not self.disabled:
            try:
                self.player.stop()
            except Exception:  # noqa: BLE001 - never fail on cleanup
                pass
