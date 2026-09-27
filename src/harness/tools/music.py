"""Music tools behind a pluggable MusicPlayer backend (optional side-effect capability).

Backends:
  auto / system  a local system audio player (afplay on macOS; aplay / paplay / ffplay on
                 Linux). Sources: a matching audio file in AI_MUSIC_DIR, else a synthesized
                 excerpt of a public-domain piece. No services, accounts or credentials.
  none/disabled  music is DISABLED; every request fails cleanly and nothing else is affected.

``auto`` degrades to DISABLED when no working player exists (no binary, no audio device).
Music never touches the repository and never influences coding verification.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from harness.tools.base import BaseTool, ToolError, ToolExecutionResult
from harness.tools.policy import sanitized_environment
from harness.tools.music_synth import find_piece, render

log = logging.getLogger("harness.music")

MAX_QUERY_CHARS = 200
DEFAULT_VOLUME = 70
AUDIO_EXTENSIONS = (".mp3", ".m4a", ".wav", ".aiff", ".aif", ".flac", ".ogg")
STARTUP_CHECK_SECONDS = 1.0


@dataclass(frozen=True)
class PlaybackState:
    action: str  # play | pause | resume | stop | volume | status
    query: str | None
    playing: bool
    backend: str
    detail: str = ""
    title: str | None = None
    source: str | None = None  # file | synthesized
    volume: int | None = None
    paused: bool = False


class MusicPlayer(ABC):
    name: str

    @property
    def available(self) -> bool:
        return True

    @property
    def unavailable_reason(self) -> str:
        return ""

    @abstractmethod
    def play(self, query: str, parameters: dict[str, Any]) -> PlaybackState:
        """Start playback of ``query``. Raise ToolError on failure."""

    @abstractmethod
    def stop(self) -> PlaybackState:
        """Stop playback. Raise ToolError on failure."""

    def pause(self) -> PlaybackState:
        raise ToolError(f"{self.name} player does not support pause")

    def resume(self) -> PlaybackState:
        raise ToolError(f"{self.name} player does not support resume")

    def set_volume(self, level: int) -> PlaybackState:
        raise ToolError(f"{self.name} player does not support volume")


class UnavailableMusicPlayer(MusicPlayer):
    """Music DISABLED: every request fails with the reason; nothing else is affected."""

    name = "disabled"

    def __init__(self, reason: str = "No music backend is configured (AI_MUSIC_BACKEND=none)") -> None:
        self.reason = reason

    @property
    def available(self) -> bool:
        return False

    @property
    def unavailable_reason(self) -> str:
        return self.reason

    def _fail(self) -> PlaybackState:
        raise ToolError(f"Music is disabled: {self.reason}")

    def play(self, query: str, parameters: dict[str, Any]) -> PlaybackState:
        return self._fail()

    def stop(self) -> PlaybackState:
        return self._fail()

    def pause(self) -> PlaybackState:
        return self._fail()

    def resume(self) -> PlaybackState:
        return self._fail()

    def set_volume(self, level: int) -> PlaybackState:
        return self._fail()


# (binary, args-before-file, volume flag builder)
_PLAYERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("afplay", ()),  # macOS
    ("paplay", ()),  # PulseAudio / PipeWire
    ("aplay", ("-q",)),  # ALSA (wav only)
    ("ffplay", ("-nodisp", "-autoexit", "-loglevel", "quiet")),
)


def detect_player_binary() -> str | None:
    for binary, _ in _PLAYERS:
        if shutil.which(binary):
            return binary
    return None


class SystemAudioPlayer(MusicPlayer):
    """Plays audio through a local system player in a background process."""

    name = "system"

    def __init__(self, binary: str, *, music_dir: str | None = None, cache_dir: Path | None = None) -> None:
        self.binary = binary
        self.args = dict(_PLAYERS).get(Path(binary).name, ())
        self.music_dir = Path(music_dir).expanduser() if music_dir else None
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "ai-coding-harness-music"
        self.volume = DEFAULT_VOLUME
        self._proc: subprocess.Popen[bytes] | None = None
        self._current: tuple[str, str, Path, str] | None = None  # query, title, path, source
        self._paused = False
        self._lock = threading.RLock()

    # --- sources -------------------------------------------------------------------------

    def _find_file(self, query: str) -> Path | None:
        if not self.music_dir or not self.music_dir.is_dir():
            return None
        words = [w for w in query.lower().replace("'", " ").split() if len(w) > 1]
        best: tuple[int, Path] | None = None
        for path in self.music_dir.rglob("*"):
            if path.suffix.lower() not in AUDIO_EXTENSIONS or not path.is_file():
                continue
            name = path.stem.lower()
            score = sum(1 for w in words if w in name)
            if score and (best is None or score > best[0]):
                best = (score, path)
        return best[1] if best else None

    def _source(self, query: str) -> tuple[str, Path, str]:
        found = self._find_file(query)
        if found is not None:
            return found.stem, found, "file"
        piece = find_piece(query)
        if piece is None:
            where = f" or in {self.music_dir}" if self.music_dir else ""
            raise ToolError(f"No music found for {query!r} (no synthesized excerpt{where})")
        path = render(piece, self.cache_dir / f"{piece.key}-v{self.volume}.wav", volume=self.volume)
        return f"{piece.composer}: {piece.title} [synthesized excerpt]", path, "synthesized"

    # --- process control -------------------------------------------------------------------

    def _command(self, path: Path) -> list[str]:
        cmd = [self.binary, *self.args]
        if Path(self.binary).name == "afplay":
            cmd += ["-v", f"{self.volume / 100:.2f}"]
        elif Path(self.binary).name == "ffplay":
            cmd += ["-volume", str(self.volume)]
        return [*cmd, str(path)]

    def _start(self, path: Path) -> None:
        self._stop_process()
        try:
            self._proc = subprocess.Popen(
                self._command(path), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, start_new_session=True, env=sanitized_environment(),
            )
        except OSError as exc:
            raise ToolError(f"Could not start {self.binary}: {exc}") from None
        try:  # returns early if the player exits (e.g. no audio device)
            code: int | None = self._proc.wait(timeout=STARTUP_CHECK_SECONDS)
        except subprocess.TimeoutExpired:
            code = None  # still running: playback started
        if code not in (None, 0):  # died immediately: typically no audio device
            err = (self._proc.stderr.read() if self._proc.stderr else b"").decode("utf-8", "replace")
            self._proc = None
            raise ToolError(f"{self.binary} failed (exit {code}): {err.strip()[:200] or 'no audio output available'}")
        self._paused = False

    def _stop_process(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=2)
            except (ProcessLookupError, subprocess.TimeoutExpired, PermissionError):
                proc.kill()

    def _playing(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _state(self, action: str, detail: str = "") -> PlaybackState:
        query, title, _, source = self._current or (None, None, None, None)
        return PlaybackState(action, query, self._playing() and not self._paused, self.name, detail,
                             title=title, source=source, volume=self.volume, paused=self._paused)

    # --- controls --------------------------------------------------------------------------

    def play(self, query: str, parameters: dict[str, Any]) -> PlaybackState:
        with self._lock:
            if "volume" in parameters and parameters["volume"] is not None:
                self.volume = max(0, min(int(parameters["volume"]), 100))
            title, path, source = self._source(query)
            self._current = (query, title, path, source)
            self._start(path)
            log.info("music: playing %s via %s", title, self.binary)
            return self._state("play", f"playing {title}")

    def pause(self) -> PlaybackState:
        with self._lock:
            if not self._playing():
                raise ToolError("Nothing is playing")
            os.killpg(self._proc.pid, signal.SIGSTOP)  # type: ignore[union-attr]
            self._paused = True
            return self._state("pause", "paused")

    def resume(self) -> PlaybackState:
        with self._lock:
            if not self._playing() or not self._paused:
                raise ToolError("Nothing is paused")
            os.killpg(self._proc.pid, signal.SIGCONT)  # type: ignore[union-attr]
            self._paused = False
            return self._state("resume", "resumed")

    def stop(self) -> PlaybackState:
        with self._lock:
            was_playing = self._playing()
            self._stop_process()
            self._paused = False
            return self._state("stop", "stopped" if was_playing else "nothing was playing")

    def set_volume(self, level: int) -> PlaybackState:
        with self._lock:
            self.volume = max(0, min(int(level), 100))
            if self._playing() and self._current is not None and not self._paused:
                query = self._current[0]
                title, path, source = self._source(query)  # re-render at the new level
                self._current = (query, title, path, source)
                self._start(path)
            return self._state("volume", f"volume {self.volume}%")


def build_music_player(backend: str, *, music_dir: str | None = None) -> MusicPlayer:
    """Choose a backend; ``auto`` degrades to DISABLED when no player is available."""
    backend = backend.lower()
    if backend in ("none", "disabled"):
        return UnavailableMusicPlayer()
    if sys.platform.startswith("win"):
        return UnavailableMusicPlayer("system audio playback is not supported on this platform")
    binary = detect_player_binary()
    if binary is None:
        return UnavailableMusicPlayer("no local audio player found (afplay/paplay/aplay/ffplay)")
    if backend in ("auto", "system"):
        return SystemAudioPlayer(binary, music_dir=music_dir)
    raise ValueError(f"Unknown music backend {backend!r}")


# --- tools ------------------------------------------------------------------------------------


class _MusicTool(BaseTool):
    def __init__(self, player: MusicPlayer) -> None:
        self.player = player


class PlayMusicTool(_MusicTool):
    name = "play_music"
    description = "Start playing music matching a query (artist, composer, piece or genre)."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "volume": {"type": "integer", "minimum": 0, "maximum": 100},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def execute(self, query: str, volume: int | None = None, **_: Any) -> ToolExecutionResult:
        query = query.strip()
        if not query or len(query) > MAX_QUERY_CHARS:
            raise ToolError(f"query must be 1-{MAX_QUERY_CHARS} characters")
        params = {"volume": volume} if volume is not None else {}
        return ToolExecutionResult.success(self.name, asdict(self.player.play(query, params)))


class PauseMusicTool(_MusicTool):
    name = "pause_music"
    description = "Pause music playback."
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    def execute(self, **_: Any) -> ToolExecutionResult:
        return ToolExecutionResult.success(self.name, asdict(self.player.pause()))


class ResumeMusicTool(_MusicTool):
    name = "resume_music"
    description = "Resume paused music."
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    def execute(self, **_: Any) -> ToolExecutionResult:
        return ToolExecutionResult.success(self.name, asdict(self.player.resume()))


class StopMusicTool(_MusicTool):
    name = "stop_music"
    description = "Stop music playback."
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    def execute(self, **_: Any) -> ToolExecutionResult:
        return ToolExecutionResult.success(self.name, asdict(self.player.stop()))


class SetVolumeTool(_MusicTool):
    name = "set_volume"
    description = "Set the music volume (0-100)."
    input_schema = {
        "type": "object",
        "properties": {"level": {"type": "integer", "minimum": 0, "maximum": 100}},
        "required": ["level"],
        "additionalProperties": False,
    }

    def execute(self, level: int, **_: Any) -> ToolExecutionResult:
        return ToolExecutionResult.success(self.name, asdict(self.player.set_volume(level)))


def build_music_tools(player: MusicPlayer) -> list[BaseTool]:
    return [PlayMusicTool(player), PauseMusicTool(player), ResumeMusicTool(player),
            StopMusicTool(player), SetVolumeTool(player)]
