"""Phase 6: Music agent (no real audio), TUI view-model/rendering, input modes, demo mode."""

from __future__ import annotations

import json
import logging
import os
import stat
import wave
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from harness.agents import AgentStatus, CoderAgent, MusicAgent, ResearcherAgent, TesterAgent
from harness.agents.music import parse_music_command
from harness.config.settings import Settings
from harness.context import InMemoryContextManager
from harness.demo import DemoScriptedClient, KNOWN_FIXES, prepare_sample_repo
from harness.main import main
from harness.orchestrator import AgentState, Orchestrator
from harness.orchestrator.events import EventLog, EventType
from harness.orchestrator.final_result import FinalResult
from harness.orchestrator.verification_manager import FinalStatus
from harness.tools import RepositoryContext
from harness.tools import music as music_module
from harness.tools.base import ToolError
from harness.tools.music import SystemAudioPlayer, UnavailableMusicPlayer, build_music_player
from harness.tools.music_synth import find_piece, render
from harness.ui import render as ui
from harness.ui.app import HarnessApp
from harness.ui.model import DashboardModel, humanize, short_command, tool_activity

from .conftest import FAKE_KEY, git
from .test_orchestration_flows import FakePlayer

TASK = "Fix the bug in this repository and play Beethoven Symphony No. 5 while you work."


# --- music: disabled mode -------------------------------------------------------------------


def test_music_disabled_when_backend_none() -> None:
    agent = MusicAgent(None, InMemoryContextManager(), settings=Settings(api_key=FAKE_KEY, music_backend="none"))
    assert agent.disabled and agent.status_text.startswith("DISABLED")
    result = agent.run("play Beethoven", AgentState(task="t"))
    assert result.status is AgentStatus.FAILURE and "Music is disabled" in result.errors[0]


def test_auto_backend_degrades_to_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(music_module.shutil, "which", lambda name: None)
    player = build_music_player("auto")
    assert not player.available and "no local audio player" in player.unavailable_reason
    monkeypatch.setattr(music_module.sys, "platform", "win32")
    assert not build_music_player("auto").available
    assert not build_music_player("disabled").available


def test_all_controls_fail_cleanly_when_disabled() -> None:
    player = UnavailableMusicPlayer()
    for control in (lambda: player.play("x", {}), player.pause, player.resume, player.stop,
                    lambda: player.set_volume(10)):
        with pytest.raises(ToolError, match="disabled"):
            control()


# --- music: system player with a silent fake binary ---------------------------------------------


def fake_binary(tmp_path: Path, name: str = "afplay", fail: bool = False) -> str:
    path = tmp_path / "bin" / name
    path.parent.mkdir(exist_ok=True)
    body = "echo 'no audio device' >&2; exit 1" if fail else "exec sleep 30"
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def test_system_player_controls(tmp_path: Path) -> None:
    player = SystemAudioPlayer(fake_binary(tmp_path), cache_dir=tmp_path / "cache")
    state = player.play("Beethoven Symphony No. 5", {"volume": 40})
    assert state.playing and state.source == "synthesized" and state.volume == 40
    assert "Beethoven" in (state.title or "") and "synthesized excerpt" in (state.title or "")
    assert player.pause().paused
    resumed = player.resume()
    assert resumed.playing and not resumed.paused
    assert player.set_volume(20).volume == 20
    assert not player.stop().playing
    with pytest.raises(ToolError, match="Nothing is playing"):
        player.pause()
    with pytest.raises(ToolError, match="No music found"):
        player.play("an unknown song xyz", {})


def test_system_player_prefers_local_files(tmp_path: Path) -> None:
    library = tmp_path / "music"
    library.mkdir()
    (library / "Moonlight Sonata - Beethoven.mp3").write_bytes(b"not really audio")
    player = SystemAudioPlayer(fake_binary(tmp_path), music_dir=str(library), cache_dir=tmp_path / "c")
    state = player.play("moonlight sonata", {})
    assert state.source == "file" and state.title == "Moonlight Sonata - Beethoven"
    player.stop()


def test_system_player_without_audio_device_fails(tmp_path: Path) -> None:
    player = SystemAudioPlayer(fake_binary(tmp_path, fail=True), cache_dir=tmp_path / "cache")
    with pytest.raises(ToolError, match="no audio device"):
        player.play("ode to joy", {})


def test_synthesized_excerpts(tmp_path: Path) -> None:
    for query, key in [("Beethoven's 5th", "beethoven-5"), ("Beethoven Symphony No. 5", "beethoven-5"),
                       ("ode to joy please", "ode-to-joy"), ("Für Elise", "fur-elise")]:
        assert getattr(find_piece(query), "key", None) == key
    assert find_piece("Taylor Swift") is None
    path = render(find_piece("beethoven 5"), tmp_path / "b5.wav")  # type: ignore[arg-type]
    with wave.open(str(path)) as wav:
        assert wav.getnframes() > 22_050 and wav.getnchannels() == 1


@pytest.mark.parametrize(
    ("command", "expected"),
    [("pause music", ("pause", None, {})), ("resume music", ("resume", None, {})),
     ("stop music", ("stop", None, {})), ("set the volume to 35%", ("volume", None, {"volume": 35})),
     ("play Beethoven Symphony No. 5", ("play", "Beethoven Symphony No. 5", {}))],
)
def test_music_commands(command: str, expected: tuple[Any, ...]) -> None:
    assert parse_music_command(command) == expected


def test_music_agent_controls_with_fake_player() -> None:
    class Controllable(FakePlayer):
        def pause(self):  # type: ignore[no-untyped-def]
            return music_module.PlaybackState("pause", None, False, "fake", paused=True)

        def set_volume(self, level):  # type: ignore[no-untyped-def]
            return music_module.PlaybackState("volume", None, True, "fake", volume=level)

    agent = MusicAgent(None, InMemoryContextManager(), player=Controllable())
    state = AgentState(task="t")
    assert agent.run("play Beethoven", state).status is AgentStatus.SUCCESS
    assert agent.run("pause music", state).summary == "Music: paused"
    assert agent.run("set volume to 30", state).summary == "Music: volume 30%"
    resume = agent.run("resume music", state)
    assert resume.status is AgentStatus.FAILURE and "does not support resume" in resume.errors[0]


# --- demo pipeline (scripted client, real tools) + music isolation ------------------------------


def demo_setup(tmp_path: Path, player: Any) -> tuple[Orchestrator, dict[str, Any], RepositoryContext]:
    settings = Settings(api_key=FAKE_KEY, provider="demo", test_timeout_seconds=60)
    repo = RepositoryContext(prepare_sample_repo(tmp_path))
    llm = DemoScriptedClient()
    orch = Orchestrator(settings, llm=llm)
    ctx = orch.context
    agents = {"coder": CoderAgent(llm, ctx, repo, settings=settings),
              "tester": TesterAgent(llm, ctx, repo, settings=settings),
              "researcher": ResearcherAgent(llm, ctx, repo, settings=settings),
              "music": MusicAgent(None, ctx, player=player)}
    return orch, agents, repo


def test_demo_pipeline_repairs_and_verifies_with_music_disabled(tmp_path: Path) -> None:
    orch, agents, repo = demo_setup(tmp_path, UnavailableMusicPlayer())
    final = orch.execute(TASK, repo, agents=agents)
    assert final.status is FinalStatus.VERIFIED_SUCCESS, final.unresolved_issues
    statuses = {n["id"]: n["status"] for n in final.plan}
    assert statuses["test-1"] == "FAILED" and statuses["repair-1"] == "SUCCESS"
    assert statuses["music-1"] == "FAILED"  # disabled, isolated
    assert all(f[2] in (repo.root / "calc.py").read_text() for f in KNOWN_FIXES)


def test_coding_failure_does_not_stop_music(tmp_path: Path) -> None:
    orch, agents, repo = demo_setup(tmp_path, FakePlayer())
    (repo.root / "calc.py").write_text("def add(a, b):\n    return a - b\n")  # no scripted fix applies
    git(repo.root, "commit", "-qam", "different bug")
    final = orch.execute(TASK, repo, agents=agents)
    assert final.status is not FinalStatus.VERIFIED_SUCCESS
    assert final.music is not None and final.music["node_status"] == "SUCCESS"


def test_music_runs_concurrently_and_finishes_while_coding(tmp_path: Path) -> None:
    orch, agents, repo = demo_setup(tmp_path, FakePlayer(delay=0.2))
    agents["coder"].llm.step_seconds = 0.1  # type: ignore[attr-defined]
    model = DashboardModel(TASK)
    seen: dict[str, dict[str, str]] = {}

    def snapshot(event: Any) -> None:
        if event.event is EventType.AGENT_COMPLETED and event.agent == "music":
            seen["at_music_done"] = {a: model.agent_status(a) for a in ("coder", "tester", "music")}

    final = orch.execute(TASK, repo, agents=agents, listeners=[model.apply, snapshot])
    assert final.status is FinalStatus.VERIFIED_SUCCESS
    assert seen["at_music_done"] == {"coder": "RUNNING", "tester": "WAITING", "music": "SUCCESS"}
    assert ("inspect-1", "music-1") in orch.last_controller.scheduler.concurrent_pairs


# --- view-model / event rendering ---------------------------------------------------------------


def emit(log: EventLog, event_type: EventType, details: str = "", **kw: Any) -> None:
    log.emit(event_type, details, **kw)


def scripted_events(model: DashboardModel) -> EventLog:
    log = EventLog("t1")
    log.subscribe(model.apply)
    emit(log, EventType.TASK_CREATED, TASK)
    emit(log, EventType.PLAN_CREATED, "3 tasks; research=no; music=yes",
         nodes=[{"id": "implement-1", "agent": "coder", "kind": "IMPLEMENT", "description": "Implement"},
                {"id": "test-1", "agent": "tester", "kind": "TEST", "description": "Test"},
                {"id": "music-1", "agent": "music", "kind": "MUSIC", "description": "Music: play Beethoven",
                 "required": False}],
         agents={"coder": "available", "researcher": "available", "tester": "available",
                 "music": "available (system)"})
    emit(log, EventType.AGENT_STARTED, "Implement", agent="coder", node="implement-1")
    emit(log, EventType.AGENT_STARTED, "Music: play Beethoven", agent="music", node="music-1")
    emit(log, EventType.TOOL_STARTED, "read_file src/auth.py", agent="coder", node="implement-1",
         tool="read_file", target="src/auth.py")
    return log


def test_model_tracks_agent_status_and_activity() -> None:
    model = DashboardModel(TASK)
    log = scripted_events(model)
    assert model.agent_status("coder") == "RUNNING" and model.agent_status("tester") == "WAITING"
    assert model.agent_status("music") == "RUNNING" and model.agent_status("researcher") == "IDLE"
    assert model.agents["coder"].activity == "reading src/auth.py"
    emit(log, EventType.TOOL_CALLED, "play_music Beethoven", agent="music", node="music-1",
         tool="play_music", target="Beethoven", summary="Symphony No. 5")
    emit(log, EventType.AGENT_COMPLETED, "Music: playing Symphony No. 5", agent="music", node="music-1",
         kind="MUSIC", usage={})
    assert model.agent_status("music") == "SUCCESS" and model.agent_status("coder") == "RUNNING"
    emit(log, EventType.AGENT_COMPLETED, "done", agent="coder", node="implement-1", kind="IMPLEMENT",
         usage={"llm_turns": 4, "estimated_input_tokens": 1000, "estimated_output_tokens": 50})
    emit(log, EventType.TEST_FAILED, "TEST_FAILURE: Verification FAILED: cmd -> FAIL (1 passed, 2 failed)",
         agent="tester", node="test-1", usage={})
    emit(log, EventType.REPAIR_STARTED, "repair-1 + test-2", agent="coder", node="repair-1", attempt=1,
         new_nodes=[{"id": "repair-1", "agent": "coder", "kind": "REPAIR", "description": "Repair attempt 1: fix"},
                    {"id": "test-2", "agent": "tester", "kind": "TEST", "description": "Re-run"}])
    assert model.agent_status("coder") == "WAITING" and model.repairs == 1
    assert model.test_status == "FAIL (1 passed, 2 failed)"
    assert model.agent_turns == 4 and model.estimated_tokens == 1050 and model.tool_calls == 1
    assert model.progress() == {"completed": 2, "pending": 2, "total": 5}


def test_disabled_music_shows_disabled() -> None:
    model = DashboardModel("fix it")
    log = EventLog("t")
    log.subscribe(model.apply)
    emit(log, EventType.PLAN_CREATED, "1 tasks", nodes=[], agents={"music": "DISABLED (no player)"})
    assert model.agent_status("music") == "DISABLED" and model.agents["music"].disabled_reason == "no player"


def test_event_stream_is_readable_and_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_API_KEY", "super-secret-key-999")
    model = DashboardModel(TASK)
    log = scripted_events(model)
    emit(log, EventType.TOOL_CALLED, "edit", agent="coder", node="implement-1", tool="edit_file",
         target="auth.py", summary="changed")
    emit(log, EventType.TOOL_STARTED, "run_tests", agent="tester", node="test-1", tool="run_tests",
         target="python3 -m pytest -rfE --tb=short --color=no -p no:cacheprovider tests/x.py")
    emit(log, EventType.TEST_PASSED, "Verification PASSED: x -> PASS (3 passed)", agent="tester", node="test-1")
    emit(log, EventType.AGENT_FAILED, "AGENT_FAILURE: token super-secret-key-999 rejected", agent="coder",
         node="implement-1")
    emit(log, EventType.VERIFICATION_PASSED, "All verification checks passed.")
    emit(log, EventType.TASK_COMPLETED, "done")
    lines = model.event_lines()
    text = "\n".join(lines)
    assert lines[0].startswith("00:00 Orchestrator created plan (3 tasks")
    for expected in ("Coder started: Implement", "Coder modified auth.py",
                     "Tester running python3 -m pytest tests/x.py", "All tests passed (3 passed)",
                     "Verification passed", "Task complete: VERIFIED SUCCESS"):
        assert expected in text, expected
    assert "super-secret-key-999" not in text and "--tb=short" not in text
    assert "NODE_READY" not in text


def test_humanize_helpers() -> None:
    assert short_command("python3 -m pytest -rfE --tb=short --color=no -p no:cacheprovider tests/a.py") == \
        "python3 -m pytest tests/a.py"
    assert tool_activity("search_code", "def authenticate") == "searching code for 'def authenticate'"
    assert tool_activity("fetch_documentation", "https://pyjwt.readthedocs.io") == \
        "reading documentation https://pyjwt.readthedocs.io"
    event = EventLog("t").emit(EventType.NODE_READY, "x")
    assert humanize(event) is None


def render_text(renderable: Any, width: int = 110) -> str:
    console = Console(record=True, width=width, color_system=None)
    console.print(renderable)
    return console.export_text()


def test_dashboard_rendering() -> None:
    model = DashboardModel(TASK, mode="DEMO MODE")
    scripted_events(model)
    text = render_text(ui.dashboard(model))
    for expected in ("AI CODING HARNESS", "DEMO MODE", "CURRENT TASK", TASK[:40], "Orchestrator: RUNNING",
                     "Coder", "RUNNING", "Researcher", "IDLE", "Tester", "WAITING", "Music",
                     "reading src/auth.py", "PROGRESS", "METRICS", "tool calls", "agent turns",
                     "elapsed", "(estimated)", "EVENTS"):
        assert expected in text, expected


def final_result(status: FinalStatus, **kw: Any) -> FinalResult:
    state = AgentState(task="t")
    state.repair_attempts = kw.pop("repairs", 0)
    state.research = kw.pop("research", {"reason": "no library"})
    return FinalResult(task_id="t", status=status, summary=kw.pop("summary", "summary text"), state=state, **kw)


def test_final_result_rendering() -> None:
    final = final_result(
        FinalStatus.VERIFIED_SUCCESS, summary="Verified: changed calc.py", files_changed=["calc.py"],
        tests_run=[{"attempt": 1, "command": "python3 -m pytest -rfE --tb=short", "status": "FAIL", "passed": 2, "failed": 1},
                   {"attempt": 2, "command": "python3 -m pytest", "status": "PASS", "passed": 3, "failed": 0}],
        tests_passed=3, repairs=1, retries=1,
        music={"node_status": "SUCCESS", "summary": "Music: playing Beethoven"},
    )
    text = render_text(ui.final_view(final))
    for expected in ("VERIFIED SUCCESS", "Summary", "Verified: changed calc.py", "Files changed", "calc.py",
                     "Tests executed", "attempt 1: python3 -m pytest -> FAIL (2 passed, 1 failed)",
                     "Test results", "3 passed, 0 failed", "Research performed", "no (no library)",
                     "Repair attempts", "Remaining issues", "none", "Music: playing Beethoven"):
        assert expected in text, expected

    blocked = render_text(ui.final_view(final_result(FinalStatus.BLOCKED, unresolved_issues=["tests_passed: FAIL"])))
    assert "BLOCKED" in blocked and "tests_passed: FAIL" in blocked and "no tests were run" in blocked
    assert "FAILED" in render_text(ui.final_view(final_result(FinalStatus.FAILED)))


def test_error_view_is_concise() -> None:
    text = render_text(ui.error_view("Tester failed to execute pytest.\nReason: pytest command unavailable.",
                                     "/tmp/harness.log"))
    assert "Tester failed to execute pytest." in text and "/tmp/harness.log" in text
    assert "Traceback" not in text


# --- app: task input and modes ------------------------------------------------------------------


def make_app(tmp_path: Path, inputs: list[str], player: Any = None) -> tuple[HarnessApp, Console]:
    orch, agents, repo = demo_setup(tmp_path, player or UnavailableMusicPlayer())
    console = Console(record=True, width=110, color_system=None)
    answers = iter(inputs)

    def ask(prompt: str) -> str:
        try:
            return next(answers)
        except StopIteration:
            raise EOFError from None

    app = HarnessApp(orch.settings, repo=repo.root, console=console, input_fn=ask, live=False,
                     agents=agents, orchestrator=orch, mode="DEMO MODE")
    return app, console


def test_interactive_task_input(tmp_path: Path) -> None:
    app, console = make_app(tmp_path, [TASK, "quit"])
    assert app.session() == 0
    text = console.export_text()
    assert "VERIFIED SUCCESS" in text and "FINAL RESULT" in text
    assert app.last_final is not None and app.last_final.status is FinalStatus.VERIFIED_SUCCESS


def test_interactive_session_ends_on_eof(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path, [])
    assert app.session() == 0 and app.last_final is None


def test_task_argument_runs_once(tmp_path: Path) -> None:
    app, console = make_app(tmp_path, ["should not be asked"])
    assert app.session(TASK) == 0
    assert console.export_text().count("FINAL RESULT") == 1


def test_app_reports_crash_without_traceback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app, console = make_app(tmp_path, [])

    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("scheduler exploded")

    monkeypatch.setattr(app.orchestrator, "execute", boom)
    app.log_path = tmp_path / "harness.log"
    assert app.run_task("Fix it") == 5
    text = console.export_text()
    assert "unexpected error: RuntimeError: scheduler exploded" in text
    assert "Traceback" not in text and "harness.log" in text


def test_plan_only_mode_without_provider(tmp_path: Path) -> None:
    repo = prepare_sample_repo(tmp_path)
    console = Console(record=True, width=110, color_system=None)
    app = HarnessApp(Settings(api_key=FAKE_KEY), repo=repo, console=console, live=False)
    assert app.session("Fix the bug and play Beethoven") == 0
    text = console.export_text()
    assert "PLANNED TASK GRAPH" in text and "music-1" in text and "NOT EXECUTED" in text


# --- non-interactive entry points --------------------------------------------------------------


@pytest.fixture
def demo_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("AI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("AI_PROVIDER", "demo")
    monkeypatch.setenv("AI_MUSIC_BACKEND", "none")  # never play audio in tests
    monkeypatch.setenv("AI_DEMO_STEP_SECONDS", "0")
    log_file = tmp_path / "logs" / "harness.log"
    monkeypatch.setenv("AI_LOG_FILE", str(log_file))
    return log_file


def test_non_interactive_json_demo(demo_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--json", "--task", "Fix the sample bug"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "VERIFIED_SUCCESS" and data["files_changed"] == ["calc.py"]
    assert data["music"] is None  # no music requested


def test_forced_tui_non_interactive(demo_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--tui", "--task", TASK]) == 0
    out = capsys.readouterr()
    assert "AI CODING HARNESS" in out.out and "VERIFIED SUCCESS" in out.out
    assert "Music is disabled" in out.out  # music degraded, coding verified
    assert "Traceback" not in out.out + out.err
    log_text = demo_env.read_text()
    assert "TASK_COMPLETED" in log_text and FAKE_KEY not in log_text


def test_log_file_redacts_credentials(demo_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from harness.logging_setup import configure_logging

    monkeypatch.setenv("AI_API_KEY", "very-secret-key-000")
    path = configure_logging(logging.INFO, log_file=str(demo_env), console=False)
    logging.getLogger("harness.test").info("calling provider with very-secret-key-000")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert path is not None and "very-secret-key-000" not in path.read_text()


def test_demo_client_is_scripted_and_honest() -> None:
    from harness.llm import Message

    client = DemoScriptedClient()
    research = client.generate([Message.system("You are the Researcher agent ..."), Message.user("q")])
    assert json.loads(research.content)["status"] == "BLOCKED"
    first = client.generate([Message.system("You are the Coder agent"), Message.user("t")], tools=[])
    assert first.tool_calls[0].name == "list_files" and first.model == "demo-scripted"


def test_core_does_not_import_ui() -> None:
    import subprocess
    import sys

    code = ("import sys, harness.orchestrator, harness.agents, harness.tools, harness.verification;"
            "print(any(m.startswith('harness.ui') or m == 'rich' for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONPATH": "src"})
    assert out.stdout.strip() == "False"
