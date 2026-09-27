from __future__ import annotations

import json
from typing import Any

import pytest

from harness.agents import AgentStatus, CoderAgent, CoderResult, ToolCallingLoop
from harness.agents.coder import CODER_SYSTEM_PROMPT, parse_final_response
from harness.config.settings import Settings
from harness.context import ContextCategory, InMemoryContextManager
from harness.llm import LLMError, LLMResponse, Message, Role, StopReason, ToolCall
from harness.orchestrator import AgentState, Orchestrator
from harness.tools import RepositoryContext, build_registry

from .conftest import BUGGY_APP, FAKE_KEY, MockLLMClient, git

TASK = "Fix the add function so it returns the sum."


def call(name: str, call_id: str | None = None, **arguments: Any) -> ToolCall:
    return ToolCall(call_id or f"call-{name}", name, arguments)


def tools(*calls: ToolCall) -> LLMResponse:
    return LLMResponse(content="", stop_reason=StopReason.TOOL_USE, tool_calls=calls)


def final(status: str = "SUCCESS", files: list[str] | None = None, **extra: Any) -> LLMResponse:
    body = {
        "status": status,
        "summary": extra.pop("summary", "Changed subtraction to addition in add()."),
        "files_changed": files if files is not None else ["app.py"],
        "next_action": extra.pop("next_action", "Run the tests."),
        "errors": extra.pop("errors", []),
    }
    return LLMResponse(content=json.dumps(body), stop_reason=StopReason.END_TURN)


def last_tool_payload(messages: list[Message]) -> list[dict[str, Any]]:
    tool_msg = next(m for m in reversed(messages) if m.role is Role.TOOL)
    return [json.loads(r.content) for r in tool_msg.tool_results]


def make_coder(
    repo: RepositoryContext, script: list[Any], max_tool_calls: int = 20
) -> tuple[CoderAgent, MockLLMClient, InMemoryContextManager]:
    llm = MockLLMClient(script)
    ctx = InMemoryContextManager()
    return CoderAgent(llm, ctx, repo, max_tool_calls=max_tool_calls), llm, ctx


def happy_path_script() -> list[Any]:
    """A scripted 'model' that inspects, searches, reads, edits, checks the diff, reports.

    Callables assert on the tool results it receives, proving the tools really ran.
    """

    def after_listing(messages: list[Message]) -> LLMResponse:
        listing, status = last_tool_payload(messages)
        assert "app.py" in listing["data"]["entries"]
        assert status["data"]["clean"] is True
        return tools(call("search_code", query="def add", file_type="py"))

    def after_search(messages: list[Message]) -> LLMResponse:
        (search,) = last_tool_payload(messages)
        assert search["data"]["files"] == ["app.py"]
        return tools(call("read_file", path="app.py"))

    def after_read(messages: list[Message]) -> LLMResponse:
        (read,) = last_tool_payload(messages)
        assert "return a - b" in read["data"]["content"]
        return tools(
            call("edit_file", path="app.py", old_text="return a - b", new_text="return a + b")
        )

    def after_edit(messages: list[Message]) -> LLMResponse:
        (edit,) = last_tool_payload(messages)
        assert edit["ok"] and edit["data"]["changed"]
        return tools(call("git_diff"))

    def after_diff(messages: list[Message]) -> LLMResponse:
        (diff,) = last_tool_payload(messages)
        assert "+    return a + b" in diff["data"]["diff"]
        return final()

    return [
        tools(call("list_files", recursive=True), call("git_status")),
        after_listing,
        after_search,
        after_read,
        after_edit,
        after_diff,
    ]


# --- acceptance ----------------------------------------------------------------------------


def test_acceptance_coder_fixes_add(git_repo: RepositoryContext, settings: Settings) -> None:
    state = Orchestrator(settings).submit(TASK)
    coder, llm, ctx = make_coder(git_repo, happy_path_script())

    result = coder.run(TASK, state)

    assert isinstance(result, CoderResult)
    assert result.status is AgentStatus.SUCCESS, result.errors
    assert (git_repo.root / "app.py").read_text() == BUGGY_APP.replace("a - b", "a + b")
    assert result.files_inspected == ["app.py"]
    assert result.files_changed == ["app.py"]
    assert result.diff_summary is not None
    assert result.diff_summary["files"] == [
        {"path": "app.py", "status": "modified", "additions": 1, "deletions": 1}
    ]
    assert "+    return a + b" in result.artifacts["diff"]
    assert [c["name"] for c in result.tool_calls] == [
        "list_files", "git_status", "search_code", "read_file", "edit_file", "git_diff"
    ]
    assert result.errors == [] and result.next_action == "Run the tests."
    # the diff it inspected is the real repository diff
    assert "+    return a + b" in git(git_repo.root, "diff")
    # shared state and context are updated
    assert state.code_changes[0]["files_changed"] == ["app.py"]
    assert state.failures == [] and state.active_agent is None
    assert ctx.entries(ContextCategory.AGENT_OUTPUT)[0].metadata["status"] == "SUCCESS"
    # the model received the system rules, task, and tool definitions
    first_messages, first_tools = llm.calls[0]
    assert first_messages[0].content == CODER_SYSTEM_PROMPT
    assert TASK in first_messages[1].content
    assert {t.name for t in first_tools or []} == set(coder.available_tools)


def test_coder_has_exactly_the_specified_tools(git_repo: RepositoryContext) -> None:
    coder, _, _ = make_coder(git_repo, [])
    assert coder.available_tools == sorted(
        ["list_files", "read_file", "search_code", "write_file", "edit_file", "terminal",
         "git_status", "git_diff"]
    )


def test_system_prompt_enforces_rules() -> None:
    for phrase in ("Inspect before modifying", "Search before reading", "Read every file",
                   "minimal change", "architecture", "git_diff AFTER your last modification",
                   "unrelated", "secrets", "destructive", "JSON"):
        assert phrase in CODER_SYSTEM_PROMPT


# --- evidence-based status ------------------------------------------------------------------


def test_success_without_diff_is_downgraded(git_repo: RepositoryContext) -> None:
    script = [
        tools(call("read_file", path="app.py")),
        tools(call("edit_file", path="app.py", old_text="a - b", new_text="a + b")),
        final(),
    ]
    coder, _, _ = make_coder(git_repo, script)
    state = AgentState(task=TASK)
    result = coder.run(TASK, state)
    assert result.status is AgentStatus.FAILURE
    assert any("without inspecting git_diff" in e for e in result.errors)
    assert state.failures and state.failures[0]["agent"] == "coder"


def test_diff_before_last_edit_does_not_count(git_repo: RepositoryContext) -> None:
    script = [
        tools(call("edit_file", path="app.py", old_text="a - b", new_text="a + b")),
        tools(call("git_diff")),
        tools(call("edit_file", path="app.py", old_text="a * b", new_text="b * a")),
        final(files=["app.py"]),
    ]
    coder, _, _ = make_coder(git_repo, script)
    result = coder.run(TASK, AgentState(task=TASK))
    assert result.status is AgentStatus.FAILURE and result.diff_summary is None


def test_success_without_changes_is_downgraded(git_repo: RepositoryContext) -> None:
    coder, _, _ = make_coder(git_repo, [tools(call("git_diff")), final(files=[])])
    result = coder.run(TASK, AgentState(task=TASK))
    assert result.status is AgentStatus.FAILURE
    assert "no file was modified" in result.errors[-1]


def test_claimed_but_unmade_changes_reported(git_repo: RepositoryContext) -> None:
    script = [
        tools(call("edit_file", path="app.py", old_text="a - b", new_text="a + b")),
        tools(call("git_diff")),
        final(files=["app.py", "other.py"]),
    ]
    coder, _, _ = make_coder(git_repo, script)
    result = coder.run(TASK, AgentState(task=TASK))
    assert result.status is AgentStatus.SUCCESS
    assert result.files_changed == ["app.py"]
    assert any("other.py" in e for e in result.errors)


def test_blocked_status_passes_through(git_repo: RepositoryContext) -> None:
    coder, _, _ = make_coder(
        git_repo, [final("BLOCKED", files=[], summary="Need credentials", next_action="")]
    )
    result = coder.run(TASK, AgentState(task=TASK))
    assert result.status is AgentStatus.BLOCKED and result.summary == "Need credentials"
    assert result.next_action.startswith("Provide the missing")


# --- tool loop limits and malformed calls ----------------------------------------------------


def test_max_tool_calls_enforced(git_repo: RepositoryContext) -> None:
    looping = [tools(call("git_status", call_id=f"c{i}")) for i in range(10)]
    coder, llm, _ = make_coder(git_repo, [*looping], max_tool_calls=3)
    # after the limit the loop asks once without tools; script returns a final answer
    llm.responses.insert(3, final("FAILURE", files=[], summary="Ran out of tool calls"))
    result = coder.run(TASK, AgentState(task=TASK))

    assert len(result.tool_calls) == 3
    assert result.metadata["limit_reached"] is True
    assert result.status is AgentStatus.FAILURE
    last_messages, last_tools = llm.calls[-1]
    assert last_tools is None and "tool call limit (3)" in last_messages[-1].content


def test_max_tool_calls_with_parallel_calls_in_one_turn(git_repo: RepositoryContext) -> None:
    burst = tools(*(call("git_status", call_id=f"c{i}") for i in range(5)))
    coder, llm, _ = make_coder(git_repo, [burst, tools(call("git_status"))], max_tool_calls=2)
    result = coder.run(TASK, AgentState(task=TASK))
    assert len(result.tool_calls) == 2
    # every requested call still gets a tool result (3 of them report the budget error)
    tool_msg = next(m for m in llm.calls[-1][0] if m.role is Role.TOOL)
    assert len(tool_msg.tool_results) == 5
    assert sum(r.is_error for r in tool_msg.tool_results) == 3
    assert result.status is AgentStatus.FAILURE
    assert "Tool call limit reached" in result.errors[0]


def test_loop_rejects_zero_budget(git_repo: RepositoryContext) -> None:
    with pytest.raises(ValueError):
        ToolCallingLoop(MockLLMClient(), build_registry(git_repo), max_tool_calls=0)


def test_malformed_tool_calls_are_reported_to_model(git_repo: RepositoryContext) -> None:
    def check(messages: list[Message]) -> LLMResponse:
        payloads = last_tool_payload(messages)
        assert [p["ok"] for p in payloads] == [False] * 5
        assert "Unknown tool" in payloads[0]["error"]
        assert "missing required" in payloads[1]["error"]
        assert "must be of type" in payloads[2]["error"]
        assert "must be a JSON object" in payloads[3]["error"]
        assert "not valid JSON" in payloads[4]["error"]
        return final("FAILURE", files=[], summary="gave up")

    script = [
        tools(
            call("delete_repo", call_id="a"),
            call("read_file", call_id="b"),
            call("read_file", call_id="c", path=123),
            ToolCall("d", "read_file", ["app.py"]),  # type: ignore[arg-type]
            ToolCall("e", "read_file", "{not json"),  # type: ignore[arg-type]
        ),
        check,
    ]
    coder, _, _ = make_coder(git_repo, script)
    result = coder.run(TASK, AgentState(task=TASK))
    assert result.status is AgentStatus.FAILURE
    assert [c["status"] for c in result.tool_calls] == ["FAILURE"] * 5
    assert (git_repo.root / "app.py").read_text() == BUGGY_APP


def test_unsafe_command_from_model_is_rejected(git_repo: RepositoryContext) -> None:
    def check(messages: list[Message]) -> LLMResponse:
        (payload,) = last_tool_payload(messages)
        assert payload["ok"] is False and "HIGH_RISK" in payload["error"]
        return final("BLOCKED", files=[], summary="Refused")

    coder, _, _ = make_coder(git_repo, [tools(call("terminal", command="rm -rf .")), check])
    result = coder.run(TASK, AgentState(task=TASK))
    assert result.status is AgentStatus.BLOCKED
    assert (git_repo.root / "app.py").exists()


def test_invalid_final_response_is_repaired_once(git_repo: RepositoryContext) -> None:
    script = [
        LLMResponse(content="All done, trust me!", stop_reason=StopReason.END_TURN),
        final("BLOCKED", files=[], summary="Needs clarification"),
    ]
    coder, llm, _ = make_coder(git_repo, script)
    result = coder.run(TASK, AgentState(task=TASK))
    assert result.status is AgentStatus.BLOCKED
    assert llm.calls[-1][1] is None and "JSON" in llm.calls[-1][0][-1].content


def test_unparseable_final_response_fails(git_repo: RepositoryContext) -> None:
    script = [
        LLMResponse(content="nope", stop_reason=StopReason.END_TURN),
        LLMResponse(content="still nope", stop_reason=StopReason.END_TURN),
    ]
    coder, _, _ = make_coder(git_repo, script)
    result = coder.run(TASK, AgentState(task=TASK))
    assert result.status is AgentStatus.FAILURE
    assert "did not return a structured" in result.errors[0]


def test_llm_error_becomes_failure(git_repo: RepositoryContext) -> None:
    def boom(messages: list[Message]) -> LLMResponse:
        raise LLMError("rate limited", retryable=True)

    coder, _, _ = make_coder(git_repo, [boom])
    state = AgentState(task=TASK)
    result = coder.run(TASK, state)
    assert result.status is AgentStatus.FAILURE and result.metadata["retryable"] is True
    assert state.failures[0]["errors"] == ["rate limited"]


def test_tool_call_records_clip_large_arguments(git_repo: RepositoryContext) -> None:
    big = "x = 1\n" * 500
    script = [tools(call("write_file", path="gen.py", content=big)), final("FAILURE", files=[])]
    coder, _, _ = make_coder(git_repo, script)
    result = coder.run(TASK, AgentState(task=TASK))
    recorded = result.tool_calls[0]["arguments"]["content"]
    assert len(recorded) < 300 and "chars]" in recorded


def test_coder_uses_settings(git_repo: RepositoryContext) -> None:
    settings = Settings(api_key=FAKE_KEY, max_tool_calls=7, max_tool_output_chars=900)
    coder = CoderAgent(MockLLMClient(), InMemoryContextManager(), git_repo, settings=settings)
    assert coder.max_tool_calls == 7 and coder.registry.max_llm_chars == 900


def test_prompt_contains_relevant_state_not_repository(git_repo: RepositoryContext) -> None:
    state = Orchestrator(Settings(api_key=FAKE_KEY)).submit(TASK)
    state.failures.append({"agent": "coder", "errors": ["previous attempt failed"]})
    coder, _, _ = make_coder(git_repo, [])
    user = coder.build_messages(TASK, state)[1].content
    assert "previous attempt failed" in user and "Implement the required code changes" in user
    assert "return a - b" not in user  # file contents are discovered via tools, not preloaded
    assert FAKE_KEY not in user


@pytest.mark.parametrize(
    ("text", "status"),
    [
        ('{"status": "SUCCESS", "summary": "x"}', "SUCCESS"),
        ('```json\n{"status": "blocked"}\n```', "blocked"),
        ('Here you go: {"status": "FAILURE"} thanks', "FAILURE"),
        ('{"summary": "no status"}', None),
        ('{"status": "MAYBE"}', None),
        ("", None),
        (None, None),
    ],
)
def test_parse_final_response(text: str | None, status: str | None) -> None:
    parsed = parse_final_response(text)
    assert (parsed or {}).get("status") == status
