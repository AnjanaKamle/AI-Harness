"""Production provider layer: DeepSeek and Qwen adapters behind LLMClient.

No network: every adapter talks to a fake transport that returns provider-format JSON
(OpenAI-compatible chat completions, with each provider's quirks). The full orchestrator runs
unchanged through both adapters. One test uses a local HTTP server to prove real timeouts.
"""

from __future__ import annotations

import http.server
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from harness.agents import CoderAgent, MusicAgent, ResearcherAgent, TesterAgent
from harness.config.settings import Settings
from harness.demo import DemoScriptedClient, prepare_sample_repo
from harness.llm import (
    LLMClient,
    LLMError,
    LLMErrorCode,
    LLMNotConfiguredError,
    LLMResponse,
    Message,
    Role,
    StopReason,
    ToolCall,
    ToolDefinition,
    ToolResult,
    create_llm_client,
)
from harness.llm.providers import DeepSeekClient, HttpResult, HttpTransport, QwenClient
from harness.llm.providers.transport import TransportNetworkError, TransportTimeout
from harness.main import main
from harness.orchestrator import Orchestrator
from harness.orchestrator.verification_manager import FinalStatus
from harness.tools import ReadFileTool, RepositoryContext, ToolRegistry
from harness.tools.music import UnavailableMusicPlayer

KEY = "sk-FAKE-provider-test-key-0000"
MODEL = "evaluator-model-placeholder"
BASE = "https://llm.example.invalid/v1"


def settings(provider: str = "deepseek", **overrides: Any) -> Settings:
    options: dict[str, Any] = {"api_key": KEY, "provider": provider, "model": MODEL,
                               "base_url": BASE, "max_retries": 2, "timeout_seconds": 30,
                               "test_timeout_seconds": 60, "research_backend": "none", **overrides}
    return Settings(**options)


# --- fake transport emitting provider wire format ------------------------------------------------


def completion(content: str | None = "", tool_calls: list[dict[str, Any]] | None = None,
               finish: str = "stop", reasoning: str | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {"id": "cmpl-1", "model": MODEL, "choices": [{"index": 0, "message": message,
            "finish_reason": finish}], "usage": {"prompt_tokens": 11, "completion_tokens": 7}}


def wire_call(name: str, arguments: Any, call_id: str = "call_1") -> dict[str, Any]:
    args = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}


class FakeTransport:
    """Records requests; replies from a queue of dicts / (status, body[, headers]) / exceptions /
    callables(payload)."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    def post_json(self, url: str, headers: dict[str, str], payload: dict[str, Any],
                  timeout: float) -> HttpResult:
        self.requests.append({"url": url, "headers": headers, "payload": payload, "timeout": timeout})
        reply = self.replies.pop(0)
        if callable(reply):
            reply = reply(payload)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, tuple):
            status, body, *rest = reply
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            return HttpResult(status, data, rest[0] if rest else {})
        return HttpResult(200, json.dumps(reply).encode())


def client(provider: str, replies: list[Any], **overrides: Any) -> tuple[Any, FakeTransport]:
    transport = FakeTransport(replies)
    cls = DeepSeekClient if provider == "deepseek" else QwenClient
    return cls(settings(provider, **overrides), transport=transport, backoff_seconds=0), transport


# --- A. scripted still works ------------------------------------------------------------------------


def test_a_scripted_provider_is_explicit_and_works() -> None:
    llm = create_llm_client(settings("SCRIPTED", model=None, base_url=None))
    assert isinstance(llm, DemoScriptedClient)
    reply = llm.generate([Message.system("You are the Coder agent"), Message.user("t")], tools=[])
    assert reply.tool_calls[0].name == "list_files"


# --- B/C/D/E. translation ---------------------------------------------------------------------------

CONVERSATION = [
    Message.system("rules"),
    Message.user("fix it"),
    Message.assistant("", (ToolCall("c1", "read_file", {"path": "a.py"}),
                           ToolCall("c2", "search_code", {"query": "def f"}))),
    Message.tool(ToolResult("c1", '{"ok": true}'), ToolResult("c2", '{"ok": false}', is_error=True)),
    Message.user("limit reached"),
]
TOOLS = [ToolDefinition("read_file", "Read a file", {"type": "object", "properties": {"path": {"type": "string"}},
                                                    "required": ["path"]})]


@pytest.mark.parametrize("provider", ["deepseek", "qwen"])
def test_b_c_d_e_request_translation(provider: str) -> None:
    llm, transport = client(provider, [completion("done")])
    llm.generate(CONVERSATION, tools=TOOLS)
    request = transport.requests[0]
    payload = request["payload"]
    assert request["url"] == f"{BASE}/chat/completions"
    assert request["headers"] == {"Authorization": f"Bearer {KEY}"}
    assert payload["model"] == MODEL and payload["stream"] is False and payload["tool_choice"] == "auto"
    assert [m["role"] for m in payload["messages"]] == ["system", "user", "assistant", "tool", "tool", "user"]
    assistant = payload["messages"][2]
    assert assistant["tool_calls"][0] == {"id": "c1", "type": "function",
                                          "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}
    assert json.loads(assistant["tool_calls"][1]["function"]["arguments"]) == {"query": "def f"}
    assert payload["messages"][3] == {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true}'}
    assert payload["messages"][4]["tool_call_id"] == "c2"
    assert payload["tools"][0] == {"type": "function", "function": {
        "name": "read_file", "description": "Read a file", "parameters": TOOLS[0].input_schema}}
    assert KEY not in json.dumps(payload)  # the credential is only ever in the header
    assert "max_tokens" not in payload and "temperature" not in payload


@pytest.mark.parametrize("provider", ["deepseek", "qwen"])
def test_d_response_translation_text_and_multiple_tool_calls(provider: str) -> None:
    llm, _ = client(provider, [
        completion("plain answer"),
        completion(None, [wire_call("read_file", {"path": "a.py"}, "x1"),
                          wire_call("search_code", {"query": "q"}, "x2")], finish="tool_calls"),
    ])
    text = llm.generate([Message.user("hi")])
    assert (text.content, text.stop_reason, text.tool_calls) == ("plain answer", StopReason.END_TURN, ())
    assert (text.usage.input_tokens, text.usage.output_tokens) == (11, 7)
    calls = llm.generate([Message.user("hi")], tools=TOOLS)
    assert calls.stop_reason is StopReason.TOOL_USE
    assert [(c.id, c.name, c.arguments) for c in calls.tool_calls] == [
        ("x1", "read_file", {"path": "a.py"}), ("x2", "search_code", {"query": "q"})]


def test_deepseek_reasoning_content_round_trip() -> None:
    llm, transport = client("deepseek", [
        completion(None, [wire_call("read_file", {"path": "a.py"})], "tool_calls", reasoning="I should read a.py"),
        completion("done"),
    ])
    first = llm.generate([Message.user("fix")], tools=TOOLS)
    history = [Message.user("fix"), first.to_message(), Message.tool(ToolResult("call_1", "{}"))]
    llm.generate(history, tools=TOOLS)
    echoed = transport.requests[1]["payload"]["messages"][1]
    assert echoed["reasoning_content"] == "I should read a.py"  # sent back, as DeepSeek requires
    assert "reasoning" not in first.content


def test_deepseek_default_endpoint_only_when_base_url_unset() -> None:
    llm = DeepSeekClient(settings("deepseek", base_url=None), transport=FakeTransport([]))
    assert llm.endpoint == "https://api.deepseek.com/chat/completions"
    explicit = DeepSeekClient(settings("deepseek", base_url="https://gateway.example.invalid/v1/"),
                              transport=FakeTransport([]))
    assert explicit.endpoint == "https://gateway.example.invalid/v1/chat/completions"  # O


def test_qwen_inline_tool_call_blocks_and_think_tags() -> None:
    content = ('<think>let me look</think>Checking.\n<tool_call>\n{"name": "read_file", '
               '"arguments": {"path": "calc.py"}}\n</tool_call>')
    llm, _ = client("qwen", [completion(content)])
    reply = llm.generate([Message.user("fix")], tools=TOOLS)
    assert reply.stop_reason is StopReason.TOOL_USE
    assert [(c.name, c.arguments) for c in reply.tool_calls] == [("read_file", {"path": "calc.py"})]
    assert reply.content == "Checking." and "<think>" not in reply.content


# --- F/G/H. errors ----------------------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["deepseek", "qwen"])
@pytest.mark.parametrize(
    ("replies", "code", "retryable", "requests"),
    [
        ([(401, {"error": {"message": f"bad key {KEY}"}})], LLMErrorCode.AUTHENTICATION_ERROR, False, 1),
        ([(403, {"error": {"message": "forbidden"}})], LLMErrorCode.AUTHENTICATION_ERROR, False, 1),
        ([(400, {"error": {"message": "unknown model"}})], LLMErrorCode.BAD_REQUEST, False, 1),
        ([(429, {"error": {"message": "slow down"}}, {"retry-after": "0"})] * 3, LLMErrorCode.RATE_LIMITED, True, 3),
        ([(503, b"upstream down")] * 3, LLMErrorCode.SERVER_ERROR, True, 3),
        ([TransportNetworkError("connection reset")] * 3, LLMErrorCode.NETWORK_ERROR, True, 3),
        ([TransportTimeout("slow")], LLMErrorCode.PROVIDER_TIMEOUT, True, 1),
        ([(200, b"not json")], LLMErrorCode.MALFORMED_RESPONSE, True, 1),
        ([{"choices": []}], LLMErrorCode.MALFORMED_RESPONSE, True, 1),
        ([completion(None, [{"id": "x", "type": "function", "function": {"arguments": "{}"}}])],
         LLMErrorCode.MALFORMED_TOOL_CALL, True, 1),
    ],
)
def test_f_g_h_provider_errors_are_structured_and_bounded(
    provider: str, replies: list[Any], code: LLMErrorCode, retryable: bool, requests: int
) -> None:
    llm, transport = client(provider, replies)
    with pytest.raises(LLMError) as info:
        llm.generate([Message.user("hi")], tools=TOOLS)
    assert info.value.code is code and info.value.retryable is retryable
    assert len(transport.requests) == requests  # auth/bad request never retried; transient bounded
    assert KEY not in str(info.value)


def test_transient_failure_then_success() -> None:
    llm, transport = client("deepseek", [(500, b"oops"), completion("recovered")])
    assert llm.generate([Message.user("hi")]).content == "recovered" and len(transport.requests) == 2


def test_masked_key_echo_is_redacted() -> None:
    llm, _ = client("deepseek", [(401, {"error": {"message": "Your api key: ****0000 is invalid"}})])
    with pytest.raises(LLMError) as info:
        llm.generate([Message.user("hi")])
    assert "0000" not in str(info.value) and "[REDACTED]" in str(info.value)


def test_insufficient_system_resource_is_transient() -> None:
    llm, transport = client("deepseek", [completion("", finish="insufficient_system_resource")])
    with pytest.raises(LLMError) as info:
        llm.generate([Message.user("hi")])
    assert info.value.code is LLMErrorCode.SERVER_ERROR and info.value.retryable


# --- I. malformed tool calls are rejected -----------------------------------------------------------


def test_i_malformed_tool_arguments_are_rejected_by_the_registry(tmp_path: Path) -> None:
    llm, _ = client("qwen", [completion(None, [wire_call("read_file", '{"path": "a.py"')], "tool_calls")])
    call = llm.generate([Message.user("x")], tools=TOOLS).tool_calls[0]
    assert call.arguments == '{"path": "a.py"'  # kept verbatim, never "repaired"
    (tmp_path / "a.py").write_text("x = 1\n")
    result, tool_result = ToolRegistry([ReadFileTool(RepositoryContext(tmp_path))]).execute_call(call)
    assert not result.ok and tool_result.is_error and "MALFORMED_TOOL_CALL" in (result.error or "")


# --- J/K/L/M/N. configuration -------------------------------------------------------------------------


def test_j_missing_provider_is_configuration_error() -> None:
    with pytest.raises(LLMNotConfiguredError) as info:
        create_llm_client(settings(None))  # type: ignore[arg-type]
    assert info.value.code is LLMErrorCode.CONFIGURATION_ERROR
    assert "No LLM provider configured. Configure AI_PROVIDER and AI_MODEL" in str(info.value)


@pytest.mark.parametrize("provider", ["deepseek", "qwen"])
def test_k_missing_model_is_configuration_error(provider: str) -> None:
    with pytest.raises(LLMNotConfiguredError, match="requires AI_MODEL; no model name is assumed"):
        create_llm_client(settings(provider, model=None))


def test_qwen_requires_explicit_endpoint() -> None:
    with pytest.raises(LLMNotConfiguredError, match="requires AI_BASE_URL"):
        create_llm_client(settings("qwen", base_url=None))


def test_l_missing_api_key(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--json", "--task", "x"]) == 2
    assert "CONFIGURATION_ERROR" in capsys.readouterr().err


@pytest.mark.parametrize(("value", "cls"), [("deepseek", DeepSeekClient), ("DEEPSEEK", DeepSeekClient),
                                            (" DeepSeek ", DeepSeekClient), ("qwen", QwenClient),
                                            ("QWEN", QwenClient), ("Qwen", QwenClient)])
def test_m_provider_selection_is_case_insensitive_and_deterministic(value: str, cls: type) -> None:
    assert type(create_llm_client(settings(value))) is cls


def test_m_provider_is_never_inferred_from_the_key(monkeypatch: pytest.MonkeyPatch,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("AI_API_KEY", "sk-looks-like-some-provider-0000")
    assert main(["--json", "--task", "Fix it"]) == 0
    out = capsys.readouterr()
    assert json.loads(out.out)["outcome"]["status"] == "CONFIGURATION_ERROR"
    assert "No LLM provider configured. Configure AI_PROVIDER and AI_MODEL before live execution." in out.err


def test_n_model_comes_only_from_configuration() -> None:
    llm, transport = client("qwen", [completion("ok")], model="org-assigned-qwen-id")
    llm.generate([Message.user("hi")])
    assert transport.requests[0]["payload"]["model"] == "org-assigned-qwen-id"
    assert llm.describe() == {"provider": "qwen", "model": "org-assigned-qwen-id",
                              "endpoint": f"{BASE}/chat/completions"}


def test_max_output_tokens_only_when_configured() -> None:
    llm, transport = client("deepseek", [completion("ok")], max_output_tokens=2048)
    llm.generate([Message.user("hi")])
    assert transport.requests[0]["payload"]["max_tokens"] == 2048


# --- real HTTP timeout (local server; no threads left behind) ----------------------------------------


class _SlowHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path.endswith("drip/chat/completions"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            for _ in range(20):  # trickles bytes forever: per-read timeouts alone never fire
                self.wfile.write(b" ")
                self.wfile.flush()
                time.sleep(0.2)
            return
        time.sleep(3)

    def log_message(self, *args: Any) -> None:
        pass


class _OkHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        body = json.dumps(completion(f"echo:{payload['model']}")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        if self.path.startswith("/chunked"):
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for part in (body[:10], body[10:]):
                self.wfile.write(f"{len(part):x}\r\n".encode() + part + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
        else:
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        pass


@pytest.mark.parametrize("prefix", ["/length", "/chunked"])
def test_real_http_success_path_through_adapter(prefix: str) -> None:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _OkHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}{prefix}"
        for cls, name in ((DeepSeekClient, "deepseek"), (QwenClient, "qwen")):
            llm = cls(settings(name, base_url=base))  # real HttpTransport
            assert llm.generate([Message.user("hi")]).content == f"echo:{MODEL}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("path", ["/slow", "/drip"])
def test_http_transport_enforces_total_deadline(path: str) -> None:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _SlowHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    threads_before = threading.active_count()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}{path}/chat/completions"
        started = time.monotonic()
        with pytest.raises(TransportTimeout):
            HttpTransport().post_json(url, {}, {"x": 1}, timeout=0.8)
        assert time.monotonic() - started < 2.5
        assert threading.active_count() <= threads_before + 1  # no helper thread left running
    finally:
        server.shutdown()
        server.server_close()


# --- 21/22/32. full orchestrator through each mocked provider ----------------------------------------


def wire_to_messages(payload: dict[str, Any]) -> list[Message]:
    """What a real server would 'see', decoded back so a scripted brain can answer."""
    messages: list[Message] = []
    pending: list[ToolResult] = []
    for m in payload["messages"]:
        if m["role"] == "tool":
            pending.append(ToolResult(m["tool_call_id"], m["content"]))
            continue
        if pending:
            messages.append(Message.tool(*pending))
            pending = []
        if m["role"] == "assistant":
            calls = tuple(ToolCall(c["id"], c["function"]["name"], json.loads(c["function"]["arguments"]))
                          for c in m.get("tool_calls") or [])
            messages.append(Message.assistant(m.get("content") or "", calls))
        else:
            messages.append(Message(Role(m["role"]), m["content"]))
    if pending:
        messages.append(Message.tool(*pending))
    return messages


def to_wire(response: LLMResponse, provider: str, turn: int) -> dict[str, Any]:
    calls = [wire_call(c.name, c.arguments, f"call_{turn}_{i}") for i, c in enumerate(response.tool_calls)]
    reasoning = f"step {turn}" if provider == "deepseek" else None  # DeepSeek reasoning mode
    if provider == "qwen" and calls and turn % 2:  # some Qwen servers inline tool calls as text
        blocks = "".join(f'<tool_call>{json.dumps({"name": c.name, "arguments": c.arguments})}</tool_call>'
                         for c in response.tool_calls)
        return completion(f"<think>thinking {turn}</think>{blocks}")
    return completion(response.content or None, calls or None,
                      "tool_calls" if calls else "stop", reasoning)


class ServerBrain:
    """A fake provider endpoint: decodes the request and answers with the scripted brain."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.brain = DemoScriptedClient()
        self.turn = 0

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.turn += 1
        tools = [ToolDefinition(t["function"]["name"], "", {}) for t in payload.get("tools", [])] or None
        return to_wire(self.brain.generate(wire_to_messages(payload), tools=tools), self.provider, self.turn)


def run_with_provider(tmp_path: Path, provider: str, task: str, *, replies: list[Any] | None = None,
                      **overrides: Any) -> tuple[Any, FakeTransport, Orchestrator]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    repo = RepositoryContext(prepare_sample_repo(tmp_path))
    config = settings(provider, **overrides)
    transport = FakeTransport(replies if replies is not None else [ServerBrain(provider)] * 60)
    llm: LLMClient = (DeepSeekClient if provider == "deepseek" else QwenClient)(
        config, transport=transport, backoff_seconds=0)
    orch = Orchestrator(config, llm=llm)
    ctx = orch.context
    agents = {"coder": CoderAgent(llm, ctx, repo, settings=config),
              "tester": TesterAgent(llm, ctx, repo, settings=config),
              "researcher": ResearcherAgent(llm, ctx, repo, settings=config),
              "music": MusicAgent(None, ctx, player=UnavailableMusicPlayer())}
    return orch.execute(task, repo, agents=agents), transport, orch


TASK = "Fix the bug in this repository."


@pytest.mark.parametrize("provider", ["deepseek", "qwen"])
def test_full_pipeline_through_mocked_provider(tmp_path: Path, provider: str) -> None:
    final, transport, _ = run_with_provider(tmp_path, provider, TASK)
    assert final.status is FinalStatus.VERIFIED_SUCCESS, final.unresolved_issues
    statuses = {n["id"]: n["status"] for n in final.plan}
    assert statuses["test-1"] == "FAILED" and statuses["repair-1"] == "SUCCESS" and statuses["test-2"] == "SUCCESS"
    assert final.state.baseline["failed_tests"]  # type: ignore[index, union-attr]
    payloads = [r["payload"] for r in transport.requests]
    assert all(p["model"] == MODEL for p in payloads)
    assert any(m["role"] == "tool" and m["tool_call_id"] for p in payloads for m in p["messages"])
    assert final.state.usage["total"]["reported_input_tokens"] > 0  # type: ignore[union-attr]
    assert KEY not in json.dumps(payloads) and KEY not in final.to_json()
    assert KEY not in json.dumps(final.state.events)  # type: ignore[union-attr]
    if provider == "deepseek":
        assert any("reasoning_content" in m for p in payloads for m in p["messages"])


def test_provider_switch_changes_nothing_else(tmp_path: Path) -> None:
    runs = {p: run_with_provider(tmp_path / p, p, TASK) for p in ("deepseek", "qwen")}
    finals = {p: r[0] for p, r in runs.items()}
    shape = {p: [(n["id"], n["agent"], n["kind"], n["status"], n["dependencies"]) for n in f.plan]
             for p, f in finals.items()}
    assert shape["deepseek"] == shape["qwen"]
    assert finals["deepseek"].files_changed == finals["qwen"].files_changed == ["calc.py"]
    assert [t["status"] for t in finals["deepseek"].tests_run] == [t["status"] for t in finals["qwen"].tests_run]
    assert type(runs["deepseek"][2].last_controller.planner) is type(runs["qwen"][2].last_controller.planner)


@pytest.mark.parametrize("provider", ["deepseek", "qwen"])
def test_authentication_failure_through_orchestrator(tmp_path: Path, provider: str) -> None:
    final, transport, _ = run_with_provider(tmp_path, provider, TASK,
                                            replies=[(401, {"error": {"message": "invalid key"}})] * 10)
    assert final.status is FinalStatus.BLOCKED and final.error_code == "AUTHENTICATION_ERROR"
    assert len(transport.requests) == 1  # never retried


@pytest.mark.parametrize("provider", ["deepseek", "qwen"])
def test_timeouts_through_orchestrator_are_bounded(tmp_path: Path, provider: str) -> None:
    final, transport, _ = run_with_provider(tmp_path, provider, TASK, max_agent_retries=2,
                                            replies=[TransportTimeout("slow")] * 10)
    assert final.status is FinalStatus.BLOCKED and final.error_code == "PROVIDER_TIMEOUT"
    assert len(transport.requests) == 3  # 1 attempt + MAX_AGENT_RETRIES, no transport-level retry


def test_rate_limit_and_server_errors_recover(tmp_path: Path) -> None:
    replies: list[Any] = [(429, {"error": {"message": "busy"}}, {"retry-after": "0"}),
                          (502, b"bad gateway")] + [ServerBrain("qwen")] * 60
    final, transport, _ = run_with_provider(tmp_path, "qwen", TASK, replies=replies)
    assert final.status is FinalStatus.VERIFIED_SUCCESS


def test_real_provider_failure_never_falls_back_to_scripted(tmp_path: Path) -> None:
    final, _, orch = run_with_provider(tmp_path, "deepseek", TASK,
                                       replies=[(500, b"down")] * 30, max_agent_retries=0)
    assert final.status is FinalStatus.BLOCKED
    assert isinstance(orch.llm, DeepSeekClient)
