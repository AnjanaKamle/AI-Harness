# AI Coding Harness

A multi-agent AI coding harness for the **LCC × DevClub AI Harness Hackathon 2026**. You give
it a coding task in plain text. An orchestrator plans the work, a Coder, a Researcher and a
Tester carry it out with sandboxed tools, and the task counts as done only when a
verification gate has checked real evidence.

```bash
export AI_API_KEY="<PROVIDED_API_KEY>"
export AI_PROVIDER="deepseek"            # or: qwen   (plus AI_BASE_URL for Qwen)
export AI_MODEL="<EVALUATOR_MODEL>"
make setup
make run
```

> **Status:** everything described here is implemented and tested (476 tests; the 2 optional
> live tests and a ripgrep test skip when their tools or credentials are absent). The
> organizers evaluate with **DeepSeek and Qwen** models. The harness has adapters for both,
> selected by configuration (`AI_PROVIDER`, `AI_MODEL`, `AI_BASE_URL`, `AI_API_KEY`). The
> exact model IDs and endpoints are evaluator configuration and are **not guessed**. With only
> `AI_API_KEY` set, `make run` reports `CONFIGURATION_ERROR: No LLM provider configured`. See
> [Model configuration](#19-model-configuration).

---

## Contents

1. [Problem](#1-problem) · 2. [Solution](#2-solution) · 3. [Architecture](#3-architecture) ·
4. [Agent responsibilities](#4-agent-responsibilities) · 5. [Orchestrator behaviour](#5-orchestrator-behaviour) ·
6. [Context management](#6-context-management) · 7. [Tool system](#7-tool-system) ·
8. [Verification and recovery](#8-verification-and-recovery) · 9. [Terminal UI](#9-terminal-ui) ·
10. [Optional Music agent](#10-optional-music-agent) · 11. [Installation](#11-installation) ·
12. [AI_API_KEY](#12-ai_api_key) · 13. [Make commands](#13-make-commands) · 14. [Testing](#14-testing) ·
15. [Example task](#15-example-task) · 16. [Architecture diagram](#16-architecture-diagram) ·
17. [Limitations](#17-limitations) · 18. [Configuration reference](#18-configuration-reference) ·
19. [Model configuration](#19-model-configuration) · 20. [Security considerations](#20-security-considerations)

---

## 1. Problem

A single LLM asked to "fix this bug" tends to:

- edit files it never read;
- dump the whole repository into its prompt;
- declare success without running anything;
- loop forever on a failing fix.

Coding agents need structure: specialised roles, safe tools, bounded context, and completion
decided by evidence rather than by the model's own claims.

## 2. Solution

The harness splits the work between specialised agents under a central orchestrator:

- The **Orchestrator** turns the task into a dependency graph, schedules agents (in parallel
  where that is safe), owns the shared state, recovers from failures, and adds repair steps
  when tests fail.
- The **Coder** inspects, searches, reads, edits minimally and reviews its own diff.
- The **Researcher** answers library/API questions with findings that are verified against
  their sources.
- The **Tester** detects how the project is tested, runs the relevant checks and classifies
  failures. It cannot modify source files.
- The **verification gate** is the only way to reach `VERIFIED_SUCCESS`, and it trusts
  evidence, not model output.
- **Music** is an optional, independent side capability, which is DISABLED when no audio
  player is available.

## 3. Architecture

```
                    USER
                     |
                ORCHESTRATOR  (planner → task graph → scheduler → recovery → verification gate)
                     |
        +------------+------------+
        |            |            |
      CODER      RESEARCHER     TESTER
        |
        +----------------------+
                               |
                         VERIFICATION
                               |
                    +----------+----------+
                    |                     |
                  FAIL                   PASS
                    |                     |
                  REPAIR               VERIFIED
                    |
                  RETRY

Optional independent capability:   MUSIC
```

| Layer | Package | Contents |
|---|---|---|
| Entry point | `harness.main` | CLI; the TUI for interactive terminals, JSON otherwise |
| Configuration | `harness.config` | `Settings`, read from environment variables only |
| LLM abstraction | `harness.llm` | `LLMClient.generate(messages, tools)`, message and tool types, provider registry and plugins |
| Orchestrator | `harness.orchestrator` | planner, task graph, scheduler, permissions, recovery, verification gate, event log, `FinalResult`, `AgentState` |
| Agents | `harness.agents` | Coder, Researcher, Tester, Music, and the tool-calling loop |
| Tools | `harness.tools` | filesystem, search, terminal (policy), git, testing, research, music |
| Verification | `harness.verification` | test detection, runner, output parsing, failure classifier |
| Context | `harness.context` | context store, per-agent context builder, relevance, summaries, usage |
| Research | `harness.research` | `ResearchFinding` and source verification |
| TUI | `harness.ui` | a presentation layer only; the core never imports it |

## 4. Agent responsibilities

| Agent | Does | Capabilities | Model use |
|---|---|---|---|
| **Coder** | inspects the repo, searches, reads, makes minimal edits, reviews `git_diff`, returns a structured result | repo read + **write**, terminal | tool-calling loop |
| **Researcher** | looks up APIs, reads documentation, checks registry metadata; returns FACT / INFERENCE / UNCERTAINTY findings | repo read, web | tool-calling loop |
| **Tester** | detects test/build/lint commands from repository evidence, runs the relevant ones, parses the results, classifies failures | repo read, terminal (verification commands only) | none; deterministic |
| **Music** | play / pause / resume / stop / volume | music only | none; deterministic |

Capabilities are enforced by the orchestrator before anything runs. Only the Coder may write
to the repository.

## 5. Orchestrator behaviour

1. **Understand and decompose.** The planner splits the task into a coding part and an
   optional music part, and decides whether research is needed. Research is planned when the
   task names a library, asks about documentation, or asks for research "if needed" and the
   project actually uses a third-party library.
2. **Plan.** It builds a DAG of task nodes. Each node has an id, description, agent, status,
   dependencies, result, retry count, priority, and a `required` flag.
   ```
   inspect-1   [coder]        <- none
   research-1  [researcher]   <- none                  (in parallel with inspect-1)
   research-2  [researcher]   <- research-1, inspect-1
   implement-1 [coder]        <- inspect-1, research-1, research-2
   test-1      [tester]       <- implement-1
   verify-1    [orchestrator] <- test-1
   music-1     [music]        <- none                  (optional, independent)
   ```
3. **Schedule.** READY nodes run concurrently, up to `AI_MAX_CONCURRENT_AGENTS` (default 3),
   on a bounded thread pool driven by asyncio. Repository access follows a readers/writer
   rule: a writing Coder never overlaps any other repository reader or writer, while music can
   run alongside anything.
4. **Own the state.** Each agent works on a private copy of `AgentState`. The orchestrator
   validates the changes and commits only what that kind of node may contribute; anything else
   is rejected and logged.
5. **Recover.** Failures are classified and handled by the recovery manager, as described in
   [Verification and recovery](#8-verification-and-recovery). A failed test adds `repair-N` and
   `test-N+1` nodes to the graph, carrying the real failure evidence.
6. **Finish.** The verification gate produces a `FinalResult`: `VERIFIED_SUCCESS`, `BLOCKED`
   or `FAILED`.

Every step is recorded as a timestamped event: TASK_CREATED, PLAN_CREATED, AGENT_STARTED,
TOOL_CALLED, TOOL_FAILED, TEST_FAILED, REPAIR_STARTED, VERIFICATION_PASSED, TASK_COMPLETED
and others. Each event carries the task ID, agent and details, with credentials redacted. The
log file also records every verification check with its outcome and every recovery decision
with its reason.

## 6. Context management

- **The whole repository is never sent.** Agents discover code through tools.
- **Per-agent context packages.** The `ContextBuilder` gives each agent a budgeted package
  (`AI_MAX_CONTEXT_CHARS`):
  - **Coder:** the task, plan, repository facts, relevant files (paths, reasons and
    signatures, not bodies), verified research, current failures.
  - **Tester:** the task, changed files, test configuration, the previous failure, the
    verification requirements.
  - **Researcher:** the questions, dependency declarations, findings so far, and only the code
    that mentions the topic.
  - **Music:** the playback command and its parameters, nothing else.
- **Relevance order.** Files are ranked: explicitly mentioned > modified in this task > search
  hits for the task's identifiers > related tests > imported by those files > documentation.
  At most 8 files are listed, and unrelated files are excluded. A test checks that unrelated
  files never reach any model call.
- **Bounded outputs.** Every tool result is capped in characters, lines and list items, and
  `search_code` and `web_search` results are capped too. Every cut is marked
  `[OUTPUT TRUNCATED: …]`. Test-failure summaries keep the assertion and error lines instead
  of cutting blindly.
- **Compression.** Older history becomes structured summaries, for example
  `Test summary: Attempt 1: FAIL … Attempt 2: PASS`, and the context store compresses old
  entries.
- **Efficiency.** Project detection is cached until a tool modifies the repository. A test run
  scoped to the changed files is skipped when it would just repeat the full suite.
- **Usage.** LLM turns, tool calls and characters sent and received are tracked per agent.
  Token counts are labelled `estimated_*` (characters ÷ 4) unless the provider reports real
  ones.

## 7. Tool system

Every tool validates its input, stays inside one repository root, and returns a structured
result. Expected failures come back as `FAILURE` results for the model to read; they are
never swallowed.

| Tool | Notes |
|---|---|
| `list_files`, `read_file`, `write_file`, `edit_file` | `edit_file` replaces text only when `old_text` matches exactly the expected number of times; `.git/` is write-protected |
| `search_code` | ripgrep when available, otherwise pure Python; results include context lines |
| `terminal` | one allow-listed command, no shell, timeout kills the whole process group, secrets removed from the environment |
| `git_status`, `git_diff`, `git_log` | read-only; the diff includes new untracked files |
| `detect_tests`, `run_tests` | `run_tests` accepts only test, build, lint or typecheck commands |
| `web_search`, `fetch_documentation`, `package_info`, `lookup_python_api` | read-only research tools, bounded, public addresses only |
| `play_music`, `pause_music`, `resume_music`, `stop_music`, `set_volume` | Music only |

A tool failure that the tool marks transient (for example a network timeout) is retried up to
`AI_MAX_TOOL_RETRIES` times.

## 8. Verification and recovery

**Baseline before any change.** A read-only `baseline-1` Tester step runs the project's tests
*before* the Coder modifies anything, in parallel with inspection. Afterwards the results are
compared:

| Case | Outcome |
|---|---|
| the test failed before the change and passes now | evidence the change works (`fixed`) |
| the test passed before and fails now | a **new failure caused by the change**: repair |
| the test failed before and still fails, and is unrelated to the change | **not blamed on the agent**: reported, does not block |
| the test failed before and still fails, and is related to the change or named in the task | must be fixed |
| the tests could not run at baseline (infrastructure) | the baseline step is optional; verification still decides |

**The verification gate** is the only path to `VERIFIED_SUCCESS`. All of these must hold:

1. every required task succeeded (a failed test that was superseded by a repair counts as
   resolved);
2. the code changes are visible in `git diff`;
3. the relevant tests actually ran and passed (only pre-existing, unrelated failures may remain);
4. **the task's behaviour is exercised by a test**: a test that failed at baseline now passes,
   a test related to the changed code ran, or the change adds a test. A suite that passes
   without touching the changed code does not count as verification;
5. no blocking failure remains, including verification that modified the repository;
6. the final repository state is inspectable.

A Coder that says "everything works" while tests fail ends `BLOCKED`.

**Failure classification** comes from exit codes, output and repository state:

| Category | Examples | Repairable by code changes |
|---|---|---|
| `CODE_FAILURE` | syntax, import, name or type errors in repository code | yes |
| `TEST_FAILURE` | assertion failures | yes |
| `TIMEOUT` | a test hangs or loops | yes |
| `UNKNOWN_FAILURE` | a non-zero exit with no recognisable cause | yes |
| `DEPENDENCY_FAILURE` | a missing third-party package or test framework | no |
| `ENVIRONMENT_FAILURE` | permissions, disk space, missing tools | no |

**Recovery** is bounded, and every decision is recorded with its reason:

| Failure | Action | Limit |
|---|---|---|
| tool failure (transient) | retry the tool | `AI_MAX_TOOL_RETRIES` (2) |
| agent failure, malformed output, retryable model error or timeout | retry the agent | `AI_MAX_AGENT_RETRIES` (2) |
| repairable test failure | the Coder repairs with the failure evidence, then tests re-run | `AI_MAX_REPAIR_ATTEMPTS` (5) |
| research failure | retry if transient; otherwise proceed if the Coder has local evidence; otherwise BLOCKED | |
| provider rate limit, server error, network error | retried with backoff by the adapter (`AI_MAX_RETRIES`), then by agent retry | bounded |
| provider timeout or malformed response | agent retry | `AI_MAX_AGENT_RETRIES` |
| non-repairable failure, integrity violation, no verification possible, `AUTHENTICATION_ERROR`, `BAD_REQUEST` | BLOCKED, never retried | |
| optional task (music) failure | recorded; never blocks coding | |

**Time limits.** An agent task that exceeds `AI_AGENT_TIMEOUT_SECONDS` is cancelled at its
next tool boundary. If it cannot be stopped, it is abandoned: the harness still finishes, and
no other repository writer can start while it might be running.

## 9. Terminal UI

In an interactive terminal, `make run` opens a live dashboard:

```
╭─────────────────────────────────────────────────────────────╮
│ AI CODING HARNESS   [DEMO MODE - scripted client, ...]      │
╰─────────────────────────────────────────────────────────────╯
 CURRENT TASK   Fix the bug ... and play Beethoven Symphony No. 5 while you work.
 Orchestrator: RUNNING
 AGENT        STATUS    CURRENT ACTIVITY
 Coder        RUNNING   reading tests/test_calc.py
 Researcher   IDLE
 Tester       WAITING
 Music        SUCCESS   playing Ludwig van Beethoven: Symphony No. 5 ...
 PROGRESS   completed 2/5 · pending 3 · retries 0 · tests not run
 METRICS    tool calls 4 · agent turns 3 · elapsed 1.2s · tokens ~6,900 (estimated)
 EVENTS     00:01 Coder started ...   00:02 1 test(s) failed ...   00:02 Repair attempt 1 ...
```

- **Final report.** After the run it shows VERIFIED SUCCESS / BLOCKED / FAILED, then the
  summary, files changed, tests executed, test results, research, repair attempts and
  remaining issues. It is rendered from the structured `FinalResult`.
- **Errors** appear as short messages; full details go to the log file. No traceback is ever
  shown.
- **Input.** `make run` prompts for tasks; `make run TASK="…"` runs one task.
- **Output format.** When stdout is not a terminal, the harness prints JSON. `--json` and
  `--tui` force either mode.
- **Presentation only.** The TUI subscribes to the orchestrator's event stream. The core
  never imports it or Rich, and a test enforces that.

## 10. Optional Music agent

The Music agent is not required for evaluation and never affects coding or verification. It
controls playback only:

- **Controls:** play, pause, resume, stop and volume.
- **Players:** `afplay` on macOS; `paplay`, `aplay` or `ffplay` on Linux.
- **Sources:** a matching file in `AI_MUSIC_DIR`, otherwise a short synthesized excerpt of a
  public-domain piece (Beethoven's 5th, "Ode to Joy" or "Für Elise"), generated with the
  standard library.
- **No accounts, services or credentials.**
- **Disabled mode:** with no player, no audio device, or `AI_MUSIC_BACKEND=none`, it is
  **DISABLED**. Requests then fail cleanly and are listed as non-blocking issues.

"Fix this bug and play Beethoven" runs the coding pipeline and the music task independently:
- a music failure does not affect coding, and a coding failure does not stop the music;
- the Coder and Tester never see the music request.

The whole system works with text only. There is no audio, image or video input.

## 11. Installation

**Prerequisites:**
- Python ≥ 3.11 as `python3`;
- `git`;
- `make`;
- network access during `make setup` (to install from PyPI).

Optional extras:
- `rg` (ripgrep) for faster search;
- a local audio player for music;
- `node`/`npm`, only for JavaScript target repositories.

```bash
git clone <repository> && cd <repository>
make setup          # creates .venv and installs the package with rich and pytest
```

Runtime dependency: `rich`, for the TUI. Development dependency: `pytest`.

## 12. AI_API_KEY

The credential is read **only** from the process environment:

```bash
export AI_API_KEY="<PROVIDED_API_KEY>"
```

- **Mandatory:** if it is missing, the harness exits with code 2 and a clear message.
- **Never read from files**, never logged or printed. It is also removed from the environment
  of every command the harness runs.
- **`.env.example`** lists every variable with empty placeholders and is never loaded.

## 13. Make commands

| Command | What it does |
|---|---|
| `make setup` | creates `.venv` and installs the dependencies (idempotent) |
| `make run` | starts the harness: the TUI in a terminal (prompts for a task), JSON otherwise. Also `make run TASK="…" [REPO=/path]` |
| `make test` | runs the full test and evaluation suite (no API key or network needed) |
| `make clean` | removes caches, build outputs and `*.egg-info`; keeps source and `.venv` |
| `make demo` | the complete pipeline on a bundled sample repository, with the scripted client (no model needed) |
| `make check` | provider health check: prints the provider, model and endpoint, sends one minimal request, reports `CONNECTED` or the error code (never the key) |
| `make test-live` | optional live tests against the configured provider (skipped unless configured) |
| `make help` | lists the targets |

## 14. Testing

```bash
make test        # 476 tests, ~50 s, deterministic; no network, credentials or live API calls
```

The tests use temporary repositories, real git and real pytest. The model is replaced by
scripted responses, so no provider is needed.

| Area | Test files |
|---|---|
| Foundation: configuration, LLM abstraction, state | `test_config.py`, `test_llm.py`, `test_foundation.py` |
| Tools: filesystem, path safety, search, terminal, git, registry | `test_tools.py` |
| Agents: Coder, Researcher, Tester, Music | `test_coder.py`, `test_research.py`, `test_repair_loop.py`, `test_music_tui.py` |
| Orchestrator: planning, graph, dependencies, scheduling, concurrency, recovery, retry limits | `test_orchestration_units.py`, `test_orchestration_flows.py` |
| Verification: pass, failure, blocked, final success | `test_verification.py`, `test_orchestration_flows.py` |
| Context: relevance, truncation, compression, agent-specific context | `test_context.py`, `test_research_flow.py` |
| TUI: startup, event rendering, final results, input modes | `test_music_tui.py` |
| Security: command restrictions, path traversal, secret handling | `test_tools.py`, `test_research.py`, `test_orchestration_units.py` |
| **Evaluation scenarios 1–5** | `test_evaluation_scenarios.py` |
| **Providers**: DeepSeek and Qwen translation, tool calls, errors, retries, timeouts (including a real local HTTP server), configuration, full pipeline through both mocked providers, provider switch | `test_providers.py` |
| **Baseline**: pre-existing vs new failures, behaviour evidence | `test_baseline.py` |
| Optional live provider tests (`make test-live`) | `tests/live/` |
| **Failure modes**: missing or invalid key, model timeout or hang, malformed output, tool timeout, missing command or test framework, bad repository path, permission error, retry exhaustion | `test_failure_modes.py` |

The evaluation scenarios are:
1. a simple bug goes Coder → Tester → VERIFIED;
2. an insufficient first fix leads to FAIL → repair → PASS;
3. research runs, then Coder, then Tester, and the result is VERIFIED;
4. coding and music run in parallel;
5. an unavailable dependency ends BLOCKED after bounded attempts.

## 15. Example task

```bash
export AI_API_KEY="<PROVIDED_API_KEY>"
make run
# Enter a task: Fix the failing tests in src/auth.py and use the correct library API if necessary.
```

What happens:
1. The planner builds the graph. If the project imports, say, `jwt`, it adds optional research
   on the `jwt` API that runs in parallel with the repository inspection.
2. The Researcher returns verified findings.
3. The Coder receives the relevant files and findings, edits, and reviews the diff.
4. The Tester runs the related tests and then the suite. If they fail, the Coder repairs using
   the actual assertion output, and the tests run again.
5. The gate verifies the result and the final report is shown.

Without a configured provider you can watch the same pipeline on the bundled sample:

```bash
make demo TASK="Fix the bug in this repository and play Beethoven Symphony No. 5 while you work."
```

In the demo:
- the first fix leaves one test failing (`test-1` FAIL), so `repair-1` runs and `test-2`
  passes, giving **VERIFIED SUCCESS**;
- the music plays alongside, or shows DISABLED if there is no audio player.

## 16. Architecture diagram

```
 ┌──────────────┐    ┌───────────────────────────── ORCHESTRATOR ──────────────────────────────┐
 │ USER / TASK  │───▶│ Planner ─▶ TaskGraph (DAG) ─▶ Scheduler (async, ≤3 agents, RW repo lock) │
 └──────────────┘    │     ▲                               │                                   │
                     │     │ repair-N / test-N+1           ▼                                   │
                     │ RecoveryManager ◀── results ── agents on private state copies           │
                     │     │ retry / repair / proceed / block        │                         │
                     │     ▼                                         ▼                         │
                     │ VerificationManager ─────────────▶ FinalResult (VERIFIED/BLOCKED/FAILED)│
                     └──────┬──────────────┬───────────────┬───────────────┬───────────────────┘
                            ▼              ▼               ▼               ▼
                         CODER        RESEARCHER        TESTER          MUSIC (optional)
                   read/write/terminal  read/web     read/verify-only     playback only
                            │              │               │
                            ▼              ▼               ▼
                    fs · search · git   docs · registry   detect · run_tests · classify
                    terminal (policy)   python API        (pytest, unittest, npm, make)
                            └──────── ContextBuilder: targeted, bounded, compressed ────────┘
 Event log ──▶ TUI (read-only view) · log file (redacted)
```

## 17. Limitations

- **Evaluator configuration still needed.** The exact DeepSeek and Qwen model IDs, and the Qwen
  endpoint (its documented endpoints are region- and workspace-specific), must come from the
  organizers via `AI_MODEL` / `AI_BASE_URL`. The adapters have been tested against mocked
  endpoints in both providers' wire formats; a live run needs the evaluator's configuration.
- **The command policy prevents accidents; it is not a sandbox.** Allowed interpreters such
  as `python3`, and the target repository's own tests, run arbitrary code with your user's
  permissions. This is the trust boundary: run the harness on repositories you trust, ideally
  inside a container.
- **`web_search`** uses DuckDuckGo's HTML endpoint, which may serve a bot check. That is
  reported as a failure, never as success, and the Researcher falls back to `package_info`,
  `lookup_python_api` and `fetch_documentation`.
- **Test detection** covers pytest, unittest, npm scripts and Makefile `test` targets. Other
  ecosystems result in `NOT_AVAILABLE`, and the task ends `BLOCKED` rather than guessing.
- **Task-behaviour evidence** comes from tests. A change that no test exercises is reported as
  unverified (`BLOCKED`) unless the Coder adds a test, which its instructions ask it to do.
- **Adapters without a request timeout.** Built-in adapters enforce a total per-request deadline
  (`AI_TIMEOUT_SECONDS`). A third-party plugin adapter that ignores it is still cut off at the
  agent time limit, but its thread may run until the call returns.
- **The scripted client** (`AI_PROVIDER=scripted`, `make demo`) only knows the bundled sample
  repository. It is selected only explicitly, and a failing real provider never falls back to
  it.

## 18. Configuration reference

All settings are environment variables. `.env.example` lists them with empty or default
values.

| Variable | Default | Purpose |
|---|---|---|
| `AI_API_KEY` | **required** | model credential |
| `AI_PROVIDER` | unset | `deepseek`, `qwen`, `scripted` (case-insensitive), or a plugin adapter |
| `AI_MODEL` | unset | model ID from the evaluator; **required** for DeepSeek and Qwen |
| `AI_BASE_URL` | unset | endpoint; **required** for Qwen, optional for DeepSeek (its documented default is `https://api.deepseek.com`) |
| `AI_TIMEOUT_SECONDS` / `AI_MAX_RETRIES` | 60 / 2 | total deadline per model request; transport retries for rate limit, server and network errors |
| `AI_MAX_OUTPUT_TOKENS` | unset | sent as `max_tokens` only when set (otherwise the provider's default applies) |
| `AI_MAX_TOOL_CALLS` | 40 | tool calls allowed per agent run |
| `AI_MAX_TOOL_OUTPUT_CHARS` / `AI_MAX_TOOL_OUTPUT_LINES` | 12000 / 400 | caps on each tool result |
| `AI_MAX_SEARCH_RESULTS` / `AI_MAX_RESEARCH_RESULTS` | 50 / 5 | result caps |
| `AI_MAX_CONTEXT_CHARS` | 16000 | budget for each agent's context package |
| `AI_COMMAND_TIMEOUT_SECONDS` / `AI_TEST_TIMEOUT_SECONDS` | 120 / 300 | command and test time limits |
| `AI_MAX_CONCURRENT_AGENTS` | 3 | scheduler concurrency |
| `AI_MAX_TOOL_RETRIES` / `AI_MAX_AGENT_RETRIES` / `AI_MAX_REPAIR_ATTEMPTS` | 2 / 2 / 5 | retry limits |
| `AI_AGENT_TIMEOUT_SECONDS` | 900 | time limit for each agent task |
| `AI_RESEARCH_BACKEND` / `AI_RESEARCH_TIMEOUT_SECONDS` | duckduckgo / 15 | web search backend (`none` disables it) |
| `AI_MUSIC_BACKEND` / `AI_MUSIC_DIR` | auto / unset | music player (`none` disables it) and local audio folder |
| `AI_LOG_FILE` | `<temp>/ai-coding-harness/harness.log` | detailed log, with credentials redacted |
| `LOG_LEVEL` | INFO | console log level (JSON mode) |
| `TASK`, `REPO` | unset | task and repository for `make run` |

## 19. Model configuration

The organizers evaluate with **DeepSeek and Qwen** models. All model configuration lives in
`harness.config.settings`, and the orchestrator, agents, tools, context and verification depend
only on `harness.llm.LLMClient`. No provider SDK is used anywhere; there are no provider
imports outside `harness/llm/providers/`.

```
                       LLMClient
                           |
             +-------------+-------------+
             |                           |
       ScriptedLLM                 real providers (OpenAI-compatible Chat Completions)
   (AI_PROVIDER=scripted)                  |   shared transport + translation
                               +-----------+-----------+
                               |                       |
                         DeepSeekClient            QwenClient
                     (AI_PROVIDER=deepseek)    (AI_PROVIDER=qwen)
```

**Configuration.** Use placeholders here; the real values are supplied by the evaluator:

```bash
export AI_API_KEY="<PROVIDED_KEY>"
export AI_PROVIDER="deepseek"
export AI_MODEL="<EVALUATOR_MODEL>"
# export AI_BASE_URL="<EVALUATOR_ENDPOINT>"   # optional for DeepSeek

export AI_API_KEY="<PROVIDED_KEY>"
export AI_PROVIDER="qwen"
export AI_MODEL="<EVALUATOR_MODEL>"
export AI_BASE_URL="<EVALUATOR_ENDPOINT>"      # required for Qwen

make check   # optional: verify connectivity (one minimal request)
make run
```

**Selection is explicit and deterministic.**
- `AI_PROVIDER` picks the adapter, case-insensitively (`deepseek` = `DEEPSEEK`).
- `AI_MODEL` is passed through unchanged, and `AI_BASE_URL` takes precedence over any default.
- The provider is never inferred from the API key's format, and **no model ID is hard-coded**.
- DeepSeek's documented endpoint (`https://api.deepseek.com`) is used only when `AI_BASE_URL`
  is unset.
- Qwen has no default endpoint: Alibaba Cloud documents region- and workspace-specific URLs,
  so `AI_BASE_URL` is required.

**Adapters** translate the harness's text-only messages, tool definitions, tool calls and tool
results to and from the Chat Completions wire format. Provider objects never leave the
adapter. Differences between the providers stay in their adapters:

| | DeepSeek | Qwen |
|---|---|---|
| default endpoint | `https://api.deepseek.com` (documented) | none: `AI_BASE_URL` required |
| reasoning output | `reasoning_content` is kept and echoed back on the next assistant turn, as the API requires | inline `<think>…</think>` is stripped |
| tool calls | structured `tool_calls` | structured `tool_calls`, or `<tool_call>{…}</tool_call>` blocks in the text (parsed) |
| extra finish reasons | `insufficient_system_resource` is treated as a transient server error | |

**Errors are structured** (`LLMErrorCode`) and bounded by the recovery policy:

| Code | Cause | Retried |
|---|---|---|
| `CONFIGURATION_ERROR` | no provider, no model, Qwen without an endpoint, unknown provider | never; execution does not start |
| `AUTHENTICATION_ERROR` | HTTP 401 / 403 | never |
| `BAD_REQUEST` | HTTP 400 / 404 / 422 (for example an unknown model) | never |
| `RATE_LIMITED` / `SERVER_ERROR` / `NETWORK_ERROR` | HTTP 429 / 5xx, connection problems | adapter backoff (`AI_MAX_RETRIES`, honours `Retry-After`), then agent retry |
| `PROVIDER_TIMEOUT` | no complete response within `AI_TIMEOUT_SECONDS` | agent retry (`AI_MAX_AGENT_RETRIES`) |
| `MALFORMED_RESPONSE` | invalid JSON or an unexpected shape | agent retry |
| `MALFORMED_TOOL_CALL` | tool call without a name (error), or invalid argument JSON (the registry rejects the call and the model sees why) | bounded |

**Timeouts are real.** The shared transport enforces one total deadline per request on the
socket (connect, headers and body, including slow-drip bodies), in the calling thread, with no
helper threads. A timeout becomes `PROVIDER_TIMEOUT` and goes to recovery; nothing is left
running, and the process exits cleanly.

**Scripted mode vs real mode.** Scripted mode is only ever chosen explicitly
(`AI_PROVIDER=scripted` or `demo`). A failing real provider never falls back to it, so a run
can't appear to succeed without the evaluation model.

**Other adapters** can still be plugged in without source changes: `AI_PROVIDER=module:factory`,
or an entry point in the `ai_coding_harness.providers` group.

## 20. Security considerations

- **Credentials:** read from the environment only; never stored, logged or printed.
  - Log and event text is redacted (API key values, common token formats and
    `key=value` secrets).
  - Every command the harness runs gets an environment with credential variables removed.
  - Research tools refuse to send anything that looks like a secret.
- **Repository boundary:** every path is resolved (following `..` and symlinks) and checked
  against the repository root. `~` and NUL bytes are rejected, `.git/` is write-protected,
  and git can never fall back to a parent repository.
- **Command policy** (`tools/policy.py`):
  - an allow-list of development commands, with argument rules for each;
  - HIGH_RISK commands are always blocked: `rm`, `sudo`, `shutdown`, `mkfs`, `dd`, `chmod`,
    `kill`, shells, `curl`/`wget`, `git push/reset/clean`, `git branch -D`, …;
  - no shell, so no pipes, redirects or substitution;
  - arguments that point outside the repository are rejected;
  - every command has a timeout, and the whole process group is killed when it expires.
- **Agent permissions:** capabilities are enforced by the orchestrator. The Tester and
  Researcher cannot write, and state changes an agent is not allowed to make are rejected.
- **Network:** only research tools use the network. They allow http and https only and
  refuse private, loopback and link-local addresses (including after redirects), with
  timeouts and size caps.
- **Integrity:** the repository is fingerprinted around verification, so tests that modify
  source files invalidate the result.
- **Model provider:** the credential travels only in the `Authorization` header of requests
  to the configured endpoint. It never appears in request bodies, model context, tool results
  (any occurrence is replaced with `[REDACTED]`), state, events, logs, the TUI or error
  messages; masked key echoes from providers are removed too.
- **Subprocesses:** every command the harness runs (terminal, tests, git, ripgrep, Python API
  lookup, music player) gets a controlled working directory and an environment with
  credentials removed. All except the background music player have enforced timeouts, and
  output is captured.
- **Trust boundary:** the harness is not a sandbox. The target repository's tests and any
  allowed interpreter run arbitrary code as the current user.
- **Secret audit:** there are no credentials in the repository or its git history.
  `.env.example` has only empty placeholders, and `.env` files are git-ignored.
