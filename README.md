# AI Harness

A Python multi-agent AI coding harness for the **LCC × DevClub AI Harness Hackathon 2026**.

```
USER / EVALUATION TASK
        │
        ▼
   ORCHESTRATOR ──┬── CODER AGENT
        │         ├── RESEARCHER AGENT
        │         ├── TESTER AGENT
        │         └── MUSIC AGENT
        ▼
SHARED STATE / CONTEXT
        │
        ▼
   VERIFICATION ──── FAIL → RECOVERY / RETRY
        │
        └─────────── PASS → VERIFIED RESULT
```

## Status

**Phase 1 of 7: foundation.** This phase includes configuration, the LLM abstraction, logging,
the core interfaces, the CLI and the test setup. The agents' behavior and the orchestrator
run loop come in later phases. `harness run` is a placeholder that exits with code 2.

## Quick start

Requires Python ≥ 3.10.

```bash
make install          # creates .venv, installs package + dev deps, copies .env.example → .env
make check            # ruff + mypy (strict) + pytest
.venv/bin/harness --provider mock ping "hello"   # offline smoke test, no API key needed
```

To use Claude, set `ANTHROPIC_API_KEY` in `.env` (or run `ant auth login`), then:

```bash
make ping
```

## Layout

```
src/harness/
  cli.py              # entry point: `harness` / `python -m harness`
  config.py           # HarnessConfig: defaults < .env < environment < CLI flags
  logging_setup.py    # text or JSON-lines logging to stderr (+ optional file)
  llm/
    base.py           # LLMClient ABC, Message, LLMResponse, Usage, LLMError, LLMRefusalError
    anthropic_client.py
    mock_client.py    # deterministic offline client for tests / dry runs
    factory.py        # create_llm_client(config) picks the provider
  core/
    task.py           # Task, TaskStatus
    state.py          # SharedState: thread-safe KV context + event log, JSON snapshot
    agent.py          # BaseAgent, AgentRole, AgentResult
    verification.py   # Verifier ABC, VerificationResult
    orchestrator.py   # agent registry; run loop comes in a later phase
  agents/             # Coder / Researcher / Tester / Music (later phases)
tests/unit/           # pytest suite (no network)
```

## Configuration

All settings are environment variables. See [.env.example](.env.example) for the full list.

| Variable | Default | Notes |
|---|---|---|
| `HARNESS_LLM_PROVIDER` | `anthropic` | `anthropic` or `mock` |
| `HARNESS_LLM_MODEL` | `claude-opus-5` | |
| `HARNESS_LLM_MAX_TOKENS` | `16000` | |
| `HARNESS_LLM_TIMEOUT_S` | `600` | per request, seconds |
| `HARNESS_LLM_MAX_RETRIES` | `2` | SDK-level retries (429 / 5xx / network) |
| `HARNESS_LLM_REFUSAL_FALLBACK` | `true` | server-side fallback if the model declines |
| `ANTHROPIC_API_KEY` | — | optional if `ant auth login` was used |
| `HARNESS_LOG_LEVEL` | `INFO` | |
| `HARNESS_LOG_FORMAT` | `text` | `text` or `json` |
| `HARNESS_LOG_FILE` | — | also write logs here |
| `HARNESS_WORKSPACE_DIR` | `workspace` | |
| `HARNESS_MAX_RECOVERY_ATTEMPTS` | `3` | used by the recovery loop (later phase) |

`harness config` prints the resolved configuration with secrets masked.

## Extending

**New agent:** subclass `BaseAgent`, set `role`, and implement `execute(state) -> AgentResult`.
Callers use `agent.run(state)`, which logs start and finish events to the shared state and
turns exceptions into a failed result.

**New LLM provider:** implement `LLMClient.complete()` and add a branch in `llm/factory.py`.
Agents depend only on `LLMClient`, so no agent code changes.

## Make targets

`make help` lists them all: `install`, `test`, `test-cov`, `lint`, `format`, `typecheck`,
`check`, `config`, `ping`, `run TASK="..."`, `clean`.
