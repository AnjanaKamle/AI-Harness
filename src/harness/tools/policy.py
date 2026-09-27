"""Centralized command policy for the terminal tool.

Every command the harness executes goes through :meth:`CommandPolicy.evaluate`. Commands
run without a shell (argv only), so pipes, redirection and substitution are unavailable by
construction; the policy also rejects them explicitly so the model gets a clear reason.

Classification:
  SAFE         allow-listed development executables, with per-command argument rules
  HIGH_RISK    destructive / privileged / system-level commands - always rejected by default
  NOT_ALLOWED  anything not on the allow-list (rejected; the allow-list is the contract)

This is a guard-rail against accidents, not a sandbox: an allowed interpreter such as
``python`` can still execute arbitrary code.
"""

from __future__ import annotations

import os
import re
import shlex
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from harness.tools.base import ToolError
from harness.tools.repository import RepositoryContext


class CommandRisk(StrEnum):
    SAFE = "SAFE"
    HIGH_RISK = "HIGH_RISK"
    NOT_ALLOWED = "NOT_ALLOWED"


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    risk: CommandRisk
    reason: str
    argv: tuple[str, ...]


HIGH_RISK_EXECUTABLES: frozenset[str] = frozenset(
    {
        # privilege escalation / identity
        "sudo", "su", "doas", "passwd", "useradd", "userdel", "usermod", "visudo",
        # deletion / destructive file ops
        "rm", "rmdir", "shred", "srm", "unlink", "truncate",
        # power / services / scheduling
        "shutdown", "reboot", "halt", "poweroff", "init", "systemctl", "service",
        "launchctl", "crontab", "at",
        # disks and filesystems
        "dd", "fdisk", "sfdisk", "parted", "diskutil", "mount", "umount", "mkswap", "swapon",
        "wipefs", "format",
        # permissions / ownership / processes
        "chmod", "chown", "chgrp", "chattr", "kill", "killall", "pkill",
        # shells and wrappers that would bypass this policy
        "sh", "bash", "zsh", "fish", "dash", "ksh", "csh", "tcsh", "env", "xargs", "eval",
        "exec", "nohup", "osascript",
        # network / remote access and downloads
        "curl", "wget", "ssh", "scp", "sftp", "rsync", "nc", "ncat", "telnet", "ftp",
    }
)
HIGH_RISK_PREFIXES: tuple[str, ...] = ("mkfs",)

SHELL_OPERATORS: frozenset[str] = frozenset(
    {"|", "||", "&", "&&", ";", ";;", ">", ">>", "<", "<<", "<<<", "2>", "2>>", "&>", "2>&1", "|&"}
)
_SUBSTITUTION = re.compile(r"`|\$\(|\$\{")

GIT_READ_ONLY: frozenset[str] = frozenset(
    {"status", "diff", "log", "show", "branch", "rev-parse", "ls-files", "blame", "grep",
     "shortlog", "describe", "cat-file", "--version"}
)
GIT_DESTRUCTIVE: frozenset[str] = frozenset(
    {"push", "reset", "clean", "rebase", "filter-branch", "filter-repo", "gc", "prune", "rm",
     "checkout", "restore", "switch", "update-ref", "reflog", "stash", "merge", "cherry-pick",
     "revert", "commit", "am", "apply", "tag", "remote", "config", "worktree", "submodule",
     "fetch", "pull", "clone", "init", "mv", "replace", "notes"}
)
GIT_BRANCH_MUTATING_FLAGS: frozenset[str] = frozenset(
    {"-d", "-D", "--delete", "-m", "-M", "--move", "-c", "-C", "--copy", "-f", "--force",
     "-u", "--set-upstream-to", "--unset-upstream", "--edit-description"}
)
FIND_FORBIDDEN: frozenset[str] = frozenset(
    {"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls"}
)
PIP_ALLOWED: frozenset[str] = frozenset(
    {"install", "list", "show", "freeze", "check", "--version", "-V"}
)
NPM_ALLOWED: frozenset[str] = frozenset(
    {"install", "i", "ci", "test", "t", "run", "run-script", "ls", "list", "outdated",
     "audit", "view", "--version", "-v"}
)

ArgRule = Callable[[list[str]], str | None]  # returns a rejection reason, or None


def _git_rule(args: list[str]) -> str | None:
    if not args:
        return "git requires a subcommand"
    sub = args[0]
    if sub.startswith("-") and sub != "--version":
        return "git global options (e.g. -C, -c, --git-dir) are not allowed"
    if sub in GIT_DESTRUCTIVE:
        return f"HIGH_RISK:git {sub} modifies repository state and is not allowed"
    if sub not in GIT_READ_ONLY:
        return f"git {sub} is not an allowed read-only git command"
    if sub == "branch" and any(a in GIT_BRANCH_MUTATING_FLAGS for a in args[1:]):
        return "HIGH_RISK:git branch may only list branches"
    if any(a == "--output" or a.startswith("--output=") for a in args[1:]):
        return "git --output is not allowed"
    return None


def _find_rule(args: list[str]) -> str | None:
    bad = sorted(FIND_FORBIDDEN.intersection(args))
    return f"find actions {bad} are not allowed" if bad else None


def _subcommand_rule(tool: str, allowed: frozenset[str]) -> ArgRule:
    def rule(args: list[str]) -> str | None:
        if not args:
            return None
        return None if args[0] in allowed else f"{tool} {args[0]} is not allowed"

    return rule


def _any(args: list[str]) -> str | None:
    return None


SAFE_EXECUTABLES: dict[str, ArgRule] = {
    "python": _any,
    "python3": _any,
    "pytest": _any,
    "pip": _subcommand_rule("pip", PIP_ALLOWED),
    "pip3": _subcommand_rule("pip", PIP_ALLOWED),
    "npm": _subcommand_rule("npm", NPM_ALLOWED),
    "node": _any,
    "git": _git_rule,
    "grep": _any,
    "rg": _any,
    "ls": _any,
    "pwd": _any,
    "cat": _any,
    "head": _any,
    "tail": _any,
    "wc": _any,
    "diff": _any,
    "echo": _any,
    "which": _any,
    "find": _find_rule,
    "tree": _any,
    "sort": _any,
    "uniq": _any,
    "make": _any,
    "ruff": _any,
    "mypy": _any,
    "black": _any,
    "tsc": _any,
    "eslint": _any,
}
_PYTHON_VERSIONED = re.compile(r"^python3\.\d+$")

ALLOWED_OUTSIDE_PATHS: frozenset[str] = frozenset({"/dev/null"})

_SECRET_ENV = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)", re.IGNORECASE)


def sanitized_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for child processes: secrets removed, pagers/prompts disabled, and the
    harness interpreter's bin directory first on PATH so ``python``/``pytest``/``pip``
    resolve even when only ``python3`` is installed system-wide."""
    env = {k: v for k, v in (os.environ if base is None else base).items()
           if not _SECRET_ENV.search(k)}
    env.update({"GIT_TERMINAL_PROMPT": "0", "GIT_PAGER": "cat", "PAGER": "cat", "CI": "1"})
    # No .pyc files: a stale cache (same size + same mtime second after an edit) would
    # otherwise make tests run old code. Also keeps target repositories clean.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    interpreter_bin = os.path.dirname(sys.executable)
    path = env.get("PATH", os.defpath)
    env["PATH"] = os.pathsep.join([interpreter_bin, path]) if path else interpreter_bin
    return env


class CommandPolicy:
    def __init__(self, extra_safe: Iterable[str] = ()) -> None:
        self.safe: dict[str, ArgRule] = dict(SAFE_EXECUTABLES)
        for name in extra_safe:
            if self._is_high_risk_name(name):
                raise ValueError(f"{name!r} is high-risk and cannot be allow-listed")
            self.safe.setdefault(name, _any)

    @staticmethod
    def parse(command: str) -> list[str]:
        if not command or not command.strip():
            raise ToolError("command must not be empty")
        if "\n" in command or "\r" in command:
            raise ToolError("multi-line commands are not allowed; run one command at a time")
        try:
            return shlex.split(command)
        except ValueError as exc:
            raise ToolError(f"could not parse command: {exc}") from exc

    @staticmethod
    def _is_high_risk_name(name: str) -> bool:
        return name in HIGH_RISK_EXECUTABLES or name.startswith(HIGH_RISK_PREFIXES)

    def evaluate(
        self, argv: list[str], repo: RepositoryContext | None = None, cwd: Path | None = None
    ) -> PolicyDecision:
        def deny(risk: CommandRisk, reason: str) -> PolicyDecision:
            return PolicyDecision(False, risk, reason, tuple(argv))

        if not argv:
            return deny(CommandRisk.NOT_ALLOWED, "empty command")

        for token in argv:
            if token in SHELL_OPERATORS:
                return deny(
                    CommandRisk.NOT_ALLOWED,
                    f"shell operator {token!r} is not supported; commands run without a shell",
                )
            if _SUBSTITUTION.search(token):
                return deny(CommandRisk.NOT_ALLOWED, "command/variable substitution is not allowed")

        exe_token = argv[0]
        if "/" in exe_token:
            return deny(
                CommandRisk.NOT_ALLOWED, "use a bare executable name (e.g. 'python'), not a path"
            )
        exe = exe_token
        if self._is_high_risk_name(exe):
            return deny(CommandRisk.HIGH_RISK, f"{exe} is a high-risk command and is blocked")

        rule = self.safe.get(exe) or (_any if _PYTHON_VERSIONED.match(exe) else None)
        if rule is None:
            return deny(CommandRisk.NOT_ALLOWED, f"{exe} is not on the allowed command list")

        reason = rule(argv[1:])
        if reason is not None:
            if reason.startswith("HIGH_RISK:"):
                return deny(CommandRisk.HIGH_RISK, reason.removeprefix("HIGH_RISK:"))
            return deny(CommandRisk.NOT_ALLOWED, reason)

        if repo is not None:
            escape = self._path_escape(argv[1:], repo, cwd or repo.root)
            if escape:
                return deny(CommandRisk.NOT_ALLOWED, escape)

        return PolicyDecision(True, CommandRisk.SAFE, "allowed", tuple(argv))

    @staticmethod
    def _path_escape(args: list[str], repo: RepositoryContext, cwd: Path) -> str | None:
        """Reject arguments that point outside the repository (absolute, ~, or ../ paths)."""
        for arg in args:
            value = arg.split("=", 1)[1] if arg.startswith("-") and "=" in arg else arg
            if not value or value in ALLOWED_OUTSIDE_PATHS:
                continue
            looks_like_path = (
                value.startswith(("/", "~")) or value == ".." or value.startswith("../")
                or "/../" in value or value.endswith("/..")
            )
            if not looks_like_path:
                continue
            if value.startswith("~"):
                return f"argument {arg!r} refers to the home directory, outside the repository"
            target = Path(value) if value.startswith("/") else cwd / value
            resolved = target.resolve()
            if resolved != repo.root and not resolved.is_relative_to(repo.root):
                return f"argument {arg!r} refers to a path outside the repository"
        return None
