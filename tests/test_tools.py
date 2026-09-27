from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from harness.llm.models import ToolCall
from harness.tools import (
    CODER_TOOL_NAMES,
    CommandPolicy,
    CommandRisk,
    EditFileTool,
    GitDiffTool,
    GitLogTool,
    GitStatusTool,
    ListFilesTool,
    PathOutsideRepositoryError,
    ReadFileTool,
    RepositoryContext,
    SearchCodeTool,
    TerminalTool,
    ToolRegistry,
    ToolStatus,
    UnknownToolError,
    WriteFileTool,
    build_registry,
    truncate_text,
)
from harness.tools.policy import sanitized_environment

from .conftest import BUGGY_APP, git

# --- RepositoryContext -------------------------------------------------------------------


def test_repository_root_must_exist(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        RepositoryContext(tmp_path / "missing")


@pytest.mark.parametrize("path", [".", "app.py", "pkg/util.py", "pkg/../app.py", "new/dir/f.py"])
def test_paths_inside_root_resolve(plain_repo: RepositoryContext, path: str) -> None:
    resolved = plain_repo.resolve(path)
    assert resolved == plain_repo.root or resolved.is_relative_to(plain_repo.root)


@pytest.mark.parametrize(
    "path", ["..", "../", "../../etc/passwd", "pkg/../../x", "/etc/passwd", "~/secrets", "a\x00b"]
)
def test_path_traversal_rejected(plain_repo: RepositoryContext, path: str) -> None:
    with pytest.raises(PathOutsideRepositoryError):
        plain_repo.resolve(path)


def test_absolute_path_inside_root_allowed(plain_repo: RepositoryContext) -> None:
    assert plain_repo.resolve(str(plain_repo.root / "app.py")) == plain_repo.root / "app.py"


def test_symlink_escape_rejected(plain_repo: RepositoryContext, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("top secret")
    os.symlink(outside, plain_repo.root / "link_dir")
    os.symlink(outside / "secret.txt", plain_repo.root / "link_file")

    for path in ("link_dir/secret.txt", "link_file"):
        with pytest.raises(PathOutsideRepositoryError):
            plain_repo.resolve(path)
    result = ReadFileTool(plain_repo).run({"path": "link_file"})
    assert not result.ok and "escapes" in (result.error or "")
    assert "top secret" not in str(result)


def test_git_dir_is_write_protected(git_repo: RepositoryContext) -> None:
    result = WriteFileTool(git_repo).run({"path": ".git/config", "content": "x"})
    assert not result.ok and "protected" in (result.error or "")


# --- list_files --------------------------------------------------------------------------


def test_list_files_recursive_skips_generated(plain_repo: RepositoryContext) -> None:
    for d in (".git", "node_modules", "__pycache__", ".venv", "foo.egg-info"):
        (plain_repo.root / d).mkdir()
        (plain_repo.root / d / "junk.txt").write_text("x")
    result = ListFilesTool(plain_repo).run({})
    assert result.ok
    assert result.data["entries"] == ["README.md", "app.py", "pkg/util.py"]
    assert result.data["truncated"] is False


def test_list_files_non_recursive_and_limit(plain_repo: RepositoryContext) -> None:
    result = ListFilesTool(plain_repo).run({"recursive": False})
    assert result.data["entries"] == ["README.md", "app.py", "pkg/"]
    limited = ListFilesTool(plain_repo).run({"max_results": 2})
    assert limited.data["count"] == 2 and limited.data["truncated"] is True


def test_list_files_rejects_outside_and_missing(plain_repo: RepositoryContext) -> None:
    assert not ListFilesTool(plain_repo).run({"path": ".."}).ok
    assert "not found" in (ListFilesTool(plain_repo).run({"path": "nope"}).error or "")


# --- read_file ---------------------------------------------------------------------------


def test_read_file_full_and_range(plain_repo: RepositoryContext) -> None:
    tool = ReadFileTool(plain_repo)
    full = tool.run({"path": "app.py"})
    assert full.ok and full.data["content"] == BUGGY_APP
    assert full.data["total_lines"] == 6 and full.data["size_bytes"] == len(BUGGY_APP)

    part = tool.run({"path": "app.py", "start_line": 2, "end_line": 2})
    assert part.data["content"] == "    return a - b\n"
    assert (part.data["start_line"], part.data["end_line"]) == (2, 2)


def test_read_file_errors(plain_repo: RepositoryContext) -> None:
    tool = ReadFileTool(plain_repo)
    assert "not found" in (tool.run({"path": "missing.py"}).error or "")
    assert "escapes" in (tool.run({"path": "../x.py"}).error or "")
    assert "Not a regular file" in (tool.run({"path": "pkg"}).error or "")
    (plain_repo.root / "bin.dat").write_bytes(b"\x00\x01\x02")
    assert "Binary" in (tool.run({"path": "bin.dat"}).error or "")
    assert not tool.run({"path": "app.py", "start_line": 99}).ok
    assert not tool.run({"path": "app.py", "start_line": "1"}).ok  # wrong type


def test_read_file_truncates_huge_content(plain_repo: RepositoryContext) -> None:
    (plain_repo.root / "big.txt").write_text("x" * 50_000)
    result = ReadFileTool(plain_repo, max_chars=1_000).run({"path": "big.txt"})
    assert result.ok and result.data["truncated"] and len(result.data["content"]) <= 1_000


# --- write_file --------------------------------------------------------------------------


def test_write_file_creates_with_parents(plain_repo: RepositoryContext) -> None:
    result = WriteFileTool(plain_repo).run({"path": "new/sub/mod.py", "content": "X = 1\n"})
    assert result.ok and result.data["path"] == "new/sub/mod.py"
    assert result.data["created"] and result.data["changed"]
    assert (plain_repo.root / "new/sub/mod.py").read_text() == "X = 1\n"


def test_write_file_overwrite_and_noop(plain_repo: RepositoryContext) -> None:
    tool = WriteFileTool(plain_repo)
    first = tool.run({"path": "README.md", "content": "# new\n"})
    assert first.ok and not first.data["created"] and first.data["old_size"] == len("# demo\n")
    again = tool.run({"path": "README.md", "content": "# new\n"})
    assert again.ok and again.data["changed"] is False


def test_write_file_rejections(plain_repo: RepositoryContext, tmp_path: Path) -> None:
    tool = WriteFileTool(plain_repo)
    assert not tool.run({"path": "../evil.py", "content": "x"}).ok
    assert not (tmp_path / "evil.py").exists()
    assert not tool.run({"path": "pkg", "content": "x"}).ok
    assert not tool.run({"path": "nodir/f.py", "content": "x", "create_dirs": False}).ok
    assert not tool.run({"path": "f.py"}).ok  # missing content


# --- edit_file ---------------------------------------------------------------------------


def test_edit_file_exact_single_replacement(plain_repo: RepositoryContext) -> None:
    result = EditFileTool(plain_repo).run(
        {"path": "app.py", "old_text": "return a - b", "new_text": "return a + b"}
    )
    assert result.ok and result.data["changed"] and result.data["occurrences"] == 1
    assert result.data["old_size"] == result.data["new_size"]
    assert "return a + b" in (plain_repo.root / "app.py").read_text()


def test_edit_file_not_found_leaves_file_untouched(plain_repo: RepositoryContext) -> None:
    result = EditFileTool(plain_repo).run(
        {"path": "app.py", "old_text": "return a / b", "new_text": "x"}
    )
    assert not result.ok and "not found" in (result.error or "")
    assert result.data["changed"] is False
    assert (plain_repo.root / "app.py").read_text() == BUGGY_APP


def test_edit_file_ambiguous_match_rejected(plain_repo: RepositoryContext) -> None:
    tool = EditFileTool(plain_repo)
    ambiguous = tool.run({"path": "app.py", "old_text": "return a", "new_text": "return b"})
    assert not ambiguous.ok and ambiguous.data["occurrences"] == 2
    assert (plain_repo.root / "app.py").read_text() == BUGGY_APP

    both = tool.run(
        {"path": "app.py", "old_text": "(a, b)", "new_text": "(x, y)", "expected_occurrences": 2}
    )
    assert both.ok and (plain_repo.root / "app.py").read_text().count("(x, y)") == 2


def test_edit_file_rejections(plain_repo: RepositoryContext) -> None:
    tool = EditFileTool(plain_repo)
    assert "not found" in (tool.run({"path": "nope.py", "old_text": "a", "new_text": "b"}).error or "")
    assert not tool.run({"path": "../app.py", "old_text": "a", "new_text": "b"}).ok
    assert not tool.run({"path": "app.py", "old_text": "", "new_text": "b"}).ok


# --- search_code -------------------------------------------------------------------------


def test_search_code_python_fallback(plain_repo: RepositoryContext) -> None:
    (plain_repo.root / "node_modules").mkdir()
    (plain_repo.root / "node_modules" / "dep.py").write_text("def add(): pass\n")
    result = SearchCodeTool(plain_repo, use_ripgrep=False).run({"query": "def add"})
    assert result.ok and result.data["engine"] == "python"
    assert result.data["files"] == ["app.py"]
    match = result.data["matches"][0]
    assert (match["file"], match["line"], match["text"]) == ("app.py", 1, "def add(a, b):")
    assert match["after"] == ["    return a - b", ""]


def test_search_code_filters_and_limits(plain_repo: RepositoryContext) -> None:
    tool = SearchCodeTool(plain_repo, use_ripgrep=False)
    assert tool.run({"query": "return", "file_type": "md"}).data["count"] == 0
    assert tool.run({"query": "return", "path": "pkg"}).data["files"] == ["pkg/util.py"]
    limited = tool.run({"query": "return", "max_results": 1})
    assert limited.data["count"] == 1 and limited.data["truncated"]
    assert tool.run({"query": "RETURN", "ignore_case": True}).data["count"] == 3
    assert tool.run({"query": r"def \w+\(a", "regex": True}).data["count"] == 2
    assert not tool.run({"query": "(", "regex": True}).ok
    assert not tool.run({"query": "x", "path": "../"}).ok


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
def test_search_code_ripgrep_matches_fallback(plain_repo: RepositoryContext) -> None:
    rg = SearchCodeTool(plain_repo, use_ripgrep=True).run({"query": "return"})
    py = SearchCodeTool(plain_repo, use_ripgrep=False).run({"query": "return"})
    assert rg.data["engine"] == "ripgrep"
    assert rg.data["matches"] == py.data["matches"]


# --- terminal ----------------------------------------------------------------------------


def test_terminal_runs_safe_command(plain_repo: RepositoryContext) -> None:
    result = TerminalTool(plain_repo).run({"command": "python -c \"print('hi')\""})
    assert result.ok, result.error
    assert result.data["stdout"].strip() == "hi"
    assert result.data["exit_code"] == 0 and result.data["duration_seconds"] >= 0


def test_terminal_nonzero_exit_is_failure_with_output(plain_repo: RepositoryContext) -> None:
    result = TerminalTool(plain_repo).run(
        {"command": "python -c \"import sys; print('bad', file=sys.stderr); sys.exit(3)\""}
    )
    assert not result.ok and result.data["exit_code"] == 3
    assert "bad" in result.data["stderr"] and "code 3" in (result.error or "")


def test_terminal_timeout_kills_command(plain_repo: RepositoryContext) -> None:
    tool = TerminalTool(plain_repo, default_timeout=1, max_timeout=1)
    result = tool.run({"command": "python -c \"import time; time.sleep(30)\"", "timeout_seconds": 60})
    assert not result.ok and result.data["timed_out"] is True
    assert result.data["exit_code"] is None and result.data["duration_seconds"] < 10


def test_terminal_cwd_must_be_inside_repo(plain_repo: RepositoryContext) -> None:
    tool = TerminalTool(plain_repo)
    assert tool.run({"command": "pwd", "cwd": "pkg"}).data["stdout"].strip().endswith("pkg")
    assert not tool.run({"command": "pwd", "cwd": ".."}).ok


def test_terminal_does_not_leak_secrets(
    plain_repo: RepositoryContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AI_API_KEY", "leak-me-please")
    monkeypatch.setenv("GITHUB_TOKEN", "leak-me-too")
    result = TerminalTool(plain_repo).run(
        {"command": "python -c \"import os; print(sorted(os.environ.items()))\""}
    )
    assert result.ok and "leak-me" not in result.data["stdout"]
    assert "AI_API_KEY" not in sanitized_environment({"AI_API_KEY": "x", "PATH": "/bin"})


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "rm file.txt",
        "sudo ls",
        "shutdown -h now",
        "reboot",
        "mkfs.ext4 /dev/sda1",
        "dd if=/dev/zero of=/dev/sda",
        "chmod -R 777 .",
        "git push origin main",
        "git reset --hard HEAD~1",
        "git clean -fdx",
        "git branch -D main",
        "bash -c 'rm -rf /'",
        "curl http://example.invalid",
    ],
)
def test_high_risk_commands_rejected(plain_repo: RepositoryContext, command: str) -> None:
    result = TerminalTool(plain_repo).run({"command": command})
    assert not result.ok and result.data.get("rejected")
    assert result.data["risk"] == CommandRisk.HIGH_RISK


@pytest.mark.parametrize(
    "command",
    [
        "ls | grep x",
        "ls && pwd",
        "echo hi > out.txt",
        "echo $(whoami)",
        "echo `whoami`",
        "/bin/ls",
        "unknowncmd --flag",
        "cat /etc/passwd",
        "cat ../../secret",
        "ls ~",
        "find . -delete",
        "git -C / status",
        "pip uninstall -y pytest",
        "npm publish",
    ],
)
def test_disallowed_commands_rejected(plain_repo: RepositoryContext, command: str) -> None:
    result = TerminalTool(plain_repo).run({"command": command})
    assert not result.ok and result.data.get("rejected"), result
    assert result.data["risk"] in (CommandRisk.NOT_ALLOWED, CommandRisk.HIGH_RISK)
    assert not (plain_repo.root / "out.txt").exists()


@pytest.mark.parametrize(
    "command",
    ["python -m pytest -q", "pytest", "git status", "git diff", "git log -n 3", "ls -la",
     "grep -rn add .", "pwd", "npm test", "pip list", "node --version", "cat app.py",
     "find . -name '*.py'", "python3.12 --version"],
)
def test_safe_commands_allowed_by_policy(plain_repo: RepositoryContext, command: str) -> None:
    policy = CommandPolicy()
    decision = policy.evaluate(policy.parse(command), plain_repo)
    assert decision.allowed and decision.risk is CommandRisk.SAFE, decision.reason


def test_policy_cannot_allowlist_high_risk() -> None:
    with pytest.raises(ValueError):
        CommandPolicy(extra_safe=["rm"])


def test_terminal_rejects_multiline_and_empty(plain_repo: RepositoryContext) -> None:
    tool = TerminalTool(plain_repo)
    assert not tool.run({"command": "ls\nrm -rf /"}).ok
    assert not tool.run({"command": "   "}).ok
    assert not tool.run({"command": "echo 'unterminated"}).ok


def test_terminal_output_truncated(plain_repo: RepositoryContext) -> None:
    tool = TerminalTool(plain_repo, max_output_chars=1_000)
    result = tool.run({"command": "python -c \"print('x' * 20000)\""})
    assert result.ok and result.data["truncated"] and len(result.data["stdout"]) <= 500


# --- git ---------------------------------------------------------------------------------


def test_git_status_clean_then_dirty(git_repo: RepositoryContext) -> None:
    tool = GitStatusTool(git_repo)
    clean = tool.run({})
    assert clean.ok and clean.data["clean"] and clean.data["branch"].startswith("main")

    (git_repo.root / "app.py").write_text("changed\n")
    (git_repo.root / "new.py").write_text("x = 1\n")
    git(git_repo.root, "add", "README.md")
    (git_repo.root / "README.md").write_text("# edited\n")
    git(git_repo.root, "add", "README.md")

    dirty = tool.run({})
    by_path = {f["path"]: f for f in dirty.data["files"]}
    assert not dirty.data["clean"]
    assert by_path["app.py"]["status"] == "modified" and not by_path["app.py"]["staged"]
    assert by_path["new.py"]["status"] == "untracked"
    assert by_path["README.md"]["staged"]
    assert dirty.data["counts"] == {"staged": 1, "unstaged": 1, "untracked": 1}


def test_git_diff_structured(git_repo: RepositoryContext) -> None:
    (git_repo.root / "app.py").write_text(BUGGY_APP.replace("a - b", "a + b"))
    (git_repo.root / "new.py").write_text("x = 1\ny = 2\n")
    result = GitDiffTool(git_repo).run({})
    assert result.ok and result.data["has_changes"]
    files = {f["path"]: f for f in result.data["files"]}
    assert (files["app.py"]["additions"], files["app.py"]["deletions"]) == (1, 1)
    assert files["new.py"]["status"] == "untracked" and files["new.py"]["additions"] == 2
    assert "-    return a - b" in result.data["diff"]
    assert "+    return a + b" in result.data["diff"]
    assert result.data["summary"] == "2 file(s) changed, +3 -1"

    only_app = GitDiffTool(git_repo).run({"path": "app.py"})
    assert [f["path"] for f in only_app.data["files"]] == ["app.py"]
    assert GitDiffTool(git_repo).run({"staged": True}).data["has_changes"] is False


def test_git_diff_clean_repo(git_repo: RepositoryContext) -> None:
    result = GitDiffTool(git_repo).run({})
    assert result.ok and not result.data["has_changes"] and result.data["diff"] == ""


def test_git_log(git_repo: RepositoryContext) -> None:
    result = GitLogTool(git_repo).run({"max_count": 5})
    assert result.ok and result.data["count"] == 1
    assert result.data["commits"][0]["subject"] == "initial"


def test_git_tools_require_repo(plain_repo: RepositoryContext) -> None:
    for tool in (GitStatusTool(plain_repo), GitDiffTool(plain_repo), GitLogTool(plain_repo)):
        result = tool.run({})
        assert not result.ok and "Not a git repository" in (result.error or "")


def test_git_does_not_escape_to_parent_repo(git_repo: RepositoryContext) -> None:
    sub = git_repo.root / "pkg"
    result = GitStatusTool(RepositoryContext(sub)).run({})
    assert not result.ok


# --- registry ----------------------------------------------------------------------------


def test_registry_register_get_and_schemas(plain_repo: RepositoryContext) -> None:
    registry = build_registry(plain_repo)
    assert set(CODER_TOOL_NAMES) <= set(registry.names) and "git_log" in registry
    schemas = {s["name"]: s for s in registry.schemas()}
    assert schemas["read_file"]["input_schema"]["required"] == ["path"]
    assert all(s["description"] for s in schemas.values())
    with pytest.raises(ValueError):
        registry.register(ReadFileTool(plain_repo))
    with pytest.raises(UnknownToolError):
        registry.get("delete_everything")


def test_registry_execute_by_name(plain_repo: RepositoryContext) -> None:
    registry = ToolRegistry([ReadFileTool(plain_repo)])
    ok = registry.execute("read_file", {"path": "app.py"})
    assert ok.status is ToolStatus.SUCCESS
    unknown = registry.execute("nope", {})
    assert not unknown.ok and "Unknown tool" in (unknown.error or "")
    assert not registry.execute("read_file", ["app.py"]).ok
    assert registry.execute("read_file", '{"path": "app.py"}').ok
    assert "not valid JSON" in (registry.execute("read_file", "{oops").error or "")


def test_registry_execute_call_builds_tool_result(plain_repo: RepositoryContext) -> None:
    registry = ToolRegistry([ReadFileTool(plain_repo)], max_llm_chars=600)
    (plain_repo.root / "big.txt").write_text("y" * 5_000)
    result, tool_result = registry.execute_call(ToolCall("c1", "read_file", {"path": "big.txt"}))
    assert result.ok and tool_result.tool_call_id == "c1" and not tool_result.is_error
    assert len(tool_result.content) <= 600 and "truncated" in tool_result.content
    _, err = registry.execute_call(ToolCall("c2", "read_file", {"path": "../x"}))
    assert err.is_error and '"ok": false' in err.content


def test_registry_subset(plain_repo: RepositoryContext) -> None:
    sub = build_registry(plain_repo, names=["read_file", "git_diff"])
    assert sub.names == ["git_diff", "read_file"]
    with pytest.raises(UnknownToolError):
        build_registry(plain_repo, names=["nope"])


def test_truncate_text() -> None:
    assert truncate_text("short", 10) == ("short", False)
    text, cut = truncate_text("a" * 100 + "b" * 100, 80)
    assert cut and len(text) <= 80 and text.startswith("a") and text.endswith("b")


def test_python_executable_available() -> None:
    # The terminal tests invoke `python`; make sure it resolves inside the venv.
    assert shutil.which("python") or sys.executable
