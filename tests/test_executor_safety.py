from app.tools.executor import Executor, WORKSPACE, should_ask_permission


def test_network_command_requires_approval() -> None:
    ask, reason = should_ask_permission("curl https://example.com")

    assert ask
    assert "approval-required" in reason


def test_sudo_is_denied_not_approval_gated() -> None:
    ask, _ = should_ask_permission("sudo ls")

    assert not ask


def test_absolute_path_outside_workspace_requires_approval() -> None:
    ask, reason = should_ask_permission("cat /tmp/outside.txt")

    assert ask
    assert "outside the workspace" in reason


def test_shell_runs_in_workspace_by_default() -> None:
    result = Executor().run_shell("pwd")

    assert result["status"] == "completed"
    assert result["exit_code"] == 0
    assert result["stdout"].strip() == str(WORKSPACE)


def test_python_runs_and_captures_output() -> None:
    result = Executor().run_python("print('hello')")

    assert result["status"] == "completed"
    assert result["exit_code"] == 0
    assert result["stdout"].strip() == "hello"


def test_outside_workspace_cwd_is_denied() -> None:
    result = Executor().run_shell("pwd", cwd="/tmp")

    assert result["status"] == "denied"
