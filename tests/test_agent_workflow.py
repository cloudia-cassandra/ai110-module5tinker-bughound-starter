from bughound_agent import BugHoundAgent
from llm_client import MockClient


def test_workflow_runs_in_offline_mode_and_returns_shape():
    agent = BugHoundAgent(client=None)  # heuristic-only
    code = "def f():\n    print('hi')\n    return True\n"
    result = agent.run(code)

    assert isinstance(result, dict)
    assert "issues" in result
    assert "fixed_code" in result
    assert "risk" in result
    assert "logs" in result

    assert isinstance(result["issues"], list)
    assert isinstance(result["fixed_code"], str)
    assert isinstance(result["risk"], dict)
    assert isinstance(result["logs"], list)
    assert len(result["logs"]) > 0


def test_offline_mode_detects_print_issue():
    agent = BugHoundAgent(client=None)
    code = "def f():\n    print('hi')\n    return True\n"
    result = agent.run(code)

    assert any(issue.get("type") == "Code Quality" for issue in result["issues"])


def test_offline_mode_proposes_logging_fix_for_print():
    agent = BugHoundAgent(client=None)
    code = "def f():\n    print('hi')\n    return True\n"
    result = agent.run(code)

    fixed = result["fixed_code"]
    assert "logging" in fixed
    assert "logging.info(" in fixed


def test_mock_client_forces_llm_fallback_to_heuristics_for_analysis():
    # MockClient returns non-JSON for analyzer prompts, so agent should fall back.
    agent = BugHoundAgent(client=MockClient())
    code = "def f():\n    print('hi')\n    return True\n"
    result = agent.run(code)

    assert any(issue.get("type") == "Code Quality" for issue in result["issues"])
    # Ensure we logged the fallback path
    assert any("Falling back to heuristics" in entry.get("message", "") for entry in result["logs"])


class _StubClient:
    """Returns a fixed analyzer reply; echoes a trivial fix for the fixer."""

    def __init__(self, analyzer_reply: str):
        self.analyzer_reply = analyzer_reply

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if "Return ONLY valid JSON" in system_prompt:
            return self.analyzer_reply
        return "def f():\n    return True\n"


def test_llm_issues_with_invalid_severities_fall_back_to_heuristics():
    reply = '[{"type": "Bug", "severity": "Critical", "msg": "x"}, {"type": "Style", "msg": "y"}]'
    agent = BugHoundAgent(client=_StubClient(reply))
    code = "def f():\n    print('hi')\n    return True\n"
    result = agent.run(code)

    assert any(issue.get("type") == "Code Quality" for issue in result["issues"])
    assert any("failed schema validation" in entry.get("message", "") for entry in result["logs"])


def test_llm_issues_keep_only_valid_entries_and_normalize_severity():
    reply = (
        '[{"type": "Reliability", "severity": "high", "msg": "bare except"},'
        ' {"type": "Bug", "severity": "Critical", "msg": "dropped"},'
        ' {"type": "Style", "severity": "Low", "msg": "   "}]'
    )
    agent = BugHoundAgent(client=_StubClient(reply))
    result = agent.run("def f():\n    return True\n")

    assert result["issues"] == [{"type": "Reliability", "severity": "High", "msg": "bare except"}]
    assert any("Dropped 2 invalid issue(s)" in entry.get("message", "") for entry in result["logs"])
