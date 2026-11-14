import json

import pytest

import run_scale_pod as pod
from summarize_scale_probe import compare_generation


def test_proposed_job_stays_below_cap_with_cleanup_reserve():
    p = pod.proposal()
    assert p["maximum_planned_usage_usd"] < p["budget_cap_usd"] - 1.5
    assert p["maximum_minutes_from_create"] == 90
    assert p["temporary_disk_gb"] == 200


def test_budget_switch_is_checked_before_api(monkeypatch):
    monkeypatch.setattr(pod, "fresh_quote", lambda: pytest.fail("API accessed before checking cap"))
    with pytest.raises(ValueError):
        pod.execute(None, None)
    with pytest.raises(ValueError):
        pod.execute(None, 100)


def test_termination_only_touches_exact_new_pod(monkeypatch):
    calls = []
    manifest = {"name": "mc-scale-unique", "pod_id": "new"}
    def api(method, path, data=None):
        calls.append((method, path))
        return {"name": "unrelated-job", "id": "new"}
    monkeypatch.setattr(pod, "api", api)
    with pytest.raises(RuntimeError, match="wrong name"):
        pod.terminate(manifest)
    assert calls == [("GET", "/pods/new")]


def test_ambiguous_creation_is_resolved_without_second_create(monkeypatch):
    calls = []
    def api(method, path, data=None):
        calls.append((method, path))
        if method == "GET":
            return [{"id": "old", "name": "other"}, {"id": "new", "name": "mc-scale-unique"}]
    monkeypatch.setattr(pod, "api", api)
    pod.terminate({"name": "mc-scale-unique"})
    assert calls == [("GET", "/pods"), ("DELETE", "/pods/new")]


def test_newly_correct_answers_do_not_hide_lost_stock_correct_answers():
    def case(name, passed):
        return {"id": name, "check_pass": passed, "loop_flag": False, "ended": True}
    before = {"cases": [case("a", True), case("b", False)]}
    after = {"cases": [case("a", False), case("b", True)]}
    result = compare_generation(before, after)
    assert result["candidate_correct"] == result["stock_correct"] == 1
    assert result["retained_stock_correct_fraction"] == 0
    assert result["lost_cases"] == ["a"] and result["gained_cases"] == ["b"]
    assert not result["screen_pass"]
