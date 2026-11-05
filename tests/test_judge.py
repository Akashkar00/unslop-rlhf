import pytest

from unslop.judge import PairwiseJudge, parse_verdict


@pytest.mark.parametrize(
    ("text", "winner"),
    [
        ('{"reason": "tighter", "winner": "A"}', "A"),
        ('Sure!\n```json\n{"reason": "x", "winner": "b"}\n```', "B"),
        ('{"winner": "Tie", "reason": "same"}', "tie"),
        ('broken {"winner": "A", reason: oops}', "A"),
    ],
)
def test_parse_verdict(text, winner):
    assert parse_verdict(text).winner == winner


@pytest.mark.parametrize("text", ["", "I prefer the first one.", '{"winner": "C"}'])
def test_parse_verdict_rejects_garbage(text):
    assert parse_verdict(text) is None


def _judge_from(rule, **kw):
    calls = []

    def complete(model, messages):
        user = messages[1]["content"]
        a = user.split("## Response A\n")[1].split("\n\n## Response B")[0]
        b = user.split("## Response B\n")[1].split("\n\nWhich response")[0]
        calls.append((a, b))
        return f'{{"reason": "", "winner": "{rule(a, b)}"}}'

    return PairwiseJudge("fake", complete_fn=complete, **kw), calls


def test_compare_runs_both_orders_and_agrees():
    judge, calls = _judge_from(lambda a, b: "A" if a == "good" else "B")
    cmp = judge.compare("p", "good", "bad")
    assert calls == [("good", "bad"), ("bad", "good")]
    assert cmp.score_x == 1.0 and cmp.consistent


def test_position_biased_judge_scores_a_tie():
    judge, _ = _judge_from(lambda a, b: "A")  # always picks whatever is shown first
    cmp = judge.compare("p", "x", "y")
    assert cmp.score_x == 0.5 and not cmp.consistent


def test_half_tie_is_inconsistent_partial_win():
    judge, _ = _judge_from(lambda a, b: "A" if a == "x" else "tie")
    cmp = judge.compare("p", "x", "y")
    assert cmp.score_x == 0.75 and not cmp.consistent


def test_cache_persists_and_skips_calls(tmp_path):
    cache = tmp_path / "cache.jsonl"
    judge, calls = _judge_from(lambda a, b: "A", cache_path=cache)
    judge.compare("p", "x", "y")
    assert len(calls) == 2

    judge2, calls2 = _judge_from(lambda a, b: "B", cache_path=cache)
    cmp = judge2.compare("p", "x", "y")
    assert calls2 == []  # served from disk, including the old verdicts
    assert cmp.score_x == 0.5


def test_unparseable_output_returns_none_after_retries():
    n = {"calls": 0}

    def complete(model, messages):
        n["calls"] += 1
        return "no json here"

    judge = PairwiseJudge("fake", complete_fn=complete, parse_retries=2)
    assert judge.compare("p", "x", "y") is None
    assert n["calls"] == 3


def test_compare_many_turns_api_errors_into_none():
    def complete(model, messages):
        if "boom" in messages[1]["content"]:
            raise RuntimeError("429 rate limited")
        return '{"winner": "A"}'

    judge = PairwiseJudge("fake", complete_fn=complete)
    results = judge.compare_many([("p", "x", "y"), ("p", "boom", "y")], workers=2)
    assert results[0] is not None and results[1] is None
    assert len(judge.errors) == 1 and "429" in judge.errors[0]


class _FakeMessages:
    def __init__(self, stop_reason="end_turn"):
        self.calls, self.stop_reason = [], stop_reason

    def create(self, **kwargs):
        from types import SimpleNamespace as NS

        self.calls.append(kwargs)
        return NS(stop_reason=self.stop_reason, stop_details=NS(category="cyber"),
                  content=[NS(type="thinking", thinking=""), NS(type="text", text='{"winner": "B"}')])


def test_anthropic_route_sends_no_sampling_params(monkeypatch):
    from types import SimpleNamespace as NS

    from unslop import judge as judge_mod

    fake = _FakeMessages()
    monkeypatch.setattr(judge_mod, "_anthropic_client", lambda: NS(messages=fake))
    j = PairwiseJudge("anthropic/claude-sonnet-5-5")
    assert j.judge_once("p", "a", "b").winner == "B"

    (call,) = fake.calls
    assert call["model"] == "claude-sonnet-5-5"
    assert "temperature" not in call and "top_p" not in call
    assert call["output_config"] == {"effort": "low"}
    assert call["system"].startswith("You are a demanding editor")
    assert [m["role"] for m in call["messages"]] == ["user"]


def test_anthropic_refusal_is_a_dropped_verdict(monkeypatch):
    from types import SimpleNamespace as NS

    from unslop import judge as judge_mod

    monkeypatch.setattr(judge_mod, "_anthropic_client", lambda: NS(messages=_FakeMessages("refusal")))
    j = PairwiseJudge("anthropic/claude-sonnet-5-5")
    assert j.compare_many([("p", "x", "y")], workers=1) == [None]
    assert "refused" in j.errors[0]
