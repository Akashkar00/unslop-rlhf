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
