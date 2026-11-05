import random

from unslop.judge import PairwiseJudge
from unslop.pairs import (
    balance_by_length,
    chosen_longer_share,
    judge_candidates,
    pair_stats,
    select_pairs,
    to_dpo_rows,
)


def test_select_pairs_respects_length_ratio():
    pairs = select_pairs([100, 110, 300, 140], random.Random(0), k=10, max_len_ratio=1.5)
    assert set(pairs) == {(0, 1), (0, 3), (1, 3)}


def _row(chosen_words, rejected_words):
    return {"chosen_words": chosen_words, "rejected_words": rejected_words}


def test_balance_by_length_is_fifty_fifty():
    rows = [_row(120, 100)] * 30 + [_row(90, 100)] * 10 + [_row(100, 100)] * 3
    out = balance_by_length(rows, random.Random(0))
    assert len(out) == 23
    assert chosen_longer_share(out) == 0.5


def _cands(*texts, finished=True):
    return [{"text": t, "finished": finished} for t in texts]


def test_pipeline_drops_truncated_and_inconsistent():
    def complete(model, messages):
        user = messages[1]["content"]
        a = user.split("## Response A\n")[1].split("\n\n## Response B")[0]
        # "good" wins when present; otherwise the judge is position-biased.
        if "good" in user:
            return '{"winner": "%s"}' % ("A" if "good" in a else "B")
        return '{"winner": "A"}'

    rows = [
        {"prompt_id": "p1", "prompt": "write", "candidates": _cands(
            "a good answer with words", "a plain answer with words")},
        {"prompt_id": "p2", "prompt": "write", "candidates": _cands(
            "one two three four five", "six seven eight nine ten")},
        {"prompt_id": "p3", "prompt": "write", "candidates": _cands(
            "cut off mid sentence and", "also cut off mid sentence", finished=False)},
    ]
    judge = PairwiseJudge("fake", complete_fn=complete)
    judged, attempted = judge_candidates(rows, judge, random.Random(0), pairs_per_prompt=1, max_len_ratio=1.5,
                                         workers=1)
    assert attempted == 2
    assert {j["prompt_id"] for j in judged} == {"p1", "p2"}  # p3 truncated

    dpo = to_dpo_rows(judged)
    assert len(dpo) == 1  # p2 flipped with position
    assert dpo[0]["chosen"][0]["content"] == "a good answer with words"
    assert dpo[0]["prompt"] == [{"role": "user", "content": "write"}]

    stats = pair_stats(rows, attempted, judged, dpo, dpo)
    assert stats["pairs_attempted"] == stats["pairs_judged"] == 2
    assert stats["position_consistent_share"] == 0.5
    assert stats["candidates_truncated_share"] == 1 / 3
