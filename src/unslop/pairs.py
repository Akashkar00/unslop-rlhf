"""Turn sampled candidates into DPO preference pairs.

Two reward-hacking guards live here:
1. Candidates are only paired when their lengths are within ``max_len_ratio``,
   so the judge has to decide on quality rather than "more".
2. After judging, pairs are rebalanced so the chosen response is the longer one
   exactly half the time. Otherwise "longer = better" is a free lesson for DPO
   whenever the judge leans long.
"""

from __future__ import annotations

import hashlib
import random
from itertools import combinations

from .judge import PairwiseJudge
from .metrics import word_count

MIN_WORDS = 5


def usable(candidate: dict) -> bool:
    # Truncated generations get cut at max_new_tokens, so keeping them would let
    # the length of the cut, not the writing, decide the label.
    return candidate["finished"] and word_count(candidate["text"]) >= MIN_WORDS


def select_pairs(lengths: list[int], rng: random.Random, k: int, max_len_ratio: float) -> list[tuple[int, int]]:
    eligible = [
        (i, j)
        for i, j in combinations(range(len(lengths)), 2)
        if max(lengths[i], lengths[j]) / max(min(lengths[i], lengths[j]), 1) <= max_len_ratio
    ]
    rng.shuffle(eligible)
    return eligible[:k]


def _pair_id(prompt_id: str, x: str, y: str) -> str:
    digest = hashlib.sha256("\x00".join((x, y)).encode()).hexdigest()[:12]
    return f"{prompt_id}:{digest}"


def judge_candidates(
    rows: list[dict],
    judge: PairwiseJudge,
    rng: random.Random,
    pairs_per_prompt: int,
    max_len_ratio: float,
    workers: int = 4,
) -> tuple[list[dict], int]:
    """Returns (judged pairs, number of pairs attempted). Pairs whose judge output
    couldn't be parsed are dropped, and the attempted count keeps that visible."""
    jobs = []
    for row in rows:
        texts = [c["text"] for c in row["candidates"] if usable(c)]
        for i, j in select_pairs([word_count(t) for t in texts], rng, pairs_per_prompt, max_len_ratio):
            jobs.append((row, texts[i], texts[j]))
    results = judge.compare_many([(r["prompt"], x, y) for r, x, y in jobs], workers=workers)
    judged = []
    for (row, x, y), cmp in zip(jobs, results, strict=True):
        if cmp is None:
            continue
        judged.append({
            "pair_id": _pair_id(row["prompt_id"], x, y),
            "prompt_id": row["prompt_id"],
            "prompt": row["prompt"],
            "x": x,
            "y": y,
            "score_x": cmp.score_x,
            "consistent": cmp.consistent,
            "reasons": list(cmp.reasons),
        })
    return judged, len(jobs)


def to_dpo_rows(judged: list[dict]) -> list[dict]:
    """Keep only pairs where both orderings named the same strict winner."""
    rows = []
    for j in judged:
        if not j["consistent"] or j["score_x"] == 0.5:
            continue
        chosen, rejected = (j["x"], j["y"]) if j["score_x"] == 1.0 else (j["y"], j["x"])
        rows.append({
            "pair_id": j["pair_id"],
            "prompt_id": j["prompt_id"],
            "prompt": [{"role": "user", "content": j["prompt"]}],
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
            "chosen_words": word_count(chosen),
            "rejected_words": word_count(rejected),
        })
    return rows


def balance_by_length(rows: list[dict], rng: random.Random) -> list[dict]:
    longer = [r for r in rows if r["chosen_words"] > r["rejected_words"]]
    shorter = [r for r in rows if r["chosen_words"] < r["rejected_words"]]
    equal = [r for r in rows if r["chosen_words"] == r["rejected_words"]]
    rng.shuffle(longer)
    rng.shuffle(shorter)
    n = min(len(longer), len(shorter))
    out = longer[:n] + shorter[:n] + equal
    rng.shuffle(out)
    return out


def chosen_longer_share(rows: list[dict]) -> float:
    unequal = [r for r in rows if r["chosen_words"] != r["rejected_words"]]
    return sum(r["chosen_words"] > r["rejected_words"] for r in unequal) / len(unequal) if unequal else float("nan")


def pair_stats(candidate_rows: list[dict], attempted: int, judged: list[dict], strict: list[dict],
               final: list[dict]) -> dict:
    n_cands = sum(len(r["candidates"]) for r in candidate_rows)
    return {
        "prompts": len(candidate_rows),
        "candidates": n_cands,
        "candidates_truncated_share": (
            sum(not c["finished"] for r in candidate_rows for c in r["candidates"]) / n_cands if n_cands else 0.0
        ),
        "pairs_attempted": attempted,
        "pairs_judged": len(judged),
        "position_consistent_share": sum(j["consistent"] for j in judged) / len(judged) if judged else 0.0,
        "strict_consistent_pairs": len(strict),
        "judge_chose_longer_share": chosen_longer_share(strict),
        "pairs_after_length_balance": len(final),
        "final_chose_longer_share": chosen_longer_share(final),
    }
