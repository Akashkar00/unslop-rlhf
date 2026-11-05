"""Cheap, judge-independent style metrics.

These are deliberately *not* mentioned in the judge rubric. If the judge were told
"penalise the word 'delve'", the model could learn to dodge a word list while
staying just as robotic, and this metric would stop meaning anything. Keeping
it out of the reward keeps it usable as an audit.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

# Stock phrases that over-represent in post-2023 assistant prose. A heuristic,
# not ground truth: "foster" also matches "foster care".
SLOP_PHRASES = (
    "a testament to", "additionally", "at the end of the day", "beacon of", "boasts",
    "bustling", "delve", "elevate your", "embark on", "ever-changing", "ever-evolving",
    "foster", "furthermore", "game-changer", "in conclusion", "in summary",
    "in today's fast-paced", "intricate", "it is important to note", "it's important to note",
    "landscape of", "meticulous", "moreover", "navigate the complexities", "nestled",
    "pivotal", "plays a crucial role", "realm of", "seamless", "showcasing",
    "symphony of", "tapestry", "unleash", "unlock the", "vibrant", "whispers of",
)
_SLOP_RE = re.compile(r"\b(?:" + "|".join(re.escape(p) for p in SLOP_PHRASES) + r")\b", re.IGNORECASE)
_NOT_JUST_RE = re.compile(r"\bnot (?:just|only|merely)\b[^.?!\n]{1,80}?\bbut\b", re.IGNORECASE)
_LIST_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)]|#{1,6})\s", re.MULTILINE)


def word_count(text: str) -> int:
    return len(text.split())


def slop_hits(text: str) -> int:
    return len(_SLOP_RE.findall(text)) + len(_NOT_JUST_RE.findall(text))


def distinct_n(texts: Sequence[str], n: int = 2) -> float:
    total, unique = 0, set()
    for t in texts:
        toks = t.lower().split()
        grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
        total += len(grams)
        unique.update(grams)
    return len(unique) / total if total else 0.0


def style_summary(texts: Sequence[str]) -> dict:
    words = sum(word_count(t) for t in texts)
    per_k = 1000 / words if words else 0.0
    return {
        "n": len(texts),
        "mean_words": words / len(texts) if texts else 0.0,
        "slop_per_1k_words": sum(slop_hits(t) for t in texts) * per_k,
        "em_dash_per_1k_words": sum(t.count("—") for t in texts) * per_k,
        "list_or_header_share": sum(bool(_LIST_RE.search(t)) for t in texts) / len(texts) if texts else 0.0,
        "distinct_2": distinct_n(texts, 2),
    }
