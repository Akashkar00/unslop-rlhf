"""Pairwise LLM judge with position swapping and a resumable on-disk cache.

Every comparison is run twice with the responses in both orders. LLM judges
have a measurable first-position bias; a verdict that flips when you swap the
order is not a preference, it's noise, and we treat it that way.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from tqdm import tqdm

from .io import append_jsonl, read_jsonl

# Bump when the rubric changes so cached verdicts from an old rubric aren't reused.
JUDGE_VERSION = "v1"

SYSTEM_PROMPT = """You are a demanding editor at a magazine known for clear, human prose. \
You compare two responses to the same request and decide which is better *writing*.

What good looks like:
- It does what the request asks, in the form it asks for.
- It sounds like a specific, thoughtful person wrote it, not a template.
- It prefers concrete detail, examples and plain words to generic claims and abstractions.
- Sentence length and rhythm vary naturally.
- No filler, throat-clearing, needless hedging, canned enthusiasm or summary paragraphs that repeat what was said.
- Its length fits the request.

Do not prefer a response because it is longer, uses more formatting, or covers more points. \
Length that adds no substance is a flaw. If the two are genuinely equal in quality, say tie."""

USER_TEMPLATE = """## Request
{prompt}

## Response A
{a}

## Response B
{b}

Which response is better writing for this request? Reply with JSON only:
{{"reason": "<one or two sentences>", "winner": "A" | "B" | "tie"}}"""

Winner = Literal["A", "B", "tie"]
CompleteFn = Callable[[str, list[dict]], str]

_WINNERS = {"a": "A", "b": "B", "tie": "tie"}
_WINNER_RE = re.compile(r'"winner"\s*:\s*"(A|B|tie)"', re.IGNORECASE)


@dataclass(frozen=True)
class Verdict:
    winner: Winner
    reason: str


@dataclass(frozen=True)
class Comparison:
    # From x's point of view: 1 = x better, 0 = y better. The two orderings are
    # averaged, so a full position flip scores 0.5.
    score_x: float
    # Both orderings returned the same verdict.
    consistent: bool
    reasons: tuple[str, str]


def parse_verdict(text: str) -> Verdict | None:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            winner = _WINNERS.get(str(obj.get("winner", "")).strip().lower())
            if winner:
                return Verdict(winner, str(obj.get("reason", "")).strip())
        except json.JSONDecodeError:
            pass
    match = _WINNER_RE.search(text)
    if match:
        return Verdict(_WINNERS[match.group(1).lower()], "")
    return None


def litellm_complete(model: str, messages: list[dict]) -> str:
    import litellm

    resp = litellm.completion(model=model, messages=messages, temperature=0.0, max_tokens=4096, num_retries=4)
    return resp.choices[0].message.content or ""


class _RateLimiter:
    def __init__(self, max_rpm: int | None):
        self._interval = 60.0 / max_rpm if max_rpm else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        if not self._interval:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self._interval
        time.sleep(max(0.0, slot - now))


class PairwiseJudge:
    def __init__(
        self,
        model: str,
        cache_path: Path | None = None,
        complete_fn: CompleteFn | None = None,
        max_rpm: int | None = None,
        parse_retries: int = 2,
    ):
        self.model = model
        self._complete = complete_fn or litellm_complete
        self._limiter = _RateLimiter(max_rpm)
        self._parse_retries = parse_retries
        self._cache_path = cache_path
        self._lock = threading.Lock()
        self._cache = {r["key"]: r for r in read_jsonl(cache_path, missing_ok=True)} if cache_path else {}

    def _key(self, prompt: str, a: str, b: str) -> str:
        blob = json.dumps([JUDGE_VERSION, self.model, prompt, a, b], ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def judge_once(self, prompt: str, a: str, b: str) -> Verdict | None:
        key = self._key(prompt, a, b)
        if (hit := self._cache.get(key)) is not None:
            return Verdict(hit["winner"], hit["reason"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(prompt=prompt, a=a, b=b)},
        ]
        verdict = None
        for _ in range(1 + self._parse_retries):
            self._limiter.wait()
            verdict = parse_verdict(self._complete(self.model, messages))
            if verdict:
                break
        if verdict is None:
            return None
        row = {"key": key, "model": self.model, **asdict(verdict)}
        with self._lock:
            self._cache[key] = row
            if self._cache_path:
                append_jsonl(self._cache_path, row)
        return verdict

    def compare(self, prompt: str, x: str, y: str) -> Comparison | None:
        first = self.judge_once(prompt, x, y)
        if first is None:
            return None
        second = self.judge_once(prompt, y, x)
        if second is None:
            return None
        s1 = {"A": 1.0, "B": 0.0, "tie": 0.5}[first.winner]
        s2 = {"A": 0.0, "B": 1.0, "tie": 0.5}[second.winner]
        return Comparison(score_x=(s1 + s2) / 2, consistent=s1 == s2, reasons=(first.reason, second.reason))

    def _compare_or_none(self, prompt: str, x: str, y: str) -> Comparison | None:
        try:
            return self.compare(prompt, x, y)
        except Exception as e:  # API error after LiteLLM's own retries
            with self._lock:
                self.errors.append(f"{type(e).__name__}: {str(e)[:200]}")
            return None

    def compare_many(self, items: Sequence[tuple[str, str, str]], workers: int = 4) -> list[Comparison | None]:
        """``items`` are (prompt, x, y). Order of results matches ``items``. A failed API
        call yields None for that item rather than aborting the batch; finished verdicts
        are cached, so re-running fills the gaps."""
        self.errors: list[str] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self._compare_or_none, *item) for item in items]
            results = [f.result() for f in tqdm(futures, desc=f"judge[{self.model}]")]
        if self.errors:
            print(f"warning: {len(self.errors)} judge calls failed for {self.model}; first: {self.errors[0]}\n"
                  "re-run the same command to retry them (verdicts so far are cached).")
        return results
