"""Writing prompts from HuggingFaceH4/no_robots (human-written, CC BY-NC 4.0).

Train prompts come from the dataset's train split and eval prompts from its test
split, with exact-duplicate removal across the two, so the eval set never
appears in training.
"""

from __future__ import annotations

import random
import re

from .config import Paths
from .io import write_jsonl

DATASET = "HuggingFaceH4/no_robots"
# Categories where the quality of the prose is the point. Chat is excluded because
# its prompts lean on a system message; Closed QA/Extract/Classify are mostly
# about correctness, not voice.
WRITING_CATEGORIES = ("Generation", "Rewrite", "Brainstorm")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def filter_rows(rows, categories=WRITING_CATEGORIES, max_chars: int = 2000) -> list[dict]:
    seen: set[str] = set()
    out = []
    for r in rows:
        if r["category"] not in categories:
            continue
        prompt = r["prompt"].strip()
        if not prompt or len(prompt) > max_chars:
            continue
        key = normalize(prompt)
        if key in seen:
            continue
        seen.add(key)
        out.append({"prompt_id": r["prompt_id"], "prompt": prompt, "category": r["category"]})
    return out


def split_prompts(train_rows: list[dict], eval_rows: list[dict], n_train: int, n_eval: int, seed: int):
    eval_keys = {normalize(r["prompt"]) for r in eval_rows}
    train_rows = [r for r in train_rows if normalize(r["prompt"]) not in eval_keys]
    rng = random.Random(seed)
    train_rows, eval_rows = train_rows[:], eval_rows[:]
    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)
    return train_rows[:n_train], eval_rows[:n_eval]


def build_prompt_splits(paths: Paths, n_train: int, n_eval: int, seed: int) -> tuple[int, int]:
    from datasets import load_dataset

    train = filter_rows(load_dataset(DATASET, split="train"))
    evals = filter_rows(load_dataset(DATASET, split="test"))
    train, evals = split_prompts(train, evals, n_train, n_eval, seed)
    write_jsonl(paths.prompts_train, train)
    write_jsonl(paths.prompts_eval, evals)
    return len(train), len(evals)
