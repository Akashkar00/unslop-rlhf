"""Settings and file layout. Every setting can be overridden with an UNSLOP_* env var."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def prompts_train(self) -> Path:
        return self.root / "data/prompts/train.jsonl"

    @property
    def prompts_eval(self) -> Path:
        return self.root / "data/prompts/eval.jsonl"

    @property
    def candidates(self) -> Path:
        return self.root / "data/gen/train_candidates.jsonl"

    @property
    def judged_pairs(self) -> Path:
        return self.root / "data/pairs/judged_train.jsonl"

    @property
    def dpo_pairs(self) -> Path:
        return self.root / "data/pairs/dpo_train.jsonl"

    @property
    def pair_stats(self) -> Path:
        return self.root / "data/pairs/pair_stats.json"

    @property
    def eval_outputs(self) -> Path:
        return self.root / "data/gen/eval_outputs.jsonl"

    def eval_judgments(self, role: str) -> Path:
        return self.root / f"data/eval/judgments_{role}.jsonl"

    @property
    def eval_judge_stats(self) -> Path:
        return self.root / "data/eval/judge_stats.json"

    @property
    def human_queue_train(self) -> Path:
        return self.root / "data/human/queue_train.jsonl"

    @property
    def human_queue_eval(self) -> Path:
        return self.root / "data/human/queue_eval.jsonl"

    @property
    def human_labels(self) -> Path:
        return self.root / "data/human/labels.jsonl"

    @property
    def judge_cache(self) -> Path:
        return self.root / "data/cache/judge_cache.jsonl"

    @property
    def adapter(self) -> Path:
        return self.root / "outputs/dpo-adapter"

    @property
    def training_log(self) -> Path:
        return self.adapter / "training_log.json"

    @property
    def report_md(self) -> Path:
        return self.root / "reports/results.md"

    @property
    def report_json(self) -> Path:
        return self.root / "reports/results.json"


@dataclass(frozen=True)
class Settings:
    base_model: str
    # Two judges from different model families. The train judge labels DPO pairs;
    # the eval judge scores the final model. Scoring with the judge you trained
    # against measures how well you fit that judge, not whether writing improved.
    train_judge: str
    eval_judge: str
    root: Path
    seed: int

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        return cls(
            base_model=os.getenv("UNSLOP_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
            train_judge=os.getenv("UNSLOP_TRAIN_JUDGE", "gemini/gemini-3.8-flash"),
            eval_judge=os.getenv("UNSLOP_EVAL_JUDGE", "groq/openai/gpt-oss-120b"),
            root=Path(os.getenv("UNSLOP_ROOT", ".")),
            seed=int(os.getenv("UNSLOP_SEED", "17")),
        )

    @property
    def paths(self) -> Paths:
        return Paths(self.root)

    def judge_model(self, role: str) -> str:
        return {"train": self.train_judge, "eval": self.eval_judge}[role]
