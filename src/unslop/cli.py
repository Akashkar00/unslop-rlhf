"""unslop pipeline CLI. GPU steps: generate, train, generate-eval. The rest run anywhere."""

from __future__ import annotations

import json
import random
from typing import Optional

import typer

from .config import Settings
from .io import read_jsonl, write_json, write_jsonl

app = typer.Typer(help="DPO for LLMs that write well, with a reward-hacking audit.", no_args_is_help=True)


def _settings() -> Settings:
    return Settings.from_env()


def _limit(rows: list[dict], limit: Optional[int]) -> list[dict]:
    return rows[:limit] if limit else rows


@app.command()
def prompts(n_train: int = 400, n_eval: int = 150) -> None:
    """Download no_robots and write the train/eval prompt splits."""
    from .prompts import build_prompt_splits

    s = _settings()
    n_tr, n_ev = build_prompt_splits(s.paths, n_train, n_eval, s.seed)
    typer.echo(f"wrote {n_tr} train / {n_ev} eval prompts")


@app.command()
def generate(
    k: int = 4,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_new_tokens: int = 512,
    batch_size: int = 8,
    limit: Optional[int] = None,
) -> None:
    """[GPU] Sample k candidates per train prompt from the base model."""
    from .generate import generate_candidates

    s = _settings()
    rows = _limit(read_jsonl(s.paths.prompts_train), limit)
    out = generate_candidates(s.base_model, rows, k=k, temperature=temperature, top_p=top_p,
                              max_new_tokens=max_new_tokens, batch_size=batch_size, seed=s.seed)
    write_jsonl(s.paths.candidates, out)
    typer.echo(f"wrote {len(out)} prompts x {k} candidates -> {s.paths.candidates}")


@app.command("build-pairs")
def build_pairs(
    pairs_per_prompt: int = 2,
    max_len_ratio: float = 1.5,
    balance: bool = True,
    human_sample: int = 150,
    workers: int = 4,
    max_rpm: Optional[int] = None,
    limit: Optional[int] = None,
) -> None:
    """Judge candidate pairs with the train judge and write DPO pairs + a human-agreement queue."""
    from .judge import PairwiseJudge
    from .pairs import balance_by_length, judge_candidates, pair_stats, to_dpo_rows

    s = _settings()
    rng = random.Random(s.seed)
    rows = _limit(read_jsonl(s.paths.candidates), limit)
    judge = PairwiseJudge(s.train_judge, cache_path=s.paths.judge_cache, max_rpm=max_rpm)
    judged, attempted = judge_candidates(rows, judge, rng, pairs_per_prompt, max_len_ratio, workers=workers)
    strict = to_dpo_rows(judged)
    final = balance_by_length(strict, rng) if balance else strict

    write_jsonl(s.paths.judged_pairs, judged)
    write_jsonl(s.paths.dpo_pairs, final)
    stats = pair_stats(rows, attempted, judged, strict, final)
    write_json(s.paths.pair_stats, stats)

    # Sample from *all* judged pairs, not just the consistent ones, so measured
    # judge-human agreement isn't inflated by only showing humans the easy cases.
    sample = rng.sample(judged, min(human_sample, len(judged)))
    write_jsonl(s.paths.human_queue_train, (
        {"item_id": f"train:{j['pair_id']}", "kind": "train", "ref": j["pair_id"],
         "prompt": j["prompt"], "x": j["x"], "y": j["y"]}
        for j in sample
    ))
    for key, value in stats.items():
        typer.echo(f"{key:>28}: {value:.3f}" if isinstance(value, float) else f"{key:>28}: {value}")


@app.command()
def train(
    beta: float = 0.1,
    lr: float = 5e-5,
    epochs: int = 2,
    lora_r: int = 16,
    batch_size: int = 2,
    grad_accum: int = 8,
    max_length: int = 1024,
) -> None:
    """[GPU] DPO + LoRA on the balanced preference pairs."""
    from .train_dpo import DPOHyperparams
    from .train_dpo import train as run_train

    s = _settings()
    hp = DPOHyperparams(beta=beta, learning_rate=lr, epochs=epochs, lora_r=lora_r, lora_alpha=2 * lora_r,
                        batch_size=batch_size, grad_accum=grad_accum, max_length=max_length)
    log = run_train(s.base_model, s.paths.dpo_pairs, s.paths.adapter, hp, s.seed)
    typer.echo(f"trained on {log['n_train_pairs']} pairs -> {s.paths.adapter}")


@app.command("generate-eval")
def generate_eval(
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 512,
    batch_size: int = 16,
    limit: Optional[int] = None,
) -> None:
    """[GPU] One sample per held-out prompt from the tuned and the base model."""
    from .generate import generate_eval_outputs

    s = _settings()
    rows = _limit(read_jsonl(s.paths.prompts_eval), limit)
    out = generate_eval_outputs(s.base_model, s.paths.adapter, rows, temperature=temperature, top_p=top_p,
                                max_new_tokens=max_new_tokens, batch_size=batch_size, seed=s.seed)
    write_jsonl(s.paths.eval_outputs, out)
    typer.echo(f"wrote {len(out)} eval generations -> {s.paths.eval_outputs}")


@app.command("judge-eval")
def judge_eval(
    roles: str = "eval,train",
    human_sample: int = 100,
    workers: int = 4,
    max_rpm: Optional[int] = None,
) -> None:
    """Judge tuned vs base on held-out prompts. 'eval' is the independent judge; 'train'
    is re-run only to measure the Goodhart gap."""
    from .judge import PairwiseJudge
    from .metrics import word_count

    s = _settings()
    outputs = read_jsonl(s.paths.eval_outputs)
    stats = json.loads(s.paths.eval_judge_stats.read_text()) if s.paths.eval_judge_stats.exists() else {}
    for role in [r.strip() for r in roles.split(",") if r.strip()]:
        judge = PairwiseJudge(s.judge_model(role), cache_path=s.paths.judge_cache, max_rpm=max_rpm)
        results = judge.compare_many([(o["prompt"], o["tuned"], o["base"]) for o in outputs], workers=workers)
        rows = [
            {"prompt_id": o["prompt_id"], "score_tuned": c.score_x, "consistent": c.consistent,
             "reasons": list(c.reasons), "tuned_words": word_count(o["tuned"]), "base_words": word_count(o["base"])}
            for o, c in zip(outputs, results, strict=True)
            if c is not None
        ]
        write_jsonl(s.paths.eval_judgments(role), rows)
        stats[role] = {"model": judge.model, "attempted": len(outputs), "judged": len(rows)}
        write_json(s.paths.eval_judge_stats, stats)
        typer.echo(f"[{role}] judged {len(rows)}/{len(outputs)} -> {s.paths.eval_judgments(role)}")

    sample = random.Random(s.seed).sample(outputs, min(human_sample, len(outputs)))
    write_jsonl(s.paths.human_queue_eval, (
        {"item_id": f"eval:{o['prompt_id']}", "kind": "eval", "ref": o["prompt_id"],
         "prompt": o["prompt"], "x": o["tuned"], "y": o["base"]}
        for o in sample
    ))


@app.command()
def report(n_boot: int = 2000) -> None:
    """Write reports/results.md and reports/results.json."""
    from .report import build_report

    s = _settings()
    results = build_report(s.paths, {"train": s.train_judge, "eval": s.eval_judge}, n_boot=n_boot, seed=s.seed)
    for line in results["verdict"]:
        typer.echo(f"- {line}")
    typer.echo(f"\nfull report -> {s.paths.report_md}")


if __name__ == "__main__":
    app()
