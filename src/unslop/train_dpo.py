"""DPO with a LoRA adapter via TRL. Needs the ``train`` extra and a CUDA GPU.

With ``ref_model=None`` and a ``peft_config``, TRL uses the base weights with
the adapter switched off as the reference policy, so only one copy of the
model sits in memory. That's what makes a 1.5B model fit on a free T4.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from .io import read_jsonl, write_json

MIN_PAIRS = 8


@dataclass(frozen=True)
class DPOHyperparams:
    beta: float = 0.1
    learning_rate: float = 5e-5
    epochs: int = 2
    lora_r: int = 16
    lora_alpha: int = 32
    batch_size: int = 2
    grad_accum: int = 8
    max_length: int = 1024
    max_prompt_length: int = 384
    holdout_frac: float = 0.05


def train(base_model: str, pairs_path: Path, out_dir: Path, hp: DPOHyperparams, seed: int) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    rows = [{k: r[k] for k in ("prompt", "chosen", "rejected")} for r in read_jsonl(pairs_path)]
    if len(rows) < MIN_PAIRS:
        raise ValueError(
            f"Only {len(rows)} preference pairs survived filtering and length balancing; need at least {MIN_PAIRS}. "
            "Raise --pairs-per-prompt in build-pairs or use more prompts."
        )
    if len(rows) < 200:
        print(f"warning: {len(rows)} pairs is enough to test the plumbing, not to move the model meaningfully.")
    random.Random(seed).shuffle(rows)
    # Never let the holdout eat more than a quarter of a small dataset.
    n_hold = min(max(2, int(len(rows) * hp.holdout_frac)), len(rows) // 4)
    train_ds, eval_ds = Dataset.from_list(rows[n_hold:]), Dataset.from_list(rows[:n_hold])

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16
    dtype = torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    model.config.use_cache = False

    args = DPOConfig(
        output_dir=str(out_dir),
        beta=hp.beta,
        learning_rate=hp.learning_rate,
        num_train_epochs=hp.epochs,
        per_device_train_batch_size=hp.batch_size,
        per_device_eval_batch_size=hp.batch_size,
        gradient_accumulation_steps=hp.grad_accum,
        max_length=hp.max_length,
        max_prompt_length=hp.max_prompt_length,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        seed=seed,
    )
    peft_config = LoraConfig(
        r=hp.lora_r, lora_alpha=hp.lora_alpha, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    log = {
        "base_model": base_model,
        "hyperparams": hp.__dict__,
        "n_train_pairs": len(train_ds),
        "n_holdout_pairs": len(eval_ds),
        "log_history": trainer.state.log_history,
    }
    write_json(out_dir / "training_log.json", log)
    return log
