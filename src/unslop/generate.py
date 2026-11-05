"""Sampling from the base / tuned model. Needs the ``train`` extra and a GPU."""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm


def load_model(base_model: str, adapter: Path | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, device_map="auto")
    else:
        # device_map="auto" on MPS/CPU can offload weights to disk; for local smoke runs just place the model.
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32).to(device)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model, tok


def _eos_ids(model, tok) -> set[int]:
    ids = {tok.eos_token_id}
    gen_eos = model.generation_config.eos_token_id
    ids.update(gen_eos if isinstance(gen_eos, list) else [gen_eos])
    return {i for i in ids if i is not None}


def sample(
    model,
    tok,
    prompts: list[str],
    *,
    n: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    batch_size: int,
    seed: int,
) -> list[list[dict]]:
    """Returns, per prompt, ``n`` dicts of {text, finished}. ``finished`` is False
    when generation hit ``max_new_tokens`` before emitting EOS."""
    import torch

    eos = _eos_ids(model, tok)
    out: list[list[dict]] = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="generate"):
        batch = prompts[start : start + batch_size]
        texts = [
            tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in batch
        ]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        torch.manual_seed(seed + start)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                num_return_sequences=n,
                pad_token_id=tok.pad_token_id,
            )
        new_tokens = gen[:, enc["input_ids"].shape[1] :]
        # generate() expands inputs with repeat_interleave, so rows come grouped per prompt.
        for b in range(len(batch)):
            group = []
            for s in range(n):
                ids = new_tokens[b * n + s].tolist()
                group.append({
                    "text": tok.decode(ids, skip_special_tokens=True).strip(),
                    "finished": any(t in eos for t in ids),
                })
            out.append(group)
    return out


def generate_candidates(base_model: str, prompt_rows: list[dict], *, k: int, temperature: float, top_p: float,
                        max_new_tokens: int, batch_size: int, seed: int) -> list[dict]:
    model, tok = load_model(base_model)
    groups = sample(model, tok, [r["prompt"] for r in prompt_rows], n=k, temperature=temperature, top_p=top_p,
                    max_new_tokens=max_new_tokens, batch_size=batch_size, seed=seed)
    return [{**row, "candidates": group} for row, group in zip(prompt_rows, groups, strict=True)]


def generate_eval_outputs(base_model: str, adapter: Path, prompt_rows: list[dict], *, temperature: float,
                          top_p: float, max_new_tokens: int, batch_size: int, seed: int) -> list[dict]:
    """One sample per prompt from the tuned model and from the base model
    (same weights with the adapter disabled), under identical seeds."""
    model, tok = load_model(base_model, adapter)
    prompts = [r["prompt"] for r in prompt_rows]
    kwargs = dict(n=1, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens,
                  batch_size=batch_size, seed=seed)
    tuned = sample(model, tok, prompts, **kwargs)
    with model.disable_adapter():
        base = sample(model, tok, prompts, **kwargs)
    return [
        {
            **row,
            "tuned": t[0]["text"],
            "tuned_finished": t[0]["finished"],
            "base": b[0]["text"],
            "base_finished": b[0]["finished"],
        }
        for row, t, b in zip(prompt_rows, tuned, base, strict=True)
    ]
