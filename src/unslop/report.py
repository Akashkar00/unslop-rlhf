"""Results report. Win rates come with bootstrap CIs, raw wins are always shown
next to length-controlled wins, and the verdict section is generated from the
numbers so the README can't claim more than the run supports."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .config import Paths
from .io import read_jsonl, write_json
from .metrics import slop_hits, style_summary, word_count
from .stats import CI, bootstrap_ci, cohen_kappa, length_controlled_win_rate, to_label

ROLES = ("eval", "train")


MAX_DROP_SHARE = 0.05


def judge_section(rows: list[dict], n_boot: int, seed: int, attempted: int | None = None) -> dict | None:
    if not rows:
        return None
    attempted = attempted if attempted is not None else len(rows)
    s = np.array([r["score_tuned"] for r in rows], dtype=float)
    lt = np.array([r["tuned_words"] for r in rows], dtype=float)
    lb = np.array([r["base_words"] for r in rows], dtype=float)
    raw = bootstrap_ci(len(s), lambda i: s[i].mean(), n_boot, seed=seed)
    lc = bootstrap_ci(len(s), lambda i: length_controlled_win_rate(s[i], lt[i], lb[i]), n_boot, seed=seed)
    return {
        "n": len(rows),
        "attempted": attempted,
        "win_rate": raw.as_dict(),
        "lc_win_rate": lc.as_dict(),
        "position_consistent_share": float(np.mean([r["consistent"] for r in rows])),
        "tuned_longer_share": float(np.mean(lt > lb)),
    }


def style_section(outputs: list[dict], n_boot: int, seed: int) -> dict | None:
    if not outputs:
        return None
    base = [o["base"] for o in outputs]
    tuned = [o["tuned"] for o in outputs]
    wb = np.array([word_count(t) for t in base], dtype=float)
    wt = np.array([word_count(t) for t in tuned], dtype=float)
    hb = np.array([slop_hits(t) for t in base], dtype=float)
    ht = np.array([slop_hits(t) for t in tuned], dtype=float)

    def slop_delta(i):
        return 1000 * (ht[i].sum() / max(wt[i].sum(), 1) - hb[i].sum() / max(wb[i].sum(), 1))

    return {
        "base": style_summary(base),
        "tuned": style_summary(tuned),
        "truncated_share": {
            "base": float(np.mean([not o["base_finished"] for o in outputs])),
            "tuned": float(np.mean([not o["tuned_finished"] for o in outputs])),
        },
        "mean_words_delta": bootstrap_ci(len(wt), lambda i: (wt[i] - wb[i]).mean(), n_boot, seed=seed).as_dict(),
        "slop_per_1k_delta": bootstrap_ci(len(wt), slop_delta, n_boot, seed=seed).as_dict(),
    }


def _agreement(judge_scores: Sequence[float], human_scores: Sequence[float]) -> dict:
    j = [to_label(s) for s in judge_scores]
    h = [to_label(s) for s in human_scores]
    decisive = [(a, b) for a, b in zip(j, h) if a != "tie" and b != "tie"]
    return {
        "n": len(j),
        "agreement": float(np.mean([a == b for a, b in zip(j, h)])) if j else float("nan"),
        "decisive_n": len(decisive),
        "decisive_agreement": float(np.mean([a == b for a, b in decisive])) if decisive else float("nan"),
        "cohen_kappa": cohen_kappa(j, h),
    }


def human_section(labels: list[dict], judged_train: list[dict], eval_judgments: list[dict],
                  n_boot: int, seed: int) -> dict:
    by_id = {lab["item_id"]: lab for lab in labels}  # a re-label overrides the earlier one
    out: dict = {}

    train = [(j, by_id[f"train:{j['pair_id']}"]) for j in judged_train if f"train:{j['pair_id']}" in by_id]
    if train:
        out["train_judge_vs_human"] = {
            "all": _agreement([j["score_x"] for j, _ in train], [h["human_score_x"] for _, h in train]),
            # The consistent subset is easier, so agreement here is an upper bound.
            "position_consistent_only": _agreement(
                [j["score_x"] for j, _ in train if j["consistent"]],
                [h["human_score_x"] for j, h in train if j["consistent"]],
            ),
        }

    evals = [lab for lab in by_id.values() if lab["kind"] == "eval"]
    if evals:
        s = np.array([lab["human_score_x"] for lab in evals], dtype=float)
        out["human_eval"] = {
            "n": len(evals),
            "win_rate": bootstrap_ci(len(s), lambda i: s[i].mean(), n_boot, seed=seed).as_dict(),
        }
        judge_by_prompt = {r["prompt_id"]: r for r in eval_judgments}
        paired = [(judge_by_prompt[lab["ref"]], lab) for lab in evals if lab["ref"] in judge_by_prompt]
        if paired:
            out["eval_judge_vs_human"] = _agreement(
                [j["score_tuned"] for j, _ in paired], [h["human_score_x"] for _, h in paired]
            )
    return out


def training_section(log: dict | None) -> dict | None:
    if not log:
        return None
    evals = [e for e in log["log_history"] if any(k.startswith("eval_") for k in e)]
    last = evals[-1] if evals else {}
    keep = ("eval_loss", "eval_rewards/accuracies", "eval_rewards/margins", "eval_rewards/chosen",
            "eval_rewards/rejected")
    return {
        "base_model": log["base_model"],
        "n_train_pairs": log["n_train_pairs"],
        "n_holdout_pairs": log["n_holdout_pairs"],
        "hyperparams": log["hyperparams"],
        "final_holdout": {k: last[k] for k in keep if k in last},
    }


def verdict(results: dict) -> list[str]:
    lines: list[str] = []
    models = results["judge_models"]
    if models["train"] == models["eval"]:
        lines.append(f"**WARN (not independent)**: `{models['eval']}` both labeled the training pairs and scored "
                     "the result, so the eval measures fit to that judge. Only the human labels are independent.")
    ev = results["judges"].get("eval")
    if ev is None:
        return [*lines, "**INCOMPLETE**: no independent-judge results yet. Run `unslop judge-eval`."]
    raw, lc = CI(**ev["win_rate"]), CI(**ev["lc_win_rate"])
    if lc.lo > 0.5:
        lines.append(f"**PASS**: the independent judge prefers the tuned model even at equal length "
                     f"(LC win rate {_pct(lc)}).")
    elif lc.hi < 0.5:
        lines.append(f"**FAIL**: the tuned model is worse at equal length (LC win rate {_pct(lc)}).")
    else:
        lines.append(f"**NO CLAIM**: after length control, the difference is not significant (LC win rate {_pct(lc)}).")
    if raw.lo > 0.5 and not lc.lo > 0.5:
        lines.append("**WARN (length hacking)**: the raw win is significant, but it goes away once length is "
                     "controlled. The model learned to write more, not better.")
    for role, j in results["judges"].items():
        dropped = 1 - j["n"] / j["attempted"] if j["attempted"] else 0.0
        if dropped > MAX_DROP_SHARE:
            lines.append(f"**WARN (judge failures)**: the {role} judge returned no usable verdict for {dropped:.0%} "
                         f"of prompts ({j['n']}/{j['attempted']} kept). Win rates cover only the survivors, "
                         "which may be a biased subset.")
    ps = results.get("pairs")
    if ps and ps.get("pairs_attempted"):
        dropped = 1 - ps["pairs_judged"] / ps["pairs_attempted"]
        if dropped > MAX_DROP_SHARE:
            lines.append(f"**WARN (judge failures)**: {dropped:.0%} of training pairs got no usable verdict.")

    tr = results["judges"].get("train")
    if tr is not None:
        gap = tr["lc_win_rate"]["point"] - lc.point
        if gap > 0.10:
            lines.append(f"**WARN (Goodhart gap)**: the judge used for training rates the tuned model {gap:+.0%} "
                         "higher than the independent judge. Part of the gain is fitting one judge's taste.")

    style = results.get("style")
    if style:
        d = CI(**style["slop_per_1k_delta"])
        if d.lo > 0:
            lines.append(f"**WARN**: stock-phrase rate went *up* by {d.point:+.2f} per 1k words [{d.lo:+.2f}, {d.hi:+.2f}].")
        elif d.hi < 0:
            lines.append(f"Stock-phrase rate fell by {-d.point:.2f} per 1k words [{d.lo:+.2f}, {d.hi:+.2f}] "
                         "(a metric the judge never saw).")
        tr_share = style["truncated_share"]
        if tr_share["tuned"] > tr_share["base"] + 0.05:
            lines.append(f"**WARN (rambling)**: {tr_share['tuned']:.0%} of tuned outputs hit the token limit vs "
                         f"{tr_share['base']:.0%} for base.")

    human = results["human"].get("human_eval")
    if human is None:
        lines.append("No human eval labels yet, so every claim above rests on LLM judges.")
    else:
        h = CI(**human["win_rate"])
        tag = "agree with" if h.lo > 0.5 else "do not clearly support" if h.hi >= 0.5 else "contradict"
        lines.append(f"Human raters {tag} the judge (tuned win rate {_pct(h)}, n={human['n']}).")
        if human["n"] < 100:
            lines.append(f"Human n={human['n']} is small; the CI is wide, so treat this as directional.")
    return lines


def _pct(ci: CI | dict) -> str:
    ci = ci if isinstance(ci, CI) else CI(**ci)
    return f"{ci.point:.1%} [{ci.lo:.1%}, {ci.hi:.1%}]"


def render_markdown(results: dict) -> str:
    md = ["# unslop: results", ""]
    md += ["## Verdict", ""] + [f"- {line}" for line in results["verdict"]] + [""]

    if results["judges"]:
        md += ["## Tuned vs base, pairwise (95% bootstrap CI)", "",
               "| Judge | Model | n judged / attempted | Win rate | Length-controlled win rate | Position-consistent "
               "| Tuned longer |",
               "|---|---|---|---|---|---|---|"]
        for role in ROLES:
            if j := results["judges"].get(role):
                md.append(f"| {role} | `{results['judge_models'][role]}` | {j['n']}/{j['attempted']} | "
                          f"{_pct(j['win_rate'])} | "
                          f"{_pct(j['lc_win_rate'])} | {j['position_consistent_share']:.0%} | "
                          f"{j['tuned_longer_share']:.0%} |")
        md += ["", "Ties and position flips score 0.5. The `train` row is shown only to expose the Goodhart gap; "
               "the `eval` judge is the one that counts.", ""]

    if style := results.get("style"):
        md += ["## Style (judge-independent)", "", "| Metric | Base | Tuned |", "|---|---|---|"]
        for key in ("mean_words", "slop_per_1k_words", "em_dash_per_1k_words", "list_or_header_share", "distinct_2"):
            md.append(f"| {key} | {style['base'][key]:.3f} | {style['tuned'][key]:.3f} |")
        t = style["truncated_share"]
        md.append(f"| truncated_share | {t['base']:.3f} | {t['tuned']:.3f} |")
        w, s = style["mean_words_delta"], style["slop_per_1k_delta"]
        md += ["", f"Mean words delta: {w['point']:+.1f} [{w['lo']:+.1f}, {w['hi']:+.1f}]. "
                   f"Stock-phrase delta per 1k words: {s['point']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}].", ""]

    if human := results.get("human"):
        md += ["## Human labels", ""]
        if he := human.get("human_eval"):
            md.append(f"- Human tuned-vs-base win rate: {_pct(he['win_rate'])} (n={he['n']})")
        for key, label in (("eval_judge_vs_human", "Eval judge vs human"),):
            if a := human.get(key):
                md.append(f"- {label}: agreement {a['agreement']:.0%}, decisive agreement "
                          f"{a['decisive_agreement']:.0%} (n={a['decisive_n']}), kappa {a['cohen_kappa']:.2f}")
        if tj := human.get("train_judge_vs_human"):
            for sub in ("all", "position_consistent_only"):
                a = tj[sub]
                md.append(f"- Train judge vs human ({sub}, n={a['n']}): agreement {a['agreement']:.0%}, "
                          f"decisive {a['decisive_agreement']:.0%}, kappa {a['cohen_kappa']:.2f}")
        md.append("")

    if ps := results.get("pairs"):
        md += ["## Preference data", "", "| Stat | Value |", "|---|---|"]
        md += [f"| {k} | {v:.3f} |" if isinstance(v, float) else f"| {k} | {v} |" for k, v in ps.items()]
        md.append("")

    if tr := results.get("training"):
        md += ["## Training", "", f"Base `{tr['base_model']}`, {tr['n_train_pairs']} train pairs, "
               f"{tr['n_holdout_pairs']} held out.", ""]
        md += [f"- {k}: {v:.4f}" for k, v in tr["final_holdout"].items()] + [""]
    return "\n".join(md)


def build_report(paths: Paths, judge_models: dict[str, str], n_boot: int = 2000, seed: int = 0) -> dict:
    import json

    judgments = {role: read_jsonl(paths.eval_judgments(role), missing_ok=True) for role in ROLES}
    judge_stats = json.loads(paths.eval_judge_stats.read_text()) if paths.eval_judge_stats.exists() else {}
    log = json.loads(paths.training_log.read_text()) if paths.training_log.exists() else None
    pairs = json.loads(paths.pair_stats.read_text()) if paths.pair_stats.exists() else None
    results = {
        "judge_models": judge_models,
        "judges": {
            role: judge_section(rows, n_boot, seed, judge_stats.get(role, {}).get("attempted"))
            for role, rows in judgments.items()
            if rows
        },
        "style": style_section(read_jsonl(paths.eval_outputs, missing_ok=True), n_boot, seed),
        "human": human_section(read_jsonl(paths.human_labels, missing_ok=True),
                               read_jsonl(paths.judged_pairs, missing_ok=True), judgments["eval"], n_boot, seed),
        "pairs": pairs,
        "training": training_section(log),
    }
    results["verdict"] = verdict(results)
    write_json(paths.report_json, results)
    paths.report_md.parent.mkdir(parents=True, exist_ok=True)
    paths.report_md.write_text(render_markdown(results), encoding="utf-8")
    return results
