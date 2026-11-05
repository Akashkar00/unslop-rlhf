import random

from unslop.config import Paths
from unslop.io import write_json, write_jsonl
from unslop.prompts import filter_rows, split_prompts
from unslop.report import build_report

MODELS = {"train": "judge-a", "eval": "judge-b"}


def _write_eval(paths: Paths, n: int, tuned_wins_by_length: bool):
    rng = random.Random(0)
    outputs, judged = [], []
    for i in range(n):
        longer = rng.random() < 0.8
        base_words = 100
        # 150/100 and 67/100 are near-symmetric in log space, so at equal length
        # a pure length effect should land on ~0.5.
        tuned_words = 150 if longer else 67
        if tuned_wins_by_length:
            score = 1.0 if rng.random() < (0.85 if longer else 0.15) else 0.0
        else:
            score = 1.0 if rng.random() < 0.7 else 0.0
        outputs.append({"prompt_id": f"p{i}", "prompt": "q", "tuned": "w " * tuned_words, "base": "w " * base_words,
                        "tuned_finished": True, "base_finished": True})
        judged.append({"prompt_id": f"p{i}", "score_tuned": score, "consistent": True, "reasons": ["", ""],
                       "tuned_words": tuned_words, "base_words": base_words})
    write_jsonl(paths.eval_outputs, outputs)
    write_jsonl(paths.eval_judgments("eval"), judged)
    return judged


def test_report_flags_length_hacking(tmp_path):
    paths = Paths(tmp_path)
    _write_eval(paths, 300, tuned_wins_by_length=True)
    results = build_report(paths, MODELS, n_boot=300)
    text = " ".join(results["verdict"])
    assert "length hacking" in text
    assert "PASS" not in text
    assert paths.report_md.exists() and paths.report_json.exists()


def test_report_passes_real_improvement_and_reads_human_labels(tmp_path):
    paths = Paths(tmp_path)
    judged = _write_eval(paths, 300, tuned_wins_by_length=False)
    write_jsonl(paths.human_labels, [
        {"item_id": f"eval:{j['prompt_id']}", "kind": "eval", "ref": j["prompt_id"],
         "human_score_x": j["score_tuned"], "labeled_at": 0}
        for j in judged[:40]
    ])
    results = build_report(paths, MODELS, n_boot=300)
    assert results["verdict"][0].startswith("**PASS**")
    assert results["human"]["human_eval"]["n"] == 40
    assert results["human"]["eval_judge_vs_human"]["agreement"] == 1.0
    assert any("small" in line for line in results["verdict"])


def test_report_warns_on_judge_failures(tmp_path):
    paths = Paths(tmp_path)
    _write_eval(paths, 300, tuned_wins_by_length=False)
    write_json(paths.eval_judge_stats, {"eval": {"model": "judge-b", "attempted": 400, "judged": 300}})
    results = build_report(paths, MODELS, n_boot=100)
    assert any("judge failures" in line and "25%" in line for line in results["verdict"])
    assert "300/400" in paths.report_md.read_text()


def test_report_without_eval_is_incomplete(tmp_path):
    results = build_report(Paths(tmp_path), MODELS, n_boot=10)
    assert results["verdict"][-1].startswith("**INCOMPLETE**")


def test_prompt_split_has_no_leakage():
    raw_train = [{"prompt_id": str(i), "prompt": f"Write about {i}", "category": "Generation"} for i in range(50)]
    raw_train.append({"prompt_id": "dup", "prompt": "  write ABOUT 3 ", "category": "Generation"})
    raw_train.append({"prompt_id": "code", "prompt": "fix my code", "category": "Coding"})
    raw_eval = [{"prompt_id": "e", "prompt": "Write about 7", "category": "Rewrite"}]
    train, evals = split_prompts(filter_rows(raw_train), filter_rows(raw_eval), n_train=100, n_eval=10, seed=0)
    assert len(train) == 49  # dup and Coding filtered, "Write about 7" removed for overlap
    assert all(r["prompt"] != "Write about 7" for r in train)
    assert len(evals) == 1


def test_report_warns_when_one_judge_does_both_jobs(tmp_path):
    _write_eval(Paths(tmp_path), 50, tuned_wins_by_length=False)
    results = build_report(Paths(tmp_path), {"train": "same", "eval": "same"}, n_boot=50)
    assert results["verdict"][0].startswith("**WARN (not independent)**")
