from unslop.synthetic import run_scenario

MODELS = {"train": "judge-a", "eval": "judge-b"}


def test_honest_scenario_passes(tmp_path):
    results = run_scenario("honest", tmp_path, MODELS, n_boot=300)
    verdict = " ".join(results["verdict"])
    assert "**PASS**" in verdict and "length hacking" not in verdict
    assert (tmp_path / "honest.md").read_text().startswith("> **SYNTHETIC")


def test_length_hack_scenario_is_caught(tmp_path):
    results = run_scenario("length_hack", tmp_path, MODELS, n_boot=300)
    verdict = " ".join(results["verdict"])
    assert "**PASS**" not in verdict
    assert "WARN (length hacking)" in verdict
    assert "Goodhart gap" in verdict and "rambling" in verdict
