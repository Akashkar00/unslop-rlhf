import ast
import json

from unslop import kaggle_runner as kr


def test_smoke_stages_limit_work():
    smoke, full = kr.stages(smoke=True), kr.stages(smoke=False)
    assert smoke["generate"].commands == ["prompts", f"generate --limit {kr.SMOKE_LIMIT}"]
    assert full["generate"].commands == ["prompts", "generate"]
    assert set(full["train"].inputs) == {"dpo_train.jsonl", "eval.jsonl"}


def test_push_kernel_writes_valid_script_and_metadata(tmp_path, monkeypatch):
    pushed = []
    monkeypatch.setattr(kr, "_kaggle", lambda *args, check=True: pushed.append(args) or "Kernel version 1 pushed")
    stage = kr.stages(smoke=True)["train"]
    cfg = {"repo": "https://github.com/x/y.git", "commit": "abc123", "env": {"UNSLOP_BASE_MODEL": "m"},
           "inputs": stage.inputs, "commands": stage.commands, "outputs": stage.outputs}

    ref = kr.push_kernel("me", stage, cfg, tmp_path, dataset="me/unslop-pairs")

    assert ref == "me/unslop-train"
    assert pushed == [("kernels", "push", "-p", str(tmp_path / "train"))]
    meta = json.loads((tmp_path / "train/kernel-metadata.json").read_text())
    assert meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"]
    assert meta["dataset_sources"] == ["me/unslop-pairs"]
    script = (tmp_path / "train/run.py").read_text()
    tree = ast.parse(script)  # must be valid Python
    cfg_node = next(n for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "CFG")
    assert ast.literal_eval(cfg_node.value) == cfg
    assert kr.OUT_TAR in script
