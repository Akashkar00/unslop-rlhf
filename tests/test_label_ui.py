"""Headless test of the Streamlit labeler: the stored score must point at the
response that was actually shown on the side you clicked, whatever the flip."""

import hashlib
from pathlib import Path

import pytest

from unslop.config import Paths
from unslop.io import read_jsonl, write_jsonl

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parents[1] / "app" / "label_ui.py")


def _flipped(item_id: str) -> bool:
    return int(hashlib.sha256(item_id.encode()).hexdigest(), 16) % 2 == 1


@pytest.mark.parametrize("item_id", ["eval:p0", "eval:p1", "eval:p2", "eval:p3"])
def test_clicking_a_records_the_left_response(tmp_path, monkeypatch, item_id):
    monkeypatch.setenv("UNSLOP_ROOT", str(tmp_path))
    paths = Paths(tmp_path)
    write_jsonl(paths.human_queue_eval, [
        {"item_id": item_id, "kind": "eval", "ref": item_id[5:], "prompt": "Write a haiku.",
         "x": "TUNED TEXT", "y": "BASE TEXT"},
    ])
    at = AppTest.from_file(APP).run()
    assert not at.exception

    rendered = [m.value for m in at.markdown]
    left = rendered[rendered.index("**A**") + 1]
    assert left == ("BASE TEXT" if _flipped(item_id) else "TUNED TEXT")

    next(b for b in at.button if b.label == "A is better").click().run()
    (label,) = read_jsonl(paths.human_labels)
    assert label["item_id"] == item_id
    assert label["human_score_x"] == (1.0 if left == "TUNED TEXT" else 0.0)
    assert any("Nothing left" in s.value for s in at.success)
