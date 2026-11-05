"""Blind pairwise labeling. Run with: uv run streamlit run app/label_ui.py

You never see which response is the tuned model, which one the judge picked,
or a stable left/right position. Order is randomised per item from a hash of
its id, so the same item keeps its order if you reload."""

from __future__ import annotations

import hashlib
import time

import streamlit as st

from unslop.config import Settings
from unslop.io import append_jsonl, read_jsonl

paths = Settings.from_env().paths
st.set_page_config(page_title="unslop labeling", layout="wide")

queue = read_jsonl(paths.human_queue_train, missing_ok=True) + read_jsonl(paths.human_queue_eval, missing_ok=True)
labels = {r["item_id"]: r for r in read_jsonl(paths.human_labels, missing_ok=True)}
skipped: set[str] = st.session_state.setdefault("skipped", set())

with st.sidebar:
    st.header("Queue")
    kinds = st.multiselect("Item kinds", ["eval", "train"], default=["eval", "train"],
                           help="eval = tuned vs base on held-out prompts; train = judge-agreement sample")
    for kind in ("eval", "train"):
        total = sum(q["kind"] == kind for q in queue)
        done = sum(q["kind"] == kind and q["item_id"] in labels for q in queue)
        st.write(f"**{kind}**: {done}/{total}")
    if skipped and st.button("Un-skip all"):
        skipped.clear()
        st.rerun()

if not queue:
    st.info("No queue yet. Run `unslop build-pairs` and/or `unslop judge-eval` first.")
    st.stop()

pending = [q for q in queue if q["kind"] in kinds and q["item_id"] not in labels and q["item_id"] not in skipped]
if not pending:
    st.success("Nothing left to label in the selected kinds. Run `unslop report`.")
    st.stop()

item = pending[0]
flip = int(hashlib.sha256(item["item_id"].encode()).hexdigest(), 16) % 2 == 1
left, right = (item["y"], item["x"]) if flip else (item["x"], item["y"])


def record(choice: str) -> None:
    if choice == "tie":
        score_x = 0.5
    else:
        picked_x = (choice == "A") != flip
        score_x = 1.0 if picked_x else 0.0
    append_jsonl(paths.human_labels, {
        "item_id": item["item_id"], "kind": item["kind"], "ref": item["ref"],
        "human_score_x": score_x, "labeled_at": time.time(),
    })


def skip() -> None:
    skipped.add(item["item_id"])


st.caption(f"{len(pending)} left. Judge the *writing*: does it do the job, sound like a person, stay concrete, "
           "and avoid padding? Don't reward length for its own sake.")
with st.container(border=True):
    st.markdown("**Request**")
    st.markdown(item["prompt"])

col_a, col_b = st.columns(2)
with col_a, st.container(border=True):
    st.markdown("**A**")
    st.markdown(left)
with col_b, st.container(border=True):
    st.markdown("**B**")
    st.markdown(right)

b1, b2, b3, b4 = st.columns(4)
b1.button("A is better", on_click=record, args=("A",), type="primary")
b2.button("B is better", on_click=record, args=("B",), type="primary")
b3.button("Tie", on_click=record, args=("tie",))
b4.button("Skip", on_click=skip)
