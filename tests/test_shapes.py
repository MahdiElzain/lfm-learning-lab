"""Checks for the shape-scene generator, answers and verifier.

Run:  .venv/bin/python tests/test_shapes.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import shapes_task as st  # noqa: E402

SCENE = {"objects": [
    {"shape": "circle", "color": "red", "cx": 40, "cy": 50, "r": 20},
    {"shape": "square", "color": "blue", "cx": 120, "cy": 60, "r": 20},
    {"shape": "circle", "color": "red", "cx": 200, "cy": 200, "r": 20},
], "background": "white", "size": 256, "bg_seed": 1}


def test_answers():
    assert st.answer_for(SCENE, {"type": "count_color", "color": "red"}) == "2"
    assert st.answer_for(SCENE, {"type": "count_shape", "shape": "triangle"}) == "0"
    assert st.answer_for(SCENE, {"type": "count_total"}) == "3"
    assert st.answer_for(SCENE, {"type": "exists", "color": "blue", "shape": "square"}) == "yes"
    assert st.answer_for(SCENE, {"type": "exists", "color": "blue", "shape": "circle"}) == "no"
    assert st.answer_for(SCENE, {"type": "leftmost"}) == "red"
    # red circle appears twice -> left_of undefined
    assert st.answer_for(SCENE, {"type": "left_of", "color": "red", "shape": "circle",
                                 "color2": "blue", "shape2": "square"}) is None


def test_normalize_verify():
    assert st.normalize("Three.") == "3" and st.normalize(" 2 shapes") == "2"
    assert st.normalize("Yes, there is") == "yes" and st.normalize("Red!") == "red"
    assert st.verify("count_color", "2", "two")["correct"]
    assert not st.verify("exists", "yes", "maybe")["valid"]
    assert st.reward("exists", "yes", "yes") == 1.0
    assert st.reward("exists", "yes", "no") == 0.1


def test_generation():
    rng = random.Random(0)
    for _ in range(200):
        sc = st.sample_scene(rng)
        for q in st.sample_questions(rng, sc):
            assert q["answer"] is not None
            assert q["answer"] in st.allowed_answers(q["type"])
    img = st.render(sc)
    assert img.size == (256, 256) and st.render(sc, 512).size == (512, 512)
    splits = st.build_scene_splits(0, 50, 10, 20)
    ids = [{s["scene_id"] for s in v} for v in splits.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    keys = [{st.scene_key(s) for s in v} for v in splits.values()]
    assert not (keys[0] & keys[2])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
