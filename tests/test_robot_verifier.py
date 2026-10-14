"""Deterministic checks of the robot-dispatch generator and verifier.

Run:  .venv/bin/python tests/test_robot_verifier.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import robot_task as rt  # noqa: E402

EX = {
    "robots": [
        {"id": "uav1", "type": "drone", "x": 1, "y": 1, "status": "idle"},      # nearest overall, wrong type
        {"id": "ugv2", "type": "rover", "x": 5, "y": 5, "status": "busy"},      # right type, busy
        {"id": "ugv3", "type": "rover", "x": 6, "y": 0, "status": "idle"},      # correct
        {"id": "ugv4", "type": "rover", "x": 9, "y": 9, "status": "idle"},      # eligible, farther
    ],
    "kind": "heavy", "target": [2, 1],
}
EX["answer"] = rt.solve(EX["robots"], EX["kind"], *EX["target"])


def test_solver():
    assert EX["answer"] == {"robot": "ugv3", "distance": 5}, EX["answer"]


def test_verifier_cases():
    v = rt.verify(EX, '{"robot": "ugv3", "distance": 5}')
    assert v["exact"] and v["valid_strict"] and not v["violation_status"]
    v = rt.verify(EX, 'Sure! ```json\n{"robot": "ugv3", "distance": 5}\n```')
    assert v["exact"] and v["parsed"] and not v["valid_strict"]          # fenced: parsed, not strict
    v = rt.verify(EX, '{"robot": "ugv3", "distance": 6}')
    assert v["robot_correct"] and not v["distance_correct"] and not v["exact"]
    v = rt.verify(EX, '{"robot": "uav1", "distance": 1}')
    assert v["violation_type"] and not v["violation_status"] and not v["robot_correct"]
    v = rt.verify(EX, '{"robot": "ugv2", "distance": 7}')
    assert v["violation_status"] and not v["violation_type"]
    v = rt.verify(EX, '{"robot": "ugv9", "distance": 1}')
    assert v["parsed"] and not v["known_robot"]
    for bad in ["", "ugv3", '{"robot": "ugv3"}', '{"robot": "ugv3", "distance": "5"}',
                '{"robot": "ugv3", "distance": true}', '{"robot": "ugv3", "distance": 5, "x": 1}']:
        assert not rt.verify(EX, bad)["parsed"], bad


def test_rewards_and_label_free():
    assert rt.reward(EX, '{"robot": "ugv3", "distance": 5}') == 1.0
    assert rt.reward(EX, '{"robot": "ugv3", "distance": 4}') == 0.5
    assert rt.reward(EX, '{"robot": "ugv4", "distance": 15}') == 0.1     # eligible but not nearest
    assert rt.reward(EX, '{"robot": "uav1", "distance": 1}') == 0.0
    assert rt.reward(EX, '{"robot": "ugv4", "distance": 15}', mode="binary") == 0.0
    lf = rt.label_free_check(EX, '{"robot": "ugv4", "distance": 15}')
    assert lf["ok"]                                       # label-free check cannot see optimality
    assert not rt.label_free_check(EX, '{"robot": "ugv4", "distance": 3}')["ok"]


def test_generator_properties():
    rng = random.Random(0)
    for _ in range(500):
        ex = rt.make_example(rng, rng.randint(3, 6))
        assert rt.solve(ex["robots"], ex["kind"], *ex["target"]) == ex["answer"]
        assert rt.verify(ex, rt.format_answer(ex["answer"]))["exact"]
    splits = rt.build_splits(1, 200, 20, 30, 20)
    sigs = [set(map(rt.scene_signature, rows)) for rows in splits.values()]
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            assert not sigs[i] & sigs[j]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
