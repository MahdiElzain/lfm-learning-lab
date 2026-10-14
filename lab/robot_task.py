"""Robot dispatch: a small text task with short, objectively checkable answers.

Each example lists 3-6 robots (type, grid position, status) and one task.
The correct answer is the *eligible* robot nearest to the task location:

    eligible  = status is "idle"  AND  robot type is allowed for the task kind
    distance  = |x_robot - x_task| + |y_robot - y_task|        (Manhattan)
    answer    = {"robot": <id of eligible robot with smallest distance>,
                 "distance": <that distance>}

The generator guarantees exactly one optimal robot (no ties) and makes about
half of the examples "traps": the robot nearest overall is NOT eligible, so a
model that ignores the rules gets them wrong.

The verifier is deterministic and needs no model. It reports:
    parsed            a JSON object with the right keys/types was found
    valid_strict      the completion is *only* that JSON object
    known_robot       the robot id exists in the prompt
    violation_status  assigned robot is not idle          (constraint violation)
    violation_type    assigned robot type not allowed     (constraint violation)
    robot_correct     the assigned robot is the optimal one
    distance_correct  reported distance equals the optimal distance
    exact             robot_correct and distance_correct
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass

TYPES = {"drone": "uav", "quadruped": "go2_", "rover": "ugv"}
STATUSES = ["idle", "busy", "charging"]

# task kind -> (sentence template, allowed robot types)
TASK_KINDS = {
    "aerial": ("photograph the roof at ({x}, {y})", {"drone"}),
    "stairs": ("climb the stairwell at ({x}, {y})", {"quadruped"}),
    "heavy": ("haul the heavy crate at ({x}, {y})", {"rover"}),
    "scan": ("scan the sector at ({x}, {y})", {"drone", "quadruped", "rover"}),
}

SYSTEM_PROMPT = (
    "You dispatch robots. Rules: aerial tasks need a drone; stair tasks need a quadruped; "
    "heavy cargo needs a rover; scan tasks accept any robot. Only idle robots can be assigned. "
    "Choose the eligible robot with the smallest Manhattan distance |x1-x2|+|y1-y2| to the task. "
    'Answer with JSON only: {"robot": "<id>", "distance": <integer>}'
)


@dataclass
class Robot:
    id: str
    type: str
    x: int
    y: int
    status: str


def manhattan(ax: int, ay: int, bx: int, by: int) -> int:
    return abs(ax - bx) + abs(ay - by)


def solve(robots: list[dict], kind: str, tx: int, ty: int) -> dict | None:
    """Return the unique optimal answer, or None if no unique answer exists."""
    allowed = TASK_KINDS[kind][1]
    eligible = [r for r in robots if r["status"] == "idle" and r["type"] in allowed]
    if not eligible:
        return None
    dists = sorted((manhattan(r["x"], r["y"], tx, ty), r["id"]) for r in eligible)
    if len(dists) > 1 and dists[0][0] == dists[1][0]:
        return None  # tie -> ambiguous, regenerate
    return {"robot": dists[0][1], "distance": dists[0][0]}


def _random_robots(rng: random.Random, n: int, grid: int) -> list[dict]:
    robots, used_ids, used_pos = [], set(), set()
    while len(robots) < n:
        rtype = rng.choice(list(TYPES))
        rid = f"{TYPES[rtype]}{rng.randint(1, 9)}"
        pos = (rng.randrange(grid), rng.randrange(grid))
        if rid in used_ids or pos in used_pos:
            continue
        used_ids.add(rid)
        used_pos.add(pos)
        status = rng.choices(STATUSES, weights=[0.55, 0.25, 0.20])[0]
        robots.append(asdict(Robot(rid, rtype, pos[0], pos[1], status)))
    return robots


def make_example(rng: random.Random, n_robots: int, grid: int = 10, want_trap: bool | None = None) -> dict:
    """Sample until we get a scene with a unique answer (and the requested trap flag)."""
    while True:
        robots = _random_robots(rng, n_robots, grid)
        kind = rng.choice(list(TASK_KINDS))
        tx, ty = rng.randrange(grid), rng.randrange(grid)
        if any((r["x"], r["y"]) == (tx, ty) for r in robots):
            continue
        answer = solve(robots, kind, tx, ty)
        if answer is None:
            continue
        nearest_overall = min(robots, key=lambda r: (manhattan(r["x"], r["y"], tx, ty), r["id"]))
        trap = nearest_overall["id"] != answer["robot"]
        if want_trap is not None and trap != want_trap:
            continue
        return {"robots": robots, "kind": kind, "target": [tx, ty], "answer": answer, "trap": trap,
                "n_robots": n_robots}


def format_user(ex: dict) -> str:
    lines = ["Robots:"]
    for r in ex["robots"]:
        lines.append(f"- {r['id']} ({r['type']}) at ({r['x']}, {r['y']}), {r['status']}")
    template = TASK_KINDS[ex["kind"]][0]
    lines.append("Task: " + template.format(x=ex["target"][0], y=ex["target"][1]) + ".")
    return "\n".join(lines)


def format_answer(answer: dict) -> str:
    return json.dumps({"robot": answer["robot"], "distance": answer["distance"]})


def chat_messages(ex: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": format_user(ex)}]


def base_model_prompt(ex: dict, shots: list[dict]) -> str:
    """Plain-text few-shot prompt for a *base* model (no chat template)."""
    parts = [SYSTEM_PROMPT, ""]
    for s in shots:
        parts += [format_user(s), "Answer: " + format_answer(s["answer"]), ""]
    parts += [format_user(ex), "Answer:"]
    return "\n".join(parts)


def scene_signature(ex: dict) -> str:
    """Identity of a scene ignoring robot names: used to block near-duplicate leakage."""
    robots = sorted((r["type"], r["x"], r["y"], r["status"]) for r in ex["robots"])
    return json.dumps([robots, ex["kind"], ex["target"]])


# --------------------------------------------------------------------------- #
# Verifier
# --------------------------------------------------------------------------- #
_JSON_OBJ = re.compile(r"\{[^{}]*\}")


def parse_completion(text: str) -> tuple[dict | None, bool]:
    """Return (parsed_object_or_None, valid_strict)."""
    stripped = text.strip()
    candidates = [stripped] + _JSON_OBJ.findall(stripped)
    for i, cand in enumerate(candidates):
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if (isinstance(obj, dict) and set(obj) == {"robot", "distance"}
                and isinstance(obj["robot"], str) and isinstance(obj["distance"], int)
                and not isinstance(obj["distance"], bool)):
            return obj, i == 0
    return None, False


def verify(ex: dict, completion: str) -> dict:
    obj, strict = parse_completion(completion)
    result = {"parsed": obj is not None, "valid_strict": strict, "known_robot": False,
              "violation_status": False, "violation_type": False, "robot_correct": False,
              "distance_correct": False, "exact": False}
    if obj is None:
        return result
    robots = {r["id"]: r for r in ex["robots"]}
    chosen = robots.get(obj["robot"])
    result["known_robot"] = chosen is not None
    if chosen is not None:
        result["violation_status"] = chosen["status"] != "idle"
        result["violation_type"] = chosen["type"] not in TASK_KINDS[ex["kind"]][1]
    result["robot_correct"] = obj["robot"] == ex["answer"]["robot"]
    result["distance_correct"] = obj["distance"] == ex["answer"]["distance"]
    result["exact"] = result["robot_correct"] and result["distance_correct"]
    return result


def label_free_check(ex: dict, completion: str) -> dict:
    """Checks that need NO ground-truth answer: format, known id, rule violations,
    and whether the reported distance matches the chosen robot. Used to filter
    teacher outputs and to select among samples without peeking at labels."""
    obj, strict = parse_completion(completion)
    out = {"parsed": obj is not None, "valid_strict": strict, "ok": False}
    if obj is None:
        return out
    chosen = {r["id"]: r for r in ex["robots"]}.get(obj["robot"])
    if chosen is None:
        return out
    eligible = chosen["status"] == "idle" and chosen["type"] in TASK_KINDS[ex["kind"]][1]
    consistent = obj["distance"] == manhattan(chosen["x"], chosen["y"], *ex["target"])
    out.update({"eligible": eligible, "self_consistent": consistent, "ok": eligible and consistent})
    return out


def reward(ex: dict, completion: str, mode: str = "shaped") -> float:
    """Reward used by GRPO and DPO pair construction.

    binary: 1 if exact else 0.
    shaped: exact 1.0 | right robot, wrong distance 0.5 |
            parsed, no rule violation 0.1 | otherwise 0.0
    """
    v = verify(ex, completion)
    if mode == "binary":
        return float(v["exact"])
    if v["exact"]:
        return 1.0
    if v["robot_correct"]:
        return 0.5
    if v["parsed"] and v["known_robot"] and not (v["violation_status"] or v["violation_type"]):
        return 0.1
    return 0.0


METRIC_KEYS = ["parsed", "valid_strict", "known_robot", "violation_status", "violation_type",
               "robot_correct", "distance_correct", "exact"]


def aggregate(results: list[dict]) -> dict:
    n = max(len(results), 1)
    agg = {k: round(sum(r[k] for r in results) / n, 4) for k in METRIC_KEYS}
    agg["n"] = len(results)
    return agg


# --------------------------------------------------------------------------- #
# Dataset builder with leakage control
# --------------------------------------------------------------------------- #
def build_splits(seed: int, n_train: int, n_val: int, n_test: int, n_test_ood: int,
                 robots_range=(3, 5), ood_robots: int = 6) -> dict[str, list[dict]]:
    """Train/val/test share the 3-5 robot distribution; test_ood uses 6 robots.

    Scenes are deduplicated by `scene_signature` across ALL splits, so no
    test scene (even with renamed robots) appears in training.
    """
    rng = random.Random(seed)
    seen: set[str] = set()
    splits: dict[str, list[dict]] = {}
    plan = [("test", n_test, robots_range), ("test_ood", n_test_ood, (ood_robots, ood_robots)),
            ("val", n_val, robots_range), ("train", n_train, robots_range)]
    for name, count, (lo, hi) in plan:
        rows = []
        while len(rows) < count:
            ex = make_example(rng, rng.randint(lo, hi), want_trap=(len(rows) % 2 == 0))
            sig = scene_signature(ex)
            if sig in seen:
                continue
            seen.add(sig)
            ex["id"] = f"{name}-{len(rows):05d}"
            ex["split"] = name
            rows.append(ex)
        splits[name] = rows
    return splits
