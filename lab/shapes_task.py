"""Synthetic shape scenes for vision-language lessons.

A *scene* is a small image with 1-6 coloured shapes on a varied background.
Several questions are asked about each scene, and every answer is computed
from the scene description, so labels are exact.

Question types (answer vocabulary in brackets):
    count_color   How many red shapes are there?          [0-6]
    count_shape   How many triangles are there?           [0-6]
    count_total   How many shapes are there in total?     [0-6]
    exists        Is there a blue circle?                 [yes/no]
    leftmost      What color is the leftmost shape?       [color]
    left_of       Is the red circle to the left of the blue square?   [yes/no]

`left_of` only makes sense when both named shapes appear exactly once, so it
is "scene-specific". All other types are well defined for ANY image, which
is what lets us ask the same question about a swapped or blank image.

Splits are made by scene id: every question about one image lives in the
same split, so the test set never shows a training image.
"""
from __future__ import annotations

import math
import random
import re

from PIL import Image, ImageDraw, ImageFilter

COLORS = {
    "red": (220, 40, 40),
    "green": (40, 170, 60),
    "blue": (40, 80, 220),
    "yellow": (235, 200, 30),
    "purple": (140, 60, 190),
}
SHAPES = ["circle", "square", "triangle"]
NUMBER_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
                "none": "0", "no": None}
ANY_IMAGE_TYPES = ["count_color", "count_shape", "count_total", "exists", "leftmost"]


# --------------------------------------------------------------------------- #
# Scene sampling and rendering
# --------------------------------------------------------------------------- #
def sample_scene(rng: random.Random, n_min: int = 1, n_max: int = 6, size: int = 256) -> dict:
    n = rng.randint(n_min, n_max)
    palette = rng.sample(list(COLORS), k=rng.randint(2, 4))  # few colours -> repeated colours -> counts > 1
    objects, tries = [], 0
    while len(objects) < n and tries < 500:
        tries += 1
        r = rng.randint(size // 14, size // 8)               # "radius" of the bounding circle
        cx, cy = rng.randint(r + 4, size - r - 4), rng.randint(r + 4, size - r - 4)
        if any(math.hypot(cx - o["cx"], cy - o["cy"]) < r + o["r"] + 6 for o in objects):
            continue                                          # no overlaps
        if any(abs(cx - o["cx"]) < 8 for o in objects):
            continue                                          # keep left/right order unambiguous
        objects.append({"shape": rng.choice(SHAPES), "color": rng.choice(palette), "cx": cx, "cy": cy, "r": r})
    background = rng.choice(["white", "gray", "beige", "gradient", "noise"])
    return {"objects": objects, "background": background, "size": size, "bg_seed": rng.randrange(10**9)}


def render(scene: dict, size: int | None = None) -> Image.Image:
    """Draw the scene. `size` lets you re-render the same scene at another resolution."""
    base = scene["size"]
    size = size or base
    s = size / base
    rng = random.Random(scene["bg_seed"])
    bg = scene["background"]
    if bg == "white":
        img = Image.new("RGB", (size, size), (250, 250, 250))
    elif bg == "gray":
        img = Image.new("RGB", (size, size), (200, 200, 200))
    elif bg == "beige":
        img = Image.new("RGB", (size, size), (238, 228, 205))
    elif bg == "gradient":
        img = Image.new("RGB", (size, size))
        top, bottom = (230, 235, 245), (170, 180, 190)
        px = img.load()
        for y in range(size):
            t = y / max(size - 1, 1)
            c = tuple(int(a + (b - a) * t) for a, b in zip(top, bottom))
            for x in range(size):
                px[x, y] = c
    else:  # light grey noise texture
        img = Image.new("RGB", (size, size))
        px = img.load()
        for y in range(size):
            for x in range(size):
                v = 200 + rng.randint(-25, 25)
                px[x, y] = (v, v, v)
        img = img.filter(ImageFilter.GaussianBlur(1))
    draw = ImageDraw.Draw(img)
    for o in scene["objects"]:
        cx, cy, r = o["cx"] * s, o["cy"] * s, o["r"] * s
        fill = COLORS[o["color"]]
        if o["shape"] == "circle":
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
        elif o["shape"] == "square":
            h = r * 0.85
            draw.rectangle([cx - h, cy - h, cx + h, cy + h], fill=fill)
        else:
            draw.polygon([(cx, cy - r), (cx - r * 0.95, cy + r * 0.75), (cx + r * 0.95, cy + r * 0.75)], fill=fill)
    return img


def blank_image(size: int = 256) -> Image.Image:
    return Image.new("RGB", (size, size), (250, 250, 250))


# --------------------------------------------------------------------------- #
# Questions and exact answers
# --------------------------------------------------------------------------- #
def answer_for(scene: dict, q: dict) -> str | None:
    """Compute the exact answer of question `q` for `scene` (None if undefined)."""
    objs = scene["objects"]
    t = q["type"]
    if t == "count_color":
        return str(sum(o["color"] == q["color"] for o in objs))
    if t == "count_shape":
        return str(sum(o["shape"] == q["shape"] for o in objs))
    if t == "count_total":
        return str(len(objs))
    if t == "exists":
        return "yes" if any(o["color"] == q["color"] and o["shape"] == q["shape"] for o in objs) else "no"
    if t == "leftmost":
        return min(objs, key=lambda o: o["cx"])["color"] if objs else None
    if t == "left_of":
        a = [o for o in objs if (o["color"], o["shape"]) == (q["color"], q["shape"])]
        b = [o for o in objs if (o["color"], o["shape"]) == (q["color2"], q["shape2"])]
        if len(a) != 1 or len(b) != 1:
            return None
        return "yes" if a[0]["cx"] < b[0]["cx"] else "no"
    raise ValueError(t)


def question_text(q: dict) -> str:
    t = q["type"]
    if t == "count_color":
        return f"How many {q['color']} shapes are there? Answer with a number."
    if t == "count_shape":
        return f"How many {q['shape']}s are there? Answer with a number."
    if t == "count_total":
        return "How many shapes are there in total? Answer with a number."
    if t == "exists":
        return f"Is there a {q['color']} {q['shape']}? Answer yes or no."
    if t == "leftmost":
        return "What color is the leftmost shape? Answer with one word."
    if t == "left_of":
        return (f"Is the {q['color']} {q['shape']} to the left of the {q['color2']} {q['shape2']}? "
                "Answer yes or no.")
    raise ValueError(t)


def sample_questions(rng: random.Random, scene: dict, k: int = 3) -> list[dict]:
    """Pick k questions of different types, balancing answers within each type."""
    objs = scene["objects"]
    combos_present = sorted({(o["color"], o["shape"]) for o in objs})
    unique = [c for c in combos_present if sum((o["color"], o["shape"]) == c for o in objs) == 1]
    types = ["count_color", "count_shape", "count_total", "exists", "leftmost"]
    if len(unique) >= 2:
        types.append("left_of")
    qs = []
    for t in rng.sample(types, k=min(k, len(types))):
        q = {"type": t}
        if t == "count_color":
            # Balance: a uniformly random colour is usually ABSENT (scenes use 2-4 of 5 colours),
            # which made "0" the answer 61% of the time. Ask about a present colour 70% of the time.
            present = sorted({o["color"] for o in objs})
            absent = [c for c in COLORS if c not in present]
            q["color"] = rng.choice(absent) if (absent and rng.random() < 0.3) else rng.choice(present)
        elif t == "count_shape":
            q["shape"] = rng.choice(SHAPES)
        elif t == "exists":
            if rng.random() < 0.5 and combos_present:             # balance yes / no
                q["color"], q["shape"] = rng.choice(combos_present)
            else:
                absent = [(c, s) for c in COLORS for s in SHAPES if (c, s) not in combos_present]
                q["color"], q["shape"] = rng.choice(absent)
        elif t == "left_of":
            (c1, s1), (c2, s2) = rng.sample(unique, 2)
            q.update(color=c1, shape=s1, color2=c2, shape2=s2)
        q["question"] = question_text(q)
        q["answer"] = answer_for(scene, q)
        qs.append(q)
    return qs


# --------------------------------------------------------------------------- #
# Answer normalisation and verification
# --------------------------------------------------------------------------- #
def allowed_answers(qtype: str) -> set[str]:
    if qtype.startswith("count"):
        return {str(i) for i in range(7)}
    if qtype in ("exists", "left_of"):
        return {"yes", "no"}
    return set(COLORS)


def normalize(text: str) -> str:
    """Lower-case, keep the first word/number, map number words to digits."""
    words = re.findall(r"[a-z]+|\d+", text.lower())
    if not words:
        return ""
    w = words[0]
    if w in NUMBER_WORDS and NUMBER_WORDS[w] is not None:
        return NUMBER_WORDS[w]
    return w


def verify(qtype: str, expected: str | None, completion: str) -> dict:
    pred = normalize(completion)
    valid = pred in allowed_answers(qtype)
    return {"pred": pred, "valid": valid, "correct": expected is not None and pred == expected}


def reward(qtype: str, expected: str, completion: str) -> float:
    """1 for a correct short answer, 0.1 for a valid-but-wrong one, 0 otherwise.
    Long answers are penalised slightly so the policy keeps answers short."""
    v = verify(qtype, expected, completion)
    r = 1.0 if v["correct"] else (0.1 if v["valid"] else 0.0)
    if len(completion.split()) > 3:
        r -= 0.1
    return r


# --------------------------------------------------------------------------- #
# Split builder
# --------------------------------------------------------------------------- #
def scene_key(scene: dict) -> tuple:
    return tuple(sorted((o["shape"], o["color"], o["cx"] // 16, o["cy"] // 16) for o in scene["objects"]))


def build_scene_splits(seed: int, n_train: int, n_val: int, n_test: int, q_per_scene: int = 3,
                       size: int = 256) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    seen = set()
    splits = {}
    for name, count in [("test", n_test), ("val", n_val), ("train", n_train)]:
        scenes = []
        while len(scenes) < count:
            sc = sample_scene(rng, size=size)
            key = scene_key(sc)
            if key in seen:            # coarse near-duplicate guard across splits
                continue
            seen.add(key)
            sc["scene_id"] = f"{name}-{len(scenes):05d}"
            sc["split"] = name
            sc["questions"] = sample_questions(rng, sc, q_per_scene)
            scenes.append(sc)
        splits[name] = scenes
    return splits
