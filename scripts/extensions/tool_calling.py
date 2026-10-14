"""Extension (runnable): tool calling with LFM2.5-350M on robot-dispatch questions.

    .venv/bin/python scripts/extensions/tool_calling.py --n 20

Instead of doing arithmetic in its head, the model may call tools:
    list_eligible_robots(task_kind)        -> idle robots allowed for the task
    manhattan_distance(x1, y1, x2, y2)     -> integer distance
LFM2.5 writes Pythonic calls between <|tool_call_start|> and <|tool_call_end|>.
We parse them with `ast` (never `eval`), run them, send results back with the
"tool" role, and let the model answer. Compare accuracy with the plain prompt.
"""
import argparse
import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lab  # noqa: E402,F401
import torch  # noqa: E402

from lab import robot_task as rt  # noqa: E402
from lab.common import DATA, OUTPUTS, read_jsonl, write_json  # noqa: E402
from lab.models import load_text_model  # noqa: E402
from lab.text_eval import evaluate_robot  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--n", type=int, default=20)
p.add_argument("--max-rounds", type=int, default=4)
args = p.parse_args()

TOOLS = [
    {"name": "list_eligible_robots", "description": "Return idle robots whose type is allowed for the task kind, "
     "with their positions.", "parameters": {"type": "object", "properties": {
         "task_kind": {"type": "string", "enum": list(rt.TASK_KINDS)}}, "required": ["task_kind"]}},
    {"name": "manhattan_distance", "description": "Return |x1-x2| + |y1-y2|.", "parameters": {
        "type": "object", "properties": {k: {"type": "integer"} for k in ("x1", "y1", "x2", "y2")},
        "required": ["x1", "y1", "x2", "y2"]}},
]
CALL_RE = re.compile(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", re.S)


def run_calls(src: str, ex: dict) -> list:
    """Parse '[f(a=1), g(b=2)]' safely with ast and execute the whitelisted tools."""
    results = []
    tree = ast.parse(src.strip(), mode="eval")
    calls = tree.body.elts if isinstance(tree.body, ast.List) else [tree.body]
    for c in calls:
        name = c.func.id
        kw = {k.arg: ast.literal_eval(k.value) for k in c.keywords}
        if name == "list_eligible_robots":
            allowed = rt.TASK_KINDS[kw["task_kind"]][1]
            results.append([{"id": r["id"], "x": r["x"], "y": r["y"]} for r in ex["robots"]
                            if r["status"] == "idle" and r["type"] in allowed])
        elif name == "manhattan_distance":
            results.append(rt.manhattan(kw["x1"], kw["y1"], kw["x2"], kw["y2"]))
        else:
            results.append(f"unknown tool {name}")
    return results


model, tok = load_text_model("text_instruct")
model.eval()
rows = read_jsonl(DATA / "robot" / "test.jsonl")[: args.n]
system = (rt.SYSTEM_PROMPT + " You may call the tools to look up eligible robots and compute distances. "
          f"Task kinds: {', '.join(rt.TASK_KINDS)}.")
log, correct, used_tools = [], 0, 0
for ex in rows:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": rt.format_user(ex)}]
    final, transcript = "", []
    for _ in range(args.max_rounds):
        ids = tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True)["input_ids"].to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=120, do_sample=False, temperature=None, top_p=None, top_k=None)
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False).replace("<|im_end|>", "").strip()
        transcript.append(text)
        m = CALL_RE.search(text)
        if not m:
            final = text
            break
        used_tools += 1
        try:
            result = run_calls(m.group(1), ex)
        except Exception as e:  # malformed call: report the error back to the model
            result = f"error: {e}"
        msgs.append({"role": "assistant", "content": text})
        msgs.append({"role": "tool", "content": json.dumps(result)})
    ok = rt.verify(ex, final)["exact"]
    correct += ok
    log.append({"id": ex["id"], "transcript": transcript, "final": final, "exact": ok})
    print(f"{ex['id']} exact={ok} rounds={len(transcript)} final={final[:60]!r}")

plain, _ = evaluate_robot(model, tok, rows)
summary = {"n": len(rows), "tool_exact": correct / len(rows), "plain_exact": plain["exact"],
           "tool_call_rounds_total": used_tools}
print(summary)
write_json(OUTPUTS / "runs" / "ext_tool_calling" / "summary.json", {"summary": summary, "log": log})
