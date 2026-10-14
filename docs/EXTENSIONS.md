# Extensions

Two kinds of extension, clearly separated:

* **Runnable**: code exists in this repository and was executed at least once on this machine
  (results below are measured).
* **Roadmap**: ideas with a concrete plan, not implemented here.

## Runnable extensions

### 1. QLoRA (4-bit base + LoRA)

`scripts/text/10_sft.py --set load_in_4bit=true` loads the frozen base in NF4 with double
quantisation (bitsandbytes) and trains LoRA adapters on top.

**When it is justified.** Measured in Lesson C: BF16 LoRA on LFM2.5-350M peaks at 3.2 GB and
even the 1.2B teacher fits in 2.9 GB with gradient checkpointing. On this 8 GB GPU QLoRA is
therefore *not needed* for these models. It is included so you can measure the trade-off
yourself and because it becomes necessary for 3B+ models.

**Measured** (`outputs/runs/text_sft/qlora_quick`, quick preset, gradient checkpointing on, single seed):

| | BF16 LoRA (Lesson C) | QLoRA (NF4 base) |
|---|---|---|
| peak allocated / reserved | 3.19 / 4.79 GB | **0.91 / 2.55 GB** |
| training time (1,000 examples) | 39 s | 61 s |
| test exact / robot | 0.097 / 0.717 | 0.117 / 0.700 |

Memory drops by ~70% with similar accuracy (differences are within single-seed noise), at ~55%
more time (de-quantisation on every forward pass, plus checkpointing).

**A pitfall the run exposed.** The reload check loads the saved adapter onto the ordinary **BF16**
base model: only **33 of 64** greedy outputs matched the in-training (4-bit base) outputs. The adapter
was trained to correct the *quantised* weights; on a different-precision base the same adapter
behaves differently. Serve a QLoRA adapter on the same 4-bit base it was trained with, or
re-evaluate after merging.

### 2. Tool calling

`scripts/extensions/tool_calling.py`: LFM2.5-350M may call `list_eligible_robots(task_kind)`
and `manhattan_distance(x1, y1, x2, y2)` in its native Pythonic tool-call format
(`<|tool_call_start|>[...]<|tool_call_end|>`). Calls are parsed with `ast` (never `eval`),
executed, and the results returned with the `tool` role.

**Measured** (40 test scenes, zero-shot LFM2.5-350M, greedy): exact **0.00 with tools vs 0.05 without**.
The model does emit well-formed tool calls (31 call rounds), but it uses them poorly. Typical
transcript: it calls `manhattan_distance(x1=5, y1=9, x2=5, y2=9)` (the task's coordinates twice),
receives 0, and answers in prose ("The Manhattan distance … is **0**.") instead of the required JSON.
Tool *syntax* is not tool *strategy*: a 350M model needs fine-tuning on tool-use trajectories for this
(a good follow-up project: generate verified trajectories with the solver and train on them).

### 3. Retrieval-augmented generation (RAG)

`scripts/extensions/rag_glossary.py`: BM25 retrieval (20 lines of Python) over this lab's
glossary, top-3 chunks in the system prompt, compared with no retrieval on 8 questions
with keyword checks.

**Measured** (8 questions, greedy): keyword accuracy **1/8 without retrieval, 5/8 with retrieval**;
for 7 of 8 questions a retrieved chunk contained every required keyword.

**A bug worth knowing about.** The first version split the glossary into one chunk per *line*.
Most glossary bullets wrap over several lines, so the facts on continuation lines (e.g. "16 query
heads and 8 KV heads") were never retrievable, and RAG scored only 2/8. Joining each bullet's
continuation lines fixed retrieval. Always print what the retriever returns before blaming the
generator.

Remaining misses: "why does LoRA's B start at zero?" (the answer misses the point that training
starts at the original model), "what does an equal-reward group contribute?" (wrong: it contributes
*zero* gradient), and "what vocabulary size does LFM2.5 have?", where the RAG answer "64,402" is
the tokenizer size given in the glossary but the keyword check expected 65,536 (the logit
dimension). That last one is a weakness of keyword checking, left unchanged on purpose:
editing the questions after seeing the answers would tune the evaluation to the model.
Eight questions are far too few for a reliable estimate.

### 4. Reward design

`lab/robot_task.py::reward` (shaped / binary) and `lab/shapes_task.py::reward` are plain
functions; `--set reward_mode=binary` already runs. Exercise F.4 asks you to remove the 0.1
"eligible but wrong" tier and compare equal-reward fractions.

### 5. Architecture comparison

`scripts/text/02_inspect_architecture.py --model text_small` prints the same report for
SmolLM2-135M (pure Llama-style Transformer) so you can compare a hybrid conv/attention
model with a pure-attention one: parameter split, cache size, LoRA target names.

## Roadmap (not implemented)

* **Post-training quantisation for deployment**: export the merged SFT model to GGUF
  (llama.cpp) and compare accuracy and CPU latency at Q8/Q4. Liquid publishes GGUF exports of
  the base models; the comparison would need the merged fine-tuned weights.
* **Thinking-teacher distillation at scale**: Qwen3.5-2B with thinking reached 87.5% on 40
  test scenes but costs ~6 s and ~1,450 tokens per label; 1,000 labels ≈ 1.6 h on this GPU.
  Distil only the final answers into the 350M student and compare with gold SFT.
* **Full PPO with a value head**, to compare its stability and memory with GRPO.
* **Learned reward model**: train a small classifier on verifier-labelled pairs and use it
  instead of the verifier in GRPO; then measure reward hacking against the true verifier.
* **More complex robotics examples**: multi-robot plans (assign K tasks to N robots, minimise
  total distance), with the plan checked by a verifier; or ROS 2 integration where a node calls
  the fine-tuned model to dispatch simulated robots. The verifier-first design of this lab
  transfers directly.
* **VLM grounding**: train the bounding-box output format the LFM2.5-VL card documents
  (`[{"label", "bbox"}]`) on the shape scenes, and score it with IoU.
