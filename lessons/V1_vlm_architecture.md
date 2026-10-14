# Lesson V1: How LFM2.5-VL-450M sees an image

Code: `scripts/vlm/01_inspect_and_infer.py` (inspection + demo), `scripts/vlm/infer.py`
(your own image and question), `scripts/vlm/02_resolution_sweep.py` (resolution and
image-token settings). All shapes below were printed by the installed model
(revision `fc6221c`, transformers 5.19.0).

## 1. Concept and purpose

A vision-language model is a language model whose input sequence contains some
"words" that are really pieces of an image. Four parts make that happen:

1. **Image processor**: resizes/tiles the image and cuts it into 16×16-pixel patches.
2. **Vision encoder** (SigLIP2 NaFlex, 86M parameters): a Transformer over patches
   that outputs one 768-dim feature per patch.
3. **Connector / projector** (8.4M): merges 2×2 neighbouring patch features
   ("pixel unshuffle") and maps them into the language model's 1024-dim embedding space.
4. **Language model** (LFM2.5-350M backbone): reads text embeddings with the
   projected image features spliced in at the positions of `<image>` placeholder
   tokens, and generates the answer as usual.

## 2. Equations

Image `I` of height `H` and width `W` (pixels), patch size `p = 16`, downsample factor `s = 2`:

```
patch grid           h = H/p,  w = W/p                          (256×256 image -> 16×16 = 256 patches)
patch vector         x_ij = flatten(I[p·i : p·(i+1), p·j : p·(j+1), :]) ∈ R^{p·p·3 = 768}
vision features      F = SigLIP2(x) ∈ R^{h·w × 768}
pixel unshuffle      F' ∈ R^{(h/s)·(w/s) × 768·s²}   (each 2×2 block of features concatenated: 3072 dims)
projector            Z = W₂ GELU(W₁ F' + b₁) + b₂ ∈ R^{(h/s)(w/s) × 1024},   W₁: 3072→2048, W₂: 2048→1024
image tokens         N_img = ⌈h/s⌉ · ⌈w/s⌉                    (256×256 -> 8×8 = 64 tokens)
```

The language model then sees the embedding sequence
`[e(<|startoftext|>), …, e(<|image_start|>), Z_1 … Z_64, e(<|image_end|>), e(question tokens) …]`
and is trained/evaluated exactly like Lesson A's text model.

**Tiling.** Images larger than the 512×512 tile (with `do_image_splitting=True`) are cut
into up to 10 tiles of 512×512 (each 32×32 patches → 16×16 = 256 tokens) plus a
downscaled **thumbnail** for global context. `min_image_tokens` / `max_image_tokens`
bound the token budget of a single (non-tiled) image by resizing it.

## 3. Results from the inspection (`outputs/runs/vlm_lesson_a/vlm/report.json`)

| Component | Class | Parameters |
|---|---|---|
| vision tower | `Siglip2VisionModel` (12 layers, width 768, 12 heads, patch 16) | 85.84M (19.1%) |
| connector | `Lfm2VlMultiModalProjector`: `linear_1` 3072→2048, GELU, `linear_2` 2048→1024 | 8.39M (1.9%) |
| language model | `Lfm2Model` (same architecture as Lesson B1) | 354.48M (79.0%) |

Processor settings actually shipped with the checkpoint: `min_image_tokens = 64`,
`max_image_tokens = 256`, `do_image_splitting = True`, `tile_size = 512`,
`max_tiles = 10`, `use_thumbnail = True`, normalisation mean = std = 0.5. (The model
card lists `min_image_tokens = 32`; the processor file says 64. The lab uses the file's values.)

One 256×256 shape scene plus a question:

```
chat template:  <|startoftext|><|im_start|>user\n<image>Is the red triangle ... ?<|im_end|>\n<|im_start|>assistant\n
input_ids             (1, 92)      64 of the 92 tokens are <image> placeholders
pixel_values          (1, 1024, 768) float32   NaFlex buffer: room for 1024 patches of 16·16·3 values
pixel_attention_mask  (1, 1024)    1 for the 256 real patches, 0 for padding
spatial_shapes        [[16, 16]]   the real patch grid
vision_tower      -> (1, 1024, 768)
projector input   -> (1, 16, 16, 768)   (padding dropped, back to a grid)
projector output  -> (1, 8, 8, 1024)    = 64 image tokens of LM width
language_model    -> (1, 92, 1024);  lm_head -> (1, 92, 65536)
```

**Demo on that scene** (truth, left to right: blue circle, blue square, red triangle, blue triangle):

| task | model output | verdict |
|---|---|---|
| caption | "Three shapes are on a tan background: a blue circle, a blue square, and a red triangle." | missed one shape |
| "How many shapes?" | 3 | wrong (4) |
| "Leftmost colour?" | Blue | right |
| JSON list | `[{"shape": "circle", "color": "blue"}, {"shape": "rectangle", "color": "blue"}, {"shape": "triangle", "color": "red"}]` | valid JSON; misses a shape; "rectangle" for square |

Peak VRAM for inference: 1.02 GB allocated.

## 4. Resolution, tiling and image-token budget

Same 90 test questions; each scene re-rendered at the given size (vector shapes, so no
blur from upscaling). Base model, greedy, batch 8, untimed warm-up first. Single run:
accuracy differences of ±4 points on 90 questions are noise. File: `results/tables/vlm_resolution_sweep.md`.

| render size | processor setting | image tokens / image | accuracy | ms / question | peak alloc GB |
|---|---|---|---|---|---|
| 128 | model-card defaults | 64.0 | 0.833 | 17.5 | 1.087 |
| 256 | model-card defaults | 64.0 | 0.844 | 17.4 | 1.087 |
| 512 | model-card defaults | 256.0 | 0.889 | 23.5 | 1.079 |
| 768 | model-card defaults | 1280.0 | 0.811 | 117.8 | 1.734 |
| 1024 | model-card defaults | 1280.0 | 0.811 | 118.2 | 1.734 |
| 512 | {'max_image_tokens': 64} | 1088.0 | 0.878 | 115.9 | 1.776 |
| 512 | {'max_image_tokens': 128} | 121.0 | 0.878 | 20.0 | 1.087 |
| 512 | {'max_image_tokens': 64, 'do_image_splitting': False} | 64.0 | 0.867 | 8.5 | 0.963 |
| 1024 | {'do_image_splitting': False} | 256.0 | 0.811 | 24.9 | 1.079 |
| 1024 | {'max_image_tokens': 64, 'do_image_splitting': False} | 64.0 | 0.867 | 9.5 | 0.963 |

Base LFM2.5-VL-450M, greedy, batch 8, 90 test questions, single run. Latency was measured while other GPU/CPU jobs shared the laptop, so treat it as indicative.

**What the numbers show.**

* **Small images are upscaled to the minimum budget.** 128 px and 256 px both become 64
  image tokens (`min_image_tokens = 64`).
* **Up to 512 px, one tile.** 512 px uses 256 tokens (32×32 patches → 16×16 after pixel
  unshuffle) and is about 35% slower than 64 tokens (23.5 vs 17.4 ms).
* **Above 512 px, tiling kicks in.** 768 and 1024 px are cut into 2×2 tiles of 256 tokens
  plus a 256-token thumbnail = **1,280 tokens**, about 7× the latency and 1.7 GB peak memory,
  with **no accuracy gain** on these simple scenes.
* **A counter-intuitive knob.** Setting only `max_image_tokens=64` on a 512 px image *raised*
  the count to **1,088 tokens**: with splitting enabled, an image that exceeds the
  single-image budget is treated as "large" and tiled (4 tiles × 256 + 64-token thumbnail).
  To really shrink the budget, also set `do_image_splitting=False`, which gives 64 tokens and
  the fastest setting measured (8.5 ms/question) at the same accuracy.
* **Hidden padding cost.** Even at 64 image tokens, the default settings took 17.5 ms/question
  versus 8.5 ms with `max_image_tokens=64`. The difference is the NaFlex patch buffer: with
  `max_image_tokens=256` the processor allocates **1,024 patch slots** (`pixel_values (1, 1024, 768)`,
  only 256 real), so the vision encoder runs over 75% padding. With `max_image_tokens=64` the
  buffer is exactly 256 slots. For small images, match the budget to the image.

Nothing here says high resolution is useless: these scenes are large flat shapes. For small
text or fine detail, tiling is what makes the model see it. Measure on *your* images.


## 5. Try it on your own image

```bash
.venv/bin/python scripts/vlm/infer.py --image path/to/photo.jpg --question "What is in this image?"
.venv/bin/python scripts/vlm/infer.py --image data/shapes/images/test-00000.png \
    --question "How many shapes are there? Answer with a number." --max-image-tokens 64
.venv/bin/python scripts/vlm/infer.py --image ... --question ... --adapter outputs/runs/vlm_sft/lora_quick/adapter
```

## 6. Exercise

See `exercises/V_vlm.md` (V.1). Solutions: `solutions/V_vlm.md`.
