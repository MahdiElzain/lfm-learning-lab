# Solutions: vision-language lessons

**V.1** 256×256 px, patch 16 → 16×16 = 256 patches; pixel unshuffle by 2 → 8×8 = **64 image
tokens** (the inspection printed 64 of 92 input tokens). 1024×1024 with tiling: 2×2 tiles of
512 px, each 32×32 patches → 16×16 = 256 tokens, so 4 × 256 = 1,024, plus a thumbnail
resized to at most `max_image_tokens` = 256 tokens → **1,280** tokens. The sweep measured
exactly 1,280 for 768 px and 1,024 px. (768 px also becomes 2×2 tiles: tiles are
512 px and the image is resized to a whole number of tiles.)

**V.2** All questions about one image share every pixel. If questions were split randomly,
a test question "How many red shapes?" could be about an image whose training questions
("Is there a red circle?", "What color is the leftmost shape?") the model already saw with
answers. It could then answer from memory of that specific image (e.g. "this background with
these positions has two red shapes") instead of counting. Scene-level splits remove that path.

**V.3** From the blank-image condition, compare `by_type` against the majority-answer rate per
type (`data/shapes/stats.json → test.majority_rate_by_type`). Question types whose correct answer
is often "0"/"no" are the most guessable without an image. After balancing, colour counts
are no longer dominated by "0" (majority answer 39% on train, was 61%).

**V.4** Read `trainable_by_component` and `adapter_mb` from both `metrics.json` files: the
connector variant trains the 8.39M-parameter projector in full on top of the 3.0M LoRA
parameters and its adapter file is larger. Its test accuracy was not better than LoRA-only in
this lab (see `results/tables/vlm_main.md`). For this task the frozen projector already maps the
features well; connector training tends to matter more when the image domain differs strongly
from pre-training.

**V.5** Accuracy alone can be high for a model that guesses priors well. The unchanged rate
asks directly: *if the image changes and the question stays, does the answer change?* A model
that ignores the image gives the same answer for the same question regardless of the image, so
its unchanged rate is **1.0** (100%). A model that looks gives the same answer only when the two
images happen to share the answer (for counts and yes/no questions that happens fairly often by
chance, so even a perfect model is well above 0). The lab's models stay around 0.3.
