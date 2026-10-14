# Troubleshooting

## Environment

**`uv venv` fails with "failed to symlink ... Operation not permitted".**
The project is on a filesystem without symlinks (exFAT, FAT32, some network
shares). `scripts/tools/install_env.sh` handles this automatically by building
the venv by hand. If you created `.venv` some other way, delete it and re-run
the script.

**`.venv/bin/<tool>: bad interpreter` after moving the drive.**
Console scripts store the absolute interpreter path. Use
`.venv/bin/python -m <module>` (e.g. `.venv/bin/python -m huggingface_hub.cli.hf`) or
re-run `install_env.sh` at the new location.

**`torch.cuda.is_available()` is False.**
Check `nvidia-smi` works. The driver must support the wheel's CUDA runtime
(12.8 here; any driver ≥ 570 does). Do not install a system CUDA toolkit for
this lab; the wheel brings its own.

**Downloads are very slow or stop.**
Model downloads resume: re-run `make download`. Package downloads resume from
the uv cache: re-run `install_env.sh`.

## GPU memory

**CUDA out of memory.**
1. Run `nvidia-smi` and see what else holds VRAM (other jobs, notebooks, a
   previous Python process that did not exit).
2. Lower `per_device_train_batch_size` and raise `gradient_accumulation_steps`
   by the same factor (the effective batch stays the same).
3. Turn on `--set gradient_checkpointing=true`.
4. For VLMs, lower `max_image_tokens` (fewer image tokens per sample).
5. For GRPO, lower `num_generations` *and* keep
   `per_device_train_batch_size * gradient_accumulation_steps` divisible by it.

**"Inference fits, so training will fit" is false.** Training also stores
activations for the backward pass, gradients, and optimizer state (AdamW keeps
two extra numbers per trainable parameter). The lab prints measured peaks for
both so you can see the gap.

**Reserved memory is much larger than allocated.** PyTorch's caching
allocator keeps freed blocks for reuse. That is normal. `torch.cuda.empty_cache()`
returns them to the driver (the scripts do this after unloading a teacher).

## Training

**Loss is `nan`.** Lower the learning rate; make sure full fine-tuning uses
FP32 master weights (`10_sft.py` does this for `method=full`); check that labels
are not all `-100` (the script prints how many tokens receive loss).

**LoRA loss does not move.** Check the printed gradient check: `grad_norm_B`
must be non-zero. `grad_norm_A` is exactly zero on the very first step
because `B` starts at zero; that is expected. Also check `target_modules`
actually matched (the script prints the wrapped module types).

**GRPO reward is flat and `frac_reward_zero_std` is near 1.** Every completion in
each group got the same reward, so all advantages are zero. Make the task harder
(or start from a weaker checkpoint), raise `temperature`, or use the shaped
reward. See Lesson F.

**DPO loss at step 0 is not ≈ 0.693.** The policy and reference differ at
initialisation; check that the reference is the adapter-disabled model.

## Evaluation

**Scores changed after reloading an adapter.** The SFT script compares greedy
outputs before and after reloading. A few differences can come from BF16
non-determinism in batched kernels; many differences mean the wrong base model
or revision was loaded.

**Base model answers ramble.** Base models do not know the chat format.
`scripts/text/evaluate.py --mode base_fewshot` uses a plain-text 2-shot
prompt and keeps the first line, which is the fair way to query a base model.
