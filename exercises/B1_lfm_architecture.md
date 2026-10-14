# Exercises: Lesson B1 (LFM2.5 architecture)

**B1.1 Parameter accounting.** From `architecture.json`, compute by hand the
parameter count of one conv layer and one attention layer from their weight
shapes. Check against the printed per-layer totals.

**B1.2 Cache size.** Using the config values, compute the generation-cache
memory (BF16) at 4,096 tokens for LFM2.5-350M. Then compute it for a
hypothetical *pure-attention* model with the same width and 16 attention
layers. What is the ratio?

**B1.3 Embedding share.** What fraction of LFM2.5-350M's parameters sit in the
embedding matrix? What does that imply for LoRA, which (by default here) does
not touch embeddings?
