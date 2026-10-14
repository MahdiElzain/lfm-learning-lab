# Exercises: vision-language lessons

**V.1 Image tokens.** For a 256×256 image, compute the number of image tokens
LFM2.5-VL produces from the patch size (16) and downsample factor (2). Check it
against the inspection script. Do the same for 1024×1024 with tiling on (512
tiles plus thumbnail). Compare with the resolution sweep table.

**V.2 Label leakage.** Why must all questions about one image stay in the same
split? Construct a concrete example of how a model could score well on test by
memorising training images if questions were split randomly.

**V.3 Blank-image prior.** Using the `blank` condition results, which question
type is most guessable without the image? Compare with the majority-answer
baseline.

**V.4 Connector training.** Compare `lora_quick` with
`lora_connector_quick`: trainable parameters, adapter size on disk, accuracy.
Was training the projector worth it on this task?

**V.5 Swapped images.** Explain why "unchanged rate" under swapping is a better
measure of visual dependence than accuracy alone. What value would a model that
ignores the image get?
