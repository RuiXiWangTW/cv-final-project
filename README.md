# cv-final-project

Instructions for Ray:

Load the two VIT classes from the 2 .py files I added.
We need to follow ResFormer training.
So ideally we want to train from multiple resolutions from 64-256 and use a subset of Imagenet that has 64 to 512 resolution images to evaluate.

You can reuse RIViT training / evaluation code to make this work.

Keep model architecture size the same for baseline ViT and SViT. Otherwise it's an unfair comparison. The only thing we can change is the parameters sent to the SPP patch embedder, aka the pyramid level list. I'd also add on additional conv layers to SPP if it seems not good enough (PatchEmbedderSPP class)
SViT can take pyramid level sizes. I'd do [1,2,4,8] as that leads to an exact 64+16+4+1 tokens for each image- or 85 tokens per any given image. You can add on any convolutional layers you think would help if performance isn't looking good. Tacking on a CNN before the SPP layer is a perfectly viable option.


Using resolutions 64x64, 128x128, 192x192, and 256x256 seem like a good idea for training. 64x64, 128x128, 192x192, 256x256, and 512x512 should work for evaluation.
RelativeViT can't train on image sizes bigger than the model was declared for- so set image_size to 256 and only evaluate up to 256x256.
SPPViT can train on anything really, if we're planning on evaluating 512x512 it might be worth doing [1,2,4,8,16] even.

If you have extra time we can evaluate what adding extra SPP levels does, so evaluate [1,2,4] vs [1,2,4,8] vs [1,2,4,8,16] but it might be a bit much training?

Evaluate both on the same dataset after 10 epochs (training is expected to take a while) and save checkpoints & evaluation data. We can then extract a few images from the checkpoints later to include in our blogs.
Thanks for handling the training work- all the 3090s dried up on the slurm cluster for CSAIL holyoke :(
