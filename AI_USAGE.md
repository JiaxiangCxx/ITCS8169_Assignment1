# AI Usage

## Tools used
OpenAI Codex and Claude Code.

## How AI assisted me
1. **Which models to try.** The assistants suggested starting from
   ImageNet-pretrained ResNets and trying a ResNet-50 pretrained on Places365,
   a scene-recognition dataset, because our classes are scenes.
2. **How to improve accuracy.** Suggestions included fine-tuning the whole
   backbone instead of freezing it, scene-specific (Places365) pretraining,
   stronger augmentation, ensembling and test-time augmentation.
3. **Implementation and writing.** Codex generated most of the code; Claude Code
   ran the follow-up experiments and drafted the report from the saved results.

## A questionable suggestion
Stronger augmentation (wider crops, color jitter, rotation, random grayscale)
was suggested to reduce overfitting, but it barely changed validation accuracy:
15 of the 16 classes are grayscale, so the color transforms do nothing for them.
An early Codex download command also used `gdown` flags the installed version
did not support.

## Verification
Every reported number comes from the saved result files. The test set was
evaluated once, after the model was chosen, and re-running the final
configuration reproduced the original run exactly.

## A decision I made
The assistant recommended combining several trained models into an ensemble as
the final model. I chose the single Places365 ResNet-50 (96.0% validation
accuracy) instead: a much cheaper model at about one point lower validation
accuracy than the ensemble.
