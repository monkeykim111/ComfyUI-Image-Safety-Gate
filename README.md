# ComfyUI-Image-Safety-Gate

ComfyUI custom node combining three safety signals with boolean OR. Tuned for
workflows that generate **illustration / anime-style** content.

1. [SmilingWolf/wd-eva02-large-tagger-v3](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3)
   — WaifuDiffusion tagger v3 (ONNX). Trained on Danbooru illustrations.
   Returns rating probabilities: `general`, `sensitive`, `questionable`,
   `explicit`. We treat `questionable + explicit` as the unsafe signal.
2. [CompVis/stable-diffusion-safety-checker](https://huggingface.co/CompVis/stable-diffusion-safety-checker)
   — CLIP-based concept similarity checker (same model the previous
   `ComfyUI-safety-checker` used). Sensitivity slider matches the old node.
3. [OwenElliott/image-safety-classifier-s](https://huggingface.co/OwenElliott/image-safety-classifier-s)
   — SwiftFormer-based NSFL / NSFW / SFW classifier. We use only the **NSFL**
   dimension (gore / violence). The NSFW dimension of this model is biased
   toward real-photo data and over-triggers on illustration content, but its
   NSFL dimension stays calm on illustrations and lights up on actual
   gore / violence — empirically the cleanest illustration-friendly NSFL
   signal available.

Why this stack:
- WD tagger covers anime-style NSFW (sexual content) without illustration
  false positives.
- CLIP covers concept matches that WD might miss.
- image-safety-classifier-s NSFL covers gore / violence, which WD does not
  separate cleanly (Danbooru classifies blood/killing under "safe" rating).

OR keeps the policy conservative: block if any signal flags the image.

## Output

- `IMAGE`: input image, unmodified (no censoring is applied)
- `nsfw` (BOOLEAN): `True` if either WD tagger or CLIP flags it as unsafe

For multi-image batches, `nsfw` is `True` if any image in the batch is flagged.

## Inputs

| name              | type    | default | notes                                                                |
|-------------------|---------|---------|----------------------------------------------------------------------|
| `images`          | IMAGE   | -       | Standard ComfyUI image tensor                                        |
| `sensitivity`     | FLOAT   | `0.6`   | CLIP safety checker sensitivity (matches the old node value)         |
| `nsfw_threshold`  | FLOAT   | `0.35`  | WD tagger threshold on `rating_questionable + rating_explicit`       |
| `nsfl_threshold`  | FLOAT   | `0.5`   | image-safety-classifier-s threshold on NSFL probability              |

Sensitivity reference (CLIP path):

- `0.0`: least sensitive
- `0.5`: explicit nudity threshold
- `1.0`: most sensitive (catches lingerie-like content)

WD tagger rating threshold reference:

- `>= 0.50`: high confidence NSFW
- `0.30 - 0.50`: borderline / sensitive content
- `< 0.30`: likely SFW

NSFL threshold reference (image-safety-classifier-s NSFL dimension):

- `>= 0.50`: clear gore / dismemberment / corpse content (default)
- `0.15 - 0.50`: light blood / wound content (lower threshold catches more)
- `< 0.15`: essentially no violence signal — most illustration content sits
  here regardless of the depicted scene

## Install

```bash
cd ComfyUI/custom_nodes
git clone <this repo> ComfyUI-Image-Safety-Gate
pip install -r ComfyUI-Image-Safety-Gate/requirements.txt
```

Restart ComfyUI.

## Model Files

On first use both models download into:

```text
ComfyUI/models/safety_checker/wd-eva02-large-tagger-v3/
├── model.onnx
└── selected_tags.csv

ComfyUI/models/safety_checker/stable-diffusion-safety-checker/
├── config.json
├── preprocessor_config.json
└── pytorch_model.bin

ComfyUI/models/safety_checker/image-safety-classifier-s/
├── config.json
└── model.safetensors
```

If files already exist (e.g., previously downloaded for `ComfyUI-safety-checker`
or `ComfyUI-image-safety-classifier`), no network access is made.

## Notes

- Both models cached in memory per process.
- WD tagger runs via `onnxruntime` (CPU or GPU automatically depending on the
  installed provider). CLIP runs on CPU to match the original node's behaviour.
- Per-image log line shows each model's decision and the final OR result, so
  disagreement cases are visible for later threshold tuning.
