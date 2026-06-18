# ComfyUI-Image-Safety-Gate

ComfyUI custom node combining three safety signals with boolean OR. Tuned for
workflows that generate **illustration / anime-style** content.

1. [SmilingWolf wd-tagger v3 family](https://huggingface.co/SmilingWolf) (ONNX)
   — WaifuDiffusion tagger v3. Trained on Danbooru illustrations. Returns
   rating probabilities `general / sensitive / questionable / explicit`. We
   treat `questionable + explicit` as the unsafe signal. Five variants are
   selectable from the node (see [WD tagger variants](#wd-tagger-variants)).
2. [CompVis/stable-diffusion-safety-checker](https://huggingface.co/CompVis/stable-diffusion-safety-checker)
   — CLIP-based concept similarity checker (same model the previous
   `ComfyUI-safety-checker` used). Sensitivity slider matches the old node.
3. [OwenElliott/image-safety-classifier-s](https://huggingface.co/OwenElliott/image-safety-classifier-s)
   — SwiftFormer-based NSFL / NSFW / SFW classifier. We use **only the NSFL
   dimension** (gore / violence). The NSFW dimension over-triggers on
   illustrations, but the NSFL dimension empirically stays calm on
   illustrations and lights up on actual gore / violence.

Why this stack:
- WD tagger covers anime-style NSFW (sexual content) without illustration
  false positives.
- CLIP catches concept matches that WD might miss.
- image-safety-classifier-s NSFL covers gore / violence, which WD does not
  separate cleanly (Danbooru classifies blood / killing under "safe" rating).

OR keeps the policy conservative: block if any of the three flags the image.

## Output

- `IMAGE`: input image, unmodified (no censoring is applied here)
- `nsfw` (BOOLEAN): `True` if **any** of WD tagger, CLIP, or NSFL flags it
  as unsafe

For multi-image batches, `nsfw` is `True` if any image in the batch is flagged.

## Inputs

| name              | type             | default        | notes                                                                 |
|-------------------|------------------|----------------|-----------------------------------------------------------------------|
| `images`          | IMAGE            | -              | Standard ComfyUI image tensor                                         |
| `wd_variant`      | choice           | `eva02-large`  | Which WD tagger v3 variant to use (see table below)                   |
| `sensitivity`     | FLOAT            | `0.6`          | CLIP safety checker sensitivity (matches the old node value)          |
| `nsfw_threshold`  | FLOAT            | `0.35`         | WD tagger threshold on `rating_questionable + rating_explicit`        |
| `nsfl_threshold`  | FLOAT            | `0.5`          | image-safety-classifier-s threshold on NSFL probability               |

### WD tagger variants

All variants are from SmilingWolf's v3 series. Larger models are slower /
heavier but more accurate. All produce the same rating layout, so swapping
variants does not require threshold retuning.

| `wd_variant`   | repo                                                                                  | size  | typical use                |
|----------------|---------------------------------------------------------------------------------------|-------|----------------------------|
| `vit`          | [SmilingWolf/wd-vit-tagger-v3](https://huggingface.co/SmilingWolf/wd-vit-tagger-v3)             | ~400MB | fastest, low VRAM         |
| `convnext`     | [SmilingWolf/wd-convnext-tagger-v3](https://huggingface.co/SmilingWolf/wd-convnext-tagger-v3)   | ~400MB | alternative to `vit`       |
| `swinv2`       | [SmilingWolf/wd-swinv2-tagger-v3](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3)       | ~400MB | alternative to `vit`       |
| `vit-large`    | [SmilingWolf/wd-vit-large-tagger-v3](https://huggingface.co/SmilingWolf/wd-vit-large-tagger-v3) | ~1.1GB | middle ground              |
| `eva02-large`  | [SmilingWolf/wd-eva02-large-tagger-v3](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3) | ~1.6GB | most accurate (default)    |

The selected variant is downloaded on first use into
`ComfyUI/models/safety_checker/wd-<variant>-tagger-v3/`. Each variant is
cached separately, so switching does not re-download.

### Sensitivity / threshold reference

CLIP sensitivity:

- `0.0`: least sensitive
- `0.5`: explicit nudity threshold
- `1.0`: most sensitive (catches lingerie-like content)

WD tagger threshold (sum of `questionable + explicit`):

- `>= 0.50`: high confidence NSFW
- `0.30 - 0.50`: borderline / sensitive content
- `< 0.30`: likely SFW

NSFL threshold (image-safety-classifier-s NSFL dimension):

- `>= 0.50`: clear gore / dismemberment / corpse content (default)
- `0.15 - 0.50`: light blood / wound content (lower catches more)
- `< 0.15`: essentially no violence signal — most illustration content sits
  here regardless of depicted scene

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/monkeykim111/ComfyUI-Image-Safety-Gate
pip install -r ComfyUI-Image-Safety-Gate/requirements.txt
```

Restart ComfyUI.

## Model Files

On first use, the three models download into the standard `safety_checker`
folder:

```text
ComfyUI/models/safety_checker/wd-<variant>-tagger-v3/
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

If files already exist (e.g., previously downloaded for
`ComfyUI-safety-checker` or `ComfyUI-image-safety-classifier`), no network
access is made.

## Notes

- All models are cached in memory per process. WD variants are cached per
  variant key, so toggling between variants in a session does not reload.
- WD tagger runs via `onnxruntime` (CPU or GPU automatically depending on the
  installed provider). CLIP runs on CPU to match the original node's exact
  numerical behaviour. NSFL runs on GPU when CUDA is available.
- Per-image log line shows each signal's decision and the final OR result,
  so disagreement cases are visible for later threshold tuning.
