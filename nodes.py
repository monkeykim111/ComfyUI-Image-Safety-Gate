from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from timm.data import create_transform
from transformers import CLIPConfig, CLIPImageProcessor, CLIPVisionModel, PreTrainedModel

try:
    import folder_paths
except ImportError:
    folder_paths = None


WD_REPO_ID = "SmilingWolf/wd-eva02-large-tagger-v3"
WD_SUBDIR = "wd-eva02-large-tagger-v3"
WD_FILES = ("model.onnx", "selected_tags.csv")
WD_RATING_CATEGORY = 9
WD_UNSAFE_RATING_NAMES = ("questionable", "explicit")

CLIP_REPO_ID = "CompVis/stable-diffusion-safety-checker"
CLIP_SUBDIR = "stable-diffusion-safety-checker"
CLIP_FILES = ("config.json", "preprocessor_config.json", "pytorch_model.bin")

NSFL_REPO_ID = "OwenElliott/image-safety-classifier-s"
NSFL_SUBDIR = "image-safety-classifier-s"
NSFL_FILES = ("config.json", "model.safetensors")
NSFL_CLASS_NAMES = ("NSFL", "NSFW", "SFW")

SAFETY_CHECKER_ROOT_SUBDIR = "safety_checker"

_CACHE: dict = {}
_LOAD_LOCK = Lock()


def get_models_root() -> Path:
    if folder_paths is not None:
        return Path(folder_paths.models_dir)
    return Path(__file__).resolve().parent.parent.parent / "models"


def get_subdir(subdir: str) -> Path:
    return get_models_root() / SAFETY_CHECKER_ROOT_SUBDIR / subdir


def files_present(model_dir: Path, names: tuple[str, ...]) -> bool:
    return all((model_dir / name).is_file() for name in names)


def download_files(repo_id: str, model_dir: Path, names: tuple[str, ...]) -> None:
    from huggingface_hub import hf_hub_download

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[image-safety-gate] Downloading {repo_id} -> {model_dir}")
    for name in names:
        hf_hub_download(repo_id=repo_id, filename=name, local_dir=str(model_dir))


def ensure_files(repo_id: str, model_dir: Path, names: tuple[str, ...]) -> Path:
    if not files_present(model_dir, names):
        download_files(repo_id, model_dir, names)
    if not files_present(model_dir, names):
        missing = ", ".join(names)
        raise FileNotFoundError(
            f"Required files missing in {model_dir} (expected: {missing})."
        )
    return model_dir


def numpy_to_pil_batch(images: torch.Tensor) -> list[Image.Image]:
    array = images.detach().cpu().numpy() if torch.is_tensor(images) else images
    if array.ndim == 3:
        array = array[None, ...]
    array = (array * 255).round().astype("uint8")
    return [Image.fromarray(frame) for frame in array]


def load_wd_tagger():
    if "wd" in _CACHE:
        return _CACHE["wd"]

    with _LOAD_LOCK:
        if "wd" in _CACHE:
            return _CACHE["wd"]

        import onnxruntime as ort

        model_dir = ensure_files(WD_REPO_ID, get_subdir(WD_SUBDIR), WD_FILES)
        tags_df = pd.read_csv(model_dir / "selected_tags.csv")

        rating_rows = tags_df[tags_df["category"] == WD_RATING_CATEGORY]
        rating_indices = {row["name"]: int(idx) for idx, row in rating_rows.iterrows()}
        for name in WD_UNSAFE_RATING_NAMES:
            if name not in rating_indices:
                raise RuntimeError(
                    f"WD tagger CSV missing expected rating '{name}'."
                )
        unsafe_indices = tuple(rating_indices[name] for name in WD_UNSAFE_RATING_NAMES)

        providers = [
            p
            for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if p in ort.get_available_providers()
        ]
        session = ort.InferenceSession(str(model_dir / "model.onnx"), providers=providers)
        input_meta = session.get_inputs()[0]
        input_name = input_meta.name
        _, height, width, _ = input_meta.shape
        target_size = int(height) if isinstance(height, int) else 448

        _CACHE["wd"] = (session, input_name, target_size, unsafe_indices, rating_indices)
        return _CACHE["wd"]


def preprocess_for_wd(pil_image: Image.Image, target_size: int) -> np.ndarray:
    image = pil_image.convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255))
    background.paste(image, mask=image.split()[3])
    image = background.convert("RGB")

    width, height = image.size
    side = max(width, height)
    square = Image.new("RGB", (side, side), (255, 255, 255))
    square.paste(image, ((side - width) // 2, (side - height) // 2))

    if side != target_size:
        square = square.resize((target_size, target_size), Image.BICUBIC)

    array = np.asarray(square, dtype=np.float32)
    array = array[..., ::-1].copy()
    return array


def run_wd_tagger(pil_images: list[Image.Image]) -> np.ndarray:
    session, input_name, target_size, unsafe_indices, _ = load_wd_tagger()
    batch = np.stack([preprocess_for_wd(p, target_size) for p in pil_images], axis=0)
    outputs = session.run(None, {input_name: batch})[0]
    return outputs[:, list(unsafe_indices)]


class ClipSafetyChecker(PreTrainedModel):
    config_class = CLIPConfig
    _no_split_modules = ["CLIPEncoderLayer"]
    _tied_weights_keys: list[str] = []
    all_tied_weights_keys: dict[str, str] = {}

    def __init__(self, config: CLIPConfig):
        super().__init__(config)
        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = nn.Linear(
            config.vision_config.hidden_size,
            config.projection_dim,
            bias=False,
        )
        self.concept_embeds = nn.Parameter(
            torch.ones(17, config.projection_dim),
            requires_grad=False,
        )
        self.special_care_embeds = nn.Parameter(
            torch.ones(3, config.projection_dim),
            requires_grad=False,
        )
        self.concept_embeds_weights = nn.Parameter(torch.ones(17), requires_grad=False)
        self.special_care_embeds_weights = nn.Parameter(torch.ones(3), requires_grad=False)

    @staticmethod
    def compute_cosine_similarity(embeds: torch.Tensor, target_embeds: torch.Tensor) -> torch.Tensor:
        if len(embeds.shape) == 1:
            embeds = embeds.unsqueeze(0)
        if len(target_embeds.shape) == 1:
            target_embeds = target_embeds.unsqueeze(0)
        if embeds.dim() == 2 and target_embeds.dim() == 2:
            embeds = embeds.unsqueeze(1)
        return F.cosine_similarity(embeds, target_embeds, dim=-1)

    def forward(self, clip_input: torch.Tensor, sensitivity: float) -> torch.Tensor:
        image_batch = self.vision_model(clip_input)[1]
        image_embeds = self.visual_projection(image_batch)
        concept_cos_dist = self.compute_cosine_similarity(image_embeds, self.concept_embeds)
        adjusted = -0.1 + 0.14 * sensitivity
        return concept_cos_dist - self.concept_embeds_weights.unsqueeze(0) + adjusted


def load_clip_safety_model():
    if "clip" in _CACHE:
        return _CACHE["clip"]

    with _LOAD_LOCK:
        if "clip" in _CACHE:
            return _CACHE["clip"]

        model_dir = ensure_files(CLIP_REPO_ID, get_subdir(CLIP_SUBDIR), CLIP_FILES)
        processor = CLIPImageProcessor.from_pretrained(str(model_dir))
        model = ClipSafetyChecker.from_pretrained(str(model_dir))
        model.eval()

        _CACHE["clip"] = (model, processor)
        return _CACHE["clip"]


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_nsfl_model():
    if "nsfl" in _CACHE:
        return _CACHE["nsfl"]

    with _LOAD_LOCK:
        if "nsfl" in _CACHE:
            return _CACHE["nsfl"]

        model_dir = ensure_files(NSFL_REPO_ID, get_subdir(NSFL_SUBDIR), NSFL_FILES)
        with (model_dir / "config.json").open("r", encoding="utf-8") as file:
            config = json.load(file)

        architecture = str(config["architecture"])
        num_classes = int(config.get("num_classes") or len(NSFL_CLASS_NAMES))
        pretrained_cfg = dict(config.get("pretrained_cfg") or {})
        class_names = tuple(pretrained_cfg.get("label_names") or NSFL_CLASS_NAMES)

        transform = create_transform(
            input_size=tuple(pretrained_cfg.get("input_size") or (3, 224, 224)),
            is_training=False,
            interpolation=str(pretrained_cfg.get("interpolation") or "bicubic"),
            mean=tuple(pretrained_cfg.get("mean") or (0.485, 0.456, 0.406)),
            std=tuple(pretrained_cfg.get("std") or (0.229, 0.224, 0.225)),
            crop_pct=float(pretrained_cfg.get("crop_pct") or 0.95),
            crop_mode=str(pretrained_cfg.get("crop_mode") or "center"),
        )

        model = timm.create_model(architecture, pretrained=False, num_classes=num_classes)
        model.load_state_dict(load_file(str(model_dir / "model.safetensors"), device="cpu"))
        model.to(resolve_device())
        model.eval()

        nsfl_idx = class_names.index("NSFL") if "NSFL" in class_names else 0
        _CACHE["nsfl"] = (model, transform, nsfl_idx)
        return _CACHE["nsfl"]


def normalize_logits(output):
    if isinstance(output, (tuple, list)):
        tensors = [item for item in output if torch.is_tensor(item)]
        if not tensors:
            raise RuntimeError("Model output did not contain a tensor.")
        output = tensors[0]
    return output


class ImageSafetyGateNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "sensitivity": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "nsfw_threshold": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "nsfl_threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("IMAGE", "nsfw")
    FUNCTION = "nsfw_checker"
    CATEGORY = "image"

    def nsfw_checker(
        self,
        images: torch.Tensor,
        sensitivity: float,
        nsfw_threshold: float,
        nsfl_threshold: float,
    ):
        pil_images = numpy_to_pil_batch(images)

        wd_unsafe_scores = run_wd_tagger(pil_images)

        clip_model, clip_processor = load_clip_safety_model()
        clip_inputs = clip_processor(pil_images, return_tensors="pt")
        with torch.no_grad():
            concept_scores_batch = clip_model(
                clip_inputs.pixel_values,
                sensitivity=sensitivity,
            ).detach().cpu()
        clip_unsafe_batch = (concept_scores_batch > 0).any(dim=-1).tolist()
        clip_max_batch = concept_scores_batch.max(dim=-1).values.tolist()

        nsfl_model, nsfl_transform, nsfl_idx = load_nsfl_model()
        nsfl_device = resolve_device()
        nsfl_inputs = torch.stack(
            [nsfl_transform(p.convert("RGB")) for p in pil_images]
        ).to(nsfl_device)
        with torch.inference_mode():
            nsfl_logits = normalize_logits(nsfl_model(nsfl_inputs))
            nsfl_probs = torch.softmax(nsfl_logits, dim=-1).detach().cpu()
        nsfl_scores = nsfl_probs[:, nsfl_idx].tolist()

        any_unsafe = False
        for i in range(len(pil_images)):
            q_score = float(wd_unsafe_scores[i, 0])
            e_score = float(wd_unsafe_scores[i, 1])
            wd_total = q_score + e_score
            wd_unsafe = wd_total >= nsfw_threshold
            clip_unsafe = bool(clip_unsafe_batch[i])
            clip_max = float(clip_max_batch[i])
            nsfl_score = float(nsfl_scores[i])
            nsfl_unsafe = nsfl_score >= nsfl_threshold

            verdict = wd_unsafe or clip_unsafe or nsfl_unsafe
            any_unsafe = any_unsafe or verdict
            print(
                f"[safety-gate] wd: questionable={q_score:.3f} explicit={e_score:.3f} "
                f"sum={wd_total:.3f}/{nsfw_threshold:.2f} -> "
                f"{'UNSAFE' if wd_unsafe else 'SAFE'} | "
                f"clip: max={clip_max:+.3f} sens={sensitivity:.2f} -> "
                f"{'UNSAFE' if clip_unsafe else 'SAFE'} | "
                f"nsfl: {nsfl_score:.3f}/{nsfl_threshold:.2f} -> "
                f"{'UNSAFE' if nsfl_unsafe else 'SAFE'} | "
                f"OR -> {'UNSAFE' if verdict else 'SAFE'}"
            )

        return (images, any_unsafe)


NODE_CLASS_MAPPINGS = {
    "ImageSafetyGate": ImageSafetyGateNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageSafetyGate": "Image Safety Gate (NSFW)",
}
