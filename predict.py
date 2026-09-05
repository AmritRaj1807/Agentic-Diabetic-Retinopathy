"""
Single-image inference for diabetic retinopathy grading.

This script mirrors the existing Test.py inference path:
    - EfficientNet-B4 + Swin Transformer Base 384 fusion architecture
    - PyTorch checkpoint state-dict loading with strict=True
    - deterministic Resize -> ToTensor -> ImageNet Normalize transform
    - CORN ordinal prediction via coral_pytorch.dataset.corn_label_from_logits

For raw fundus images, it also reuses the existing Preprocessing.py functions
before applying the Test.py tensor transform. Use --input-preprocessed only when
the image has already been processed by Preprocessing.py.

Research/education use only. This model is not a substitute for professional
medical evaluation.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torch import amp
from torchvision import transforms

import timm
from coral_pytorch.dataset import corn_label_from_logits


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "exp_parallel_effb4_swinb384_corn_fusion_head"
    / "models"
    / "model_epoch_015_qwk_0.8943.pth"
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class InferenceConfig:
    """Configuration matching the existing Test.py model and transform."""

    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH
    num_classes: int = 5
    image_size: int = 384
    efficientnet_name: str = "efficientnet_b4.ra2_in1k"
    swin_name: str = "swin_base_patch4_window12_384.ms_in22k_ft_in1k"
    fusion_hidden_dim: int = 1024
    fusion_dropout: float = 0.3
    id2label: dict[int, str] = field(
        default_factory=lambda: {
            0: "No DR",
            1: "Mild",
            2: "Moderate",
            3: "Severe",
            4: "Proliferative DR",
        }
    )


@dataclass
class InferenceBundle:
    """Loaded model, transform, config, and device for repeated predictions."""

    model: nn.Module
    transform: transforms.Compose
    cfg: InferenceConfig
    device: torch.device


class ParallelEfficientNetSwinCORN(nn.Module):
    """
    Dual-backbone CORN model copied from Test.py.

    Module names and layer shapes intentionally match Test.py/Train.py so that
    the saved checkpoint loads with strict=True.
    """

    def __init__(self, cfg: InferenceConfig):
        super().__init__()

        self.efficientnet = timm.create_model(
            cfg.efficientnet_name,
            pretrained=False,
            num_classes=0,
        )

        self.swin = timm.create_model(
            cfg.swin_name,
            pretrained=False,
            num_classes=0,
        )

        eff_dim = getattr(self.efficientnet, "num_features", None)
        swin_dim = getattr(self.swin, "num_features", None)

        if eff_dim is None or swin_dim is None:
            raise ValueError("Could not determine feature dimensions from timm models.")

        fusion_dim = eff_dim + swin_dim

        self.fusion_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, cfg.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.fusion_dropout),
            nn.Linear(cfg.fusion_hidden_dim, cfg.num_classes - 1),
        )

    @staticmethod
    def to_vector(features: torch.Tensor) -> torch.Tensor:
        """Convert backbone outputs into 2D feature vectors, as in Test.py."""

        if features.ndim == 2:
            return features

        if features.ndim == 3:
            return features.mean(dim=1)

        if features.ndim == 4:
            if (
                features.shape[1] > features.shape[-1]
                and features.shape[1] > features.shape[-2]
            ):
                return features.mean(dim=(2, 3))

            return features.mean(dim=(1, 2))

        raise ValueError(f"Unexpected feature shape: {features.shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eff_features = self.to_vector(self.efficientnet(x))
        swin_features = self.to_vector(self.swin(x))

        fused = torch.cat([eff_features, swin_features], dim=1)

        return self.fusion_head(fused)


def build_eval_transform(cfg: InferenceConfig) -> transforms.Compose:
    """Build the exact deterministic tensor transform used by Test.py."""

    return transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def extract_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """
    Extract model weights from the checkpoint formats supported by Test.py.

    Supported formats:
        - {"model_state_dict": ...}
        - {"state_dict": ...}
        - {"model": ...}
        - a raw state_dict
    """

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary.")

    possible_keys = ["model_state_dict", "state_dict", "model"]

    for key in possible_keys:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]

    if all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint

    raise ValueError(
        "Could not find model weights in checkpoint. Expected one of: "
        "'model_state_dict', 'state_dict', 'model', or a raw state_dict."
    )


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove a DataParallel 'module.' prefix, matching Test.py."""

    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value

    return cleaned


def select_device(requested_device: str = "auto") -> torch.device:
    """Select CUDA or CPU, with a useful error if CUDA was explicitly requested."""

    requested = requested_device.lower()

    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("Device must be one of: auto, cuda, cpu.")

    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but it is not available. "
                "Run without --device cuda to use CPU instead."
            )
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(cfg: InferenceConfig, device: torch.device) -> nn.Module:
    """Build the Test.py architecture, load checkpoint weights, and set eval mode."""

    checkpoint_path = Path(cfg.checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}\n"
            "Pass --checkpoint with the correct .pth file, or place the default "
            "checkpoint under outputs/exp_parallel_effb4_swinb384_corn_fusion_head/models/."
        )

    model = ParallelEfficientNetSwinCORN(cfg)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = extract_state_dict(checkpoint)
        state_dict = clean_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model checkpoint: {checkpoint_path}") from exc

    model.to(device)
    model.eval()

    return model


def load_inference_bundle(
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "auto",
) -> InferenceBundle:
    """
    Load the model once for repeated predictions.

    A future Streamlit/FastAPI/Flask app can call this at startup and pass the
    returned bundle into predict_image() to avoid reloading the checkpoint.
    """

    cfg = InferenceConfig(
        checkpoint_path=Path(checkpoint_path)
        if checkpoint_path is not None
        else DEFAULT_CHECKPOINT_PATH
    )

    selected_device = (
        select_device(device) if isinstance(device, str) else torch.device(device)
    )

    transform = build_eval_transform(cfg)
    model = load_model(cfg, selected_device)

    return InferenceBundle(
        model=model,
        transform=transform,
        cfg=cfg,
        device=selected_device,
    )


def validate_image_path(image_path: str | Path) -> Path:
    """Validate that an image path exists and has a supported extension."""

    path = Path(image_path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    if path.suffix.lower() not in IMAGE_EXTS:
        raise ValueError(
            f"Unsupported image extension '{path.suffix}'. "
            f"Supported extensions: {', '.join(sorted(IMAGE_EXTS))}"
        )

    return path


def load_preprocessed_fundus_image(image_path: Path, cfg: InferenceConfig) -> Image.Image:
    """
    Apply the existing Preprocessing.py pipeline to a raw fundus image in memory.

    Preprocessing.py returns OpenCV BGR arrays. The model transform expects a PIL
    RGB image, so the final CLAHE image is converted before tensor conversion.
    """

    try:
        import cv2
        from Preprocessing import Config as PreprocessingConfig
        from Preprocessing import preprocess_image
    except Exception as exc:
        raise RuntimeError(
            "Could not import the existing Preprocessing.py pipeline. "
            "Install the project requirements, including opencv-python-headless."
        ) from exc

    pre_cfg = PreprocessingConfig()
    pre_cfg.image_size = (cfg.image_size, cfg.image_size)
    pre_cfg.crop_mode = "auto"
    pre_cfg.output_ext = ".jpg"

    try:
        processed = preprocess_image(image_path, pre_cfg)
        image_bgr = processed["clahe"]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    except Exception as exc:
        raise RuntimeError(f"Failed to preprocess image with Preprocessing.py: {image_path}") from exc

    return Image.fromarray(image_rgb)


def load_rgb_image(image_path: Path) -> Image.Image:
    """Load an already-preprocessed image with PIL and convert it to RGB."""

    try:
        with Image.open(image_path) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise RuntimeError(f"Unsupported or corrupt image file: {image_path}") from exc


def prepare_image_tensor(
    image_path: str | Path,
    bundle: InferenceBundle,
    apply_fundus_preprocessing: bool = True,
) -> torch.Tensor:
    """Load, preprocess, transform, and batch one image for model inference."""

    path = validate_image_path(image_path)

    if apply_fundus_preprocessing:
        image = load_preprocessed_fundus_image(path, bundle.cfg)
    else:
        image = load_rgb_image(path)

    tensor = bundle.transform(image).unsqueeze(0)

    return tensor.to(bundle.device, non_blocking=bundle.device.type == "cuda")


def corn_class_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert CORN ordinal logits into class probabilities.

    The four sigmoid outputs are interpreted as conditional probabilities of
    passing each ordinal threshold. For five classes:
        P(class 0) = 1 - P(y > 0)
        P(class k) = P(y > 0) * ... * P(y > k-1) * (1 - P(y > k))
        P(class 4) = P(y > 0) * ... * P(y > 3)
    """

    threshold_probs = torch.sigmoid(logits.float())
    num_classes = threshold_probs.shape[1] + 1

    class_probs = [1.0 - threshold_probs[:, 0]]
    cumulative = threshold_probs[:, 0]

    for class_idx in range(1, num_classes - 1):
        class_probs.append(cumulative * (1.0 - threshold_probs[:, class_idx]))
        cumulative = cumulative * threshold_probs[:, class_idx]

    class_probs.append(cumulative)

    return torch.stack(class_probs, dim=1)


def prediction_dict(
    image_path: Path,
    logits: torch.Tensor,
    pred_class: int,
    cfg: InferenceConfig,
) -> dict[str, Any]:
    """Build a stable dictionary suitable for web UI integration."""

    class_probs = corn_class_probabilities(logits)
    ordinal_probs = torch.sigmoid(logits.float())

    class_scores = {
        str(index): float(score)
        for index, score in enumerate(class_probs.squeeze(0).detach().cpu().tolist())
    }

    threshold_scores = {
        f"P(level > {index})": float(score)
        for index, score in enumerate(ordinal_probs.squeeze(0).detach().cpu().tolist())
    }

    confidence = class_scores[str(pred_class)]

    return {
        "image": image_path.name,
        "image_path": str(image_path),
        "predicted_class": pred_class,
        "predicted_label": cfg.id2label[pred_class],
        "confidence": confidence,
        "class_scores": class_scores,
        "class_score_type": "corn_derived_class_probabilities",
        "ordinal_probabilities": threshold_scores,
        "logits": [float(value) for value in logits.squeeze(0).detach().cpu().tolist()],
        "disclaimer": (
            "Research/education use only. This model is not a substitute for "
            "professional medical evaluation."
        ),
    }


def predict_image(
    image_path: str | Path,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "auto",
    bundle: InferenceBundle | None = None,
    apply_fundus_preprocessing: bool = True,
) -> dict[str, Any]:
    """
    Predict the diabetic retinopathy grade for one fundus image.

    Args:
        image_path: Path to an input fundus image.
        checkpoint_path: Optional checkpoint override. Ignored if bundle is supplied.
        device: "auto", "cuda", "cpu", or a torch.device. Ignored if bundle is supplied.
        bundle: Optional preloaded InferenceBundle for repeated predictions.
        apply_fundus_preprocessing: If True, reuse Preprocessing.py before Test.py transform.

    Returns:
        Dictionary with predicted class, label, confidence, CORN-derived class
        probabilities, ordinal threshold probabilities, and raw logits.
    """

    path = validate_image_path(image_path)

    if bundle is None:
        bundle = load_inference_bundle(
            checkpoint_path=checkpoint_path,
            device=device,
        )

    image_tensor = prepare_image_tensor(
        path,
        bundle=bundle,
        apply_fundus_preprocessing=apply_fundus_preprocessing,
    )

    use_amp = bundle.device.type == "cuda"

    with torch.inference_mode():
        with amp.autocast(device_type=bundle.device.type, enabled=use_amp):
            logits = bundle.model(image_tensor)

        pred_tensor = corn_label_from_logits(logits.float())

    pred_class = int(pred_tensor.item())

    return prediction_dict(
        image_path=path,
        logits=logits.float(),
        pred_class=pred_class,
        cfg=bundle.cfg,
    )


def format_prediction(result: dict[str, Any], device: torch.device) -> str:
    """Format one prediction for readable terminal output."""

    pred_class = int(result["predicted_class"])
    pred_label = result["predicted_label"]
    class_scores = result["class_scores"]

    lines = [
        "=" * 50,
        "DIABETIC RETINOPATHY SCREENING PREDICTION",
        "=" * 50,
        "",
        f"Image        : {result['image']}",
        f"Device       : {device.type.upper()}",
        f"Predicted DR : {pred_class} - {pred_label}",
        "",
        "CORN-derived class probabilities:",
    ]

    labels = InferenceConfig().id2label

    for class_id, label in labels.items():
        score = class_scores[str(class_id)] * 100.0
        lines.append(f"  {class_id} - {label:<17}: {score:6.2f}%")

    lines.extend(
        [
            "",
            f"Confidence   : {result['confidence'] * 100.0:.2f}%",
            "",
            result["disclaimer"],
            "",
            "=" * 50,
        ]
    )

    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Predict diabetic retinopathy grade for one or more fundus images.",
    )

    parser.add_argument(
        "--image",
        nargs="+",
        required=True,
        help="Path(s) to input fundus image file(s).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to model checkpoint. Defaults to the known best checkpoint.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Inference device. Default: auto.",
    )
    parser.add_argument(
        "--input-preprocessed",
        action="store_true",
        help=(
            "Use this only when the image has already been processed by "
            "Preprocessing.py. Otherwise the raw fundus preprocessing is applied."
        ),
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""

    args = parse_args(argv)

    try:
        bundle = load_inference_bundle(
            checkpoint_path=args.checkpoint,
            device=args.device,
        )

        for index, image_path in enumerate(args.image):
            result = predict_image(
                image_path=image_path,
                bundle=bundle,
                apply_fundus_preprocessing=not args.input_preprocessed,
            )

            if index > 0:
                print()

            print(format_prediction(result, bundle.device))

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
