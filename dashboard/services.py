"""Service layer for the Streamlit diabetic retinopathy dashboard.

The UI imports this module instead of reaching directly into model internals.
Future model outputs can be added to the dictionaries returned here without
requiring the dashboard layout to be rewritten.
"""

from __future__ import annotations

import csv
import io
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = PROJECT_ROOT / "demo_images"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "confusion_matrix.png"
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "exp_parallel_effb4_swinb384_corn_fusion_head"
    / "models"
    / "model_epoch_015_qwk_0.8943.pth"
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

DR_LABELS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

MODEL_PERFORMANCE = {
    "held_out": {
        "Accuracy": "71.75%",
        "QWK": "0.7813",
        "Micro F1": "0.7175",
        "Weighted F1": "0.7131",
        "Macro F1": "0.6400",
    },
    "internal_validation": {
        "QWK": "0.8943",
    },
    "per_class_f1": {
        "0 - No DR": "0.8034",
        "1 - Mild": "0.4444",
        "2 - Moderate": "0.6772",
        "3 - Severe": "0.7746",
        "4 - Proliferative DR": "0.5000",
    },
    "per_class_recall": {
        "0 - No DR": "0.8103",
        "1 - Mild": "0.4348",
        "2 - Moderate": "0.6957",
        "3 - Severe": "0.8088",
        "4 - Proliferative DR": "0.3500",
    },
}


class DashboardInputError(ValueError):
    """Raised when an uploaded or selected image cannot be used."""


def get_checkpoint_status() -> dict[str, Any]:
    """Return lightweight checkpoint information without loading the model."""

    try:
        from predict import DEFAULT_CHECKPOINT_PATH as predict_checkpoint_path

        checkpoint = Path(predict_checkpoint_path)
    except Exception:
        checkpoint = DEFAULT_CHECKPOINT_PATH

    return {
        "path": str(checkpoint),
        "exists": checkpoint.is_file(),
        "name": checkpoint.name,
    }


def get_demo_images() -> list[dict[str, Any]]:
    """Load optional demo-image metadata from demo_images/demo_labels.csv."""

    labels_path = DEMO_DIR / "demo_labels.csv"
    demos: list[dict[str, Any]] = []

    if labels_path.is_file():
        with labels_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                path = DEMO_DIR / row["filename"]
                if path.is_file():
                    demos.append(
                        {
                            "filename": row["filename"],
                            "path": path,
                            "reference_class": int(row["actual_class"]),
                            "reference_label": row["actual_label"],
                            "source_image": row.get("source_image", ""),
                        }
                    )

    if demos:
        return demos

    return [
        {"filename": path.name, "path": path}
        for path in sorted(DEMO_DIR.glob("*"))
        if path.suffix.lower() in IMAGE_EXTS
    ]


def open_image_from_bytes(file_bytes: bytes) -> Image.Image:
    """Validate and open uploaded image bytes as RGB."""

    try:
        with Image.open(io.BytesIO(file_bytes)) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise DashboardInputError("The uploaded file is not a readable image.") from exc


def validate_image_filename(filename: str) -> None:
    """Validate a user-facing image filename before saving temporary bytes."""

    suffix = Path(filename).suffix.lower()

    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise DashboardInputError("Supported image formats are JPG, JPEG, and PNG.")


def image_metadata(image: Image.Image, filename: str, file_size: int | None) -> dict[str, Any]:
    """Return display metadata for an input image."""

    return {
        "filename": filename,
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "file_size_bytes": file_size,
    }


def _quality_status(value: float, low: float | None, high: float | None) -> str:
    if low is not None and value < low:
        return "Review"
    if high is not None and value > high:
        return "Review"
    return "No issue flagged"


def analyze_image_quality(image: Image.Image) -> dict[str, Any]:
    """Compute preliminary, non-clinical image-quality heuristics."""

    rgb = image.convert("RGB")
    arr = np.asarray(rgb).astype(np.float32)
    gray = np.asarray(rgb.convert("L")).astype(np.float32)

    width, height = rgb.size
    min_side = min(width, height)
    brightness = float(gray.mean())
    contrast = float(gray.std())

    try:
        import cv2

        sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    except Exception:
        gy, gx = np.gradient(gray)
        sharpness = float(np.mean(gx * gx + gy * gy))

    black_pixels = np.all(arr < 12.0, axis=2)
    black_border_fraction = float(black_pixels.mean())

    metrics = [
        {
            "name": "Resolution",
            "value": f"{width} x {height}",
            "status": "No issue flagged" if min_side >= 384 else "Review",
            "detail": "Minimum side is compared with the 384 px model input size.",
        },
        {
            "name": "Brightness",
            "value": f"{brightness:.1f}",
            "status": _quality_status(brightness, 35.0, 220.0),
            "detail": "Mean grayscale intensity, reported as a heuristic only.",
        },
        {
            "name": "Contrast",
            "value": f"{contrast:.1f}",
            "status": "No issue flagged" if contrast >= 25.0 else "Review",
            "detail": "Grayscale standard deviation, reported as a heuristic only.",
        },
        {
            "name": "Sharpness",
            "value": f"{sharpness:.1f}",
            "status": "No issue flagged" if sharpness >= 4.0 else "Review",
            "detail": "Variance of Laplacian or gradient energy, not a clinical quality model.",
        },
        {
            "name": "Dark field proportion",
            "value": f"{black_border_fraction * 100:.1f}%",
            "status": "Review" if black_border_fraction > 0.72 else "No issue flagged",
            "detail": "Fraction of near-black pixels; fundus borders can make this naturally high.",
        },
    ]

    issue_count = sum(1 for metric in metrics if metric["status"] == "Review")
    overall = (
        "Potential quality issue detected"
        if issue_count
        else "Suitable for model analysis"
    )

    return {
        "title": "Preliminary image quality checks",
        "overall": overall,
        "issue_count": issue_count,
        "metrics": metrics,
        "implemented_as": "deterministic heuristics",
        "warnings": [
            "These checks are not a trained clinical image-quality assessment model."
        ],
    }


def load_dashboard_bundle() -> Any:
    """Load the existing inference bundle once for repeated dashboard predictions."""

    from predict import load_inference_bundle

    return load_inference_bundle()


def _predict_path(path: Path, bundle: Any, display_name: str) -> dict[str, Any]:
    from predict import predict_image

    result = predict_image(
        image_path=path,
        bundle=bundle,
        apply_fundus_preprocessing=True,
    )

    result["image"] = display_name
    result["metadata"] = {
        "checkpoint": str(bundle.cfg.checkpoint_path),
        "image_size": bundle.cfg.image_size,
        "device": bundle.device.type,
        "preprocessing": (
            "Existing Preprocessing.py raw fundus preprocessing followed by "
            "Resize -> ToTensor -> ImageNet Normalize."
        ),
        "architecture": (
            "EfficientNet-B4 + Swin Transformer Base 384 + CORN ordinal fusion head"
        ),
    }
    result["available_outputs"] = [
        "fused_corn_prediction",
        "corn_derived_class_scores",
        "ordinal_threshold_probabilities",
        "raw_corn_logits",
    ]
    result["unavailable_outputs"] = [
        "separate_efficientnet_prediction",
        "separate_swin_prediction",
        "model_disagreement",
        "uncertainty_estimation",
        "calibration",
        "grad_cam",
        "transformer_attention",
        "agentic_decision_policy",
    ]

    return result


def predict_uploaded_image(
    file_bytes: bytes,
    filename: str,
    bundle: Any,
) -> dict[str, Any]:
    """Persist uploaded bytes briefly so predict.py can use its real path API."""

    validate_image_filename(filename)
    suffix = Path(filename).suffix.lower()

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="dr_dashboard_",
            suffix=suffix,
            delete=False,
        ) as handle:
            handle.write(file_bytes)
            temp_path = Path(handle.name)

        return _predict_path(temp_path, bundle=bundle, display_name=filename)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def predict_demo_image(path: Path, bundle: Any) -> dict[str, Any]:
    """Run the real predict.py pipeline on a demo image."""

    return _predict_path(path, bundle=bundle, display_name=path.name)
