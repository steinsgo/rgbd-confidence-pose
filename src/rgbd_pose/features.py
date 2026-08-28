from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ImageFeatures:
    uv: np.ndarray
    descriptors: np.ndarray
    # Optional dense-feature bookkeeping.  OpenCV features leave these as
    # ``None``; DINOv2 uses them to report how many image patches were kept by
    # an ROI or polygon mask.
    candidate_count: int | None = None
    grid_shape: tuple[int, int] | None = None


def rectangular_roi_mask(
    image_shape: Sequence[int], roi_xyxy: Sequence[int] | None
) -> np.ndarray | None:
    """Build a boolean feature mask from ``(x0, y0, x1, y1)`` pixels.

    The upper bounds are exclusive, matching NumPy slicing.  The mask is
    intended for OpenCV feature extraction; it does not crop or alter the
    image, so returned keypoint coordinates remain in full-image coordinates.
    """
    if roi_xyxy is None:
        return None
    if len(image_shape) < 2:
        raise ValueError("image_shape must contain height and width")
    if len(roi_xyxy) != 4:
        raise ValueError("roi_xyxy must contain (x0, y0, x1, y1)")
    try:
        x0, y0, x1, y1 = (int(value) for value in roi_xyxy)
    except (TypeError, ValueError) as exc:
        raise ValueError("roi_xyxy must contain integer coordinates") from exc
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must have positive height and width")
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(
            f"ROI {(x0, y0, x1, y1)} must lie inside image bounds "
            f"(0, 0, {width}, {height})"
        )
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def polygon_mask(
    image_shape: Sequence[int],
    polygon_xy: Sequence[Sequence[float]] | Sequence[float] | np.ndarray | None,
) -> np.ndarray | None:
    """Build a boolean feature mask from polygon vertices in image pixels.

    ``polygon_xy`` may be an ``N x 2`` vertex array or a flat sequence of
    ``x0, y0, x1, y1, ...`` values.  Vertices are rounded to the nearest
    pixel, must be inside the image, and must contain at least three points.
    The image itself is not cropped or modified; keypoint coordinates remain
    in full-image coordinates.
    """
    if polygon_xy is None:
        return None
    if len(image_shape) < 2:
        raise ValueError("image_shape must contain height and width")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must have positive height and width")
    try:
        vertices = np.asarray(polygon_xy, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("polygon_xy must contain numeric coordinates") from exc
    if vertices.ndim == 1:
        if vertices.size < 6 or vertices.size % 2 != 0:
            raise ValueError(
                "polygon_xy must contain at least three x,y pairs"
            )
        vertices = vertices.reshape(-1, 2)
    elif vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("polygon_xy must have shape (N, 2) or be a flat x,y sequence")
    if vertices.shape[0] < 3:
        raise ValueError("polygon_xy must contain at least three vertices")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("polygon_xy must contain only finite coordinates")
    if not np.all(
        (vertices[:, 0] >= 0)
        & (vertices[:, 0] < width)
        & (vertices[:, 1] >= 0)
        & (vertices[:, 1] < height)
    ):
        raise ValueError(
            "polygon vertices must lie inside image bounds "
            f"(0, 0, {width}, {height})"
        )

    rounded = np.rint(vertices).astype(np.int32)
    if not np.all(
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    ):
        raise ValueError(
            "rounded polygon vertices must lie inside image bounds "
            f"(0, 0, {width}, {height})"
        )

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install opencv-python to rasterize polygon masks") from exc
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [rounded], 1)
    return mask.astype(bool)


def build_feature_mask(
    image_shape: Sequence[int],
    *,
    roi_xyxy: Sequence[int] | None = None,
    polygon_xy: Sequence[Sequence[float]] | Sequence[float] | np.ndarray | None = None,
) -> np.ndarray | None:
    """Build an optional feature mask, intersecting rectangle and polygon.

    Supplying both forms is allowed and restricts features to their
    intersection.  ``None`` is returned when neither restriction is given.
    """
    masks: list[np.ndarray] = []
    rectangle = rectangular_roi_mask(image_shape, roi_xyxy)
    polygon = polygon_mask(image_shape, polygon_xy)
    if rectangle is not None:
        masks.append(rectangle)
    if polygon is not None:
        masks.append(polygon)
    if not masks:
        return None
    result = masks[0].copy()
    for item in masks[1:]:
        result &= item
    return result


def intersect_feature_masks(
    *masks: np.ndarray | None,
) -> np.ndarray | None:
    """Intersect optional boolean masks while preserving ``None`` semantics."""

    present = [np.asarray(mask, dtype=bool) for mask in masks if mask is not None]
    if not present:
        return None
    shape = present[0].shape
    if any(mask.shape != shape for mask in present[1:]):
        raise ValueError("feature masks must have identical shapes")
    result = present[0].copy()
    for mask in present[1:]:
        result &= mask
    return result


def load_feature_mask(
    path: str | Path,
    image_shape: Sequence[int],
) -> np.ndarray:
    """Load a binary or grayscale mask image without changing the source."""

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install Pillow to load feature masks") from exc
    mask_path = Path(path)
    if not mask_path.is_file():
        raise FileNotFoundError(f"Feature mask file not found: {mask_path}")
    with Image.open(mask_path) as image:
        mask = np.asarray(image.convert("L")) > 0
    expected_shape = (int(image_shape[0]), int(image_shape[1]))
    if mask.shape != expected_shape:
        raise ValueError(
            f"Feature mask resolution mismatch: {mask.shape} vs {expected_shape}"
        )
    return mask.astype(bool)


def extract_opencv_features(
    image_rgb: np.ndarray,
    method: str = "sift",
    mask: np.ndarray | None = None,
    max_features: int = 2000,
) -> ImageFeatures:
    """Extract normalized SIFT or ORB descriptors."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install opencv-python for the OpenCV baseline") from exc
    image = np.asarray(image_rgb, np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must have shape HxWx3")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if mask is None:
        cv_mask = None
    else:
        mask_array = np.asarray(mask, bool)
        if mask_array.shape != image.shape[:2]:
            raise ValueError(
                "feature mask must have the same height and width as image_rgb"
            )
        cv_mask = mask_array.astype(np.uint8) * 255
    if method.lower() == "sift":
        detector = cv2.SIFT_create(nfeatures=max_features)
    elif method.lower() == "orb":
        detector = cv2.ORB_create(nfeatures=max_features)
    else:
        raise ValueError("method must be 'sift' or 'orb'")
    keypoints, descriptors = detector.detectAndCompute(gray, cv_mask)
    if descriptors is None or len(keypoints) == 0:
        return ImageFeatures(np.empty((0, 2)), np.empty((0, 0)))
    uv = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    descriptors = descriptors.astype(np.float32)
    descriptors /= np.maximum(np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-8)
    return ImageFeatures(uv, descriptors)


def load_dinov2_model(
    model_name: str = "dinov2_vits14",
    device: str = "cuda",
):
    """Load one DINOv2 model on an explicitly requested device.

    The caller should reuse the returned model for both images in a pair.  A
    CUDA request fails clearly when CUDA is unavailable instead of silently
    falling back to CPU.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install torch and torchvision for DINOv2 features") from exc

    actual_device = torch.device(device)
    if actual_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "DINOv2 requested CUDA, but torch.cuda.is_available() is false; "
            "run with --dino-device cpu or fix the WSL CUDA installation"
        )
    # Avoid the unauthenticated GitHub API repository-validation request.  It
    # is rate-limited in restricted/networked environments; the hub loader
    # still fetches the named official repository when validation is skipped.
    model = torch.hub.load(
        "facebookresearch/dinov2",
        model_name,
        skip_validation=True,
    )
    model = model.eval().to(actual_device)
    return model, str(actual_device)


def extract_dinov2_patch_features(
    image_rgb: np.ndarray,
    model_name: str = "dinov2_vits14",
    device: str = "cuda",
    max_side: int = 560,
    *,
    model=None,
    mask: np.ndarray | None = None,
) -> ImageFeatures:
    """Extract DINOv2 patch tokens and their pixel-center coordinates.

    A supplied model is reused without reloading.  ``mask`` is applied at
    patch-center coordinates after inference, preserving coordinates in the
    original image.  The first call without a model downloads official
    pretrained weights through torch.hub.
    """
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("Install torch and torchvision for DINOv2 features") from exc

    # Pillow-backed arrays can be read-only; make a contiguous writable copy
    # before handing the buffer to torch.from_numpy.
    image = np.array(image_rgb, dtype=np.uint8, copy=True)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must have shape HxWx3")
    if max_side < 14:
        raise ValueError("max_side must be at least one DINOv2 patch")
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    resized_h = max(14, int(height * scale) // 14 * 14)
    resized_w = max(14, int(width * scale) // 14 * 14)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    tensor = functional.interpolate(tensor, (resized_h, resized_w), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    if model is None:
        model, actual_device = load_dinov2_model(model_name, device)
    else:
        actual_device = str(torch.device(device))
    with torch.inference_mode():
        output = model.forward_features(tensor.to(actual_device))
        tokens = output["x_norm_patchtokens"][0].float().cpu().numpy()

    grid_h, grid_w = resized_h // 14, resized_w // 14
    yy, xx = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
    uv = np.stack([(xx.ravel() + 0.5) * 14 / scale, (yy.ravel() + 0.5) * 14 / scale], axis=1)
    if tokens.shape[0] != uv.shape[0]:
        raise RuntimeError(
            "DINOv2 patch-token count does not match the expected patch grid"
        )
    if mask is not None:
        mask_array = np.asarray(mask, bool)
        if mask_array.shape != image.shape[:2]:
            raise ValueError(
                "DINOv2 patch mask must have the same height and width as image_rgb"
            )
        pixel_x = np.clip(np.rint(uv[:, 0]).astype(int), 0, width - 1)
        pixel_y = np.clip(np.rint(uv[:, 1]).astype(int), 0, height - 1)
        keep = mask_array[pixel_y, pixel_x]
        uv = uv[keep]
        tokens = tokens[keep]
    return ImageFeatures(
        uv.astype(np.float32),
        tokens.astype(np.float32),
        candidate_count=int(grid_h * grid_w),
        grid_shape=(int(grid_h), int(grid_w)),
    )
