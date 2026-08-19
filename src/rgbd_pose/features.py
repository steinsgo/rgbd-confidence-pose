from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImageFeatures:
    uv: np.ndarray
    descriptors: np.ndarray


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
    gray = cv2.cvtColor(np.asarray(image_rgb, np.uint8), cv2.COLOR_RGB2GRAY)
    cv_mask = None if mask is None else (np.asarray(mask, bool).astype(np.uint8) * 255)
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


def extract_dinov2_patch_features(
    image_rgb: np.ndarray,
    model_name: str = "dinov2_vits14",
    device: str = "cuda",
    max_side: int = 560,
) -> ImageFeatures:
    """Extract DINOv2 patch tokens and their pixel-center coordinates.

    The first invocation downloads official pretrained weights through torch.hub.
    """
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("Install torch and torchvision for DINOv2 features") from exc

    image = np.asarray(image_rgb, np.uint8)
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    resized_h = max(14, int(height * scale) // 14 * 14)
    resized_w = max(14, int(width * scale) // 14 * 14)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    tensor = functional.interpolate(tensor, (resized_h, resized_w), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    actual_device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", model_name).eval().to(actual_device)
    with torch.inference_mode():
        output = model.forward_features(tensor.to(actual_device))
        tokens = output["x_norm_patchtokens"][0].float().cpu().numpy()

    grid_h, grid_w = resized_h // 14, resized_w // 14
    yy, xx = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
    uv = np.stack([(xx.ravel() + 0.5) * 14 / scale, (yy.ravel() + 0.5) * 14 / scale], axis=1)
    return ImageFeatures(uv.astype(np.float32), tokens.astype(np.float32))
