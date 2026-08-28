"""Confidence-aware RGB-D pose estimation."""

from .geometry import backproject_pixels, transform_points
from .features import (
    build_feature_mask,
    intersect_feature_masks,
    load_feature_mask,
    load_dinov2_model,
    polygon_mask,
    rectangular_roi_mask,
)
from .dino_quality import feature_mask_kind, summarize_dino_quality
from .pose import (
    PoseEstimate,
    estimate_pose_comparison,
    estimate_pose_ransac,
    summarize_pose_estimate,
    weighted_rigid_transform,
)
from .plane_mask import PlaneModel, fit_plane_from_pixels, tabletop_foreground_mask
from .realsense_sequence import RealSenseFrame, load_realsense_sequence_frame
from .sequence_quality import QualityThresholds, screen_realsense_session

__all__ = [
    "PoseEstimate",
    "PlaneModel",
    "RealSenseFrame",
    "QualityThresholds",
    "backproject_pixels",
    "build_feature_mask",
    "estimate_pose_comparison",
    "estimate_pose_ransac",
    "feature_mask_kind",
    "fit_plane_from_pixels",
    "intersect_feature_masks",
    "load_feature_mask",
    "load_realsense_sequence_frame",
    "load_dinov2_model",
    "polygon_mask",
    "rectangular_roi_mask",
    "screen_realsense_session",
    "summarize_dino_quality",
    "summarize_pose_estimate",
    "tabletop_foreground_mask",
    "transform_points",
    "weighted_rigid_transform",
]
