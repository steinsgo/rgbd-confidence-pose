"""Confidence-aware RGB-D pose estimation."""

from .geometry import backproject_pixels, transform_points
from .features import build_feature_mask, polygon_mask, rectangular_roi_mask
from .pose import PoseEstimate, estimate_pose_ransac, weighted_rigid_transform
from .realsense_sequence import RealSenseFrame, load_realsense_sequence_frame
from .sequence_quality import QualityThresholds, screen_realsense_session

__all__ = [
    "PoseEstimate",
    "RealSenseFrame",
    "QualityThresholds",
    "backproject_pixels",
    "build_feature_mask",
    "estimate_pose_ransac",
    "load_realsense_sequence_frame",
    "polygon_mask",
    "rectangular_roi_mask",
    "screen_realsense_session",
    "transform_points",
    "weighted_rigid_transform",
]
