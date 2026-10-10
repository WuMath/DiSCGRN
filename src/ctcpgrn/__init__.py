"""CTCP-GRN: continuous-time, change-point GRN inference."""

from .model import FitResult, fit_exact_segment
from .segmentation import SegmentationResult, segment_shared_change_points

__all__ = [
    "FitResult",
    "SegmentationResult",
    "fit_exact_segment",
    "segment_shared_change_points",
]

