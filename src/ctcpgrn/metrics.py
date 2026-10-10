"""Evaluation metrics for change points and directed signed networks."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score, roc_auc_score


CHANGE_EDGE_RELATIVE_THRESHOLD = 0.10


def _off_diagonal(matrix: np.ndarray) -> np.ndarray:
    return matrix[~np.eye(matrix.shape[0], dtype=bool)]


def change_edge_labels(
    truth_before: np.ndarray,
    truth_after: np.ndarray,
    relative_threshold: float = CHANGE_EDGE_RELATIVE_THRESHOLD,
) -> np.ndarray:
    """Mark changes exceeding a prespecified fraction of a typical true edge.

    The reference is the median absolute nonzero off-diagonal weight in the
    first regime. If that regime has no edges, use the second regime instead.
    """
    if relative_threshold < 0:
        raise ValueError("relative_threshold must be nonnegative")
    reference = _off_diagonal(np.abs(truth_before))
    active = reference[reference > 0]
    if active.size == 0:
        reference = _off_diagonal(np.abs(truth_after))
        active = reference[reference > 0]
    threshold = relative_threshold * float(np.median(active)) if active.size else 0.0
    return (_off_diagonal(np.abs(truth_after - truth_before)) > threshold).astype(int)


def edge_ranking_metrics(truth: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y_true = (_off_diagonal(np.abs(truth)) > 0).astype(int)
    y_score = _off_diagonal(np.abs(score))
    result = {"aupr": float(average_precision_score(y_true, y_score))}
    result["auroc"] = float(roc_auc_score(y_true, y_score)) if np.unique(y_true).size == 2 else float("nan")
    return result


def binary_edge_metrics(truth: np.ndarray, estimate: np.ndarray, threshold: float) -> dict[str, float]:
    true_edge = _off_diagonal(np.abs(truth) > 0)
    pred_edge = _off_diagonal(np.abs(estimate) >= threshold)
    tp = int(np.sum(true_edge & pred_edge))
    fp = int(np.sum(~true_edge & pred_edge))
    fn = int(np.sum(true_edge & ~pred_edge))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    shd = fp + fn
    correct_sign = np.sum(true_edge & pred_edge & (_off_diagonal(np.sign(truth)) == _off_diagonal(np.sign(estimate))))
    signed_precision = correct_sign / np.sum(pred_edge) if np.sum(pred_edge) else 0.0
    signed_recall = correct_sign / np.sum(true_edge) if np.sum(true_edge) else 0.0
    signed_f1 = (
        2 * signed_precision * signed_recall / (signed_precision + signed_recall)
        if signed_precision + signed_recall
        else 0.0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "signed_f1": float(signed_f1),
        "shd": float(shd),
    }


def edge_change_metrics(truth_before: np.ndarray, truth_after: np.ndarray, est_before: np.ndarray, est_after: np.ndarray) -> dict[str, float]:
    true_change = change_edge_labels(truth_before, truth_after)
    score = _off_diagonal(np.abs(est_after - est_before))
    if np.unique(true_change).size < 2:
        return {"change_aupr": float("nan"), "change_auroc": float("nan")}
    return {
        "change_aupr": float(average_precision_score(true_change, score)),
        "change_auroc": float(roc_auc_score(true_change, score)),
    }


def change_point_metrics(true_times: list[float], estimated_times: list[float], tolerance: float) -> dict[str, float]:
    if not true_times and not estimated_times:
        return {"cp_precision": 1.0, "cp_recall": 1.0, "cp_f1": 1.0, "cp_mae": 0.0}
    if not true_times or not estimated_times:
        return {"cp_precision": 0.0, "cp_recall": 0.0, "cp_f1": 0.0, "cp_mae": float("nan")}
    distances = np.abs(np.asarray(true_times)[:, None] - np.asarray(estimated_times)[None, :])
    rows, cols = linear_sum_assignment(distances)
    accepted = distances[rows, cols] <= tolerance
    tp = int(np.sum(accepted))
    precision = tp / len(estimated_times)
    recall = tp / len(true_times)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mae = float(np.mean(distances[rows[accepted], cols[accepted]])) if tp else float("nan")
    return {"cp_precision": precision, "cp_recall": recall, "cp_f1": f1, "cp_mae": mae}
