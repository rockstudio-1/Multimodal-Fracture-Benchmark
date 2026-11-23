"""Micro-averaged crack segmentation metrics.

Overlap scores treat a pixel as positive when its value is nonzero.
Tolerance F1 dilates with a square window of radius ``tolerance_radius``
(Chebyshev distance). clDice follows Shit et al., CVPR 2021, using the
iterative soft skeleton with ``skeleton_iterations`` steps.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _as_bool(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    return array > 0


def _empty_scores() -> dict[str, float]:
    return {
        "iou": 1.0,
        "dice": 1.0,
        "precision": 1.0,
        "recall": 1.0,
    }


def _ratio(numerator: float, denominator: float, empty_value: float) -> float:
    if denominator == 0:
        return empty_value
    return numerator / denominator


def overlap_scores(tp: float, fp: float, fn: float) -> dict[str, float]:
    if tp == 0 and fp == 0 and fn == 0:
        return _empty_scores()
    precision = _ratio(tp, tp + fp, 1.0)
    recall = _ratio(tp, tp + fn, 1.0)
    iou = _ratio(tp, tp + fp + fn, 1.0)
    dice = _ratio(2 * tp, 2 * tp + fp + fn, 1.0)
    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
    }


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    size = 2 * radius + 1
    return ndimage.binary_dilation(mask, structure=np.ones((size, size), dtype=bool))


def _soft_erode(image: np.ndarray) -> np.ndarray:
    horizontal = ndimage.minimum_filter(image, size=(3, 1), mode="nearest")
    vertical = ndimage.minimum_filter(image, size=(1, 3), mode="nearest")
    return np.minimum(horizontal, vertical)


def _soft_dilate(image: np.ndarray) -> np.ndarray:
    return ndimage.maximum_filter(image, size=(3, 3), mode="nearest")


def soft_skeleton(mask: np.ndarray, iterations: int = 5) -> np.ndarray:
    """Iterative soft skeleton used by clDice."""
    image = mask.astype(np.float64)
    opened = _soft_dilate(_soft_erode(image))
    skeleton = np.maximum(image - opened, 0.0)
    for _ in range(iterations):
        image = _soft_erode(image)
        opened = _soft_dilate(_soft_erode(image))
        delta = np.maximum(image - opened, 0.0)
        skeleton = skeleton + np.maximum(delta - skeleton * delta, 0.0)
    return skeleton


def _pair(hit_pred: float, pred: float, hit_gt: float, gt: float) -> tuple[float, float]:
    if pred == 0 and gt == 0:
        return 1.0, 1.0
    precision = 1.0 if pred == 0 else hit_pred / pred
    recall = 1.0 if gt == 0 else hit_gt / gt
    if pred == 0:
        recall = 0.0
    if gt == 0:
        precision = 0.0
    return precision, recall


def _harmonic(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


class MetricAccumulator:
    """Accumulate a split, then compute one micro-averaged report."""

    def __init__(self, tolerance_radius: int = 2, skeleton_iterations: int = 5) -> None:
        self.tolerance_radius = tolerance_radius
        self.skeleton_iterations = skeleton_iterations
        self.reset()

    def reset(self) -> None:
        self.images = 0
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tol_pred_hit = 0.0
        self.tol_pred = 0.0
        self.tol_gt_hit = 0.0
        self.tol_gt = 0.0
        self.skel_pred_hit = 0.0
        self.skel_pred = 0.0
        self.skel_gt_hit = 0.0
        self.skel_gt = 0.0

    def update(self, prediction: np.ndarray, ground_truth: np.ndarray) -> None:
        pred = _as_bool(prediction)
        gt = _as_bool(ground_truth)
        if pred.shape != gt.shape:
            raise ValueError(f"shape mismatch: prediction {pred.shape} vs ground truth {gt.shape}")
        self.images += 1
        self.tp += float(np.logical_and(pred, gt).sum())
        self.fp += float(np.logical_and(pred, np.logical_not(gt)).sum())
        self.fn += float(np.logical_and(np.logical_not(pred), gt).sum())

        pred_hit = np.logical_and(pred, dilate(gt, self.tolerance_radius))
        gt_hit = np.logical_and(gt, dilate(pred, self.tolerance_radius))
        self.tol_pred_hit += float(pred_hit.sum())
        self.tol_pred += float(pred.sum())
        self.tol_gt_hit += float(gt_hit.sum())
        self.tol_gt += float(gt.sum())

        skel_pred = soft_skeleton(pred, self.skeleton_iterations) > 0
        skel_gt = soft_skeleton(gt, self.skeleton_iterations) > 0
        self.skel_pred_hit += float(np.logical_and(skel_pred, gt).sum())
        self.skel_pred += float(skel_pred.sum())
        self.skel_gt_hit += float(np.logical_and(skel_gt, pred).sum())
        self.skel_gt += float(skel_gt.sum())

    def compute(self) -> dict[str, float]:
        scores = overlap_scores(self.tp, self.fp, self.fn)
        tol_precision, tol_recall = _pair(self.tol_pred_hit, self.tol_pred, self.tol_gt_hit, self.tol_gt)
        skel_precision, skel_recall = _pair(self.skel_pred_hit, self.skel_pred, self.skel_gt_hit, self.skel_gt)
        return {
            "num_images": self.images,
            "iou": scores["iou"],
            "dice": scores["dice"],
            "precision": scores["precision"],
            "recall": scores["recall"],
            "tolerance_f1": _harmonic(tol_precision, tol_recall),
            "cldice": _harmonic(skel_precision, skel_recall),
            "tolerance_radius": self.tolerance_radius,
            "skeleton_iterations": self.skeleton_iterations,
        }
