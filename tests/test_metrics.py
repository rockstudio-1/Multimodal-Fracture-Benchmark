import unittest

import numpy as np

from fracturebench.metrics import MetricAccumulator, overlap_scores


class MetricTests(unittest.TestCase):
    def test_perfect_overlap(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 10:14] = 255
        acc = MetricAccumulator()
        acc.update(mask, mask)
        report = acc.compute()
        self.assertAlmostEqual(report["iou"], 1.0)
        self.assertAlmostEqual(report["dice"], 1.0)
        self.assertAlmostEqual(report["tolerance_f1"], 1.0)
        self.assertAlmostEqual(report["cldice"], 1.0)

    def test_empty_pair(self):
        blank = np.zeros((16, 16), dtype=np.uint8)
        acc = MetricAccumulator()
        acc.update(blank, blank)
        report = acc.compute()
        self.assertAlmostEqual(report["iou"], 1.0)
        self.assertAlmostEqual(report["cldice"], 1.0)

    def test_complete_miss(self):
        gt = np.zeros((16, 16), dtype=np.uint8)
        pred = np.zeros((16, 16), dtype=np.uint8)
        gt[4:8, 4:8] = 1
        pred[10:14, 10:14] = 1
        scores = overlap_scores(0, 16, 16)
        self.assertAlmostEqual(scores["iou"], 0.0)
        acc = MetricAccumulator(tolerance_radius=2)
        acc.update(pred, gt)
        report = acc.compute()
        self.assertLess(report["tolerance_f1"], 0.2)
        self.assertAlmostEqual(report["iou"], 0.0)

    def test_one_pixel_shift_is_within_tolerance(self):
        gt = np.zeros((32, 32), dtype=np.uint8)
        gt[10:20, 8:12] = 1
        pred = np.roll(gt, 1, axis=1)
        acc = MetricAccumulator(tolerance_radius=2)
        acc.update(pred, gt)
        report = acc.compute()
        self.assertGreater(report["tolerance_f1"], report["iou"])
        self.assertGreater(report["tolerance_f1"], 0.9)


if __name__ == "__main__":
    unittest.main()
