import unittest

import numpy as np

import kgcn.src.models.metrics as metrics


class TestMetricsReport(unittest.TestCase):
    def test_print(self):
        y_true = np.array([0, 1, 2, 0])
        y_pred = y_true
        # expected_confusion_matrix = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]])
        metrics.report_multiclass_metrics(y_true, y_pred)
