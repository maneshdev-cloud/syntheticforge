import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import evaluate_synthetic


class EvaluateTests(unittest.TestCase):
    def test_identical_data_has_zero_distance_metrics(self):
        data = pd.DataFrame({"x": [1, 2, 3, 4], "kind": ["a", "b", "a", "b"]})
        report = evaluate_synthetic(data, data.copy())
        self.assertEqual(report["similarity"]["numeric_ks_statistic"]["x"], 0.0)
        self.assertEqual(
            report["similarity"]["categorical_total_variation_distance"]["kind"], 0.0
        )
        self.assertEqual(report["privacy"]["exact_match_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
