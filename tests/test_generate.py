import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate import generate_synthetic


class GenerateTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "score": [41, 55, 62, 77, 83, 91],
                "hours": [2, 3, 4, 6, 7, 8],
                "group": ["a", "a", "b", "b", "c", "c"],
            }
        )

    def test_copula_shape_and_columns(self):
        result = generate_synthetic(self.data, n_rows=25, method="copula", seed=7)
        self.assertEqual(result.shape, (25, 3))
        self.assertEqual(list(result.columns), list(self.data.columns))

    def test_marginal_categories_come_from_source(self):
        result = generate_synthetic(self.data, n_rows=25, method="marginal", seed=7)
        self.assertTrue(set(result["group"]).issubset({"a", "b", "c"}))


if __name__ == "__main__":
    unittest.main()
