"""Basic tests for the from-scratch linear regression model."""
import unittest

import numpy as np

from linear_regression import LinearRegression


class TestLinearRegression(unittest.TestCase):
    def test_learns_simple_line(self):
        x = np.arange(0, 20, dtype=float)
        y = 3.0 * x + 5.0
        model = LinearRegression(learning_rate=0.001, n_iterations=5_000)
        model.fit(x, y)

        self.assertAlmostEqual(model.weight, 3.0, delta=0.05)
        self.assertAlmostEqual(model.bias, 5.0, delta=0.1)
        np.testing.assert_allclose(model.predict(np.array([1.0, 2.0])), [8.0, 11.0], atol=0.2)

    def test_loss_decreases(self):
        x = np.arange(1, 11, dtype=float)
        y = 2.0 * x + 1.0
        model = LinearRegression(learning_rate=0.01, n_iterations=500)
        model.fit(x, y)

        self.assertLess(model.loss_history[-1], model.loss_history[0])

    def test_rejects_mismatched_lengths(self):
        model = LinearRegression()
        with self.assertRaises(ValueError):
            model.fit(np.array([1, 2, 3]), np.array([1, 2]))

    def test_rejects_empty_data(self):
        model = LinearRegression()
        with self.assertRaises(ValueError):
            model.fit(np.array([]), np.array([]))


if __name__ == "__main__":
    unittest.main()
