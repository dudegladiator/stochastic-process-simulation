import csv
import math
import random
import tempfile
import unittest
from pathlib import Path

import generate_data
from simulation import brownian_motion, geometric_brownian_motion, ornstein_uhlenbeck, vasicek


class SimulationTests(unittest.TestCase):
    def test_brownian_terminal_moments_match_theory(self):
        paths = brownian_motion(8_000, 100, 2.0, random.Random(1))
        terminal = [path[-1] for path in paths]

        self.assertAlmostEqual(sum(terminal) / len(terminal), 0.0, delta=0.06)
        self.assertAlmostEqual(sample_variance(terminal), 2.0, delta=0.12)

    def test_gbm_is_positive_and_terminal_mean_matches_theory(self):
        s0, mu, sigma, horizon = 100.0, 0.08, 0.20, 1.0
        paths = geometric_brownian_motion(
            8_000, 100, horizon, s0, mu, sigma, random.Random(2)
        )
        terminal = [path[-1] for path in paths]

        self.assertTrue(all(value > 0.0 for path in paths for value in path))
        self.assertAlmostEqual(
            sum(terminal) / len(terminal), s0 * math.exp(mu * horizon), delta=0.8
        )

    def test_ou_reverts_deterministically_when_volatility_is_zero(self):
        path = ornstein_uhlenbeck(
            1, 10, 1.0, x0=2.0, kappa=1.5, theta=0.5, sigma=0.0,
            rng=random.Random(3),
        )[0]
        expected = 0.5 + (2.0 - 0.5) * math.exp(-1.5)

        self.assertAlmostEqual(path[-1], expected)

    def test_vasicek_has_same_transition_as_ou(self):
        ou_path = ornstein_uhlenbeck(
            2, 12, 1.0, x0=0.03, kappa=0.7, theta=0.05, sigma=0.02,
            rng=random.Random(4),
        )
        rate_path = vasicek(
            2, 12, 1.0, r0=0.03, kappa=0.7, theta=0.05, sigma=0.02,
            rng=random.Random(4),
        )

        self.assertEqual(rate_path, ou_path)

    def test_fixed_seed_is_reproducible(self):
        first = brownian_motion(3, 5, 1.0, random.Random(42))
        second = brownian_motion(3, 5, 1.0, random.Random(42))

        self.assertEqual(first, second)

    def test_histogram_output_uses_bin_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            original_output_dir = generate_data.OUTPUT_DIR
            generate_data.OUTPUT_DIR = Path(directory)
            try:
                generate_data.write_distribution(
                    "histogram.csv", [0.2, 1.2, 2.2], 0.0, 3.0,
                    theoretical_density=lambda value: value, bins=3,
                )
            finally:
                generate_data.OUTPUT_DIR = original_output_dir

            with (Path(directory) / "histogram.csv").open() as output:
                rows = list(csv.DictReader(output))

        self.assertEqual([float(row["edge"]) for row in rows], [0.0, 1.0, 2.0, 3.0])
        for actual, expected in zip(
            [float(row["empirical"]) for row in rows],
            [1 / 3, 1 / 3, 1 / 3, 0.0],
        ):
            self.assertAlmostEqual(actual, expected, places=7)


def sample_variance(values):
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


if __name__ == "__main__":
    unittest.main()
