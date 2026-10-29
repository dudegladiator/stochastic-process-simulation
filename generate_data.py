"""Generate reproducible CSV data for the report figures and diagnostics."""

import csv
import math
import random
from pathlib import Path

from simulation import brownian_motion, geometric_brownian_motion, ornstein_uhlenbeck, vasicek

OUTPUT_DIR = Path("data")
HORIZON = 1.0
N_STEPS = 252
N_SAMPLE_PATHS = 10
N_DISTRIBUTION_PATHS = 10_000


def write_paths(filename, paths, theoretical_mean):
    with (OUTPUT_DIR / filename).open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["time", *[f"path_{i + 1}" for i in range(len(paths))], "mean"])
        for step in range(N_STEPS + 1):
            time = step * HORIZON / N_STEPS
            writer.writerow([
                f"{time:.8f}",
                *[f"{path[step]:.8f}" for path in paths],
                f"{theoretical_mean(time):.8f}",
            ])


def write_distribution(filename, values, lower, upper, theoretical_density, bins=60):
    width = (upper - lower) / bins
    counts = [0] * bins
    for value in values:
        index = int((value - lower) / width)
        if 0 <= index < bins:
            counts[index] += 1

    with (OUTPUT_DIR / filename).open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["edge", "empirical", "theoretical"])
        for index in range(bins + 1):
            edge = lower + index * width
            empirical_density = (
                counts[index] / (len(values) * width) if index < bins else 0.0
            )
            writer.writerow([
                f"{edge:.8f}", f"{empirical_density:.8f}",
                f"{theoretical_density(edge):.8f}",
            ])


def normal_density(mean, variance):
    scale = math.sqrt(2.0 * math.pi * variance)
    return lambda x: math.exp(-((x - mean) ** 2) / (2.0 * variance)) / scale


def sample_variance(values):
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    models = [
        {
            "name": "Brownian motion",
            "slug": "brownian",
            "sample": lambda count, seed: brownian_motion(count, N_STEPS, HORIZON, random.Random(seed)),
            "mean_at": lambda t: 0.0,
            "terminal_mean": 0.0,
            "terminal_variance": HORIZON,
            "range": (-4.0, 4.0),
            "density": normal_density(0.0, HORIZON),
        },
        {
            "name": "Geometric Brownian motion",
            "slug": "gbm",
            "sample": lambda count, seed: geometric_brownian_motion(
                count, N_STEPS, HORIZON, 100.0, 0.08, 0.20, random.Random(seed)
            ),
            "mean_at": lambda t: 100.0 * math.exp(0.08 * t),
            "terminal_mean": 100.0 * math.exp(0.08 * HORIZON),
            "terminal_variance": 100.0**2 * math.exp(2 * 0.08 * HORIZON)
            * (math.exp(0.20**2 * HORIZON) - 1.0),
            "range": (
                100.0 * math.exp((0.08 - 0.5 * 0.20**2) * HORIZON - 4 * 0.20 * math.sqrt(HORIZON)),
                100.0 * math.exp((0.08 - 0.5 * 0.20**2) * HORIZON + 4 * 0.20 * math.sqrt(HORIZON)),
            ),
            "density": lambda x: 0.0 if x <= 0 else math.exp(
                -(math.log(x / 100.0) - (0.08 - 0.5 * 0.20**2) * HORIZON) ** 2
                / (2 * 0.20**2 * HORIZON)
            ) / (x * 0.20 * math.sqrt(2 * math.pi * HORIZON)),
        },
        {
            "name": "Ornstein--Uhlenbeck process",
            "slug": "ou",
            "sample": lambda count, seed: ornstein_uhlenbeck(
                count, N_STEPS, HORIZON, 1.5, 2.0, 0.5, 0.35, random.Random(seed)
            ),
            "mean_at": lambda t: 0.5 + (1.5 - 0.5) * math.exp(-2.0 * t),
            "terminal_mean": 0.5 + (1.5 - 0.5) * math.exp(-2.0 * HORIZON),
            "terminal_variance": 0.35**2 / (2 * 2.0) * (1 - math.exp(-2 * 2.0 * HORIZON)),
        },
        {
            "name": "Vasicek short-rate model",
            "slug": "vasicek",
            "sample": lambda count, seed: vasicek(
                count, N_STEPS, HORIZON, 0.03, 1.2, 0.05, 0.02, random.Random(seed)
            ),
            "mean_at": lambda t: 0.05 + (0.03 - 0.05) * math.exp(-1.2 * t),
            "terminal_mean": 0.05 + (0.03 - 0.05) * math.exp(-1.2 * HORIZON),
            "terminal_variance": 0.02**2 / (2 * 1.2) * (1 - math.exp(-2 * 1.2 * HORIZON)),
        },
    ]

    for model in models[2:]:
        standard_deviation = math.sqrt(model["terminal_variance"])
        model["range"] = (
            model["terminal_mean"] - 4 * standard_deviation,
            model["terminal_mean"] + 4 * standard_deviation,
        )
        model["density"] = normal_density(model["terminal_mean"], model["terminal_variance"])

    diagnostics = []
    for index, model in enumerate(models):
        write_paths(
            f"{model['slug']}_paths.csv",
            model["sample"](N_SAMPLE_PATHS, 100 + index),
            model["mean_at"],
        )
        distribution_paths = model["sample"](N_DISTRIBUTION_PATHS, 200 + index)
        terminal = [path[-1] for path in distribution_paths]
        write_distribution(
            f"{model['slug']}_distribution.csv", terminal,
            model["range"][0], model["range"][1], model["density"],
        )
        diagnostics.append([
            model["name"],
            sum(terminal) / len(terminal), model["terminal_mean"],
            sample_variance(terminal), model["terminal_variance"],
        ])

    with (OUTPUT_DIR / "terminal_diagnostics.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow([
            "model", "empirical_mean", "theoretical_mean",
            "empirical_variance", "theoretical_variance",
        ])
        writer.writerows(diagnostics)


if __name__ == "__main__":
    main()
