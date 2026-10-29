"""Dependency-free simulators for the stochastic processes in the report."""

import math
import random
from typing import Callable

Path = list[float]
Paths = list[Path]


def _simulate(n_paths: int, n_steps: int, initial: float, transition: Callable[[float], float]) -> Paths:
    paths: Paths = []
    for _ in range(n_paths):
        path = [initial]
        for _ in range(n_steps):
            path.append(transition(path[-1]))
        paths.append(path)
    return paths


def brownian_motion(
    n_paths: int, n_steps: int, horizon: float, rng: random.Random
) -> Paths:
    """Simulate standard Brownian motion using Gaussian increments."""
    increment_scale = math.sqrt(horizon / n_steps)
    return _simulate(
        n_paths, n_steps, 0.0,
        lambda current: current + increment_scale * rng.gauss(0.0, 1.0),
    )


def geometric_brownian_motion(
    n_paths: int,
    n_steps: int,
    horizon: float,
    s0: float,
    mu: float,
    sigma: float,
    rng: random.Random,
) -> Paths:
    """Simulate GBM with its exact lognormal time-step transition."""
    dt = horizon / n_steps
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * math.sqrt(dt)
    return _simulate(
        n_paths, n_steps, s0,
        lambda current: current * math.exp(drift + diffusion * rng.gauss(0.0, 1.0)),
    )


def ornstein_uhlenbeck(
    n_paths: int,
    n_steps: int,
    horizon: float,
    x0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rng: random.Random,
) -> Paths:
    """Simulate an OU process with its exact Gaussian transition."""
    dt = horizon / n_steps
    decay = math.exp(-kappa * dt)
    transition_std = sigma * math.sqrt((1.0 - decay**2) / (2.0 * kappa))
    return _simulate(
        n_paths, n_steps, x0,
        lambda current: theta + (current - theta) * decay
        + transition_std * rng.gauss(0.0, 1.0),
    )


def vasicek(
    n_paths: int,
    n_steps: int,
    horizon: float,
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rng: random.Random,
) -> Paths:
    """Simulate the Vasicek short rate, mathematically an OU process."""
    return ornstein_uhlenbeck(
        n_paths, n_steps, horizon, r0, kappa, theta, sigma, rng
    )
