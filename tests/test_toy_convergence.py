from __future__ import annotations

import numpy as np

from scintillation.fitting.convergence import (
    ToyConvergenceMonitor,
    ToyConvergenceSettings,
)


def test_adaptive_toys_stop_after_stable_checkpoints() -> None:
    rng = np.random.default_rng(12345)
    toys = rng.multivariate_normal(
        [1.0, 2.0],
        [[1.0, 0.35], [0.35, 0.7]],
        size=600,
    )
    settings = ToyConvergenceSettings(
        enabled=True,
        max_toys=600,
        min_toys=100,
        batch_size=50,
        quantile_tolerance=0.5,
        correlation_tolerance=0.5,
        stable_checks=2,
    )
    monitor = ToyConvergenceMonitor(settings)
    status = None
    for stop in range(50, 601, 50):
        status = monitor.update(toys[:stop], attempted_toys=stop)
        if status.converged:
            break
    assert status is not None
    assert status.converged
    assert status.valid_toys >= 100
    assert status.attempted_toys < 600


def test_fixed_toys_never_report_early_convergence() -> None:
    toys = np.arange(400, dtype=float).reshape(200, 2)
    settings = ToyConvergenceSettings(
        enabled=False,
        max_toys=200,
        min_toys=50,
        batch_size=50,
        quantile_tolerance=1.0,
        correlation_tolerance=1.0,
        stable_checks=1,
    )
    status = ToyConvergenceMonitor(settings).update(toys[:50], attempted_toys=50)
    assert not status.converged
    assert status.reason == "fixed_toy_count"
