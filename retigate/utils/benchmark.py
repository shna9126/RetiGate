import time
import numpy as np
from typing import Callable, Dict


def benchmark(name: str,
              fn: Callable,
              n_runs: int = 200,
              warmup: int = 10) -> Dict:
    """
    Run fn() n_runs times after warmup, return timing statistics.
    All experiment scripts use this function for every timing claim.
    """
    for _ in range(warmup):
        fn()

    times_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - t0) * 1000)

    times_ms = np.array(times_ms)
    ci95 = 1.96 * np.std(times_ms, ddof=1) / np.sqrt(n_runs)

    return {
        'name': name,
        'n': n_runs,
        'mean_ms': float(np.mean(times_ms)),
        'std_ms': float(np.std(times_ms, ddof=1)),
        'ci95_ms': float(ci95),
        'p50_ms': float(np.percentile(times_ms, 50)),
        'p95_ms': float(np.percentile(times_ms, 95)),
        'p99_ms': float(np.percentile(times_ms, 99)),
        'min_ms': float(np.min(times_ms)),
        'max_ms': float(np.max(times_ms)),
    }
