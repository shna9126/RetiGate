# experiments/02_Fidelity_Evaluations/ci_bootstrap.py
# Takes your existing per-sequence mAP numbers and computes CI

import numpy as np
import pandas as pd

# Your per-sequence mAP from results/02_Accuracy_Logs/14_kitti_map.txt
sequence_maps = [
    25.22, 55.18, 54.77, 82.99, 40.04,
    39.99, 82.34, 62.23, 71.77, 34.30,
    42.46, 49.65, 34.23, 40.39, 51.34,
    34.75, 29.46, 38.44, 88.02, 34.21,
    63.03
]

def bootstrap_ci(data, n_boot=10000, ci=95):
    data   = np.array(data)
    means  = [np.mean(
                  np.random.choice(data, size=len(data),
                                   replace=True)
              ) for _ in range(n_boot)]
    lower  = np.percentile(means, (100-ci)/2)
    upper  = np.percentile(means, 100-(100-ci)/2)
    return np.mean(data), lower, upper

mean, lo, hi = bootstrap_ci(sequence_maps)
print(f"Mean mAP@50: {mean:.2f}% "
      f"[{lo:.2f}%, {hi:.2f}%] (95% CI, bootstrap n=10000)")
