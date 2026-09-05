import numpy as np


def balanced_prior_correction(y_true) -> float:
    y = np.asarray(y_true).astype(int)
    positives = max(int(y.sum()), 1)
    negatives = max(int(len(y) - y.sum()), 1)
    return float(np.log(negatives / positives))


def calibrate_balanced_probabilities(probability, correction: float) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-9, 1.0 - 1e-9)
    logit = np.log(p / (1.0 - p)) - correction
    return 1.0 / (1.0 + np.exp(-logit))
