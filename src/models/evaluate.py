import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    safe = np.clip(predicted, 1e-9, None)
    return {
        "mae": float(mean_absolute_error(actual, safe)),
        "rmse": float(np.sqrt(mean_squared_error(actual, safe))),
        "mean_poisson_deviance": float(mean_poisson_deviance(actual, safe)),
    }


def classification_metrics(actual: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = (probability >= threshold).astype(int)
    return {
        "average_precision": float(average_precision_score(actual, probability)),
        "roc_auc": float(roc_auc_score(actual, probability)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "brier_score": float(brier_score_loss(actual, probability)),
        "log_loss": float(log_loss(actual, np.column_stack([1 - probability, probability]), labels=[0, 1])),
    }
