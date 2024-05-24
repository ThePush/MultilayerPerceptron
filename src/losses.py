import numpy as np


def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute the binary cross-entropy loss.

    Parameters:
    - pred (numpy.ndarray): The predicted values.
    - true (numpy.ndarray): The true values.

    Returns:
    - float: The binary cross-entropy loss."""
    epsilon = 1e-15
    pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(pred) + (1 - y_true) * np.log(1 - pred))
