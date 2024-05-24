import numpy as np
from loguru import logger


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute the accuracy of predictions.

    Parameters:
    - y_pred (np.ndarray): The predicted probabilities, which will be rounded to 0 or 1.
    - y_true (np.ndarray): The true binary labels (0 or 1).

    Returns:
    - float: The accuracy of the predictions, ranging from 0 to 1.
    """
    return np.mean(np.round(y_pred).astype(int) == y_true)


def true_positives(y_pred: np.ndarray, y_true: np.ndarray) -> int:
    """
    Calculate the number of true positive predictions.

    Parameters:
    - y_pred (np.ndarray): Predicted labels, must be binary (0 or 1).
    - y_true (np.ndarray): Actual true labels, must be binary (0 or 1).

    Returns:
    - int: The count of true positive instances.
    """
    return np.sum((y_pred == 1) & (y_true == 1))


def false_positives(y_pred: np.ndarray, y_true: np.ndarray) -> int:
    """
    Calculate the number of false positive predictions.

    Parameters:
    - y_pred (np.ndarray): Predicted labels, must be binary (0 or 1).
    - y_true (np.ndarray): Actual true labels, must be binary (0 or 1).

    Returns:
    - int: The count of false positive instances.
    """
    return np.sum((y_pred == 1) & (y_true == 0))


def true_negatives(y_pred: np.ndarray, y_true: np.ndarray) -> int:
    """
    Calculate the number of true negative predictions.

    Parameters:
    - y_pred (np.ndarray): Predicted labels, must be binary (0 or 1).
    - y_true (np.ndarray): Actual true labels, must be binary (0 or 1).

    Returns:
    - int: The count of true negative instances.
    """
    return np.sum((y_pred == 0) & (y_true == 0))


def false_negatives(y_pred: np.ndarray, y_true: np.ndarray) -> int:
    """
    Calculate the number of false negative predictions.

    Parameters:
    - y_pred (np.ndarray): Predicted labels, must be binary (0 or 1).
    - y_true (np.ndarray): Actual true labels, must be binary (0 or 1).

    Returns:
    - int: The count of false negative instances.
    """
    return np.sum((y_pred == 0) & (y_true == 1))


def precision(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculate the precision of the predictions.

    Precision is the ratio of true positives to the total number of positive predictions (true positives + false positives).

    Parameters:
    - y_pred (np.ndarray): Predicted labels, must be binary (0 or 1).
    - y_true (np.ndarray): Actual true labels, must be binary (0 or 1).

    Returns:
    - float: The precision of the predictions, ranging from 0 to 1.
    """
    y_pred = np.round(y_pred).astype(int)
    TP = true_positives(y_pred, y_true)
    FP = false_positives(y_pred, y_true)
    return TP / (TP + FP) if (TP + FP) != 0 else 0


def recall(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculate the recall of the predictions.

    Recall is the ratio of true positives to the total number of actual positives (true positives + false negatives).

    Parameters:
    - y_pred (np.ndarray): Predicted labels, must be binary (0 or 1).
    - y_true (np.ndarray): Actual true labels, must be binary (0 or 1).

    Returns:
    - float: The recall of the predictions, ranging from 0 to 1.
    """
    y_pred = np.round(y_pred).astype(int)
    TP = true_positives(y_pred, y_true)
    FN = false_negatives(y_pred, y_true)
    return TP / (TP + FN) if (TP + FN) != 0 else 0


def f1_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculate the F1 score of the predictions.

    The F1 score is the harmonic mean of precision and recall.

    Parameters:
    - y_pred (np.ndarray): Predicted labels, must be binary (0 or 1).
    - y_true (np.ndarray): Actual true labels, must be binary (0 or 1).

    Returns:
    - float: The F1 score of the predictions, ranging from 0 to 1.
    """
    y_pred = np.round(y_pred).astype(int)
    P = precision(y_pred, y_true)
    R = recall(y_pred, y_true)
    return 2 * (P * R) / (P + R) if (P + R) != 0 else 0


def print_metrics(y_pred, y_true) -> tuple:
    """
    Print the accuracy, precision, recall, and F1 score of the predictions.

    Parameters:
    - y_pred (np.ndarray): Predicted labels, must be binary (0 or 1).
    - y_true (np.ndarray): Actual true labels, must be binary (0 or 1).
    """
    _accuracy = accuracy(y_pred, y_true)
    _precision = precision(y_pred, y_true)
    _recall = recall(y_pred, y_true)
    _f1 = f1_score(y_pred, y_true)
    logger.info(f" Accuracy: {_accuracy:.2f}")
    logger.info(f" Precision: {_precision:.2f}")
    logger.info(f" Recall: {_recall:.2f}")
    logger.info(f" F1 Score: {_f1:.2f}")
    return _accuracy, _precision, _recall, _f1
