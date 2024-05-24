import numpy as np


# Activation functions and their derivatives
def leaky_relu(Z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Compute the leaky ReLU function.

    Parameters:
    - Z (numpy.ndarray): The input values.
    - alpha (float): The slope of the leaky ReLU function for negative values.

    Returns:
    - numpy.ndarray: The leaky ReLU values."""
    return np.where(Z > 0, Z, alpha * Z)


def leaky_relu_derivative(A: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Compute the derivative of the leaky ReLU function.

    Parameters:
    - A (numpy.ndarray): The activations of the leaky ReLU function.
    - alpha (float): The slope of the leaky ReLU function for negative values.

    Returns:
    - numpy.ndarray: The derivative of the leaky ReLU function."""
    dZ = np.ones_like(A)
    dZ[A < 0] = alpha
    return dZ


def sigmoid(X: np.ndarray) -> np.ndarray:
    """Compute the sigmoid function.

    Parameters:
    - X (numpy.ndarray): The input values.

    Returns:
    - numpy.ndarray: The sigmoid values."""
    X = np.clip(X, -500, 500)
    return 1 / (1 + np.exp(-X))


def sigmoid_derivative(activations: np.ndarray) -> np.ndarray:
    """Compute the derivative of the sigmoid function.

    Parameters:
    - activations (numpy.ndarray): The activations of the sigmoid function.

    Returns:
    - numpy.ndarray: The derivative of the sigmoid function."""
    return activations * (1 - activations)


def relu(X: np.ndarray) -> np.ndarray:
    """Compute the ReLU function.

    Parameters:
    - X (numpy.ndarray): The input values.

    Returns:
    - numpy.ndarray: The ReLU values."""
    return np.maximum(0, X)


def relu_derivative(activations: np.ndarray) -> np.ndarray:
    """Compute the derivative of the ReLU function.

    Parameters:
    - activations (numpy.ndarray): The activations of the ReLU function.

    Returns:
    - numpy.ndarray: The derivative of the ReLU function."""
    return activations > 0


def softmax(X: np.ndarray) -> np.ndarray:
    """Compute the softmax function.

    Parameters:
    - X (numpy.ndarray): The input values.

    Returns:
    - numpy.ndarray: The softmax values."""
    shift_X = X - np.max(X, axis=0, keepdims=True)
    e_x = np.exp(shift_X)
    return e_x / e_x.sum(axis=0, keepdims=True)