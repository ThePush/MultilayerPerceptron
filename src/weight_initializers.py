import numpy as np


def xavier_normal(input_shape: int, output_shape: int) -> np.ndarray:
    """Initialize the weights using the Xavier normal initialization.

    Parameters:
    - input_shape (int): The number of input units.
    - output_shape (int): The number of output units.

    Returns:
    - numpy.ndarray: The initialized weights."""
    std_dev = np.sqrt(2.0 / (input_shape + output_shape))
    return np.random.normal(0, std_dev, (output_shape, input_shape))


def xavier_uniform(input_shape: int, output_shape: int) -> np.ndarray:
    """Initialize the weights using the Xavier uniform initialization.

    Parameters:
    - input_shape (int): The number of input units.
    - output_shape (int): The number of output units.

    Returns:
    - numpy.ndarray: The initialized weights."""
    std_dev = np.sqrt(6.0 / (input_shape + output_shape))
    return np.random.uniform(-std_dev, std_dev, (output_shape, input_shape))


def he_normal(input_shape: int, output_shape: int) -> np.ndarray:
    """Initialize the weights using the He normal initialization.

    Parameters:
    - input_shape (int): The number of input units.
    - output_shape (int): The number of output units.

    Returns:
    - numpy.ndarray: The initialized weights."""
    std_dev = np.sqrt(2.0 / input_shape)
    return np.random.normal(0, std_dev, (output_shape, input_shape))


def he_uniform(input_shape: int, output_shape: int) -> np.ndarray:
    """Initialize the weights using the He uniform initialization.

    Parameters:
    - input_shape (int): The number of input units.
    - output_shape (int): The number of output units.

    Returns:
    - numpy.ndarray: The initialized weights."""
    std_dev = np.sqrt(6.0 / input_shape)
    return np.random.uniform(-std_dev, std_dev, (output_shape, input_shape))
