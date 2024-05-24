import time
import numpy as np
from dataclasses import dataclass, field
from loguru import logger

from src.weight_initializers import (
    xavier_normal,
    xavier_uniform,
    he_normal,
    he_uniform,
)
from src.activations import (
    sigmoid,
    sigmoid_derivative,
    relu,
    relu_derivative,
    leaky_relu,
    leaky_relu_derivative,
    softmax,
)


@dataclass
class Layer:
    input_shape: int
    output_shape: int
    activation_function: str
    weights_initializer: str = "xavier_normal"
    weights: np.ndarray = field(init=False, repr=False)
    biases: np.ndarray = field(init=False, repr=False)
    optimizer: str = "gradient_descent"
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    m_w: np.ndarray = field(init=False, repr=False)
    v_w: np.ndarray = field(init=False, repr=False)
    m_b: np.ndarray = field(init=False, repr=False)
    v_b: np.ndarray = field(init=False, repr=False)
    t: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        self.activation_functions_lookup = {
            "sigmoid": sigmoid,
            "relu": relu,
            "leaky_relu": leaky_relu,
            "softmax": softmax,
            "": lambda x: x,
        }
        self.derivative_functions_lookup = {
            "sigmoid": sigmoid_derivative,
            "relu": relu_derivative,
            "leaky_relu": leaky_relu_derivative,
            "softmax": lambda x: 1,
        }
        self.initializer_functions_lookup = {
            "xavier_normal": xavier_normal,
            "xavier_uniform": xavier_uniform,
            "he_normal": he_normal,
            "he_uniform": he_uniform,
        }
        self.weights = self._init_weights(
            self.input_shape,
            self.output_shape,
            self.weights_initializer,
        )
        self.biases = np.zeros(shape=(self.output_shape, 1))
        self.activations = np.zeros(shape=(1, self.output_shape))
        self.m_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.biases)
        self.v_b = np.zeros_like(self.biases)

    def _init_weights(
        self,
        input_shape: int,
        output_shape: int,
        initializer: str,
    ) -> np.ndarray:
        """Initialize the weights using the specified initializer.

        Parameters:
        - input_shape (int): The number of input neurons.
        - output_shape (int): The number of output neurons.

        Raises:
        - ValueError: If the initializer is not found.

        Returns:
        - numpy.ndarray: The initialized weights."""
        initializer_func = self.initializer_functions_lookup.get(initializer)
        if not initializer_func:
            logger.error("Initializer not found")
            raise ValueError("Initializer not found")
        np.random.seed(int(time.time()))
        return initializer_func(input_shape, output_shape)

    def get_activations(self) -> np.ndarray:
        """Return the activations of the layer.

        Parameters:
        None

        Returns:
        - numpy.ndarray: The activations of the layer."""
        return self.activations

    def forward(self, previous_activations: np.ndarray) -> np.ndarray:
        """Perform the forward propagation step.

        Parameters:
        - previous_activations (numpy.ndarray): The activations from the previous layer.

        Raises:
        - ValueError: If the activation function is not found.

        Returns:
        - numpy.ndarray: The activations of the current layer."""
        activation_func = self.activation_functions_lookup.get(self.activation_function)
        if not activation_func:
            logger.error("Activation function not found")
            raise ValueError("Activation function not found")
        if not self.activation_function:
            self.activations = previous_activations
        else:
            self.activations = activation_func(
                self.weights.dot(previous_activations) + self.biases
            )
        return self.activations

    def backward(
        self,
        dz: np.ndarray,
        previous_activations: np.ndarray,
        targets: np.ndarray,
        lambda_reg: float,
    ) -> tuple:
        """Perform the backward propagation step.

        Parameters:
        - dz (numpy.ndarray): The gradient of the activations.
        - previous_activations (numpy.ndarray): The activations from the previous layer.
        - targets (numpy.ndarray): The target values.

        Raises:
        - ValueError: If the derivative function is not found.

        Returns:
        - numpy.ndarray: The gradient of the activations.
        - numpy.ndarray: The gradient of the weights.
        - numpy.ndarray: The gradient of the biases."""
        if dz is None:
            dz = self.activations - targets
        targets_length = targets.shape[0]
        dw = (dz.dot(previous_activations.T) / targets_length) + (
            lambda_reg * self.weights
        )
        db = np.sum(dz, axis=1, keepdims=True) / targets_length

        derivative_func = self.derivative_functions_lookup.get(self.activation_function)
        if not derivative_func:
            logger.error("Derivative function not found")
            raise ValueError("Derivative function not found")
        next_dz = self.weights.T.dot(dz) * derivative_func(previous_activations)
        return next_dz, dw, db

    def update(
        self,
        dw: np.ndarray,
        db: np.ndarray,
        learning_rate: float,
        momentum: float,
    ) -> None:
        """Update the weights and biases using the specified optimizer algorithm."""
        if self.optimizer == "gradient_descent":
            self.weights -= learning_rate * dw
            self.biases -= learning_rate * db
        elif self.optimizer == "adam":
            self.t += 1
            # Update biased first moment estimate
            self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * dw
            self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db
            # Update biased second raw moment estimate
            self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (dw**2)
            self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (db**2)
            # Compute bias-corrected first moment estimate
            m_w_hat = self.m_w / (1 - self.beta1**self.t)
            m_b_hat = self.m_b / (1 - self.beta1**self.t)
            # Compute bias-corrected second raw moment estimate
            v_w_hat = self.v_w / (1 - self.beta2**self.t)
            v_b_hat = self.v_b / (1 - self.beta2**self.t)
            # Update weights and biases
            self.weights -= learning_rate * (
                m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            )
            self.biases -= learning_rate * (m_b_hat / (np.sqrt(v_b_hat) + self.epsilon))
        elif self.optimizer == "nesterov":
            prev_v_w = np.copy(self.v_w)
            prev_v_b = np.copy(self.v_b)
            self.v_w = momentum * self.v_w - learning_rate * dw
            self.v_b = momentum * self.v_b - learning_rate * db
            self.weights += -momentum * prev_v_w + (1 + momentum) * self.v_w
            self.biases += -momentum * prev_v_b + (1 + momentum) * self.v_b
