import os
import pickle
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from loguru import logger


from Layer import Layer
from src.losses import binary_cross_entropy
from src.metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
)

matplotlib.use("TkAgg")


@dataclass
class NeuralNetwork:
    epochs: int = 5000
    learning_rate: float = 0.00314
    lambda_reg: float = 0.0
    hidden_layers: list = field(default_factory=lambda: [8, 8])
    activation: str = "sigmoid"
    initializer: str = "xavier_normal"
    optimizer: str = "gradient_descent"
    momentum: float = 0.9
    input_shape: int = field(init=False, repr=False)
    output_shape: int = field(init=False, repr=False)
    layers: list = field(init=False, default_factory=list, repr=False)
    accuracy_history: dict = field(
        init=False,
        default_factory=lambda: {"train": [], "valid": []},
        repr=False,
    )
    log_loss_history: dict = field(
        init=False,
        default_factory=lambda: {"train": [], "valid": []},
        repr=False,
    )
    precision_history: dict = field(
        init=False,
        default_factory=lambda: {"train": [], "valid": []},
        repr=False,
    )
    recall_history: dict = field(
        init=False,
        default_factory=lambda: {"train": [], "valid": []},
        repr=False,
    )
    f1_score_history: dict = field(
        init=False,
        default_factory=lambda: {"train": [], "valid": []},
        repr=False,
    )
    best_params: dict = field(
        init=False,
        default_factory=dict,
        repr=False,
    )
    best_loss: float = field(
        init=False,
        default=np.inf,
        repr=False,
    )

    def _init_layers(
        self,
        input_shape: int,
        output_shape: int,
        hidden_layers_shapes_list: list,
        activation: str,
        initializer: str,
        lambda_reg: float,
        optimizer: str,
    ) -> None:
        """Initialize the layers of the neural network.

        Parameters:
        - input_shape (int): The number of input neurons.
        - output_shape (int): The number of output neurons.
        - hidden_layers_shapes_list (list): The number of neurons in each layer.
        - activation (str): The activation function.
        - initializer (str): The weight initializer.
        - lambda_reg (float): The regularization parameter.
        - optimizer (str): The optimizer algorithm.

        Returns:
        None"""
        self.layers.append(
            Layer(
                input_shape=1,
                output_shape=input_shape,
                activation_function="",
                weights_initializer=initializer,
                optimizer=optimizer,
            )
        )
        for layer_shape in hidden_layers_shapes_list:
            self.layers.append(
                Layer(
                    input_shape=input_shape,
                    output_shape=layer_shape,
                    activation_function=activation,
                    weights_initializer=initializer,
                    optimizer=optimizer,
                )
            )
            input_shape = layer_shape
        self.layers.append(
            Layer(
                input_shape=input_shape,
                output_shape=output_shape,
                activation_function="softmax",
                weights_initializer=initializer,
                optimizer=optimizer,
            )
        )

    def _forward_propagation(self, x: np.ndarray) -> np.ndarray:
        """Perform the forward propagation step.

        Parameters:
        - x (numpy.ndarray): The input data.

        Returns:
        - numpy.ndarray: The output of the neural network."""
        tmp_activations = x
        for layer in self.layers:
            tmp_activations = layer.forward(tmp_activations)
        return tmp_activations[0]

    def _backward_propagation(self, targets):
        tmp_dz = None
        reversed_layers = list(reversed(self.layers))

        for idx, layer in enumerate(reversed_layers[:-1]):
            tmp_dz, dw, db = layer.backward(
                tmp_dz,
                reversed_layers[idx + 1].get_activations(),
                targets,
                self.lambda_reg,
            )
            layer.update(dw, db, self.learning_rate, self.momentum)

    def _compute_metrics(
        self, x: np.ndarray, y: np.ndarray, validation_data: tuple
    ) -> None:
        """Compute the metrics for the current epoch.

        Parameters:
        - x (numpy.ndarray): The input data.
        - y (numpy.ndarray): The target data.
        - validation_data (tuple): The validation data.

        Returns:
        None"""
        y_train_pred = self._forward_propagation(x)
        y_train_true = y[0]
        self.log_loss_history["train"].append(
            binary_cross_entropy(y_train_pred, y_train_true)
        )
        self.accuracy_history["train"].append(accuracy(y_train_pred, y_train_true))
        self.precision_history["train"].append(precision(y_train_pred, y_train_true))
        self.recall_history["train"].append(recall(y_train_pred, y_train_true))
        self.f1_score_history["train"].append(f1_score(y_train_pred, y_train_true))

        y_test_pred = self._forward_propagation(validation_data[0])
        y_test_true = validation_data[1][0]
        self.log_loss_history["valid"].append(
            binary_cross_entropy(y_test_pred, y_test_true)
        )
        self.accuracy_history["valid"].append(accuracy(y_test_pred, y_test_true))
        self.precision_history["valid"].append(precision(y_test_pred, y_test_true))
        self.recall_history["valid"].append(recall(y_test_pred, y_test_true))
        self.f1_score_history["valid"].append(f1_score(y_test_pred, y_test_true))

        # if self.lambda_reg > 0:
        #    reg_loss = (
        #        sum(np.sum(layer.weights**2) for layer in self.layers) * self.lambda_reg
        #    )
        #    total_loss = self.log_loss_history["train"][-1] + reg_loss
        #    logger.info(f"Regularization Loss: {reg_loss:.4f}, Total Loss: {total_loss:.4f}")

    def _log_metrics(self, epoch: int) -> None:
        """Log the metrics for the current epoch.

        Parameters:
        - epoch (int): The current epoch.

        Returns:
        None"""
        print(
            f"Epoch: {epoch+1} | loss: {self.log_loss_history['train'][-1]:.4f} - "
            f"acc: {self.accuracy_history['train'][-1]:.4f} - "
            f"precision: {self.precision_history['train'][-1]:.4f} - "
            f"recall: {self.recall_history['train'][-1]:.4f} - "
            f"f1_score: {self.f1_score_history['train'][-1]:.4f} | "
            f"val_loss: {self.log_loss_history['valid'][-1]:.4f} - "
            f"val_acc: {self.accuracy_history['valid'][-1]:.4f} - "
            f"val_precision: {self.precision_history['valid'][-1]:.4f} - "
            f"val_recall: {self.recall_history['valid'][-1]:.4f} - "
            f"val_f1_score: {self.f1_score_history['valid'][-1]:.4f}"
        )

    def _fit_on_batch(
        self,
        epoch: int,
        n_batches: int,
        n_samples: int,
        x_shuffled: np.ndarray,
        y_shuffled: np.ndarray,
        batch_norm: bool,
        batch_size: int,
    ) -> None:
        """Fit the model on a batch of data.

        Parameters:
        - n_batches (int): The number of batches.
        - n_samples (int): The number of samples.
        - x_shuffled (numpy.ndarray): The shuffled input data.
        - y_shuffled (numpy.ndarray): The shuffled target data.
        - batch_size (int): The batch size.

        Returns:
        None"""
        with tqdm(
            total=n_batches,
            desc=f"Epoch {epoch+1}/{self.epochs}",
            unit="batch",
            ncols=75,
            mininterval=0.001,
        ) as pbar:
            for batch_index in range(n_batches):
                start = batch_index * batch_size
                end = min(start + batch_size, n_samples)
                x_batch = x_shuffled[:, start:end]
                y_batch = y_shuffled[:, start:end]
                if batch_norm:
                    x_batch = (x_batch - x_batch.min()) / (
                        x_batch.max() - x_batch.min()
                    ) + 1e-15

                self._forward_propagation(x_batch)
                self._backward_propagation(y_batch)
                pbar.update(1)

    def _early_stopping(self, epoch: int, patience: int) -> bool:
        """Check if early stopping should be applied.

        Parameters:
        - epoch (int): The current epoch.

        Returns:"""
        if epoch - np.argmin(self.log_loss_history["valid"]) >= patience:
            logger.info(
                f"Early stopping at epoch {epoch+1} after {patience} epochs without improvement."
            )
            return True
        return False

    def _save_best_model(self, save_model: bool, current_loss: float) -> None:
        """Save the best model based on the validation loss.

        Parameters:
        - current_loss (float): The current validation loss.

        Returns:
        None"""
        if save_model and current_loss < self.best_loss:
            self.best_loss = current_loss
            self._save_model("model.pkl")

    def fit(
        self,
        x,
        y,
        validation_data,
        batch_size=None,
        patience=30,
        batch_norm=True,
        save_model=True,
        plot_results=True,
    ) -> None:
        self.input_shape = x.shape[0]
        self.output_shape = np.unique(y).shape[0]
        self._init_layers(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            hidden_layers_shapes_list=self.hidden_layers,
            activation=self.activation,
            initializer=self.initializer,
            lambda_reg=self.lambda_reg,
            optimizer=self.optimizer,
        )

        if batch_size is None:
            batch_size = x.shape[1]
            logger.debug(
                f"Batch size not provided. Using full batch of size {batch_size}"
            )
            if batch_size <= 0:
                logger.error("Batch size must be greater than 0")
                raise ValueError("Batch size must be greater than 0")
        n_samples = x.shape[1]
        n_batches = n_samples // batch_size
        if n_samples % batch_size != 0:
            n_batches += (
                1  # Account for the last batch that may be smaller than batch_size
            )

        for epoch in range(self.epochs):
            permutation = np.random.permutation(n_samples)
            x_shuffled = x[:, permutation]
            y_shuffled = y[:, permutation]

            self._fit_on_batch(
                epoch,
                n_batches,
                n_samples,
                x_shuffled,
                y_shuffled,
                batch_norm,
                batch_size,
            )
            self._compute_metrics(x, y, validation_data)
            self._log_metrics(epoch)
            self._save_best_model(save_model, self.log_loss_history["valid"][-1])
            if self._early_stopping(epoch, patience):
                break

        logger.info(
            f"\033[1m Best valid loss: {min(self.log_loss_history['valid'])} at epoch {np.argmin(self.log_loss_history['valid'])+1}\033[0m"
        )
        logger.info(
            f" Best valid accuracy: {max(self.accuracy_history['valid'])} at epoch {np.argmax(self.accuracy_history['valid'])+1}"
        )
        if plot_results:
            self._plot_results(show=True, save=True)

    def _save_model(self, path: str) -> None:
        """Save the model parameters to a file using pickle.

        Parameters:
        - path (str): The path to save the model file.

        Raises:
        - FileNotFoundError: If the model file is not saved at the given path.

        Returns:
        None"""
        model_params = {
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "hidden_layers": self.hidden_layers,
            "weights_biases": [(layer.weights, layer.biases) for layer in self.layers],
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "activation": self.activation,
            "initializer": self.initializer,
            "lambda_reg": self.lambda_reg,
            "optimizer": self.optimizer,
        }
        with open(path, "wb") as f:
            pickle.dump(model_params, f)
        if not os.path.exists(path):
            logger.error("Model file not saved")
            raise FileNotFoundError("Model file not saved")
        else:
            print(f"Model successfully saved at path: {path}")

    def load_model(self, path: str) -> None:
        """Load the model parameters from a file using pickle.

        Parameters:
        - path (str): The path to load the model file.

        Raises:
        - FileNotFoundError: If the model file is not found at the given path.

        Returns:
        None"""
        if not os.path.exists(path):
            logger.error(f"Model file not found at {path}")
            raise FileNotFoundError(f"Model file not found at {path}")

        with open(path, "rb") as f:
            model_params = pickle.load(f)
        self.epochs = model_params["epochs"]
        self.learning_rate = model_params["learning_rate"]
        self.hidden_layers = model_params["hidden_layers"]
        self.activation = model_params["activation"]
        self.initializer = model_params["initializer"]
        self.lambda_reg = model_params["lambda_reg"]
        self.optimizer = model_params["optimizer"]
        self._init_layers(
            model_params["input_shape"],
            model_params["output_shape"],
            model_params["hidden_layers"],
            model_params["activation"],
            model_params["initializer"],
            model_params["lambda_reg"],
            model_params["optimizer"],
        )
        for layer, (weights, biases) in zip(
            self.layers, model_params["weights_biases"]
        ):
            layer.weights = weights
            layer.biases = biases

        logger.info(f"Model loaded from {path}")
        logger.info(self)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts outputs for the given preprocessed data.

        Parameters:
        - x (numpy.ndarray): Already normalized and preprocessed data ready for prediction.

        Returns:
        - numpy.ndarray: The prediction results.
        """
        return self._forward_propagation(x)

    def _plot_results(self, show: bool = False, save: bool = True) -> None:
        """Plot the loss, accuracy, precision, recall, and F1 score history.

        Parameters:
        - show (bool): Whether to display the plot.
        - save (bool): Whether to save the plot.

        Returns:
        None
        """
        epoch_of_best_loss = np.argmin(self.log_loss_history["valid"])
        epoch_of_best_accuracy = np.argmax(self.accuracy_history["valid"])
        best_loss = self.log_loss_history["valid"][epoch_of_best_loss]
        best_accuracy = self.accuracy_history["valid"][epoch_of_best_accuracy]

        plt.figure(figsize=(18, 10))

        # Loss Plot
        plt.subplot(2, 3, 1)
        plt.plot(self.log_loss_history["train"], label="Training Loss")
        plt.plot(self.log_loss_history["valid"], label="Validation Loss")
        plt.scatter(
            epoch_of_best_loss,
            best_loss,
            color="red",
            label="Best Validation Loss",
            zorder=5,
        )
        plt.annotate(
            f"{best_loss:.4f}",
            (float(epoch_of_best_loss), float(best_loss)),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
        plt.title("Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Binary Cross-Entropy Loss")
        plt.legend()

        # Accuracy Plot
        plt.subplot(2, 3, 2)
        plt.plot(self.accuracy_history["train"], label="Training Accuracy")
        plt.plot(self.accuracy_history["valid"], label="Validation Accuracy")
        plt.scatter(
            epoch_of_best_accuracy,
            best_accuracy,
            color="red",
            label="Best Validation Accuracy",
            zorder=5,
        )
        plt.annotate(
            f"{best_accuracy:.2%}",
            (float(epoch_of_best_accuracy), float(best_accuracy)),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
        plt.title("Accuracy History")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        # Precision Plot
        plt.subplot(2, 3, 3)
        plt.plot(self.precision_history["train"], label="Training Precision")
        plt.plot(self.precision_history["valid"], label="Validation Precision")
        plt.title("Precision History")
        plt.xlabel("Epoch")
        plt.ylabel("Precision")
        plt.legend()

        # Recall Plot
        plt.subplot(2, 3, 4)
        plt.plot(self.recall_history["train"], label="Training Recall")
        plt.plot(self.recall_history["valid"], label="Validation Recall")
        plt.title("Recall History")
        plt.xlabel("Epoch")
        plt.ylabel("Recall")
        plt.legend()

        # F1 Score Plot
        plt.subplot(2, 3, 5)
        plt.plot(self.f1_score_history["train"], label="Training F1 Score")
        plt.plot(self.f1_score_history["valid"], label="Validation F1 Score")
        plt.title("F1 Score History")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.legend()

        plt.tight_layout()
        if save:
            if not os.path.exists("results/"):
                os.makedirs("results/")
            model_name = f"{self.hidden_layers}_{self.activation}_{self.initializer}_{self.optimizer}_{self.best_loss}.png"
            plt.savefig("results/" + model_name)
        if show:
            plt.show()
