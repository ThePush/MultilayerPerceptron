import pandas as pd
import matplotlib
from dataclasses import dataclass
from loguru import logger


from config.config import (
    MODEL_PATH,
    COLUMNS,
    TARGET,
)
from src.preprocessing import (
    stratified_train_test_split,
    min_max_normalize_features,
    one_hot_encode,
    standardize_dataset,
)
from src.losses import binary_cross_entropy
from src.metrics import print_metrics
from NeuralNetwork import NeuralNetwork

matplotlib.use("TkAgg")


@dataclass
class MultilayerPerceptronInterface:
    def mlp_split(
        self,
        dataset_path: str,
        split: float,
    ) -> None:
        """
        Split the dataset into training and testing sets.

        Parameters:
        - dataset_path (str): Path to the dataset.
        - split (float): Percentage of the dataset to use for testing.

        Returns:
        - None
        """
        pass

    def mlp_train(
        self,
        dataset_path: str,
        layer: list[int],
        epochs: int,
        batch_size: int,
        learning_rate: float,
        initializer: str,
        activation: str,
        optimizer: str,
        momentum: float,
        lambda_reg: float,
        patience: int,
        random_state: int,
        plot_results: bool,
    ) -> NeuralNetwork:
        """
        Train a multilayer perceptron model on the provided dataset.

        Parameters:
        - dataset_path (str): Path to the dataset.
        - layer (list[int]): The number of neurons in each hidden layer.
        - epochs (int): Number of epochs for training.
        - batch_size (int): Batch size for training.
        - learning_rate (float): Learning rate for optimizer.
        - initializer (str): Type of weight initializer to use.
        - activation (str): Type of activation function to use.
        - optimizer (str): Type of optimizer to use.
        - momentum (float): Momentum for Nesterov optimizer.
        - lambda_reg (float): Regularization parameter for L2 regularization.
        - patience (int): Number of epochs to wait before early stopping.
        - random_state (int): Random seed for reproducibility.
        - plot_results (bool): Flag to plot the results of the model.
        """
        pass

    def mlp_predict(self, dataset_path: str) -> None:
        """
        Load the model and predict on the provided dataset.

        Parameters:
        - dataset_path (str): Path to the dataset.

        Returns:
        - None
        """
        pass


@dataclass
class MultilayerPerceptron(MultilayerPerceptronInterface):
    @staticmethod
    def preprocess_data(dataset_path: str) -> pd.DataFrame:
        """
        Load and preprocess the dataset.

        Parameters:
        - dataset_path (str): Path to the dataset.

        Returns:
        - pd.DataFrame: Preprocessed dataset.
        """
        try:
            df = pd.read_csv(dataset_path, header=None)
            df = df.dropna()
            if df.shape[0] == 0:
                logger.error("Dataset is empty")
                raise ValueError("Dataset is empty")
            df.columns = COLUMNS
            if df[TARGET].dtype == "object":
                df[TARGET] = df[TARGET].map({"M": 1, "B": 0})
            return df
        except Exception as e:
            logger.error(f"Error: preprocessing failed: {e}")
            raise ValueError(f"Error: preprocessing failed: {e}")

    def mlp_split(
        self,
        dataset_path: str,
        split: float,
    ) -> None:
        """
        Split the dataset into training and testing sets.

        Parameters:
        - dataset_path (str): Path to the dataset.
        - split (float): Percentage of the dataset to use for testing.

        Returns:
        - None
        """
        try:
            dataset = self.preprocess_data(dataset_path)
            stratified_train_test_split(
                dataset,
                TARGET,
                test_size=split,
                random_state=None,
                save=True,
            )
        except Exception as e:
            logger.error(f"Error: splitting failed: {e}")
            raise ValueError(f"Error: splitting failed: {e}")

    def mlp_train(
        self,
        dataset_path: str,
        layer: list[int],
        epochs: int,
        batch_size: int,
        learning_rate: float,
        initializer: str,
        activation: str,
        optimizer: str,
        momentum: float,
        lambda_reg: float,
        patience: int,
        random_state: int,
        plot_results: bool,
    ) -> NeuralNetwork:
        """
        Train a multilayer perceptron model on the provided dataset.

        Parameters:
        - dataset_path (str): Path to the dataset.
        - layer (list[int]): The number of neurons in each hidden layer.
        - epochs (int): Number of epochs for training.
        - batch_size (int): Batch size for training.
        - learning_rate (float): Learning rate for optimizer.
        - initializer (str): Type of weight initializer to use.
        - activation (str): Type of activation function to use.
        - optimizer (str): Type of optimizer to use.
        - momentum (float): Momentum for Nesterov optimizer.
        - lambda_reg (float): Regularization parameter for L2 regularization.
        - patience (int): Number of epochs to wait before early stopping.
        - random_state (int): Random seed for reproducibility.
        - plot_results (bool): Flag to plot the results of the model.
        """
        try:
            df = self.preprocess_data(dataset_path)

            (
                X_train,
                X_test,
                y_train,
                y_test,
            ) = stratified_train_test_split(
                df,
                TARGET,
                test_size=0.3,
                random_state=random_state,
                save=False,
            )

            X_train = X_train.drop("id", axis=1)
            X_test = X_test.drop("id", axis=1)
            X_train, X_test, y_train, y_test = standardize_dataset(
                X_train, X_test, y_train, y_test
            )
            X_train = X_train.T.to_numpy()
            X_test = X_test.T.to_numpy()
            y_train = y_train.T.to_numpy()
            y_test = y_test.T.to_numpy()

            model = NeuralNetwork(
                epochs=epochs,
                learning_rate=learning_rate,
                hidden_layers=layer,
                activation=activation,
                initializer=initializer,
                lambda_reg=lambda_reg,
                optimizer=optimizer,
                momentum=momentum,
            )
            model.fit(
                x=X_train,
                y=y_train,
                validation_data=(X_test, y_test),
                batch_size=batch_size,
                patience=patience,
                save_model=True,
                batch_norm=True,
                plot_results=plot_results,
            )
            return model
        except Exception as e:
            logger.error(f"Error: training failed: {e}")
            raise ValueError(f"Error: training failed: {e}")

    def mlp_predict(self, dataset_path: str) -> tuple:
        """
        Load the model and predict on the provided dataset.

        Parameters:
        - dataset_path (str): Path to the dataset.

        Returns:
        - tuple: Accuracy, precision, recall, f1, loss
        """

        try:
            df = self.preprocess_data(dataset_path)

            X = df.drop("id", axis=1)
            X = X.drop(TARGET, axis=1)
            X = min_max_normalize_features(X).T.to_numpy()

            y = df[TARGET]
            y = one_hot_encode(y).T.to_numpy()

            model = NeuralNetwork()
            model.load_model(MODEL_PATH)
            y_pred = model.predict(X)

            accuracy, precision, recall, f1 = print_metrics(y_pred, y[0])
            loss = binary_cross_entropy(y_pred, y[0])
            logger.info(f" Loss: {loss:.2f}")
            return accuracy, precision, recall, f1, loss
        except Exception as e:
            logger.error(f"Error: prediction failed: {e}")
            raise ValueError(f"Error: prediction failed: {e}")
