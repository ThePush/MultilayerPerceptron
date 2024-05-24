from itertools import product
import numpy as np
from loguru import logger
from MultilayerPerceptron import MultilayerPerceptron


class HyperparameterTuner:
    def __init__(
        self,
        mlp: MultilayerPerceptron,
        param_grid: dict,
        dataset_train_path: str = "data/train.csv",
        dataset_test_path: str = "data/test.csv",
    ) -> None:
        """
        Initialize the HyperparameterTuner with an MLP instance and a parameter grid.

        Parameters:
        - mlp (MultilayerPerceptron): An instance of the MultilayerPerceptron class.
        - param_grid (dict): Dictionary containing hyperparameters and their possible values.
        """
        self.mlp = mlp
        self.param_grid = param_grid
        self.best_params = None
        self.best_score = np.inf
        self.dataset_train_path = dataset_train_path
        self.dataset_test_path = dataset_test_path

    def _evaluate_model(
        self,
        params: dict,
    ) -> float:
        """
        Train and evaluate the MLP model with given parameters.

        Parameters:
        - params (dict): Dictionary of parameters to set for the MLP.
        - X_train, y_train: Training data.
        - X_val, y_val: Validation data.

        Returns:
        - score (float): The evaluation score (accuracy) of the model.
        """
        self.mlp.mlp_train(
            dataset_path=self.dataset_train_path,
            layer=params["layer"],
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
            initializer=params["initializer"],
            activation=params["activation"],
            optimizer=params["optimizer"],
            momentum=params.get("momentum", 0.0),
            lambda_reg=params.get("lambda_reg", 0.0),
            patience=params["patience"],
            random_state=42,
            plot_results=False,
        )

        (
            accuracy,
            precision,
            recall,
            f1,
            loss,
        ) = self.mlp.mlp_predict(self.dataset_test_path)
        return loss

    def grid_search(
        self,
    ) -> tuple:
        """
        Perform grid search to find the best hyperparameters.

        Parameters:
        - X, y: Dataset features and target labels.
        - test_size (float): Proportion of the dataset to include in the validation split.
        - random_state (int): Seed used by the random number generator.

        Returns:
        - best_params (dict): The best hyperparameters found during the search.
        - best_score (float): The best score achieved by the best hyperparameters.
        """

        keys, values = zip(*self.param_grid.items())
        for i, v in enumerate(product(*values)):
            logger.info(f"Fitting combination {i+1} / {len(list(product(*values)))}")
            params = dict(zip(keys, v))
            score = self._evaluate_model(params)
            logger.info(f"Evaluated params: {params} => Score: {score:.4f}")

            if score < self.best_score:
                self.best_score = score
                self.best_params = params

        return self.best_params, self.best_score


if __name__ == "__main__":
    param_grid = {
        "layer": [[8, 8], [16, 16], [24, 24, 24]],
        "epochs": [200],
        "batch_size": [1, 8, None],
        "learning_rate": [0.01, 0.001, 0.00314],
        "initializer": ["xavier_normal", "he_normal"],
        "activation": ["relu", "sigmoid"],
        "optimizer": ["nesterov"],
        "momentum": [0.9],
        "lambda_reg": [0.0, 0.01],
        "patience": [20],
    }
    mlp = MultilayerPerceptron()
    tuner = HyperparameterTuner(
        mlp=mlp,
        param_grid=param_grid,
        dataset_train_path="data/train.csv",
        dataset_test_path="data/test.csv",
    )
    best_params, best_score = tuner.grid_search()
    logger.info("Grid search completed.")
    logger.info(f"Best hyperparameters: {best_params}")
    logger.info(f"Best score: {best_score:.4f}")
    logger.success("Hyperparameter tuning completed.")