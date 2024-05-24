import argparse
from loguru import logger
from MultilayerPerceptron import MultilayerPerceptron

if __name__ == "__main__":
    mlp = MultilayerPerceptron()
    parser = argparse.ArgumentParser(
        description="Run a multilayer perceptron model on provided data."
    )

    parser.add_argument(
        "--split",
        type=float,
        help="Percentage of the dataset to use for testing",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Flag to train the model",
    )
    parser.add_argument(
        "--layer",
        type=int,
        nargs="+",
        default=[24, 24, 24],
        help="The number of neurons in each hidden layer",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10_000,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.00314,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--initializer",
        type=str,
        default="he_uniform",
        choices=["xavier_uniform", "xavier_normal", "he_normal", "he_uniform"],
        help="Type of weight initializer to use",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "relu"],
        help="Type of activation function to use",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="nesterov",
        choices=["gradient_descent", "adam", "nesterov"],
        help="Type of optimizer to use",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.99,
        help="Momentum for Nesterov optimizer",
    )
    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=0.0,
        help="Regularization parameter for L2 regularization",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/train.csv",
        help="Path to the dataset to use for training",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Flag to plot the results of the model",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Number of epochs to wait before early stopping",
    )

    parser.add_argument(
        "--predict",
        type=str,
        help="Path to the dataset to use for prediction",
    )

    args = parser.parse_args()

    if args.split:
        try:
            mlp.mlp_split(args.dataset, args.split)
        except Exception as e:
            logger.error(f"Error splitting dataset: {e}")

    if args.train:
        try:
            model = mlp.mlp_train(
                dataset_path=args.dataset,
                layer=args.layer,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                initializer=args.initializer,
                activation=args.activation,
                optimizer=args.optimizer,
                momentum=args.momentum,
                lambda_reg=args.lambda_reg,
                patience=args.patience,
                random_state=42,
                plot_results=args.plot,
            )
        except Exception as e:
            logger.error(f"Error training model: {e}")

    if args.predict:
        try:
            mlp.mlp_predict(args.predict)
        except Exception as e:
            logger.error(f"Error predicting with model: {e}")
