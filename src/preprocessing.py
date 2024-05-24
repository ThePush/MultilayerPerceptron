import pandas as pd
import numpy as np
import os


def min_max_normalize_features(
    features: pd.DataFrame, skip_columns: list = []
) -> pd.DataFrame:
    """
    Normalize the features of a DataFrame with an option to skip certain columns.

    Parameters:
    - features (pd.DataFrame): DataFrame containing the features to normalize.
    - skip_columns (list of str): List of column names to skip normalization.

    Returns:
    - pd.DataFrame: DataFrame with normalized features, excluding skipped columns.
    """
    # Normalize all columns not in skip_columns
    result = features.copy()  # Copy the original DataFrame to preserve it
    for column in features.columns:
        if column not in skip_columns:
            result[column] = (features[column] - features[column].min()) / (
                features[column].max() - features[column].min()
            )

    return result


def one_hot_encode(targets: pd.Series) -> pd.DataFrame:
    """
    One-hot encode the target classes.

    Parameters:
    - targets (pd.Series): Series containing the target classes.

    Returns:
    - pd.DataFrame: DataFrame with one-hot encoded target classes.
    """
    return pd.get_dummies(targets, dtype=int)


def standardize_dataset(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    skip_columns: list = [],
) -> tuple:
    """
    Standardize the features and one-hot encode the target classes.

    Parameters:
    - X_train (pd.DataFrame): Training set features.
    - X_test (pd.DataFrame): Test set features.
    - y_train (pd.Series): Training set target.
    - y_test (pd.Series): Test set target.
    - skip_columns (list of str): List of column names to skip normalization.

    Returns:
    - X_train_std (pd.DataFrame): Standardized training set features.
    - X_test_std (pd.DataFrame): Standardized test set features.
    - y_train_ohe (pd.DataFrame): One-hot encoded training set target.
    - y_test_ohe (pd.DataFrame): One-hot encoded test set target.
    """
    # Standardize the features
    X_train_std = min_max_normalize_features(X_train, skip_columns)
    X_test_std = min_max_normalize_features(X_test, skip_columns)

    # One-hot encode the target classes
    y_train_ohe = one_hot_encode(y_train)
    y_test_ohe = one_hot_encode(y_test)

    return X_train_std, X_test_std, y_train_ohe, y_test_ohe


def stratified_train_test_split(
    dataset: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 17,
    save: bool = False,
    train_path: str = "data/train.csv",
    test_path: str = "data/test.csv",
) -> tuple:
    """
    Split the dataset into train and test sets while preserving the percentage of samples for each class,
    ensure that the target column remains in the same position as in the original dataset,
    save the splits to CSV files, and return the split datasets.

    Parameters:
    - dataset (pd.DataFrame): The dataset.
    - target (str): The name of the target column.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Random state for reproducibility.
    - train_path (str): Path to save the training set CSV.
    - test_path (str): Path to save the test set CSV.
    """

    X = dataset.drop(columns=[target])
    y = dataset[target]

    class_counts = y.value_counts()
    test_counts = (class_counts * test_size).round().astype(int)

    train_indices = []
    test_indices = []

    for class_value, count in test_counts.items():
        class_indices = y[y == class_value].index.tolist()
        np.random.seed(random_state)
        np.random.shuffle(class_indices)
        test_indices.extend(class_indices[:count])
        train_indices.extend(class_indices[count:])

    X_train = X.loc[train_indices]
    X_test = X.loc[test_indices]
    y_train = y.loc[train_indices]
    y_test = y.loc[test_indices]

    train_df = pd.concat([X_train, y_train], axis=1)[dataset.columns]
    test_df = pd.concat([X_test, y_test], axis=1)[dataset.columns]

    if save:
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        train_df.to_csv(train_path, index=False, header=False)
        test_df.to_csv(test_path, index=False, header=False)

    return X_train, X_test, y_train, y_test
