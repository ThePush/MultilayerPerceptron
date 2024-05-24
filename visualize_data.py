import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import warnings
from skimpy import skim

from config.config import (
    TARGET,
    COLUMNS,
    DATASET_PATH,
)

matplotlib.use("TkAgg")
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    df = pd.read_csv(DATASET_PATH)
    columns_titles = COLUMNS
    df.columns = columns_titles

    print(df.head())
    print(df.shape)
    print(df.info())
    skim(df)
    print(df.describe(include="O"))

    plt.figure(figsize=(10, 6))
    sns.countplot(
        x="diagnosis",
        data=df,
        palette=["purple", "blue"],
    )
    plt.xlabel("Diagnosis")
    plt.ylabel("Count")
    plt.title("Diagnosis Count")
    axes = plt.gca()
    axes.set_facecolor("lightgrey")
    plt.show()

    df.hist(bins=20, figsize=(20, 20))
    plt.show()

    cols = [
        "diagnosis",
        "radius_mean",
        "texture_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
    ]
    sns.pairplot(
        df[cols],
        hue="diagnosis",
        palette=["purple", "blue"],
    )
    plt.show()

    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    columns_titles.pop(0)
    plt.figure(figsize=(12, 9))
    plt.title("Correlation Heatmap")
    sns.heatmap(
        df[columns_titles].corr(),
        annot=True,
        cmap="coolwarm",
        linewidths=0.2,
    )
    plt.show()

    corr = df.corr()
    features = corr.index[abs(corr[TARGET]) > 0.5]
    plt.figure(figsize=(12, 9))
    plt.title("Correlation Heatmap")
    sns.heatmap(
        df[features].corr(),
        annot=True,
        cmap="coolwarm",
        linewidths=0.2,
    )
    plt.show()
