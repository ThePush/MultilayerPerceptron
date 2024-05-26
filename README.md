<div align="center">
      <center><h1>Multilayer Perceptron</h1></center>
    <img src="https://github.com/ThePush/mlp/assets/91064070/3c8eaba5-f4e5-4f68-8e0a-8fd557460826" width="500" height="500">
</div>

This repository contains the implementation of a Multilayer Perceptron (MLP) or Deep neural network from scratch, using only maths. It aims to predict whether a cancer is malignant or benign using the Wisconsin breast cancer dataset.


## Table of Contents

-   [Introduction](#introduction)
-   [Objectives](#objectives)
-   [Installation](#installation)
-   [Usage](#usage)
    -   [Command Line Interface](#command-line-interface)
    -   [Graphical User Interface](#graphical-user-interface)
    -   [Hyperparameter Tuning](#hyperparameter-tuning)
    -   [Visualizing the Data](#visualizing-the-data)
-   [Project Structure](#project-structure)

## Introduction

The Multilayer Perceptron (MLP) is a type of artificial neural network that consists of multiple layers of neurons, each layer fully connected to the next. This project is an implementation of an MLP from scratch without the use of high-level machine learning libraries. It features optimization algorithms such as Adam, Nesterov momentum, L2 regularization, Xavier and He weights inializer etc.

The MLP is trained to classify whether a breast cancer tumor is malignant or benign based on features extracted from cell images.

## Objectives

The main objectives of this project are:

1. Implement the core algorithms of an MLP including forward propagation, backward propagation, and gradient descent.
2. Train the MLP on the Wisconsin breast cancer dataset.
3. Evaluate the performance of the model using various metrics.
4. Provide both a command-line interface (CLI) and a graphical user interface (GUI) for ease of use.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/MultilayerPerceptron.git
    cd MultilayerPerceptron
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Command Line Interface

The CLI allows you to train and evaluate the MLP model using command-line arguments.

![image](https://github.com/ThePush/mlp/assets/91064070/9cb55bbf-0e0f-4d64-8f9f-30f90f332675)



#### Splitting the Dataset

To split the dataset into training and test sets, you can use the following command:

```bash
python mlp.py --split=0.25 --dataset=data/data.csv
```

#### Training the Model

To train and save the model and then save the plot of the results, you can use the following command:

```bash
python mlp.py --train --dataset=data/train.csv --layer 24 24 24 --batch_size=8 --learning_rate=0.00314 --initializer=he_uniform --activation=sigmoid --optimizer=nesterov --plot --patience 20
```

#### Evaluating the Model

To evaluate the model on the test set, you can use the following command:

```bash
python mlp.py --predict=data/test.csv
```

### Graphical User Interface

The GUI provides an easy-to-use interface for interacting with the MLP model.

![image](https://github.com/ThePush/mlp/assets/91064070/94216e39-b02d-4166-9a61-e7d55489a9e6)

To start the GUI, run:

```bash
python gui.py
```


### Hyperparameter Tuning

To tune the hyperparameters of the model, you can use the following command:

```bash
python HyperparameterTuner.py
```

The script does not take any arguments and will output the best hyperparameters found during the tuning process.

### Visualizing the Data

To visualize the data, you can use the following command:

```bash
python visualize_data.py
```

## Project Structure

The project structure is as follows:

```
├── config
│   ├── config.py
├── data
│   ├── data.csv
│   ├── test.csv
│   └── train.csv
├── evaluation.py
├── gui.py
├── HyperparameterTuner.py
├── Layer.py
├── mlp.py
├── model.pkl
├── MultilayerPerceptron.py
├── NeuralNetwork.py
├── README.md
├── results
├── src
│   ├── activations.py
│   ├── losses.py
│   ├── metrics.py
│   ├── preprocessing.py
│   └── weight_initializers.py
└── visualize_data.py
```
