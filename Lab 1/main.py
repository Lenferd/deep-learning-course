#!/usr/bin/env python
import numpy as np

from helpers.readers import get_mlxtend_mnist_5000
from helpers.viewer import plot_digit2
from propagation import logistic_regression
from loss import *
from backward_propagation import cross_entropy_backward_propagation

from helpers.metrics import classification_report

if __name__ == '__main__':
    X, y = get_mlxtend_mnist_5000()
    X = X / 255

    # Use only label 0 and 0 or 1 as result
    y_new = np.zeros(y.shape)
    y_new[np.where(y == 0.0)[0]] = 1
    y = y_new

    # Divide data to train and test
    print(X.shape)
    print(y.shape)
    size_train = 4000   # change to 60000 after
    size_test = X.shape[0] - size_train    # amount of rows

    X_train, X_test = X[:size_train].T, X[size_train:].T    # (5000, 784 (28x28)
    # Here transposition will not work
    y_train, y_test = y[:size_train].reshape(1, size_train), y[size_train:].reshape(1, size_test)

    print(X_train.shape)
    print(y_train.shape)

    # Shuffle training set (why?)
    np.random.seed(42)
    shuffled_indexes = np.random.permutation(size_train)

    X_train, y_train = X_train[:, shuffled_indexes], y_train[:, shuffled_indexes]

    index = 0
    plot_digit2(X_train[:, index], y_train[:, index])
    print(X_train[:, index])

#     Calculation block
    learning_rate = 1

    X = X_train
    Y = y_train

    amInputs = X.shape[1]
    amFeatures = X.shape[0]

    W = np.random.randn(amFeatures, 1)
    b = np.zeros((1, 1))

    # Loop
    for i in range(200):
        U, A = logistic_regression(W, X, b)

        # Expected Y and actual result after Activation
        cost = compute_cross_entropy_loss(Y, A)

        # Gradient
        dW, dB = cross_entropy_backward_propagation(X, Y, A)

        # Correction
        W = W - dW * learning_rate
        b = b - dB * learning_rate

        if i % 100 == 0:
            print("Epoch {}, cost {}".format(i, cost))

    # Z, A = logistic_regression(W, X_test, b)
    #
    # classification_report(A, y_test)
