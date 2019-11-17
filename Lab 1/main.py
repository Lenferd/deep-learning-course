#!/usr/bin/env python
import numpy as np

from helpers.readers import get_mlxtend_mnist_5000
from helpers.viewer import plot_digit2

from loss import *
from activations import *

from backward_propagation import cross_entropy_backward_propagation_regression

from helpers.metrics import classification_report


def log_debug(message, enabled=0):
    if enabled:
        print("{}".format(message))


if __name__ == '__main__':
    shape_debug = 1

    X, y = get_mlxtend_mnist_5000()
    X = X / 255

    # Use only label 0 and 0 or 1 as result
    y_new = np.zeros(y.shape)
    y_new[np.where(y == 0.0)[0]] = 1
    y = y_new

    # Divide data to train and test
    log_debug("X.shape {}".format(X.shape), shape_debug)
    log_debug("y.shape {}".format(y.shape), shape_debug)

    size_train = 4000   # change to 60000 after
    size_test = X.shape[0] - size_train    # amount of rows

    X_train, X_test = X[:size_train].T, X[size_train:].T    # (5000, 784 (28x28)
    # Here transposition will not work
    y_train, y_test = y[:size_train].reshape(1, size_train), y[size_train:].reshape(1, size_test)

    log_debug("X_train.shape {}".format(X_train.shape))
    log_debug("y_train.shape {}".format(y_train.shape))

    # Shuffle training set (why?)
    np.random.seed(42)
    shuffled_indexes = np.random.permutation(size_train)

    X_train, y_train = X_train[:, shuffled_indexes], y_train[:, shuffled_indexes]

    # Show what we have
    # index = 0
    # plot_digit2(X_train[:, index], y_train[:, index])
    # debug(X_train[:, index])

#     Calculation block
    learning_rate = 0.3

    X = X_train
    Y = y_train

    sBatch = X.shape[1]
    sHidden = 64
    sFeatures = X.shape[0]

    # Why  (x,y) (?)
    W1 = np.random.randn(sFeatures, sHidden)
    b1 = np.zeros((sHidden, 1))  # ?

    W2 = np.random.randn(sHidden, 1)
    b2 = np.zeros((1, 1))

    log_debug("sBatch {}".format(sBatch), shape_debug)
    log_debug("sHidden {}".format(sHidden), shape_debug)
    log_debug("sFeatures {}".format(sFeatures), shape_debug)

    log_debug("X.shape {}".format(X.shape), shape_debug)
    log_debug("W1.shape {}".format(W1.shape), shape_debug)
    log_debug("b1.shape {}".format(b1.shape), shape_debug)

    log_debug("W2.shape {}".format(W2.shape), shape_debug)
    log_debug("b2.shape {}".format(b2.shape), shape_debug)

    # Loop
    for i in range(2000):
        shape_debug_loop = 1
        # Propagation
        U1 = np.matmul(W1.T, X) + b1
        log_debug("U1.shape {}".format(U1.shape), shape_debug_loop)

        Y1_hat = Sigmoid.calculate(U1)
        log_debug("Y1_hat.shape {}".format(Y1_hat.shape), shape_debug_loop)

        U2 = np.matmul(W2.T, Y1_hat) + b2
        Y2_hat = Sigmoid.calculate(U2)
        log_debug("Y2_hat.shape {}".format(Y2_hat.shape), shape_debug_loop)

        # Loss: expected Y and actual result after Activation
        cost = compute_cross_entropy_loss(Y, Y2_hat)

        # Gradient W2
        dW2 = (1. / sHidden) * np.matmul(Y1_hat, (Y2_hat - Y).T)
        log_debug("dW2.shape {}".format(dW2.shape), shape_debug)

        # What is keepdims and axis 1
        db2 = (1. / sHidden) * np.sum(Y2_hat - Y, axis=1, keepdims=True)
        log_debug("db2.shape {}".format(db2.shape))

        # np.matmul(W2, (A2 - Y).T)
        # dL/dy(1)
        dA1 = np.matmul(W2, Y2_hat - Y)
        log_debug("dA1.shape {}".format(dA1.shape))

        # dL/du(1)
        dU1 = dA1 * Y1_hat * (1 - Y1_hat)
        log_debug("dU1.shape {}".format(dU1.shape))

        dW1 = (1. / sBatch) * np.matmul(dU1, X.T)
        log_debug("dW1.shape {}".format(dW1.shape))
        db1 = (1. / sHidden) * np.sum(dU1, axis=1, keepdims=True)
        log_debug("db1.shape {}".format(db1.shape))

        # Correction

        debug_correction = 0
        log_debug("W2.shape before {}".format(W2.shape), debug_correction)
        W2 = W2 - dW2 * learning_rate
        log_debug("W2.shape after {}".format(W2.shape), debug_correction)
        b2 = b2 - db2 * learning_rate

        W1 = W1 - dW1 * learning_rate
        b1 = b1 - db1 * learning_rate

        if i % 1 == 0:
            print("Epoch {}, cost {}".format(i, cost))

    # Z, A = logistic_regression(W, X_test, b)
    #
    # classification_report(A, y_test)
