#!/usr/bin/env python
import numpy as np

from helpers.readers import get_mlxtend_mnist
from helpers.viewer import plot_digit2

if __name__ == '__main__':
    X, y = get_mlxtend_mnist()

    # Use only label 0 and 0 or 1 as result
    y_new = np.zeros(y.shape)
    y_new[np.where(y == 0.0)[0]] = 1
    y = y_new

    # Divide data to train and test
    size_train = 60000
    size_test = X.shape[0] - size_train    # amount of rows

    X_train, X_test = X[:size_train].T, X[size_train:].T    # (5000, 784 (28x28)
    y_train, y_test = y[:size_train].T, y[size_train:].T    # (5000,), reshape can be used instead

    # Shuffle training set (why?)
    np.random.seed(42)
    shuffled_indexes = np.random.permutation(size_train)

    X_train, y_train = X_train[:, shuffled_indexes], y_train[:, shuffled_indexes]
