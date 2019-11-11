#!/usr/bin/env python
import numpy as np

from helpers.readers import get_mlxtend_mnist
from helpers.viewer import plot_digit2

if __name__ == '__main__':
    X, y = get_mlxtend_mnist()

    X_train = X[:].T
    plot_digit2(X_train[:, 3])

