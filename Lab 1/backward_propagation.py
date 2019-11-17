import numpy as np


def cross_entropy_backward_propagation_regression(X, expected, actual):
    size = X.shape[0]
    dW = (1.0 / size) * np.matmul(X, (actual - expected).T)
    dB = (1.0 / size) * np.sum(actual - expected, axis=1, keepdims=True)

    return dW, dB
