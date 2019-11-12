import numpy as np
from activations import Sigmoid


def logistic_regression(W, X, b):
    U = np.matmul(W.T, X) + b   # Calculation
    A = Sigmoid.calculate(U)    # Activation
    return U, A
