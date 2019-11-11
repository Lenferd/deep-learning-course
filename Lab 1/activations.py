import numpy as np


class Sigmoid:
    @staticmethod
    def calculate(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivative(x):
        return Sigmoid.calculate(x) * (1 - Sigmoid.calculate(x))


class Softmax:
    def calculate(x):
        vectSum = np.sum(x)
        return np.exp(x) / vectSum
