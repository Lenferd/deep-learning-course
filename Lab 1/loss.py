import numpy as np


def compute_cross_entropy_loss(expected, result):
    assert(expected.shape[1] == result.shape[1])
    size = expected.shape[1]

    loss = (-1. / size) * (
            np.sum(np.multiply(expected, np.log(result)))
            + np.sum(np.multiply((1 - expected), np.log(1 - result)))
    )

    return loss
