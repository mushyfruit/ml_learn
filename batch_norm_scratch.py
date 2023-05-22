# Remembering ML fundamentals
import numpy as np


# Example of batch norm from "scratch"
def batch_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)

    x_norm = (x - mean) / np.sqrt(var + eps)

    # gamma and beta will be trainable parameters.
    # allows for learning an ideal shift + scaling factor.
    output = gamma * x_norm + beta

    return output, gamma, beta, x_norm, mean, var, eps
