import numpy as np


def binary_cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


# negative log ensures we return positive values
# For correct class p(i) = 1.0, -log(q(i)) measures "surprise"
# If predicted probability is high, loq(q(i)) goes towards 0, thus
# we incur very low cross entropy cost for this prediction.
def categorical_cross_entropy(y_true, y_pred, num_classes):
    y_true_one_hot = np.eye(num_classes)[y_true]
    loss = -np.sum(y_true_one_hot * np.log(y_pred))
    return loss
