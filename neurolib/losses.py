import numpy as np


def mse(y_true, y_pred):
    """
    Calculates the Mean Squared Error (MSE) loss.

    Args:
        y_true: The ground truth values, as a NumPy array.
        y_pred: The predicted values, as a NumPy array of the same shape as y_true.

    Returns:
        The mean squared error as a float.
    
    Mean Squared Error (MSE) loss function and its derivative.

    MSE is a common loss function used for regression tasks, where the goal is to
    predict a continuous value. It measures the average of the squared differences
    between the true values (y_true) and the predicted values (y_pred).
    """

    loss = np.mean(np.power(y_true - y_pred, 2))

    return loss


def mse_prime(y_true, y_pred):
    """
    Calculates the derivative of the MSE loss with respect to the predicted values.

    Args:
        y_true: The ground truth values, as a NumPy array.
        y_pred: The predicted values, as a NumPy array of the same shape as y_true.

    Returns:
        The derivative of the MSE loss with respect to y_pred, as a NumPy array.
    
    Mean Squared Error (MSE) loss function and its derivative.
    
    MSE is a common loss function used for regression tasks, where the goal is to
    predict a continuous value. It measures the average of the squared differences
    between the true values (y_true) and the predicted values (y_pred).
    """

    loss_prime = 2 * (y_pred - y_true) / np.size(y_true)

    return loss_prime


def binary_cross_entropy(y_true, y_pred):
    
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)


def categorical_cross_entropy(y_true, y_pred):
    """
    Calculates the categorical cross-entropy loss.

    Args:
        y_true: The ground truth labels, as a one-hot encoded NumPy array of shape (batch_size, num_classes).
        y_pred: The predicted probabilities, as a NumPy array of shape (batch_size, num_classes).

    Returns:
        The categorical cross-entropy loss, as a float.
    """

    # Clip y_pred to avoid log(0) errors
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    # Calculate the loss
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    return loss

def categorical_cross_entropy_prime(y_true, y_pred):
    """
    Calculates the derivative of the categorical cross-entropy loss with respect to the predicted probabilities.

    Args:
        y_true: The ground truth labels, as a one-hot encoded NumPy array of shape (batch_size, num_classes).
        y_pred: The predicted probabilities, as a NumPy array of shape (batch_size, num_classes).

    Returns:
        The derivative of the categorical cross-entropy loss, as a NumPy array of shape (batch_size, num_classes).
    """

    # Clip y_pred to avoid division by zero errors
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    
    # Add epsilon to y_pred to avoid division by zero
    epsilon = 1e-7
    y_pred = y_pred + epsilon
    
    # Calculate the derivative
    loss_prime = -y_true / y_pred
    
    return loss_prime