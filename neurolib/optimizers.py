import numpy as np


def stochastic_gradient_descent(X, y, learning_rate=0.01, epochs=100):
    """
    **Stochastic Gradient Descent (SGD) Optimizer**

    This function applies the Stochastic Gradient Descent optimization algorithm
    to update the weights and biases of a neural network.

    **Parameters:**

    - `X` (numpy.ndarray): The input data.
    - `y` (numpy.ndarray): The ground truth labels.
    - `learning_rate` (float, optional): The learning rate for weight updates. Default is 0.01.
    - `epochs` (int, optional): The number of training epochs. Default is 100.

    **Key Features:**

    - Updates weights and biases using Stochastic Gradient Descent.
    - Suitable for large datasets due to its stochastic nature.

    **Usage:**

    ```python
    weights, bias = stochastic_gradient_descent(X_train, y_train, learning_rate=0.01, epochs=100)
    ```

    **Example:**

    Assume X_train and y_train are the training data and labels:

    ```python
    weights, bias = stochastic_gradient_descent(X_train, y_train, learning_rate=0.01, epochs=100)
    ```

    **Returns:**

    - `weights` (numpy.ndarray): Updated weights after optimization.
    - `bias` (float): Updated bias after optimization.
    """

    m, n = X.shape
    weights = np.zeros(n)
    bias = 0

    for epoch in range(epochs):
        for i in range(m):
            # Compute gradient for each training example
            y_pred = np.dot(X[i], weights) + bias
            error = y_pred - y[i]
            gradient_weights = 2 * error * X[i]
            gradient_bias = 2 * error

            # Update weights and bias
            weights -= learning_rate * gradient_weights
            bias -= learning_rate * gradient_bias

    return weights, bias


def adam_optimizer(X, y, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, epochs=100):
    """
    **Adam Optimizer**

    This function applies the Adam optimization algorithm to update the weights and biases of a neural network.

    **Parameters:**

    - `X` (numpy.ndarray): The input data.
    - `y` (numpy.ndarray): The ground truth labels.
    - `learning_rate` (float, optional): The learning rate for weight updates. Default is 0.001.
    - `beta1` (float, optional): Exponential decay rate for the first moment estimate. Default is 0.9.
    - `beta2` (float, optional): Exponential decay rate for the second raw moment estimate. Default is 0.999.
    - `epsilon` (float, optional): Small constant to prevent division by zero. Default is 1e-8.
    - `epochs` (int, optional): The number of training epochs. Default is 100.

    **Key Features:**

    - Updates weights and biases using the Adam optimization algorithm.
    - Combines the benefits of momentum and RMSprop.
    - Efficient for a wide range of neural network architectures.

    **Usage:**

    ```python
    weights, bias = adam_optimizer(X_train, y_train, learning_rate=0.001, epochs=100)
    ```

    **Example:**

    Assume X_train and y_train are the training data and labels:

    ```python
    weights, bias = adam_optimizer(X_train, y_train, learning_rate=0.001, epochs=100)
    ```

    **Returns:**

    - `weights` (numpy.ndarray): Updated weights after optimization.
    - `bias` (float): Updated bias after optimization.
    """

    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    m_t = np.zeros(n)
    v_t = np.zeros(n)
    t = 0

    for epoch in range(epochs):
        for i in range(m):
            t += 1
            # Compute gradient for each training example
            y_pred = np.dot(X[i], weights) + bias
            error = y_pred - y[i]
            gradient_weights = 2 * error * X[i]
            gradient_bias = 2 * error

            # Update biased first moment estimate
            m_t = beta1 * m_t + (1 - beta1) * gradient_weights
            # Update biased second raw moment estimate
            v_t = beta2 * v_t + (1 - beta2) * (gradient_weights ** 2)

            # Correct bias in first moment estimate
            m_t_hat = m_t / (1 - beta1 ** t)
            # Correct bias in second moment estimate
            v_t_hat = v_t / (1 - beta2 ** t)

            # Update weights and bias
            weights -= learning_rate * m_t_hat / (np.sqrt(v_t_hat) + epsilon)
            bias -= learning_rate * gradient_bias

    return weights, bias