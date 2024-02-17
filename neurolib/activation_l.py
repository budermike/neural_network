import numpy as np
from .layer import Layer  # Assuming the 'Layer' class is defined in a file named 'layer.py'


class Activation(Layer):
    """
    Represents an activation layer in a neural network.

    Args:
        activation: The activation function to apply.
        activation_prime: The derivative of the activation function.
    
    **Activation Layer**

    This class implements an activation layer for a neural network. It applies a
    non-linear activation function to its input during the forward pass and calculates
    the derivative of the activation function during the backward pass.

    **Key Features:**

    - Inherits from the `Layer` class, providing a common interface for neural network layers.
    - Stores the activation function and its derivative for efficient calculations.
    - Implements the `forward` and `backward` methods for forward and backward propagation.

    **Usage:**

    1. Create an `Activation` object with the desired activation function and its derivative.
    2. Call `forward(input)` to apply the activation function during the forward pass.
    3. Call `backward(output_gradient, learning_rate)` to calculate the gradient during backpropagation.

    **Example:**

    ```python
    activation_layer = Activation(activation=np.tanh, activation_prime=lambda x: 1 - np.tanh(x)**2)
    output = activation_layer.forward(input_data)
    gradient = activation_layer.backward(output_gradient, learning_rate=0.01)

    """

    def __init__(self, activation, activation_prime):
        super().__init__()  # Call the constructor of the parent class (Layer)
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        """
        Performs the forward pass through the activation layer.

        Args:
            input: The input data to be transformed by the activation function.

        Returns:
            np.ndarray: The output of the activation function.
        """

        self.input = input  # Store the input for use in the backward pass
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate, L1, L2, L1_reg, L2_reg):
        """
        Performs the backward pass through the activation layer.

        Args:
            output_gradient: The gradient of the loss function with respect to the output of this layer.
            learning_rate: The learning rate used for updating weights.

        Returns:
            np.ndarray: The gradient of the loss function with respect to the input of this layer.
        """

        return np.multiply(output_gradient, self.activation_prime(self.input))