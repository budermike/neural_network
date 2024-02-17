import numpy as np
from .layer import Layer


class Dense(Layer):
    """
    Represents a dense layer in a neural network.

    Args:
        input_size: The number of input units for the layer.
        output_size: The number of output units for the layer.
    
    **Dense Layer**
    
    This class implements a fully-connected (dense) layer for a neural network.
    It performs a linear transformation of its input, followed by optional activation.
    
    **Key Features:**
    
    - Inherits from the `Layer` class, providing a common interface for neural network layers.
    - Initializes weights and biases with random values for learning.
    - Implements the `forward` method for forward propagation.
    - Implements the `backward` method for backpropagation and parameter updates.
    
    **Usage:**
    
    1. Create a `Dense` object with specified input and output sizes.
    2. Call `forward(input)` to perform the linear transformation and activation during the forward pass.
    3. Call `backward(output_gradient, learning_rate)` to calculate gradients and update weights and biases during backpropagation.
    
    **Example:**
    
    ```python
    dense_layer = Dense(10, 5)  # Create a dense layer with 10 inputs and 5 outputs
    output = dense_layer.forward(input_data)  # Perform forward pass
    gradient = dense_layer.backward(output_gradient, learning_rate=0.01)  # Perform backpropagation
    
    
    """

    def __init__(self, input_size, output_size):
        super().__init__()  # Call the constructor of the parent class (Layer)
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        """
        Performs the forward pass through the dense layer.

        Args:
            input: The input data to the layer, as a NumPy array of shape (batch_size, input_size).

        Returns:
            The output of the layer, as a NumPy array of shape (batch_size, output_size).
        """

        self.input = input  # Store the input for the backward pass
        output = np.dot(self.weights, self.input) + self.bias

        return output

    def backward(self, output_gradient, learning_rate, L1, L2, L1_reg, L2_reg):
        """
        Performs the backward pass through the dense layer.

        Args:
            output_gradient: The gradient of the loss function with respect to the output of this layer.
            learning_rate: The learning rate used for updating weights.

        Returns:
            The gradient of the loss function with respect to the input of this layer.
        """
        
        weights_gradient = np.dot(output_gradient, self.input.T)  # Calculate gradients of weights
        if L1:
            l1_grad = L1_reg * np.sign(self.weights)
            weights_gradient += l1_grad 
        if L2:
            l2_grad = 0.5 * L2_reg * self.weights
            weights_gradient += l2_grad
        
        input_gradient = np.dot(self.weights.T, output_gradient)  # Calculate gradient of input
        self.weights -= learning_rate * weights_gradient  # Update weights using gradient descent
        self.bias -= learning_rate * np.sum(output_gradient, axis=1, keepdims=True) # Update bias

        return input_gradient