class Layer:
    """
    Abstract base class for layers in a neural network.

    All concrete layer classes should inherit from this class and implement
    the `forward` and `backward` methods.
    
    **Layers Class (Abstract Base Class)**

    This class serves as a blueprint for defining various types of layers in a neural network.
    It provides a common interface for forward and backward propagation, ensuring consistency
    across different layer implementations.

    **Key Features:**

    - Defines the abstract methods `forward` and `backward` that must be implemented by
    concrete layer classes.
    - Stores the input and output of the layer for backpropagation.
    - Acts as a foundation for building specific layer types (e.g., Dense, Convolutional, etc.).

    **Usage:**

    1. Create concrete subclasses of `Layers` to implement specific layer functionalities.
    2. Override the `forward` method to define the layer's forward computation.
    3. Override the `backward` method to define the layer's backward computation and parameter updates.

    **Example:**

    ```python
    class Dense(Layers):  # Example of a concrete layer subclass
        def __init__(self, input_size, output_size):
            # ... (Initialize weights and biases)

        def forward(self, input):
            # ... (Perform matrix multiplication and activation)

        def backward(self, output_gradient, learning_rate):
            # ... (Calculate gradients and update weights and biases)

    """

    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        """
        Performs the forward pass through the layer.

        Must be implemented by concrete subclasses.

        Args:
            input: The input data to the layer.

        Returns:
            The output of the layer.
        """

        raise NotImplementedError("Subclasses must implement the forward method.")

    def backward(self, output_gradient, learning_rate, L1, L2, L1_reg, L2_reg):
        """
        Performs the backward pass through the layer.

        Must be implemented by concrete subclasses.

        Args:
            output_gradient: The gradient of the loss function with respect to the output of this layer.
            learning_rate: The learning rate used for updating weights.

        Returns:
            The gradient of the loss function with respect to the input of this layer.
        """

        raise NotImplementedError("Subclasses must implement the backward method.")