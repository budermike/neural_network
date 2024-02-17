from .activation_l import Activation
from .layer import Layer
import numpy as np
import matplotlib.pyplot as plt


#helper function to plot activation functions
def plot_func(x, y, title):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("activation(x)")
    plt.grid(True)
    plt.show()
    

# --- Sigmoid Activation --- Achtung wurde angepasst also noch Docstring anpassen!!!
class Sigmoid(Activation):
    """
    Sigmoid activation function.

    Squashes values to the range [0, 1], suitable for binary classification problems.
    
    **Sigmoid Activation**

    This class implements the sigmoid activation function for a neural network.
    It squashes input values to the range [0, 1], introducing non-linearity.

    **Key Features:**

    - Inherits from the `Activation` class, providing a common interface for activation layers.
    - Implements the `sigmoid` function and its derivative, `sigmoid_prime`.

    **Usage:**

    1. Create a `Sigmoid` object to use as an activation layer in your model.
    2. Call `forward(input)` during the forward pass to apply the sigmoid activation.
    3. Call `backward(output_gradient, learning_rate)` during backpropagation to calculate the gradient of the sigmoid function.

    **Example:**

    ```python
    activation_layer = Sigmoid()  # Create a sigmoid activation layer
    output = activation_layer.forward(input_data)  # Apply sigmoid activation
    gradient = activation_layer.backward(output_gradient, learning_rate=0.01)  # Calculate gradient

    """

    def __init__(self):
        """
        Initializes the Sigmoid activation function.
        """

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)


# --- Softmax ---  Achtung wurde angepasst also noch Docstring anpassen!!!
# Description: Outputs probabilities that sum to 1, used for multi-class classification. Not suitable for hidden layers due to computational cost.
class Softmax(Layer):
    """
    Softmax activation function, used for multi-class classification.

    Outputs probabilities that sum to 1. Not suitable for hidden layers
    due to computational cost.
    
    **Softmax Activation**

    This class implements the softmax activation function for a neural network.
    It's primarily used in the output layer of multi-class classification models to produce probabilities that sum to 1.
    However, it's not generally suitable for hidden layers due to its computational cost.

    **Key Features:**

    - Inherits from the `Activation` class, providing a common interface for activation layers.
    - Transforms raw output scores into probabilities, enabling class prediction.
    - Ensures the output probabilities are normalized to sum to 1.

    **Usage:**

    1. Create a `Softmax` object.
    2. Call `forward(input)` during the forward pass to apply the softmax activation.
    3. Call `backward(output_gradient, learning_rate)` during backpropagation to calculate the gradient.

    **Example:**

    ```python
    output_layer = Softmax()  # Use Softmax in the output layer
    output = output_layer.forward(logits)  # Apply softmax to logits (unnormalized scores)
    predicted_class = np.argmax(output, axis=1)  # Get the class with the highest probability

    """
    
    def forward(self, input):
        """
        Initializes the Softmax forward activation function.
        """
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, output_gradient, learning_rate, L1, L2, L1_reg, L2_reg):
        """
        Initializes the Softmax backward activation function.
        """
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)


# --- Hyperbolic Tangent (Tanh) --- Achtung wurde angepasst also noch Docstring anpassen!!!
class Tanh(Activation):
    """
    Hyperbolic Tangent (Tanh) activation function.

    Squashes values to the range [-1, 1], useful for preserving information about the sign of inputs.
    Suitable for general-purpose activation in neural networks.
    
    **Hyperbolic Tangent (Tanh) Activation**

    This class implements the hyperbolic tangent (tanh) activation function for a neural network.
    It squashes input values to the range [-1, 1], introducing non-linearity and helping to prevent vanishing gradients.

    **Key Features:**

    - Inherits from the `Activation` class, providing a common interface for activation layers.
    - Implements the `tanh` function and its derivative, `tanh_prime`.

    **Usage:**

    1. Create a `Tanh` object to use as an activation layer in your model.
    2. Call `forward(input)` during the forward pass to apply the tanh activation.
    3. Call `backward(output_gradient, learning_rate)` during backpropagation to calculate the gradient of the tanh function.

    **Example:**

    ```python
    activation_layer = Tanh()  # Create a tanh activation layer
    output = activation_layer.forward(input_data)  # Apply tanh activation
    gradient = activation_layer.backward(output_gradient, learning_rate=0.01)  # Calculate gradient

    """

    def __init__(self):
        """
        Initializes the Tanh activation function.
        """

        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)


# --- Rectified Linear Unit (ReLU) ---
class ReLU(Activation):
    """
    Rectified Linear Unit (ReLU) activation function.

    Simple and efficient activation function, often used as a default choice.
    Can suffer from the "dying ReLU" problem if not properly initialized.
    Suitable for general-purpose activation in neural networks.
    
    **Rectified Linear Unit (ReLU) Activation**

    This class implements the rectified linear unit (ReLU) activation function for a neural network.
    It outputs the input directly if it's positive, otherwise, it outputs zero. This introduces non-linearity while being computationally efficient.

    **Key Features:**

    - Inherits from the `Activation` class, providing a common interface for activation layers.
    - Implements the `relu` function and its derivative, `relu_prime`.

    **Usage:**

    1. Create a `ReLU` object to use as an activation layer in your model.
    2. Call `forward(input)` during the forward pass to apply the ReLU activation.
    3. Call `backward(output_gradient, learning_rate)` during backpropagation to calculate the gradient of the ReLU function.

    **Example:**

    ```python
    activation_layer = ReLU()  # Create a ReLU activation layer
    output = activation_layer.forward(input_data)  # Apply ReLU activation
    gradient = activation_layer.backward(output_gradient, learning_rate=0.01)  # Calculate gradient

    """

    def __init__(self):
        """
        Initializes the ReLU activation function.
        """

        relu = lambda x: np.maximum(0, x)
        relu_prime = lambda x: np.where(x > 0, 1, 0)
        super(ReLU, self).__init__(relu, relu_prime)


# --- Leaky ReLU ---
class LeakyReLU(Activation):
    """
    Leaky Rectified Linear Unit (Leaky ReLU) activation function.

    Addresses the dying ReLU problem by allowing small negative values to pass through with a scaled slope.
    Introduces a hyperparameter `alpha` to control the slope.
    Suitable for general-purpose activation, especially when dealing with sparse inputs.
    
    **Leaky Rectified Linear Unit (Leaky ReLU) Activation**

    This class implements the leaky rectified linear unit (Leaky ReLU) activation function for a neural network.
    It addresses the "dying ReLU" problem by allowing a small, non-zero gradient for negative inputs.

    **Key Features:**

    - Inherits from the `Activation` class, providing a common interface for activation layers.
    - Introduces a hyperparameter `alpha` to control the slope for negative values.
    - Implements the `leaky_relu` function and its derivative, `leaky_relu_prime`.

    **Usage:**

    1. Create a `LeakyReLU` object with the desired `alpha` value.
    2. Call `forward(input)` during the forward pass to apply the Leaky ReLU activation.
    3. Call `backward(output_gradient, learning_rate)` during backpropagation to calculate the gradient of the Leaky ReLU function.

    **Example:**

    ```python
    activation_layer = LeakyReLU(alpha=0.1)  # Create a Leaky ReLU layer with alpha=0.1
    output = activation_layer.forward(input_data)  # Apply Leaky ReLU activation
    gradient = activation_layer.backward(output_gradient, learning_rate=0.01)  # Calculate gradient

    """

    def __init__(self, alpha=0.01):
        """
        Initializes the Leaky ReLU activation function with a given alpha.

        Args:
            alpha (float, optional): Slope for negative values. Defaults to 0.01.
        """

        self.alpha = np.asarray(alpha, dtype=np.float32)  # Ensure alpha is a float32 array
        leaky_relu = lambda x: np.maximum(self.alpha * x, x)
        leaky_relu_prime = lambda x: np.where(x > 0, 1, self.alpha)
        super(LeakyReLU, self).__init__(leaky_relu, leaky_relu_prime)


# --- Parametric ReLU (PReLU) ---
# Description: Similar to Leaky ReLU, but learns the optimal slope for negative values during training. More flexible, but introduces more parameters.
class PReLU(Activation):
    """
    Parametric Rectified Linear Unit (PReLU) activation function.

    Similar to Leaky ReLU, but instead of a fixed `alpha`, learns the optimal
    slope for negative values during training. Introduces more parameters
    but can be more flexible.

    Attributes:
        alpha (float or np.ndarray): Slope for negative values.
    
    **Parametric Rectified Linear Unit (PReLU) Activation**

    This class implements the parametric rectified linear unit (PReLU) activation function for a neural network.
    It's similar to Leaky ReLU, but instead of a fixed `alpha`, it learns the optimal slope for negative values during training.
    This introduces more parameters but can lead to greater flexibility and potentially better performance.

    **Key Features:**

    - Inherits from the `Activation` class, providing a common interface for activation layers.
    - Learns the `alpha` parameter during training for adaptive negative slope.
    - Offers potential performance improvements over Leaky ReLU in some cases.

    **Usage:**

    1. Create a `PReLU` object with an optional initial `alpha` value (defaults to 0.01).
    2. Call `forward(input)` during the forward pass to apply the PReLU activation.
    3. Call `backward(output_gradient, learning_rate)` during backpropagation to calculate the gradient and update `alpha`.

    **Example:**

    ```python
    activation_layer = PReLU(alpha=0.1)  # Create a PReLU layer with initial alpha=0.1
    output = activation_layer.forward(input_data)  # Apply PReLU activation
    gradient = activation_layer.backward(output_gradient, learning_rate=0.01)  # Calculate gradient and update alpha

    """

    def __init__(self, alpha=0.01):
        """
        Initializes the PReLU activation function.

        Args:
            alpha (float or np.ndarray, optional): Slope for negative values.
                Defaults to 0.01.
        """
        self.alpha = np.asarray(alpha, dtype=np.float32)
        pr_relu = lambda x: np.maximum(self.alpha * x, x)
        pr_relu_prime = lambda x: np.where(x > 0, 1, self.alpha)
        super(PReLU, self).__init__(pr_relu, pr_relu_prime)


# --- Exponential Linear Unit (ELU) ---
# Description: Pushes mean unit activations closer to zero, potentially speeding up learning. More computationally expensive than ReLU variants.
class ELU(Activation):
    """
    Exponential Linear Unit (ELU) activation function.

    Pushes mean unit activations closer to zero, potentially speeding up learning.
    More computationally expensive than ReLU variants.

    Attributes:
        alpha (float): Hyperparameter controlling the value to which negative
            inputs are pushed.
    
    **Exponential Linear Unit (ELU) Activation**

    This class implements the exponential linear unit (ELU) activation function for a neural network.
    It's characterized by its ability to push mean unit activations closer to zero, which can potentially accelerate learning.
    However, it incurs higher computational cost compared to ReLU variants.

    **Key Features:**

    - Inherits from the `Activation` class, providing a common interface for activation layers.
    - Introduces a hyperparameter `alpha` to control the value to which negative inputs are pushed.
    - Possesses smooth and non-monotonic behavior, potentially aiding in feature learning.

    **Usage:**

    1. Create an `ELU` object with an optional `alpha` value (defaults to 1.0).
    2. Call `forward(input)` during the forward pass to apply the ELU activation.
    3. Call `backward(output_gradient, learning_rate)` during backpropagation to calculate the gradient.

    **Example:**

    ```python
    activation_layer = ELU(alpha=0.5)  # Create an ELU layer with alpha=0.5
    output = activation_layer.forward(input_data)  # Apply ELU activation
    gradient = activation_layer.backward(output_gradient, learning_rate=0.01)  # Calculate gradient

    """

    def __init__(self, alpha=1.0):
        """
        Initializes the ELU activation function.

        Args:
            alpha (float, optional): Hyperparameter controlling the value to
                which negative inputs are pushed. Defaults to 1.0.
        """

        self.alpha = alpha
        elu = lambda x: np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
        elu_prime = lambda x: np.where(x > 0, 1, self.alpha * np.exp(x))
        super().__init__(elu, elu_prime)


# --- Swish ---
# Description: A smooth, non-monotonic function that can outperform ReLU in some cases.
class Swish(Activation):
    """
    Swish activation function, a smooth, non-monotonic function that can
    outperform ReLU in some cases.

    Attributes:
        beta (float): Hyperparameter controlling the shape of the function.
    
    **Swish Activation**
    
    This class implements the Swish activation function for a neural network.
    It's a smooth, non-monotonic function that has demonstrated potential to outperform ReLU in various tasks.
    
    **Key Features:**
    
    - Inherits from the `Activation` class, providing a common interface for activation layers.
    - Introduces a hyperparameter `beta` to control the shape of the function, allowing for flexibility.
    - Offers a balance of non-linearity and smoothness, which can aid in optimization and generalization.
    
    **Usage:**
    
    1. Create a `Swish` object with an optional `beta` value (defaults to 1.0).
    2. Call `forward(input)` during the forward pass to apply the Swish activation.
    3. Call `backward(output_gradient, learning_rate)` during backpropagation to calculate the gradient.
    
    **Example:**
    
    ```python
    activation_layer = Swish(beta=0.5)  # Create a Swish layer with beta=0.5
    output = activation_layer.forward(input_data)  # Apply Swish activation
    gradient = activation_layer.backward(output_gradient, learning_rate=0.01)  # Calculate gradient
    
    """

    def __init__(self, beta=1.0):
        """
        Initializes the Swish activation function.

        Args:
            beta (float, optional): Hyperparameter controlling the shape of
                the function. Defaults to 1.0.
        """

        self.beta = beta
        swish = lambda x: x * (1 / (1 + np.exp(-self.beta * x)))  # Using sigmoid definition
        swish_prime = lambda x: x * (1 + self.beta * (1 - (1 / (1 + np.exp(-self.beta * x)))))
        super().__init__(swish, swish_prime)