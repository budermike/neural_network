# Neurolib

Welcome to Neurolib, a journey into the fascinating world of neuroinformatics and neural networks through the creation of a Python library.
<br/><br/><br/>

### About

Neurolib is my personal exploration into neuroinformatics, driven by a deep curiosity and passion for understanding neural networks. While I may not have prior experience in neuroinformatics, I am excited to embark on this journey and share my progress with you.
<br/><br/><br/>

### Goals

The primary goal of Neurolib is to develop a versatile and efficient Python library for neural network experimentation and implementation. Through this project, I aim to learn, grow, and contribute to the field of neuroinformatics.
<br/><br/><br/>

### To-Do

- Improve training performance: Address overfitting issue on the MNIST dataset and enhance accuracy for self-written numbers.
- Looking that all activation functions are Pickel compatible.
- Test convolutional layer, optimizer and reshape
<br/><br/><br/>

## Code Documentation

### General Methods (network.py)
<br/>

### - one_hot_encode(labels)

#### Description:
Performs one-hot encoding for the given labels.

#### Parameters:
- `labels` (numpy.ndarray): The input labels to be one-hot encoded.

#### Returns:
- numpy.ndarray: The one-hot encoded labels.
<br/>

### - predict(network, input)

#### Description:
Performs a forward pass through the neural network.

#### Parameters:
- `network` (list): List of layers comprising the neural network.
- `input` (numpy.ndarray): The input data for prediction.

#### Returns:
- numpy.ndarray: The output prediction of the neural network.
<br/>

### - train(network, loss, loss_prime, x_train, y_train, epochs=1000, learning_rate=0.01, verbose=True, detailed_verbose=True, statistics=True, L1=False, L2=False, L1_reg=0.0, L2_reg=0.0)

#### Description:
Trains the neural network using the specified training data and hyperparameters.

#### Parameters:
- `network` (list): List of layers comprising the neural network.
- `loss` (function): The loss function used for training.
- `loss_prime` (function): The derivative of the loss function.
- `x_train` (numpy.ndarray): The input training data.
- `y_train` (numpy.ndarray): The target training data.
- `epochs` (int): Number of training epochs. (default: 1000)
- `learning_rate` (float): Learning rate for gradient descent. (default: 0.01)
- `verbose` (bool): Whether to print training progress. (default: True)
- `detailed_verbose` (bool): Whether to print detailed training progress. (default: True)
- `statistics` (bool): Whether to print training statistics. (default: True)
- `L1` (bool): Whether to apply L1 regularization. (default: False)
- `L2` (bool): Whether to apply L2 regularization. (default: False)
- `L1_reg` (float): L1 regularization parameter. (default: 0.0)
- `L2_reg` (float): L2 regularization parameter. (default: 0.0)
<br/>

### - show_para(epochs, learning_rate, network, L1_reg, L2_reg)

#### Description:
Displays the hyperparameters used for training.

#### Parameters:
- `epochs` (int): Number of training epochs.
- `learning_rate` (float): Learning rate for gradient descent.
- `network` (list): List of layers comprising the neural network.
- `L1_reg` (float): L1 regularization parameter.
- `L2_reg` (float): L2 regularization parameter.
<br/>

### - accuracy(network, x_test, y_test)

#### Description:
Evaluates the accuracy of the trained network on the test data.

#### Parameters:
- `network` (list): List of layers comprising the neural network.
- `x_test` (numpy.ndarray): The input test data.
- `y_test` (numpy.ndarray): The target test data.
<br/>

### - test_network(network, x_test, y_test, title, show_img=False, reshape_x=None, reshape_y=None)

#### Description:
Tests the trained network on the test data and displays results.

#### Parameters:
- `network` (list): List of layers comprising the neural network.
- `x_test` (numpy.ndarray): The input test data.
- `y_test` (numpy.ndarray): The target test data.
- `title` (str): Title for the test results.
- `show_img` (bool): Whether to display images. (default: False)
- `reshape_x` (int): Reshape dimension for x data.
- `reshape_y` (int): Reshape dimension for y data.
<br/>

### - own_image_predict(network, x_test, title, reshape_x, reshape_y)

#### Description:
Makes predictions on custom images using the trained network.

#### Parameters:
- `network` (list): List of layers comprising the neural network.
- `x_test` (numpy.ndarray): The input test data.
- `title` (str): Title for the prediction results.
- `reshape_x` (int): Reshape dimension for x data.
- `reshape_y` (int): Reshape dimension for y data.
<br/>

### - add_parent_dir(dir_path)

#### Description:
Adds the parent directory of the given directory to the system path.

#### Parameters:
- `dir_path` (str): The directory path whose parent directory will be added to the system path.

#### Returns:
- parent directory
<br/>

### - save_network(network, directory, filename)

#### Description:
Saves the trained network in the specified directory with the specified filename.

#### Parameters:
- `network` (list): List of layers comprising the neural network.
- `directory` (str): The directory where the network should be stored.
- `filename` (str): The name of the file to save the trained network.

#### Filetype: .pkl (pickle file)
<br/>

### - load_network(system_path, file_name)

#### Description:
Loads the trained network from the specified file in the given system path.

#### Parameters:
- `system_path` (str): The system path where the network file is located.
- `file_name` (str): The name of the file containing the trained network.

#### Returns:
- The loaded network object.

#### Filetype: .pkl (pickle file)
<br/>

### Activation Functions (activations.py)
<br/>

### - Sigmoid()

#### Description:
Sigmoid activation function.

Squashes values to the range [0, 1], suitable for binary classification problems.

**Usage:**

```python
activation_layer = Sigmoid()  # Create a sigmoid activation layer
output = activation_layer.forward(input_data)  # Apply sigmoid activation
gradient = activation_layer.backward(output_gradient, learning_rate=0.01)  # Calculate gradient
```
<br/>

### - Softmax()

#### Description:
Softmax activation function, used for multi-class classification.

Outputs probabilities that sum to 1. Not suitable for hidden layers due to computational cost.

**Usage:**

```python
output_layer = Softmax()  # Use Softmax in the output layer
output = output_layer.forward(logits)  # Apply softmax to logits (unnormalized scores)
predicted_class = np.argmax(output, axis=1)  # Get the class with the highest probability
```
<br/>

### - Hyperbolic Tangent (Tanh())

#### Description:
Hyperbolic Tangent (Tanh) activation function.

Squashes values to the range [-1, 1], useful for preserving information about the sign of inputs.
Suitable for general-purpose activation in neural networks.

**Usage:**

```python
activation_layer = Tanh()  # Create a tanh activation layer
output = activation_layer.forward(input_data)  # Apply tanh activation
gradient = activation_layer.backward(output_gradient, learning_rate=0.01)  # Calculate gradient
```
<br/>

### - Rectified Linear Unit (ReLU())

#### Description:
Rectified Linear Unit (ReLU) activation function.

Simple and efficient activation function, often used as a default choice.
Can suffer from the "dying ReLU" problem if not properly initialized.
Suitable for general-purpose activation in neural networks.

**Usage:**

```python
activation_layer = ReLU()  # Create a ReLU activation layer
output = activation_layer.forward(input_data)  # Apply ReLU activation
gradient = activation_layer.backward(output_gradient, learning_rate=0.01)  # Calculate gradient
```
<br/>

### - Leaky Rectified Linear Unit (LeakyReLU())

#### Description:
Leaky Rectified Linear Unit (Leaky ReLU) activation function.

Addresses the dying ReLU problem by allowing small negative values to pass through with a scaled slope.
Introduces a hyperparameter `alpha` to control the slope.
Suitable for general-purpose activation, especially when dealing with sparse inputs.

**Usage:**

```python
activation_layer = LeakyReLU(alpha=0.1)  # Create a Leaky ReLU layer with alpha=0.1
output = activation_layer.forward(input_data)  # Apply Leaky ReLU activation
gradient = activation_layer.backward(output_gradient, learning_rate=0.01)  # Calculate gradient
```
<br/>

### - Parametric Rectified Linear Unit (PReLU())

#### Description:
Parametric Rectified Linear Unit (PReLU) activation function.

Similar to Leaky ReLU, but instead of a fixed `alpha`, learns the optimal slope for negative values during training.
Introduces more parameters but can be more flexible.

**Usage:**

```python
activation_layer = PReLU(alpha=0.1)  # Create a PReLU layer with initial alpha=0.1
output = activation_layer.forward(input_data)  # Apply PReLU activation
gradient = activation_layer.backward(output_gradient, learning_rate=0.01)  # Calculate gradient and update alpha
```
<br/>

### - Exponential Linear Unit (ELU())

#### Description:
Exponential Linear Unit (ELU) activation function.

Pushes mean unit activations closer to zero, potentially speeding up learning.
More computationally expensive than ReLU variants.

**Usage:**

```python
activation_layer = ELU(alpha=0.5)  # Create an ELU layer with alpha=0.5
output = activation_layer.forward(input_data)  # Apply ELU activation
gradient = activation_layer.backward(output_gradient, learning_rate=0.01)  # Calculate gradient
```
<br/>

### - Swish()

#### Description:
Swish activation function, a smooth, non-monotonic function that can outperform ReLU in some cases.

**Usage:**

```python
activation_layer = Swish(beta=0.5)  # Create a Swish layer with beta=0.5
output = activation_layer.forward(input_data)  # Apply Swish activation
gradient = activation_layer.backward(output_gradient, learning_rate=0.01)  # Calculate gradient
```
<br/>

### Loss Functions (losses.py)

### - Mean Squared Error (MSE) Loss Function

#### Description:
Mean Squared Error (mse()) loss function and its derivative (mse_prime()).

MSE is a common loss function used for regression tasks, where the goal is to predict a continuous value. It measures the average of the squared differences between the true values (`y_true`) and the predicted values (`y_pred`).

**Usage:**

```python
loss = mse(y_true, y_pred)  # Calculate the MSE loss
gradient = mse_prime(y_true, y_pred)  # Calculate the gradient of the MSE loss
```
<br/>

### - Binary Cross-Entropy Loss Function

#### Description:
Binary Cross-Entropy loss function and its derivative.

Binary Cross-Entropy (BCE) is a common loss function used for binary classification tasks, where the goal is to predict probabilities of two classes (0 and 1). It measures the difference between the true binary labels (`y_true`) and the predicted probabilities (`y_pred`).

**Usage:**

```python
loss = binary_cross_entropy(y_true, y_pred)  # Calculate the binary cross-entropy loss
gradient = binary_cross_entropy_prime(y_true, y_pred)  # Calculate the gradient of the binary cross-entropy loss
```
<br/>

### - Categorical Cross-Entropy Loss Function

#### Description:
Categorical Cross-Entropy loss function and its derivative.

Categorical Cross-Entropy (CCE) is a common loss function used for multi-class classification tasks, where the goal is to predict the probability distribution over multiple classes. It measures the difference between the true labels (`y_true`) and the predicted probabilities (`y_pred`).

**Usage:**

```python
loss = categorical_cross_entropy(y_true, y_pred)  # Calculate the categorical cross-entropy loss
gradient = categorical_cross_entropy_prime(y_true, y_pred)  # Calculate the gradient of the categorical cross-entropy loss
```
<br/>

### Activation Layer (activation_l.py)

#### Description:
Activation layer in a neural network, applying a non-linear activation function during the forward pass and computing its derivative during the backward pass.

**Activation Layer**

This class represents an activation layer in a neural network. It applies a non-linear activation function to its input during the forward pass and calculates the derivative of the activation function during the backward pass.

#### Key Features:
- Inherits from the `Layer` class, providing a common interface for neural network layers.
- Stores the activation function and its derivative for efficient calculations.
- Implements the `forward` and `backward` methods for forward and backward propagation.

#### Usage:
1. Create an `Activation` object with the desired activation function and its derivative.
2. Call `forward(input)` to apply the activation function during the forward pass.
3. Call `backward(output_gradient, learning_rate)` to calculate the gradient during backpropagation.

#### Example:
```python
activation_layer = Activation(activation=np.tanh, activation_prime=lambda x: 1 - np.tanh(x)**2)
output = activation_layer.forward(input_data)
gradient = activation_layer.backward(output_gradient, learning_rate=0.01)
```
<br/>

### Dense Layer (dense_l.py)

#### Description:
Dense layer in a neural network, performing a linear transformation of its input followed by optional activation.

**Dense Layer**

This class represents a dense layer in a neural network. It performs a linear transformation of its input, followed by optional activation.

#### Key Features:
- Inherits from the `Layer` class, providing a common interface for neural network layers.
- Initializes weights and biases with random values for learning.
- Implements the `forward` method for forward propagation.
- Implements the `backward` method for backpropagation and parameter updates.

#### Usage:
1. Create a `Dense` object with specified input and output sizes.
2. Call `forward(input)` to perform the linear transformation and activation during the forward pass.
3. Call `backward(output_gradient, learning_rate)` to calculate gradients and update weights and biases during backpropagation.

#### Example:
```python
dense_layer = Dense(10, 5)  # Create a dense layer with 10 inputs and 5 outputs
output = dense_layer.forward(input_data)  # Perform forward pass
gradient = dense_layer.backward(output_gradient, learning_rate=0.01)  # Perform backpropagation
```
<br/><br/><br/>

## Contribution Guidelines

To ensure a smooth and collaborative development process, please adhere to the following guidelines when contributing to Neurolib:

1. **Fork the Repository**: Before making any changes, fork the Neurolib repository to your GitHub account.

2. **Branch Naming Convention**: When working on a new feature or bug fix, create a descriptive branch name that reflects the purpose of your changes.

3. **Code Style**: Follow the existing code style and formatting conventions used in the project.

4. **Commit Messages**: Write clear and descriptive commit messages that summarize the purpose of your changes.

5. **Pull Requests**: When submitting a pull request, provide a detailed description of the changes made and any relevant context. Ensure that your code passes all tests and does not introduce any regressions.

6. **Testing**: If applicable, include tests to cover the functionality added or modified by your changes.

7. **Documentation**: Update the documentation as needed to reflect any changes or additions to the codebase.
<br/><br/><br/>

## Collaboration

Join me in shaping the future of neural networks and neuroinformatics! Whether you're an experienced practitioner or just starting out, your contributions, ideas, and insights are invaluable.

Let's collaborate and advance the field together. Feel free to reach out for questions, feedback, or collaboration opportunities:
- Email: mathmeetsart01@gmail.com
- Instagram: [art_meets_math](https://www.instagram.com/art_meets_math/)
<br/><br/><br/>

**About Me**

I'm a Swiss dude fascinated by the hidden beauty within mathematics and the power of code to reveal it. My passion led me to pursue a degree in math and neuroinformatics, combining my love for numbers with the mysteries of the brain. I'm excited to explore the intersection of math, art, and technology, and I ultimately aspire to make meaningful contributions in the field of science.

Feel free to get in touch, share your thoughts, and join me on this exciting journey!
