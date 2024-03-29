import os
import sys
import numpy as np

# Get the absolute path to the parent directory (Neural Network)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from neurolib.dense_l import Dense
from neurolib.activations import Sigmoid, Tanh, Softmax
from neurolib.losses import categorical_cross_entropy, categorical_cross_entropy_prime
from neurolib.network import train, test_network, accuracy, one_hot_encode, save_network


def load_mnist():
    # Define the path to the parent directory of the current file
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Define the path to the data folder containing the MNIST dataset
    data_folder = os.path.join(parent_dir, 'data', 'MNIST')
    
    # Load training images
    with open(os.path.join(data_folder, 'train-images.idx3-ubyte'), 'rb') as f:
        f.seek(16)
        train_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28*28)
        train_images = train_images.reshape(-1,784, 1)  # Reshape to (60000, 784, 1)

    # Load training labels
    with open(os.path.join(data_folder, 'train-labels.idx1-ubyte'), 'rb') as f:
        f.seek(8)
        train_labels = np.frombuffer(f.read(), dtype=np.uint8)

    # Load test images
    with open(os.path.join(data_folder, 't10k-images.idx3-ubyte'), 'rb') as f:
        f.seek(16)
        test_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28*28)
        test_images = test_images.reshape(-1, 784, 1)  # Reshape to (10000, 784, 1)

    # Load test labels
    with open(os.path.join(data_folder, 't10k-labels.idx1-ubyte'), 'rb') as f:
        f.seek(8)
        test_labels = np.frombuffer(f.read(), dtype=np.uint8)

    # Normalize pixel values to the range [0, 1]
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # One-hot encode the labels
    train_labels = one_hot_encode(train_labels).reshape(-1, 10, 1)
    test_labels = one_hot_encode(test_labels).reshape(-1, 10, 1)

    return (train_images, train_labels), (test_images, test_labels)


# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = load_mnist()

# Define your neural network architecture
network = [
    Dense(28*28, 32),  # Input layer to hidden layer
    Tanh(),
    Dense(32, 64),
    Sigmoid(),
    Dense(64, 10),  # Hidden layer to output layer
    Softmax()
]

#training
epochs = 3
learning_rate = 0.01
L1_reg = 0.0
L2_reg = 0.0

train(network, loss=categorical_cross_entropy, loss_prime=categorical_cross_entropy_prime, x_train=train_images, y_train=train_labels, epochs=epochs, learning_rate=learning_rate, detailed_verbose=True, L2=True, L2_reg=0.001)

print("\n--- Hyperparameters used ---\n")
print(f"epochs: {epochs}")
print(f"learning rate: {learning_rate}")
print(f"L1: {L1_reg}")
print(f"L2: {L2_reg}\n")

#testing
test_network(network=network, x_test=test_images[:10], y_test=test_labels[:10], title="Testing Network")
accuracy(network=network, x_test=test_images[:100], y_test=test_labels[:100])

#save trained network
#save_network(network, "D:\\Benutzer\\OneDrive\\Documents\\1 Mike\\Coding\\Neural Network\\training_models\\trained_networks", "MNIST_trained.pkl" )

