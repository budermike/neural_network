import os
import sys
import numpy as np

# Get the absolute path to the parent directory (Neural Network)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from neurolib.network import one_hot_encode, load_network, test_network, accuracy

def load_mnist():
    # Define the path to the parent directory of the current file
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Define the path to the data folder containing the MNIST dataset
    data_folder = os.path.join(parent_dir, 'Neural Network', 'data', 'MNIST')
    
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

#def own_data():

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = load_mnist()

# Load self made dataset
own_image = None

# Load network
loaded_network = load_network("D:\\Benutzer\\OneDrive\\Documents\\1 Mike\\Coding\\Neural Network\\training_models\\trained_networks", "MNIST_trained.pkl")

# predict
test_network(loaded_network, test_images[:10], test_labels[:10], title="Predictions", show_img=True, reshape_x=28, reshape_y=28)
accuracy(loaded_network, test_images[:100], test_labels[:100])


