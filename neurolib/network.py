import numpy as np
import os
import dill
import matplotlib.pyplot as plt


def one_hot_encode(labels):
    num_labels = labels.shape[0]
    num_classes = np.max(labels) + 1
    one_hot = np.zeros((num_labels, num_classes))
    one_hot[np.arange(num_labels), labels] = 1
    return one_hot


def predict(network, input):
    """
    Perform a forward pass through the neural network.

    Args:
        input_data (numpy.ndarray): The input data for prediction.
        network (list): List of layers comprising the neural network.

    Returns:
        numpy.ndarray: The output prediction of the neural network.
        
    Example Usage:
        # Make predictions
        for input_sample in input_data_to_predict:
        
            # Perform forward pass to get predictions
            prediction = predict(network, input)

            # Display the input and corresponding prediction
            print(f"Input: {input}, Prediction: {prediction}")
    """
    
    output = input
    for layer in network:
        output = layer.forward(output)
        #print(f"Output: {output}") if layer == network[0] else None
    return output


def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True, detailed_verbose = True, statistics=True, L1=False, L2=False, L1_reg=0.0, L2_reg=0.0):
    epoch_statistics = []
    
    for e in range(epochs):
        error = 0
        training_data = len(x_train)
        current_data = 0
        total_gradient_magnitude = 0
        initial_values_captured = False  # A flag to track if we've captured initial values
        
        for x, y in zip(x_train, y_train):
            # forward
            current_data += 1
            output = predict(network, x)

            # error
            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            total_gradient_magnitude += np.linalg.norm(grad)
            
            # Capture initial values only once
            if not initial_values_captured:
                initial_error = error
                initial_grad_mag = total_gradient_magnitude
                initial_values_captured = True 
                
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate, L1, L2, L1_reg, L2_reg)
                
            if detailed_verbose:
                error /= len(x_train)
                average_gradient_magnitude = total_gradient_magnitude / len(x_train)
                print('%d/%d, data=%d/%d, error=%f, avg_gradient_mag=%f' % (e + 1, epochs, training_data, current_data, error, average_gradient_magnitude))
            
        # Store statistics (regardless of 'statistics' flag for now)
        epoch_statistics.append({
            'epoch': e + 1,
            'initial_error': initial_error,
            'initial_grad_mag': initial_grad_mag,
            'final_error': error,
            'final_grad_mag': average_gradient_magnitude
        })

        if verbose:
            error /= len(x_train)
            average_gradient_magnitude = total_gradient_magnitude / len(x_train)
            print('%d/%d, error=%f, avg_gradient_mag=%f' % (e + 1, epochs, error, average_gradient_magnitude))

        if statistics:  # Check the 'statistics' flag
            print('\n--- Training Summary ---')
            for stats in epoch_statistics:
                print('\nEpoch %d:' % stats['epoch'])
                print('  Initial Error: %f, Initial Grad Mag: %f' % (stats['initial_error'], stats['initial_grad_mag']))
                print('  Final Error: %f, Final Grad Mag: %f' % (stats['final_error'], stats['final_grad_mag']))
            print("")


def show_para(epochs, learning_rate, network, L1_reg, L2_reg):
    print("\n--- Hyperparameters used ---\n")
    print(f"epochs: {epochs}")
    print(f"learning rate: {learning_rate}")
    print(f"L1: {L1_reg}")
    print(f"L2: {L2_reg}\n")

def accuracy(network, x_test, y_test):
    # Testing code (evaluate accuracy)
    correct_predictions = 0
    print("\n--- Network Accuracy ---\n")
    for x, y in zip(x_test, y_test):
        output = x
        for layer in network:
            output = layer.forward(output)

        predicted_label = np.argmax(output)
        true_label = np.argmax(y)  # Find the index of the true label
        if predicted_label == true_label:
            correct_predictions += 1

    accuracy = correct_predictions / len(y_test) *100
    print(f"Test accuracy: {accuracy}%\n")


def test_network(network, x_test, y_test, title, show_img=False, reshape_x=None, reshape_y=None):
    print(f"\n--- {title} ---\n")
    pred_count = 0
    for x, y in zip(x_test, y_test):
        pred_count += 1
        output = predict(network, x)
        print(f'{pred_count}. pred:', np.argmax(output), '\ttrue:', np.argmax(y))
        if show_img:
            if reshape_x is None or reshape_y is None:
                raise ValueError("reshape_x and reshape_y are required when show_img=True")
            #reshape image
            image = x.reshape(reshape_x, reshape_y)
            label = np.argmax(y) 
            #Display image
            plt.figure(figsize=(2, 2))
            plt.imshow(image, cmap='gray')
            plt.title(f"Original Image {pred_count}. Label: {label}")
            plt.show()
    print("")


#add plotting of images
def own_image_predict(network, x_test, title, reshape_x, reshape_y):
    print(f"\n--- {title} ---\n")
    pred_count = 0
    for x in x_test:
        pred_count += 1
        output = predict(network, x)
        #print(output)
        print('pred:', np.argmax(output))
        #reshape image
        image = x.reshape(reshape_x, reshape_y)
        #Display image
        plt.figure(figsize=(2, 2))
        plt.imshow(image, cmap='gray')
        plt.title(f"Original Image {pred_count}.")
        plt.show()
    print("")


def add_parent_dir(dir_path):
    """
    Add the parent directory of the given directory to the system path.
    
    Parameters:
    dir_path (str): The directory path whose parent directory will be added to the system path.
    
    Returns:
    parent directory
    """
    
    return os.path.abspath(os.path.join(dir_path, '..'))


def save_network(network, directory, filename):
    """
    Save the trained network in the specified directory with the specified filename.
    
    Parameters:
    network (list): add the defined network structure
    directory (str): The directory where the network should be stored.
    filename (str): The name of the file to save the trained network.
    
    Returns:
    None
    """
    
    os.makedirs(directory, exist_ok=True)  # Create the folder if it doesn't exist
    output_file = os.path.join(directory, filename)  # Save the trained network (assume some code to save the network)
    with open(output_file, 'wb') as file:
        dill.dump(network, file)
        
    print(f"Network saved at: {output_file}")


def load_network(system_path, file_name):
    """
    Load the trained network from the specified file in the given system path.
    
    Parameters:
    system_path (str): The system path where the network file is located.
    file_name (str): The name of the file containing the trained network.
    
    Returns:
    The loaded network object.
    """
    
    network_file = os.path.join(system_path, file_name)
    with open(network_file, 'rb') as file:
        loaded_network = dill.load(file)
    return loaded_network