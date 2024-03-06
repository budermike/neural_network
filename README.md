# Neurolib

Welcome to Neurolib, a journey into the fascinating world of neuroinformatics and neural networks through the creation of a Python library.

### About

Neurolib is my personal exploration into neuroinformatics, driven by a deep curiosity and passion for understanding neural networks. While I may not have prior experience in neuroinformatics, I am excited to embark on this journey and share my progress with you.

### Goals

The primary goal of Neurolib is to develop a versatile and efficient Python library for neural network experimentation and implementation. Through this project, I aim to learn, grow, and contribute to the field of neuroinformatics.

### To-Do

- Improve training performance: Address overfitting issue on the MNIST dataset and enhance accuracy for self-written numbers.

## Code Documentation

### General Methods (network.py)

### - one_hot_encode(labels)

#### Description:
Performs one-hot encoding for the given labels.

#### Parameters:
- `labels` (numpy.ndarray): The input labels to be one-hot encoded.

#### Returns:
- numpy.ndarray: The one-hot encoded labels.

### - predict(network, input)

#### Description:
Performs a forward pass through the neural network.

#### Parameters:
- `network` (list): List of layers comprising the neural network.
- `input` (numpy.ndarray): The input data for prediction.

#### Returns:
- numpy.ndarray: The output prediction of the neural network.

#### Example Usage:
```python
# Make predictions
for input_sample in input_data_to_predict:
    
    # Perform forward pass to get predictions
    prediction = predict(network, input)

    # Display the input and corresponding prediction
    print(f"Input: {input}, Prediction: {prediction}")
```

## Contribution Guidelines

To ensure a smooth and collaborative development process, please adhere to the following guidelines when contributing to Neurolib:

1. **Fork the Repository**: Before making any changes, fork the Neurolib repository to your GitHub account.

2. **Branch Naming Convention**: When working on a new feature or bug fix, create a descriptive branch name that reflects the purpose of your changes.

3. **Code Style**: Follow the existing code style and formatting conventions used in the project.

4. **Commit Messages**: Write clear and descriptive commit messages that summarize the purpose of your changes.

5. **Pull Requests**: When submitting a pull request, provide a detailed description of the changes made and any relevant context. Ensure that your code passes all tests and does not introduce any regressions.

6. **Testing**: If applicable, include tests to cover the functionality added or modified by your changes.

7. **Documentation**: Update the documentation as needed to reflect any changes or additions to the codebase.

## Collaboration

Join me in shaping the future of neural networks and neuroinformatics! Whether you're an experienced practitioner or just starting out, your contributions, ideas, and insights are invaluable.

Let's collaborate and advance the field together. Feel free to reach out for questions, feedback, or collaboration opportunities:
- Email: mathmeetsart01@gmail.com
- Instagram: [art_meets_math](https://www.instagram.com/art_meets_math/)

**About Me**

I'm a Swiss dude fascinated by the hidden beauty within mathematics and the power of code to reveal it. My passion led me to pursue a degree in math and neuroinformatics, combining my love for numbers with the mysteries of the brain. I'm excited to explore the intersection of math, art, and technology, and I ultimately aspire to make meaningful contributions in the field of science.

Feel free to get in touch, share your thoughts, and join me on this exciting journey!
