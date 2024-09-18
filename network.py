import numpy as np
import random

def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    """The ReLU function."""
    return np.maximum(0, z)

def relu_prime(z):
    """Derivative of the ReLU function."""
    return (z > 0).astype(float)

class Network(object):
    def __init__(self, sizes):
        """Initialize the network's parameters.

        - `sizes`: list containing the number of neurons in each layer.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Initialize biases for each layer except the input layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # Initialize weights with He initialization for ReLU layers
        self.weights = [np.random.randn(y, x) * np.sqrt(2 / x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network given input `a`."""
        # Hidden layers with ReLU activation
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = relu(np.dot(w, a) + b)
        # Output layer with sigmoid activation
        a = sigmoid(np.dot(self.weights[-1], a) + self.biases[-1])
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0, test_data=None):
        """Train the network using mini-batch stochastic gradient descent.

        - `training_data`: list of tuples `(x, y)` for training.
        - `epochs`: number of epochs to train for.
        - `mini_batch_size`: size of each mini-batch.
        - `eta`: learning rate.
        - `lmbda`: regularization parameter.
        - `test_data`: optional test data to evaluate performance.
        """
        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            # Create mini-batches
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            # Update network for each mini-batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n)
            # Evaluate on test data if provided
            if test_data:
                accuracy, _, _ = self.evaluate(test_data)
                print(f"Epoch {j}: {accuracy} / {n_test}")
            else:
                print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update network's weights and biases using backpropagation.

        - `mini_batch`: list of tuples `(x, y)`.
        - `eta`: learning rate.
        - `lmbda`: regularization parameter.
        - `n`: total size of the training data.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # Gradient for biases
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # Gradient for weights

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # Accumulate the gradients
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # Update weights and biases with L2 regularization
        self.weights = [
            (1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
            for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (eta / len(mini_batch)) * nb
            for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, x, y):
        """Perform backpropagation to compute gradients.

        - `x`: input data.
        - `y`: expected output.
        Returns a tuple `(nabla_b, nabla_w)` representing the gradients.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # Gradient for biases
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # Gradient for weights

        # Feedforward
        activation = x
        activations = [x]  # Store activations layer by layer
        zs = []            # Store z vectors layer by layer

        # Hidden layers with ReLU activation
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = relu(z)
            activations.append(activation)

        # Output layer with sigmoid activation
        z = np.dot(self.weights[-1], activation) + self.biases[-1]
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)

        # Backward pass
        # Output layer error
        delta = self.cross_entropy_delta(activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Backpropagate the error to hidden layers
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = relu_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (nabla_b, nabla_w)

    def cross_entropy_delta(self, output_activations, y):
        """Compute the error delta for the output layer."""
        return output_activations - y  # For cross-entropy loss with sigmoid activation

    def evaluate(self, test_data):
        """Evaluate the network's performance on test data.

        Returns a tuple `(accuracy, correct, incorrect)`:
        - `accuracy`: number of correct predictions.
        - `correct`: list of indices of correct predictions.
        - `incorrect`: list of indices of incorrect predictions.
        """
        test_results = [
            (np.argmax(self.feedforward(x)), np.argmax(y))
            for (x, y) in test_data
        ]

        correct = [i for i, (x_pred, y_true) in enumerate(test_results) if x_pred == y_true]
        incorrect = [i for i, (x_pred, y_true) in enumerate(test_results) if x_pred != y_true]
        accuracy = len(correct)
        return accuracy, correct, incorrect

    def cost_derivative(self, output_activations, y):
        """Compute the derivative of the cost function."""
        return output_activations - y  # Not used with cross-entropy loss