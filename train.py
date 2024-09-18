import mnist_loader
import network
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Load the data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Initialize the network
net = network.Network([784, 64, 32, 10])

# Train the network
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# Save the trained model's weights and biases
model_data = {
    "weights": net.weights,  # Save the network's weights
    "biases": net.biases  # Save the network's biases
}

# Save to a file using pickle
with open('model_weights.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model weights and biases saved successfully.")

# Evaluate the network and get indices of correct and incorrect predictions
accuracy, correct, incorrect = net.evaluate(test_data)
print(f"Accuracy: {accuracy} / {len(test_data)}")

# Function to show a sample of correct and incorrect images
def show_images(test_data, correct, incorrect, num_samples=5):
    # Show correctly predicted images
    print(f"Displaying {num_samples} correctly predicted images:")
    plt.figure(figsize=(10, 4))
    for i in range(num_samples):
        index = correct[i]
        image = test_data[index][0].reshape(28, 28)  # Reshape to 28x28
        predicted_label = np.argmax(net.feedforward(test_data[index][0]))
        actual_label = np.argmax(test_data[index][1])
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f"P: {predicted_label}, A: {actual_label}")
        plt.axis('off')

    # Show incorrectly predicted images
    print(f"Displaying {num_samples} incorrectly predicted images:")
    for i in range(num_samples):
        index = incorrect[i]
        image = test_data[index][0].reshape(28, 28)  # Reshape to 28x28
        predicted_label = np.argmax(net.feedforward(test_data[index][0]))
        actual_label = np.argmax(test_data[index][1])
        plt.subplot(2, num_samples, num_samples + i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f"P: {predicted_label}, A: {actual_label}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Show images
show_images(test_data, correct, incorrect, num_samples=10)