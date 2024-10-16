import numpy as np
import pickle
import random
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps
import os

# Import your Network class
from network import Network

# Import your center_image function
from backend import center_image

def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    
    # Invert the image to match the inversion done during prediction
    image = ImageOps.invert(image)
    
    # Apply centering and scaling if not already applied
    image = center_image(image)
    
    # Do not binarize the image
    image_np = np.array(image).astype('float32') / 255.0
    image_np = image_np.flatten().reshape(784, 1)
    return image_np

def load_user_data(data_dir):
    data = []
    labels_csv = os.path.join(data_dir, 'labels.csv')
    with open(labels_csv, 'r') as f:
        for line in f:
            filename, label = line.strip().split(',')
            image_path = os.path.join(data_dir, filename)
            x = preprocess_image(image_path)
            y = np.zeros((10, 1))
            y[int(label)] = 1.0
            data.append((x, y))
    return data

# Initialize your network
net = Network([784, 128, 64, 32, 10])

# Load pre-trained weights and biases
with open('model_weights.pkl', 'rb') as f:
    model_data = pickle.load(f)
net.weights = model_data['weights']
net.biases = model_data['biases']

# Load your user data
user_data = load_user_data('data/processedData')

# Split user data into training and validation sets
user_train_data, user_val_data = train_test_split(user_data, test_size=0.1, random_state=42)

# If not combining with MNIST data
training_data_to_use = user_train_data

# Fine-tune the model
net.SGD(
    training_data=training_data_to_use,
    epochs=40,
    mini_batch_size=32,
    eta=0.005,
    lmbda=0.1,
    test_data=user_val_data
)

# Save the fine-tuned model
fine_tuned_model = {
    'weights': net.weights,
    'biases': net.biases
}

with open('fine_tuned_model_weights.pkl', 'wb') as f:
    pickle.dump(fine_tuned_model, f)
