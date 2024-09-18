from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import numpy as np
from PIL import Image, ImageOps
import io
import base64  # Import for encoding the image to base64
import pickle
from network import Network  # Import your Network class

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for the entire app

# Load the saved model weights and biases
with open('model_weights.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Initialize the network and load weights
net = Network([784, 128, 64, 32, 10])  # Make sure the architecture matches the saved model
net.weights = model_data['weights']  # Load the saved weights
net.biases = model_data['biases']  # Load the saved biases

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    image_file = request.files['image'].read()
    
    # Convert the image to a 28x28 grayscale image
    image = Image.open(io.BytesIO(image_file)).convert('L')
    image = image.resize((28, 28))

    # Invert the image (model was trained with black background)
    inverted_image = ImageOps.invert(image)
    
    # Convert the image to a numpy array and normalize it
    image_np = np.array(inverted_image).astype('float32') / 255.0  # Normalize to 0-1 range
    image_np = image_np.flatten().reshape(784, 1)  # Flatten and reshape for the network input (784x1)

    # Use the network to make a prediction
    output = net.feedforward(image_np)
    predicted_digit = np.argmax(output)  # Get the index of the highest output value
    
    # Convert the processed image (inverted_image) to base64 for display
    buffered = io.BytesIO()
    inverted_image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the predicted digit and the processed image (as base64)
    return jsonify({'digit': int(predicted_digit), 'processed_image': img_base64})

if __name__ == '__main__':
    app.run(debug=True)