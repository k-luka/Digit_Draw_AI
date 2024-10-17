from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageOps
import io
import base64
import pickle
import os
from datetime import datetime
from network import Network

app = Flask(__name__)
CORS(app)

# Load the fine-tuned model weights and biases
with open('fine_tuned_model_weights.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Initialize the network and load weights
net = Network([784, 128, 64, 32, 10])
net.weights = model_data['weights']
net.biases = model_data['biases']

def center_image(image):
    # Convert to binary to find the bounding box
    np_image = np.array(image)
    np_image = (np_image > 0).astype(np.uint8)
    coords = np.argwhere(np_image)
    if coords.size == 0:
        return Image.new('L', (28, 28))  # Return blank if no digit found
    
    # Crop to bounding box
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = image.crop((x0, y0, x1, y1))

    # Calculate the scaling factor to fit into a 20x20 box, keeping aspect ratio
    scale_factor = min(20 / cropped.width, 20 / cropped.height)
    new_width = int(cropped.width * scale_factor)
    new_height = int(cropped.height * scale_factor)

    # Resize the digit without smoothing
    resized_digit = cropped.resize((new_width, new_height), resample=Image.NEAREST)

    # Center the resized digit on a 28x28 canvas
    new_image = Image.new('L', (28, 28))
    paste_x = (28 - new_width) // 2
    paste_y = (28 - new_height) // 2
    new_image.paste(resized_digit, (paste_x, paste_y))
    
    return new_image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read the image file from the request
        image_file = request.files['image'].read()
        image = Image.open(io.BytesIO(image_file)).convert('L')
        
        # Invert the image to match MNIST format (black background, white digits)
        inverted_image = ImageOps.invert(image)
        
        # Center and scale the image
        centered_image = center_image(inverted_image)
        
        # Convert the processed image to a numpy array and normalize
        image_np = np.array(centered_image).astype('float32') / 255.0
        image_np = image_np.flatten().reshape(784, 1)

        # Feedforward to get the output
        output = net.feedforward(image_np)
        predicted_digit = np.argmax(output)

        # Convert the processed image back to base64 for display (optional)
        buffered = io.BytesIO()
        centered_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({'digit': int(predicted_digit), 'processed_image': img_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/submit_data', methods=['POST'])
def submit_data():
    try:
        # Read the image file and label from the request
        image_file = request.files['image'].read()
        label = request.form.get('label')

        # Validate the label
        if label is None or not label.isdigit() or not (0 <= int(label) <= 9):
            return jsonify({'status': 'error', 'message': 'Invalid label provided.'}), 400

        # Open the image and convert to grayscale
        image = Image.open(io.BytesIO(image_file)).convert('L')
        
        # Invert the image to match MNIST format
        inverted_image = ImageOps.invert(image)
        
        # Center and scale the image
        centered_image = center_image(inverted_image)

        # Define the directory to save processed data
        save_directory = 'data/processedData'  # Updated directory
        os.makedirs(save_directory, exist_ok=True)

        # Generate a unique filename based on the current timestamp
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        filename = f'user_{timestamp}.png'
        filepath = os.path.join(save_directory, filename)

        # Save the processed image
        centered_image.save(filepath)

        # Append the filename and label to the labels.csv file
        label_file = os.path.join(save_directory, 'labels.csv')
        with open(label_file, 'a') as f:
            f.write(f'{filename},{label}\n')

        return jsonify({'status': 'success', 'message': 'Data saved successfully.'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
