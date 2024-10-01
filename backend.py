from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageOps
import io
import base64  # Import for encoding the image to base64
import base64
import pickle
import os
from datetime import datetime
from network import Network

app = Flask(__name__)
CORS(app)

# Load the saved model weights and biases
with open('fine_tuned_model_weights.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Initialize the network and load weights
net = Network([784, 128, 64, 32, 10])
net.weights = model_data['weights']
net.biases = model_data['biases']

def center_image(image):
    np_image = np.array(image)
    np_image = (np_image > 0).astype(np.uint8)
    coords = np.argwhere(np_image)
    if coords.size == 0:
        return image
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = np_image[y0:y1, x0:x1]

    new_image = Image.new('L', (28, 28))
    paste_x = (28 - cropped.shape[1]) // 2
    paste_y = (28 - cropped.shape[0]) // 2
    new_image.paste(Image.fromarray(cropped * 255), (paste_x, paste_y))

    return new_image

def binarize_image(image):
    threshold = 128
    return image.point(lambda p: 255 if p > threshold else 0, 'L')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_file = request.files['image'].read()
        image = Image.open(io.BytesIO(image_file)).convert('L')
        inverted_image = ImageOps.invert(image)
        centered_image = center_image(inverted_image)
        binarized_image = binarize_image(centered_image)

        image_np = np.array(binarized_image).astype('float32') / 255.0
        image_np = image_np.flatten().reshape(784, 1)

        output = net.feedforward(image_np)
        predicted_digit = np.argmax(output)

        buffered = io.BytesIO()
        binarized_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({'digit': int(predicted_digit), 'processed_image': img_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/submit_data', methods=['POST'])
def submit_data():
    try:
        image_file = request.files['image'].read()
        label = request.form.get('label')

        if label is None or not label.isdigit() or not (0 <= int(label) <= 9):
            return jsonify({'status': 'error', 'message': 'Invalid label provided.'}), 400

        image = Image.open(io.BytesIO(image_file)).convert('L')
        inverted_image = ImageOps.invert(image)
        centered_image = center_image(inverted_image)
        binarized_image = binarize_image(centered_image)

        save_directory = 'data/newData'
        os.makedirs(save_directory, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        filename = f'user_{timestamp}.png'
        filepath = os.path.join(save_directory, filename)

        binarized_image.save(filepath)

        label_file = os.path.join(save_directory, 'labels.csv')
        with open(label_file, 'a') as f:
            f.write(f'{filename},{label}\n')

        return jsonify({'status': 'success', 'message': 'Data saved successfully.'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
