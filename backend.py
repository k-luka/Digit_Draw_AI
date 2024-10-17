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
import requests  
import base64   

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
    resized_digit = cropped.resize((new_width, new_height))

    # Center the resized digit on a 28x28 canvas
    new_image = Image.new('L', (28, 28))
    paste_x = (28 - new_width) // 2
    paste_y = (28 - new_height) // 2
    new_image.paste(resized_digit, (paste_x, paste_y))
    
    return new_image

def upload_to_github(local_file_path, github_file_path, commit_message):
    """
    Uploads a file to GitHub using the GitHub API.
    """
    # Get environment variables
    github_token = os.environ.get('GITHUB_TOKEN')
    github_repo = os.environ.get('GITHUB_REPO')
    github_username = os.environ.get('GITHUB_USERNAME')

    if not github_token or not github_repo or not github_username:
        raise Exception('GitHub configuration is missing.')

    # Read the file content and encode it in base64
    with open(local_file_path, 'rb') as file:
        content = base64.b64encode(file.read()).decode('utf-8')

    # Prepare the API URL
    github_api_url = f'https://api.github.com/repos/{github_repo}/contents/{github_file_path}'

    headers = {
        'Authorization': f'token {github_token}',
        'Accept': 'application/vnd.github.v3+json'
    }

    # Check if the file already exists to get its SHA
    response = requests.get(github_api_url, headers=headers, params={'ref': 'data-upload'})
    if response.status_code == 200:
        sha = response.json().get('sha')
        print(f'Existing file found. SHA: {sha}')
    elif response.status_code == 404:
        sha = None
        print('File does not exist. It will be created.')
    else:
        print(f'Failed to check file existence: {response.status_code}')
        print('Response:', response.json())
        raise Exception(f'Failed to check file existence for {github_file_path}.')

    data = {
        'message': commit_message,
        'content': content,
        'branch': 'data-upload'
    }

    if sha:
        data['sha'] = sha

    # Make the request to create/update the file
    response = requests.put(github_api_url, headers=headers, json=data)

    if response.status_code in [200, 201]:
        print(f'File {github_file_path} uploaded successfully.')
    else:
        print(f'Failed to upload {github_file_path}:', response.json())
        raise Exception(f'Failed to upload {github_file_path} to GitHub.')


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

        image_np = np.array(centered_image).astype('float32') / 255.0
        image_np = image_np.flatten().reshape(784, 1)

        output = net.feedforward(image_np)
        predicted_digit = np.argmax(output)

        buffered = io.BytesIO()
        centered_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({'digit': int(predicted_digit), 'processed_image': img_base64})
    except Exception as e:
        print('Error:', e)
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

        # Save the processed image locally
        save_directory = 'data/processedData'
        os.makedirs(save_directory, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        filename = f'user_{timestamp}.png'
        filepath = os.path.join(save_directory, filename)

        centered_image.save(filepath)

        # Append the filename and label to the labels.csv file
        label_file = os.path.join(save_directory, 'labels.csv')
        if not os.path.exists(label_file):
            # Initialize labels.csv with headers
            with open(label_file, 'w') as f:
                f.write('filename,label\n')
        with open(label_file, 'a') as f:
            f.write(f'{filename},{label}\n')

        # Upload the image and labels.csv to GitHub
        upload_to_github(filepath, f'data/processedData/{filename}', f'Add {filename}')
        upload_to_github(label_file, 'data/processedData/labels.csv', 'Update labels.csv')

        return jsonify({'status': 'success', 'message': 'Data saved and uploaded successfully.'}), 200

    except Exception as e:
        print('Error:', e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
