from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
# Note: I converted the image to 28x28 greyscale here for consistency and safety.
# Need to remove the covertion in the front end.

# TODO: import my model here!!!

app = Flask(__name__)

# TODO: Initialize model here!!!

# Define route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    # Get image file from request
    image_file = request.files['image'].read()

    # Convert image to 28x28 greyscale
    image = Image.open(io.BytesIO(image_file)).convert('L') # 'L' is greyscale
    image = image.resize((28, 28))

    # normalize
    image_np = np.array(image).astype('flaot32') / 255.0
    image_np = image_np.reshape(1, 28, 28) 

    # predicted_digit = model.predict(image_np)

    return jsonify({'digit': int(predicted_digit)})

if __name__ == '__main__':
    app.run(debug=True)