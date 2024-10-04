# Digit Draw - Handwritten Digit Recognition

**Digit Draw** is a web-based application that allows users to draw digits (0-9) on a canvas, and a custom-built neural network model predicts the digit in real time. The project also includes a training mode where users can contribute to the dataset by drawing and labeling digits, adding a small amount of new data to the model.

## Features

- **Real-time digit prediction:** Draw any digit on the canvas, and the custom neural network will predict what you wrote.
- **Training mode:** Use the slider to switch into training mode, where you can draw, label, and submit new training data to the model.
- **Interactive front-end:** Easy-to-use interface with a responsive canvas.
- **Custom neural network backend:** A neural network model built entirely from scratch, trained to recognize handwritten digits.
- **Modern web technologies:** Built using HTML, CSS, JavaScript, and Flask.

## Live Demo

Check out the live version of the project at: [digitdraw.org](https://www.digitdraw.org)

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/digit-draw.git
    ```

2. Navigate to the project directory:
    ```bash
    cd digit-draw
    ```

3. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Start the Flask development server:
    ```bash
    flask run
    ```

5. Open your browser and go to `http://localhost:5000` to see the website in action.

## How It Works

1. **Drawing Interface:** Users can draw digits on a canvas.
2. **Prediction Mode:** The drawn image is processed (inverted and resized) to match the format used during training, then sent to a fully custom neural network model for prediction.
3. **Training Mode:** Users can switch to training mode with a slider, where they draw, label, and submit digits, helping to augment the training data.
4. **Neural Network:** The model is custom-built and trained from scratch on the MNIST dataset, supplemented with a small amount of user-submitted training data.
5. **Prediction/Training Display:** The predicted digit or submission confirmation is displayed on the website.

## Technologies Used

- **Front-end:** HTML, CSS, JavaScript (Canvas API)
- **Back-end:** Flask (Python)
- **Machine Learning Model:** Custom-built neural network (not using any external libraries like PyTorch)
- **Deployment:** Hosted on Railway

## Dataset

The neural network was initially trained on the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits. Users can also contribute to the dataset by submitting new labeled data in training mode.

## Future Improvements

- Add support for recognizing letters.
- Enhance the accuracy of the neural network with more training.
- Improve the front-end design and add mobile responsiveness.
- Implement more user-friendly feedback in training mode.

## Contributing

Feel free to submit issues or pull requests. Contributions are always welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
