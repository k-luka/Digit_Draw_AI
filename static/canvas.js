const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
let drawing = false;

// Start drawing when mouse is pressed down
canvas.addEventListener('mousedown', (event) => {
  drawing = true;
  ctx.beginPath(); // Start a new path to avoid connecting lines
  ctx.moveTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
});

// Stop drawing when mouse is released
canvas.addEventListener('mouseup', () => drawing = false);

// Draw while mouse is moving
canvas.addEventListener('mousemove', draw);

function draw(event) {
  if (!drawing) return;

  // Get the position of the canvas relative to the viewport
  const rect = canvas.getBoundingClientRect();
  
  // Adjust the mouse position to account for the canvas's position
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  
  ctx.lineWidth = 30;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  
  ctx.lineTo(x, y);
  ctx.stroke();
}


// Reset the canvas
const resetButton = document.getElementById('resetBtn');
resetButton.addEventListener('click', () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
});

// Function to downscale the image
function getDownscaledImageData() {
  const downscaleCanvas = document.createElement('canvas');
  downscaleCanvas.width = 28;
  downscaleCanvas.height = 28;
  const downscaleCtx = downscaleCanvas.getContext('2d');

  // Disable image smoothing for pixelated effect
  downscaleCtx.imageSmoothingEnabled = false;

  downscaleCtx.fillStyle = "#ffffff"; // Set fill color to white
  downscaleCtx.fillRect(0, 0, downscaleCanvas.width, downscaleCanvas.height);

  // Downscale the original canvas into a 28x28 pixel canvas
  downscaleCtx.drawImage(canvas, 0, 0, downscaleCanvas.width, downscaleCanvas.height);

  return downscaleCanvas;
}

// Submit button functionality for prediction
const submitButton = document.getElementById('submitBtn');
submitButton.addEventListener('click', () => {
  const downscaleCanvas = getDownscaledImageData();

  // Convert the canvas to a blob and send it to the server
  downscaleCanvas.toBlob((blob) => {
    const formData = new FormData();
    formData.append('image', blob, 'digit.png');

    // Send the image to the Flask backend for prediction
    fetch('/predict', { // Changed to relative path
      method: 'POST',
      body: formData,
    })
      .then(response => response.json())
      .then(data => {
        if (data.error) {
          alert(`Error: ${data.error}`);
          return;
        }
        console.log('Prediction response:', data);
        // Display the predicted digit
        const resultElement = document.getElementById('predictionResult');
        if (resultElement) {
          resultElement.innerText = `Predicted Digit: ${data.digit}`;
        } else {
          alert(`Predicted Digit: ${data.digit}`);
        }
        // Display the processed image
        const processedImageElement = document.getElementById('processedImage');
        if (processedImageElement) {
          processedImageElement.src = `data:image/png;base64,${data.processed_image}`;
        }
      })
      .catch(error => {
        console.error('Error:', error);
        alert('An error occurred during prediction.');
      });
  }, 'image/png');
});

// Submit Data functionality
const submitDataButton = document.getElementById('submitDataBtn');
submitDataButton.addEventListener('click', () => {
  const labelInput = document.getElementById('digitLabel');
  const label = labelInput.value.trim();

  // Validate label (should be a single digit between 0 and 9)
  if (!/^\d$/.test(label)) {
    alert('Please enter a valid digit between 0 and 9.');
    return;
  }

  const downscaleCanvas = getDownscaledImageData();

  // Convert the canvas to a blob and send it to the server along with the label
  downscaleCanvas.toBlob((blob) => {
    const formData = new FormData();
    formData.append('image', blob, 'digit.png');
    formData.append('label', label);

    // Send the image and label to the Flask backend for saving
    fetch('/submit_data', { // Changed to relative path
      method: 'POST',
      body: formData,
    })
      .then(response => response.json())
      .then(data => {
        if (data.error) {
          alert(`Error: ${data.error}`);
          return;
        }
        console.log('Data submission response:', data);
        alert('Your data has been submitted. Thank you!');
        // Clear the canvas and label input
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        labelInput.value = '';
      })
      .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while submitting data.');
      });
  }, 'image/png');
});

// Training Mode Toggle Functionality
const trainingModeToggle = document.getElementById('trainingMode');
const submitDataBtn = document.getElementById('submitDataBtn');
const labelContainer = document.getElementById('labelContainer');

trainingModeToggle.addEventListener('change', (event) => {
  if (event.target.checked) {
    // Training Mode is ON: Show Submit Data button and label input
    submitDataBtn.style.display = 'inline-block';
    labelContainer.style.display = 'block';
  } else {
    // Training Mode is OFF: Hide Submit Data button and label input
    submitDataBtn.style.display = 'none';
    labelContainer.style.display = 'none';
  }
});
