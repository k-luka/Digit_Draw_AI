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
  
  ctx.lineWidth = 40;
  ctx.lineCap = 'round';
  
  ctx.lineTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
  ctx.stroke();
}

// Reset the canvas
const resetButton = document.getElementById('resetBtn');
resetButton.addEventListener('click', () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
});

// Preprocess the image data (convert to grayscale, normalize)
function preprocessImage(imageData) {
  const grayImageData = [];
  
  for (let i = 0; i < imageData.length; i += 4) {
    const r = imageData[i];
    const g = imageData[i + 1];
    const b = imageData[i + 2];
    
    // Convert to grayscale using luminosity formula
    const gray = 0.299 * r + 0.587 * g + 0.114 * b;
    
    // Normalize to 0-1 range (MNIST uses this format)
    grayImageData.push(gray / 255);
  }
  
  return grayImageData;
}

// Submit functionality
const submitButton = document.getElementById('submitBtn');
let img = null; // Store the reference to the image element

submitButton.addEventListener('click', () => {
  const downscaleCanvas = document.createElement('canvas');
  downscaleCanvas.width = 28;
  downscaleCanvas.height = 28;
  const downscaleCtx = downscaleCanvas.getContext('2d');

  downscaleCtx.fillStyle = "#ffffff"; // Set fill color to white
  downscaleCtx.fillRect(0, 0, downscaleCanvas.width, downscaleCanvas.height);
  
  // Downscale the original canvas into a 28x28 pixel canvas
  downscaleCtx.drawImage(canvas, 0, 0, 28, 28);

  // Convert the canvas to an image (blob)
  downscaleCanvas.toBlob((blob) => {
    const formData = new FormData();
    formData.append('image', blob, 'digit.png');

    // Log before sending the request
    console.log('Sending image to backend...');

    // Send the image to the Flask backend for prediction
    fetch('http://localhost:5000/predict', {
      method: 'POST',
      body: formData,
    })
    .then(response => response.json())
    .then(data => {
      console.log('Response from backend:', data);  // Log the response

      // Display the predicted digit
      const resultElement = document.getElementById('predictionResult');
      if (resultElement) {
        resultElement.innerText = `Predicted Digit: ${data.digit}`;
      } else {
        alert(`Predicted Digit: ${data.digit}`);
      }
    })
    .catch(error => {
      console.error('Error:', error);
    });
  }, 'image/png');
});
