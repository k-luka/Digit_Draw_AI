document.addEventListener('DOMContentLoaded', () => {
  const canvas = document.getElementById('drawCanvas');
  const ctx = canvas.getContext('2d');
  let drawing = false;

  // Function to resize the canvas based on its displayed size
  function resizeCanvas() {
    // Get the computed style of the canvas
    const computedStyle = window.getComputedStyle(canvas);
    const width = parseInt(computedStyle.getPropertyValue('width'));
    const height = parseInt(computedStyle.getPropertyValue('height'));

    // Only resize if the size has changed
    if (canvas.width !== width || canvas.height !== height) {
      // Create a temporary canvas to preserve the current drawing
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = canvas.width;
      tempCanvas.height = canvas.height;
      const tempCtx = tempCanvas.getContext('2d');
      tempCtx.drawImage(canvas, 0, 0);

      // Resize the canvas
      canvas.width = width;
      canvas.height = height;

      // Restore the drawing
      ctx.drawImage(tempCanvas, 0, 0, width, height);
    }
  }

  // Initial resize
  resizeCanvas();

  // Resize the canvas when the window is resized
  window.addEventListener('resize', resizeCanvas);

  // Set up event listeners for mouse events
  canvas.addEventListener('mousedown', startDrawing);
  canvas.addEventListener('mouseup', stopDrawing);
  canvas.addEventListener('mouseout', stopDrawing); // Stop drawing when mouse leaves the canvas
  canvas.addEventListener('mousemove', draw);

  // Set up event listeners for touch events
  canvas.addEventListener('touchstart', startDrawing);
  canvas.addEventListener('touchend', stopDrawing);
  canvas.addEventListener('touchcancel', stopDrawing); // Handle touch cancel
  canvas.addEventListener('touchmove', draw);

  // Prevent scrolling when touching the canvas
  ['touchstart', 'touchend', 'touchmove'].forEach(eventType => {
    canvas.addEventListener(eventType, (event) => event.preventDefault(), { passive: false });
  });

  function startDrawing(event) {
    drawing = true;
    ctx.beginPath(); // Start a new path to avoid connecting lines
    const { x, y } = getCoords(event);
    ctx.moveTo(x, y);
  }

  function stopDrawing() {
    drawing = false;
  }

  function draw(event) {
    if (!drawing) return;

    const { x, y } = getCoords(event);

    // Adjust line width based on screen size
    ctx.lineWidth = window.innerWidth <= 700 ? 20 : 30;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    ctx.lineTo(x, y);
    ctx.stroke();
  }

  // Helper function to get coordinates for mouse or touch events
  function getCoords(event) {
    const rect = canvas.getBoundingClientRect();
    let x, y;

    if (event.touches && event.touches.length > 0) {
      // Touch event
      x = event.touches[0].clientX - rect.left;
      y = event.touches[0].clientY - rect.top;
    } else {
      // Mouse event
      x = event.clientX - rect.left;
      y = event.clientY - rect.top;
    }

    return { x, y };
  }

  // Reset the canvas
  const resetButton = document.getElementById('resetBtn');
  resetButton.addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#ffffff"; // Reset background to white
    ctx.fillRect(0, 0, canvas.width, canvas.height);
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
          // Update the processed image preview
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
          ctx.fillStyle = "#ffffff"; // Reset background to white
          ctx.fillRect(0, 0, canvas.width, canvas.height);
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
});
