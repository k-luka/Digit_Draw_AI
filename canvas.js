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
  
  ctx.lineWidth = 28;
  ctx.lineCap = 'round';
  
  ctx.lineTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
  ctx.stroke();
}

// Reset the canvas
const resetButton = document.getElementById('resetBtn');
resetButton.addEventListener('click', () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
});

// Submit functionality
const submitButton = document.getElementById('submitBtn');
submitButton.addEventListener('click', () => {
  const downscaleCanvas = document.createElement('canvas');
  downscaleCanvas.width = 28;
  downscaleCanvas.height = 28;
  const downscaleCtx = downscaleCanvas.getContext('2d');
  
  // Downscale canvas into 28x28 pixel
  downscaleCtx.drawImage(canvas, 0, 0, 28, 28);
  
  // Get the image data (to use for model input)
  const imageData = downscaleCtx.getImageData(0, 0, 28, 28);
  console.log(imageData.data); // Log the image data (this can be used for the neural network)
});
