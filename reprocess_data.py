import os
from PIL import Image
from backend import center_image  # Ensure this is the updated function

def reprocess_image(image_path):
    # Open image in grayscale mode
    image = Image.open(image_path).convert('L')
    
    # Center and scale the image using the updated center_image function
    processed_image = center_image(image)
    
    return processed_image

def reprocess_data(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Use os.path.join for paths
    labels_csv = os.path.join(data_dir, 'labels.csv')
    output_labels_csv = os.path.join(output_dir, 'labels.csv')
    
    # Check if labels.csv exists
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"Could not find labels.csv in {data_dir}. Please check the file's location.")
    
    # Open the input and output CSV files
    with open(labels_csv, 'r') as infile, open(output_labels_csv, 'w') as outfile:
        for line in infile:
            filename, label = line.strip().split(',')
            input_image_path = os.path.join(data_dir, filename)
            output_image_path = os.path.join(output_dir, filename)
            
            # Reprocess the image using our updated center_image function
            processed_image = reprocess_image(input_image_path)
            
            # Save the processed image to the output directory
            processed_image.save(output_image_path)
            
            # Write the filename and label to the new labels.csv
            outfile.write(f'{filename},{label}\n')

if __name__ == '__main__':
    # Set the base directory to the script’s directory
    base_dir = os.path.dirname(__file__)
    
    # Define data and output directories relative to the script
    data_dir = os.path.join(base_dir, 'data', 'newData')
    output_dir = os.path.join(base_dir, 'data', 'processedData')
    
    # Run the reprocessing function
    reprocess_data(data_dir, output_dir)
