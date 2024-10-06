import argparse
import cv2
import numpy as np
import json
import os
from tqdm import tqdm

def shift_object(image, mask, x_shift, y_shift):
    # Get the dimensions of the image
    height, width = image.shape[:2]
    
    # Create a background image (original image without the object)
    background = image.copy()
    background[mask > 0] = 0
    
    # Create an image of just the object
    object_only = image.copy()
    object_only[mask == 0] = 0
    
    # Create a new mask and object image with the shift applied
    new_mask = np.zeros_like(mask)
    new_object = np.zeros_like(image)
    
    for y in range(height):
        for x in range(width):
            new_x = x + x_shift
            new_y = y + y_shift
            if 0 <= new_x < width and 0 <= new_y < height:
                if mask[y, x] > 0:
                    new_mask[new_y, new_x] = mask[y, x]
                    new_object[new_y, new_x] = object_only[y, x]
    
    # Combine the shifted object with the background
    result = background + new_object
    
    return result, new_mask

def process_images(input_dir, segmented_dir, config_file, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the configuration file
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Process each image
    for image_info in tqdm(config, desc="Processing images"):
        image_file = image_info['file']
        x_shift = image_info['x']
        y_shift = image_info['y']
        
        # Load the original image
        input_path = os.path.join(input_dir, image_file)
        original_image = cv2.imread(input_path)
        if original_image is None:
            print(f"Warning: Image file {input_path} not found. Skipping.")
            continue
        
        # Load the segmented image
        segmented_path = os.path.join(segmented_dir, f"segmented_{image_file}")
        segmented_image = cv2.imread(segmented_path)
        if segmented_image is None:
            print(f"Warning: Segmented image file {segmented_path} not found. Skipping.")
            continue
        
        # Create a binary mask from the segmented image
        mask = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        
        # Shift the object
        result, shifted_mask = shift_object(original_image, mask, x_shift, y_shift)
        
        # Save the result
        output_path = os.path.join(output_dir, f"shifted_{image_file}")
        cv2.imwrite(output_path, result)
    
    print(f"Shifted images saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Shift segmented objects in images based on x and y coordinates.")
    parser.add_argument("--input_dir", required=True, help="Path to the input directory containing original images")
    parser.add_argument("--segmented_dir", required=True, help="Path to the directory containing segmented images")
    parser.add_argument("--config", required=True, help="Path to the JSON configuration file")
    parser.add_argument("--output_dir", required=True, help="Path to the output directory to save shifted images")
    args = parser.parse_args()
    
    process_images(args.input_dir, args.segmented_dir, args.config, args.output_dir)

if __name__ == "__main__":
    main()