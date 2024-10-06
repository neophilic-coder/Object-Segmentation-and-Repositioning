import argparse
import cv2
import numpy as np
import torch
from transformers import AutoProcessor, CLIPSegForImageSegmentation
import os
from tqdm import tqdm

def segment_object(image_path, class_prompt, model, processor):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = processor(text=[class_prompt], images=[image], padding="max_length", return_tensors="pt")

    # Generate the segmentation mask
    with torch.no_grad():
        outputs = model(**inputs)
    preds = outputs.logits.squeeze().sigmoid()

    # Create a binary mask
    mask = (preds > 0.5).float().numpy()

    # Apply the red mask to the original image
    red_mask = np.zeros_like(image)
    red_mask[:, :, 0] = 255  # Red channel
    masked_image = np.where(mask[:, :, None], red_mask, image)

    return masked_image

def process_directory(input_dir, class_prompt, output_dir):
    # Load the CLIP segmentation model (only once for all images)
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Process each image
    for image_file in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, f"segmented_{image_file}")

        result = segment_object(input_path, class_prompt, model, processor)
        cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    print(f"Segmentation results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Segment objects in multiple images based on a text prompt.")
    parser.add_argument("--input_dir", required=True, help="Path to the input directory containing images")
    parser.add_argument("--object_class", required=True, help="Text prompt for the object to segment")
    parser.add_argument("--output_dir", required=True, help="Path to the output directory to save segmented images")

    args = parser.parse_args()

    process_directory(args.input_dir, args.object_class, args.output_dir)

if __name__ == "__main__":
    main()
