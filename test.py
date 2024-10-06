import argparse
import cv2
import numpy as np
import torch
from transformers import AutoProcessor, CLIPSegForImageSegmentation

def segment_object(image_path, class_prompt):
    # Load the CLIP segmentation model
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

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

    # Resize the mask to match the input image dimensions
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))  # Resize to (width, height)

    # Apply the red mask to the original image
    red_mask = np.zeros_like(image)
    red_mask[:, :, 0] = 255  # Red channel
    masked_image = np.where(mask_resized[:, :, None], red_mask, image)

    return masked_image

def main():
    parser = argparse.ArgumentParser(description="Segment an object in an image based on a text prompt.")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--object_class", required=True, help="Text prompt for the object to segment")
    parser.add_argument("--output", required=True, help="Path to save the output image")

    args = parser.parse_args()

    result = segment_object(args.image, args.object_class)
    cv2.imwrite(args.output, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"Segmentation result saved to {args.output}")

if __name__ == "__main__":
    main()
