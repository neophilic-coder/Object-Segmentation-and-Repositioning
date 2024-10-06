import argparse
import cv2
import numpy as np
import torch
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from diffusers import StableDiffusionInpaintPipeline
import torch
print(torch.cuda.is_available())


def segment_and_reposition(image_path, class_prompt, x_shift, y_shift):
    # Load the CLIP segmentation model
    seg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = processor(text=[class_prompt], images=[image], padding="max_length", return_tensors="pt")

    # Generate the segmentation mask
    with torch.no_grad():
        outputs = seg_model(**inputs)
    preds = outputs.logits.squeeze().sigmoid()

    # Create a binary mask
    mask = (preds > 0.5).float().numpy()

    # Shift the mask
    shifted_mask = np.zeros_like(mask)
    if x_shift > 0:
        shifted_mask[:, x_shift:] = mask[:, :-x_shift]
    elif x_shift < 0:
        shifted_mask[:, :x_shift] = mask[:, -x_shift:]
    else:
        shifted_mask = mask

    if y_shift > 0:
        shifted_mask = np.roll(shifted_mask, y_shift, axis=0)
        shifted_mask[:y_shift, :] = 0
    elif y_shift < 0:
        shifted_mask = np.roll(shifted_mask, y_shift, axis=0)
        shifted_mask[y_shift:, :] = 0

    # Invert the mask for inpainting
    inpaint_mask = 1 - shifted_mask

    # Load the inpainting model
    inpaint_model = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16
    ).to("cuda")

    # Prepare the image and mask for inpainting
    image_pil = Image.fromarray(image)
    mask_pil = Image.fromarray((inpaint_mask * 255).astype(np.uint8))

    # Perform inpainting
    result = inpaint_model(
        prompt="background",
        image=image_pil,
        mask_image=mask_pil,
        num_inference_steps=50
    ).images[0]

    # Composite the shifted object onto the inpainted result
    result_np = np.array(result)
    object_region = image * mask[:, :, None]
    shifted_object = np.zeros_like(image)
    
    if x_shift > 0:
        shifted_object[:, x_shift:] = object_region[:, :-x_shift]
    elif x_shift < 0:
        shifted_object[:, :x_shift] = object_region[:, -x_shift:]
    else:
        shifted_object = object_region

    if y_shift > 0:
        shifted_object = np.roll(shifted_object, y_shift, axis=0)
        shifted_object[:y_shift, :] = 0
    elif y_shift < 0:
        shifted_object = np.roll(shifted_object, y_shift, axis=0)
        shifted_object[y_shift:, :] = 0

    final_result = np.where(shifted_mask[:, :, None], shifted_object, result_np)

    return final_result

def main():
    parser = argparse.ArgumentParser(description="Segment and reposition an object in an image.")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--object_class", required=True, help="Text prompt for the object to segment")
    parser.add_argument("--x", type=int, default=0, help="Horizontal shift (positive: right, negative: left)")
    parser.add_argument("--y", type=int, default=0, help="Vertical shift (positive: up, negative: down)")
    parser.add_argument("--output", required=True, help="Path to save the output image")

    args = parser.parse_args()

    result = segment_and_reposition(args.image, args.object_class, args.x, args.y)
    cv2.imwrite(args.output, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"Repositioned image saved to {args.output}")

if __name__ == "__main__":
    main()