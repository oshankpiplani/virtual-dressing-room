#!/usr/bin/env python
import cv2
import numpy as np
import argparse
import os

def create_cloth_mask(input_path, mask_output_path, masked_output_path=None):
    # Load the image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not read image from {input_path}")
        return False
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to separate white background from cloth
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Optional: remove noise and smooth edges
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Save the mask
    cv2.imwrite(mask_output_path, mask)
    print(f"Mask saved to {mask_output_path}")
    
    # If masked output path is provided, create the masked image
    if masked_output_path:
        # Apply mask to original image
        cloth_only = cv2.bitwise_and(image, image, mask=mask)
        cv2.imwrite(masked_output_path, cloth_only)
        print(f"Masked image saved to {masked_output_path}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a mask for cloth images")
    parser.add_argument("--input", required=True, help="Path to input cloth image")
    parser.add_argument("--output", required=True, help="Path to save output mask")
    parser.add_argument("--masked-output", help="Optional path to save masked cloth image")
    
    args = parser.parse_args()
    
    success = create_cloth_mask(args.input, args.output, args.masked_output)
    
    if success:
        print("Cloth mask creation completed successfully")
    else:
        print("Cloth mask creation failed")
        exit(1)