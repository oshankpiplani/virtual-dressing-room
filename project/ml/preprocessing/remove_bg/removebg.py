#!/usr/bin/env python3
import argparse
import io
import os
import sys
from PIL import Image
from rembg import remove

def process_image(input_path, output_path):
    """Process image by removing background and standardizing size"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Load input image
        with open(input_path, 'rb') as f:
            input_image = f.read()
        
        # Remove background
        output_image = remove(input_image)
        
        # Convert to PIL Image for processing
        img = Image.open(io.BytesIO(output_image))
        
        # Find bounding box of content
        bbox = img.getbbox()
        if bbox:
            cropped = img.crop(bbox)
        else:
            cropped = img
        
        # Set canvas size (768x1024 for VITON-HD)
        canvas_width, canvas_height = 768, 1024
        
        # Calculate scaling factor
        scale_factor = min(
            canvas_width / cropped.width,
            canvas_height / cropped.height,
            1.0  # Don't scale up
        )
        new_size = (int(cropped.width * scale_factor), 
                   int(cropped.height * scale_factor))
        
        # Resize image
        resized = cropped.resize(new_size, Image.Resampling.LANCZOS)
        
        # Create white canvas
        canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
        
        # Calculate position to center the image
        offset_x = (canvas_width - resized.width) // 2
        offset_y = (canvas_height - resized.height) // 2
        
        # Handle transparency
        mask = resized.split()[3] if resized.mode == 'RGBA' else None
        
        # Paste the resized image onto the canvas
        canvas.paste(resized, (offset_x, offset_y), mask=mask)
        
        # Save the result
        canvas.save(output_path)
        return True
        
    except Exception as e:
        print(f"Error processing image: {str(e)}", file=sys.stderr)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove background from image and standardize size"
    )
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output image path")
    
    args = parser.parse_args()
    
    success = process_image(args.input, args.output)
    sys.exit(0 if success else 1)