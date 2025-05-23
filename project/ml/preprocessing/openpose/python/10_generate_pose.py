#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import os
import json
import argparse
from sys import platform

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process an image with OpenPose")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to save output visualization")
    parser.add_argument("--json-output", help="Path to save JSON keypoints")

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    json_output_path = args.json_output

    print(f"Processing: {input_path}")
    print(f"Output image: {output_path}")
    if json_output_path:
        print(f"Output JSON: {json_output_path}")

    # Import OpenPose
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        if platform == "win32":
            sys.path.append(dir_path + '/../bin/python/openpose/Release')
            os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../x64/Release;' + dir_path + '/../bin;'
            import pyopenpose as op
        else:
            sys.path.append('/Users/dineshkotwani/project/ml/preprocessing/pose_generation/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        sys.exit(1)

    # Create output directories
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if json_output_path:
        os.makedirs(os.path.dirname(json_output_path), exist_ok=True)

    # Set up OpenPose parameters for black background
    params = {
        "model_folder": "C:/Users/singh/Downloads/openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended/openpose/models/",
        "display": 0,
        "render_pose": 1,  # Regular rendering
        "model_pose": "BODY_25",
        "net_resolution": "-1x368",
        "disable_blending": True,  # This will keep background black
        "render_threshold": 0.05,
        "number_people_max": 1
    }

    # Initialize OpenPose
    op_wrapper = op.WrapperPython()
    op_wrapper.configure(params)
    op_wrapper.start()

    # Process the image
    image_to_process = cv2.imread(input_path)
    if image_to_process is None:
        print(f"Error loading image: {input_path}")
        sys.exit(1)

    datum = op.Datum()
    datum.cvInputData = image_to_process
    op_wrapper.emplaceAndPop(op.VectorDatum([datum]))

    # Create output with black background
    if hasattr(datum, 'cvOutputData'):
        # Convert to black background by thresholding
        gray = cv2.cvtColor(datum.cvOutputData, cv2.COLOR_BGR2GRAY)
        _, black_bg = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        black_bg = cv2.cvtColor(black_bg, cv2.COLOR_GRAY2BGR)
        
        # Combine with original pose (white on black)
        result = cv2.bitwise_and(datum.cvOutputData, black_bg)
        cv2.imwrite(output_path, result)
        print(f"Saved pose image with black background: {output_path}")
    else:
        print("Warning: No output data generated from OpenPose")
        # Create blank black image as fallback
        blank_image = np.zeros((image_to_process.shape[0], image_to_process.shape[1], 3), dtype=np.uint8)
        cv2.imwrite(output_path, blank_image)

    # Save JSON if requested
    if json_output_path:
        keypoints_data = {"people": []}
        
        if hasattr(datum, 'poseKeypoints') and datum.poseKeypoints is not None:
            try:
                for person_idx in range(datum.poseKeypoints.shape[0]):
                    keypoints_data["people"].append({
                        "pose_keypoints_2d": datum.poseKeypoints[person_idx].flatten().tolist()
                    })
            except (IndexError, AttributeError, ValueError):
                keypoints_data["people"].append({"pose_keypoints_2d": []})
        else:
            keypoints_data["people"].append({"pose_keypoints_2d": []})
        
        with open(json_output_path, 'w') as f:
            json.dump(keypoints_data, f, indent=2)
        print(f"Saved JSON keypoints: {json_output_path}")

    print("Processing complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())