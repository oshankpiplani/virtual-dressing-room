# test_openpose.py
import cv2
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("openpose_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("OpenPose-Test")

def test_openpose_installation():
    try:
        # Windows-specific setup
        if os.name == 'nt':
            openpose_path = r"C:\Users\singh\project\ml\preprocessing\openpose"
            python_path = os.path.join(openpose_path, "python", "openpose", "Release")
            bin_path = os.path.join(openpose_path, "bin")
            release_path = os.path.join(openpose_path, "x64", "Release")
            
            os.environ['PATH'] = f"{bin_path};{release_path};" + os.environ['PATH']
            sys.path.append(python_path)
            
            logger.info(f"Added to PATH: {bin_path}, {release_path}")
            logger.info(f"Added to Python path: {python_path}")
        
        import pyopenpose as op
        logger.info("Successfully imported pyopenpose")
        return True
    except Exception as e:
        logger.error(f"Import failed: {str(e)}")
        return False

def test_image_processing(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error("Could not read test image")
            return False
            
        logger.info(f"Image loaded successfully. Dimensions: {img.shape}")
        return True
    except Exception as e:
        logger.error(f"Image test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_image = r"C:\Users\singh\project\backend\services\processed\image\person_26_nobg.jpg"
    
    logger.info("Starting OpenPose installation test...")
    if not test_openpose_installation():
        sys.exit(1)
        
    logger.info("Starting image processing test...")
    if not test_image_processing(test_image):
        sys.exit(1)
        
    logger.info("All tests passed successfully!")