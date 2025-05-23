#!/usr/bin/env python3

import os
import sys
import json
import argparse
import subprocess
import logging
import boto3
from botocore.config import Config
import psycopg2
import urllib.request
import time
import uuid
import shutil
import io
from datetime import datetime
from dotenv import load_dotenv
import warnings
from PIL import Image

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("preprocessor")

# Load environment variables
load_dotenv()

# AWS Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET")
AWS_S3_BUCKET_NAME = os.getenv("AWS_BUCKET", "virtual-dressing-room")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_CONFIG = Config(
    connect_timeout=5,
    read_timeout=5,
    retries={'max_attempts': 3}
)

# Database configuration
DB_PARAMS = {
    'host': os.getenv("DB_HOST", "localhost"),
    'database': os.getenv("DB_NAME", "postgres"),
    'user': os.getenv("DB_USER", "postgres"),
    'password': os.getenv("DB_PASS", "Sherlock@23"),
    'port': os.getenv("DB_PORT", "5432")
}

# Directory setup
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(WORKING_DIR, "temp")
PROCESSED_DIR = os.path.join(WORKING_DIR, "processed")

# S3 folder structure
S3_FOLDERS = {
    "cloth": "datasets/cloth",
    "image": "datasets/image", 
    "image_parse": "datasets/image-parse",
    "openpose": "datasets/openpose-img",
    "openpose_json": "datasets/openpose-json",
    "cloth_mask": "datasets/cloth-mask",
    "results": "results"
}

# Environment configurations
ENV_CONFIGS = {
    "remove_bg": {
        "type": "conda",
        "name": "tf115",
        "script_path": r"C:\Users\singh\project\ml\preprocessing\remove_bg\removebg.py",
        "python_path": r"C:\Users\singh\anaconda3\envs\tf115\python.exe",
        "timeout": 300
    },
    "cloth_mask": {
        "type": "conda",
        "name": "tf115",
        "script_path": r"C:\Users\singh\project\ml\preprocessing\remove_bg\cloth_mask.py",
        "python_path": r"C:\Users\singh\anaconda3\envs\tf115\python.exe",
        "timeout": 300
    },
    "inf_pgn": {
        "type": "conda",
        "name": "tf115",
        "script_path": r"C:\Users\singh\project\ml\preprocessing\segmentation\CIHP_PGN\inf_pgn.py",
        "python_path": r"C:\Users\singh\anaconda3\envs\tf115\python.exe",
        "timeout": 600
    },
    "openpose": {
        "type": "conda",
        "name": "openpose",
        "script_path": r"C:\Users\singh\project\ml\preprocessing\openpose\python\10_generate_pose.py",
        "python_path": r"C:\Users\singh\anaconda3\envs\openpose\python.exe",
        "openpose_dir": r"C:\Users\singh\project\ml\preprocessing\openpose",
        "timeout": 900
    },
    "virtual_try_on": {
        "type": "conda",
        "name": "schp",
        "script_path": r"C:\Users\singh\project\ml\inferrence\VITON-HD\test.py",
        "python_path": r"C:\Users\singh\anaconda3\envs\schp\python.exe",
        "default_args": ["--name", "viton-hd"],
        "timeout": 1800
    }
}

# Create directories
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
for subdir in ["cloth", "image", "image-parse", "openpose-img", "openpose-json", "cloth-mask", "try_on_results"]:
    os.makedirs(os.path.join(PROCESSED_DIR, subdir), exist_ok=True)

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
    config=S3_CONFIG
)

def clean_old_datasets():
    """Clean up old dataset directories"""
    try:
        for item in os.listdir(TEMP_DIR):
            item_path = os.path.join(TEMP_DIR, item)
            if item.startswith("datasets-") and os.path.isdir(item_path):
                shutil.rmtree(item_path)
                logger.info(f"Removed old dataset: {item}")
    except Exception as e:
        logger.error(f"Error cleaning datasets: {str(e)}")

def download_with_retry(url, local_path, max_retries=3, delay=5):
    """Download with retry logic"""
    for attempt in range(max_retries):
        try:
            if download_file_from_s3(url, local_path):
                if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                    return True
                logger.warning(f"Downloaded empty file, retrying... (attempt {attempt + 1})")
        except Exception as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {str(e)}")
        
        if attempt < max_retries - 1:
            time.sleep(delay)
    
    return False

def download_file_from_s3(s3_url, local_path):
    """Download file from S3 with retries"""
    logger.info(f"Downloading {s3_url} to {local_path}")
    
    try:
        if s3_url.startswith("https://"):
            parts = s3_url.replace("https://", "").split(".")
            bucket = parts[0]
            key = "/".join(s3_url.split("/")[3:])
        elif s3_url.startswith("s3://"):
            parts = s3_url[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
        else:
            raise ValueError(f"Invalid S3 URL format: {s3_url}")
        
        s3_client.download_file(bucket, key, local_path)
        return True
    except Exception as e:
        logger.error(f"Failed to download {s3_url}: {str(e)}")
        return False

def upload_file_to_s3(local_path, s3_folder, filename=None):
    """Upload file to S3"""
    if not os.path.exists(local_path):
        logger.error(f"File not found for upload: {local_path}")
        return None
        
    filename = filename or os.path.basename(local_path)
    s3_key = f"{s3_folder}/{filename}"
    
    try:
        s3_client.upload_file(local_path, AWS_S3_BUCKET_NAME, s3_key)
        return f"https://{AWS_S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
    except Exception as e:
        logger.error(f"Failed to upload {local_path}: {str(e)}")
        return None

def validate_image(file_path):
    """Verify an image file is valid"""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception as e:
        logger.error(f"Invalid image file {file_path}: {str(e)}")
        return False

def update_db_status(conn, image_id, step, status):
    """Update processing status in database"""
    try:
        cursor = conn.cursor()
        
        # Use the specific status columns from your schema
        cursor.execute(
            f"UPDATE preprocessing_steps SET {step}_status = %s, {step}_timestamp = %s WHERE image_id = %s",
            (status, datetime.now(), image_id)
        )
        
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        if conn:
            conn.rollback()
        return False

def run_in_env(env_config, input_file, output_file, args=None):
    """Run command in specified environment with special handling for OpenPose"""
    args = args or []
    python_path = env_config["python_path"]
    script_path = env_config["script_path"]
    timeout = env_config.get("timeout", 300)
    
    # Special handling for OpenPose
    if env_config.get("name") == "openpose":
        # Set OpenPose environment variables
        openpose_dir = env_config["openpose_dir"]
        os.environ["PATH"] = f"{os.path.join(openpose_dir, 'bin')};{os.path.join(openpose_dir, 'x64', 'Release')};{os.environ['PATH']}"
        os.environ["PYTHONPATH"] = f"{os.path.join(openpose_dir, 'python', 'openpose', 'Release')};{os.environ.get('PYTHONPATH', '')}"
        
        logger.debug(f"Updated PATH for OpenPose: {os.environ['PATH']}")
        logger.debug(f"Updated PYTHONPATH for OpenPose: {os.environ['PYTHONPATH']}")

    if env_config.get("name") == "schp" and input_file is None and output_file is None:
        cmd = [python_path, script_path] + env_config.get("default_args", []) + args
    else:
        cmd = [python_path, script_path, "--input", input_file, "--output", output_file] + args
    
    logger.info(f"Executing: {' '.join(cmd)}")
    
    try:
        # For OpenPose, we need to run in a new shell to get the updated environment
        use_shell = (env_config.get("name") == "openpose")
        
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=use_shell
        )
        
        logger.debug(f"Command output: {result.stdout}")
        if result.stderr:
            logger.warning(f"Command stderr: {result.stderr}")
            
        # Verify output file was created if specified
        if output_file and not os.path.exists(output_file):
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=cmd,
                output=result.stdout,
                stderr="Output file was not created"
            )
            
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after {timeout} seconds")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed. Return code: {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        logger.error(f"Standard output: {e.stdout}")
        return False

def prepare_val_files(image_path, script_dir):
    """Create val_id.txt and val.txt for segmentation"""
    try:
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        
        val_id_path = os.path.join(script_dir, "val_id.txt")
        val_txt_path = os.path.join(script_dir, "val.txt")
        
        with open(val_id_path, 'w') as f:
            f.write(f"{img_name}\n")
        
        with open(val_txt_path, 'w') as f:
            f.write(f"/images/{img_name}.jpg /labels/{img_name}.png\n")
        
        return val_id_path, val_txt_path
    except Exception as e:
        logger.error(f"Error creating val files: {str(e)}")
        return None, None

def prepare_dataset(job_id, files):
    """Prepare dataset directory for virtual try-on"""
    try:
        dataset_dir = os.path.join(TEMP_DIR, f"datasets-{job_id}")
        test_dir = os.path.join(dataset_dir, "test")
        
        os.makedirs(test_dir, exist_ok=True)
        for subdir in ["cloth", "image", "image-parse", "openpose-img", "openpose-json", "cloth-mask"]:
            os.makedirs(os.path.join(test_dir, subdir), exist_ok=True)
        
        # Verify all input files exist
        for key, path in files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file {key} not found at {path}")
        
        # Copy files to dataset directory
        shutil.copy2(files["cloth"], os.path.join(test_dir, "cloth", f"cloth_{job_id}.jpg"))
        shutil.copy2(files["image"], os.path.join(test_dir, "image", f"person_{job_id}.jpg"))
        shutil.copy2(files["parse"], os.path.join(test_dir, "image-parse", f"person_{job_id}.png"))
        shutil.copy2(files["pose_img"], os.path.join(test_dir, "openpose-img", f"person_{job_id}_rendered.png"))
        shutil.copy2(files["pose_json"], os.path.join(test_dir, "openpose-json", f"person_{job_id}_keypoints.json"))
        shutil.copy2(files["cloth_mask"], os.path.join(test_dir, "cloth-mask", f"cloth_{job_id}.jpg"))
        
        # Create test pairs file
        pairs_path = os.path.join(dataset_dir, "test_pairs.txt")
        with open(pairs_path, 'w') as f:
            f.write(f"person_{job_id}.jpg cloth_{job_id}.jpg\n")
        print(dataset_dir)    
        
        return dataset_dir
    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}")
        return None

def process_job(job_id):
    """Process a virtual try-on job"""
    logger.info(f"Starting job {job_id}")
    conn = None
    person_orig = None
    cloth_orig = None
    
    try:
        # Initialize database connection
        conn = psycopg2.connect(**DB_PARAMS)
        
        # Update the main job status in images table
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE images SET status = 'processing' WHERE id = %s",
                (job_id,)
            )
            conn.commit()

        # Get job details
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT person_image_path, cloth_image_path FROM images WHERE id = %s",
                (job_id,)
            )
            person_url, cloth_url = cursor.fetchone()
        
        # Generate unique filenames
        run_id = str(uuid.uuid4())[:8]
        person_orig = os.path.join(TEMP_DIR, f"person_{job_id}_{run_id}.jpg")
        cloth_orig = os.path.join(TEMP_DIR, f"cloth_{job_id}_{run_id}.jpg")
        
        # Download images with retries
        if not download_with_retry(person_url, person_orig):
            raise Exception("Failed to download person image after 3 attempts")
        
        if not download_with_retry(cloth_url, cloth_orig):
            raise Exception("Failed to download cloth image after 3 attempts")
            
        # Validate downloaded images
        if not validate_image(person_orig):
            raise Exception("Invalid person image file")
            
        if not validate_image(cloth_orig):
            raise Exception("Invalid cloth image file")

        # 1. Remove background with better error handling
        update_db_status(conn, job_id, "remove_bg", "processing")
        person_nobg = os.path.join(PROCESSED_DIR, "image", f"person_{job_id}_nobg.jpg")
        
        try:
            if not run_in_env(ENV_CONFIGS["remove_bg"], person_orig, person_nobg):
                raise Exception("Background removal failed")
            
            # Verify output was created
            if not os.path.exists(person_nobg) or os.path.getsize(person_nobg) == 0:
                raise Exception("Background removal produced empty output")
                
            update_db_status(conn, job_id, "remove_bg", "completed")
        except Exception as e:
            update_db_status(conn, job_id, "remove_bg", "failed")
            raise e

        # 2. Create cloth mask
        update_db_status(conn, job_id, "cloth_mask", "processing")
        cloth_mask = os.path.join(PROCESSED_DIR, "cloth-mask", f"cloth_{job_id}.jpg")
        cloth_masked = os.path.join(PROCESSED_DIR, "cloth-mask", f"cloth_{job_id}_masked.jpg")
        
        try:
            if not run_in_env(ENV_CONFIGS["cloth_mask"], cloth_orig, cloth_mask, ["--masked-output", cloth_masked]):
                raise Exception("Cloth mask creation failed")
                
            if not os.path.exists(cloth_mask) or os.path.getsize(cloth_mask) == 0:
                raise Exception("Cloth mask produced empty output")
                
            update_db_status(conn, job_id, "cloth_mask", "completed")
        except Exception as e:
            update_db_status(conn, job_id, "cloth_mask", "failed")
            raise e

        # 3. Image parsing
        update_db_status(conn, job_id, "segmentation", "processing")
        parse_output = os.path.join(PROCESSED_DIR, "image-parse", f"person_{job_id}.png")
        val_id, val_txt = prepare_val_files(person_nobg, os.path.dirname(ENV_CONFIGS["inf_pgn"]["script_path"]))
        
        try:
            if not run_in_env(ENV_CONFIGS["inf_pgn"], person_nobg, parse_output, ["--val-id-file", val_id, "--val-file", val_txt]):
                raise Exception("Image parsing failed")
                
            if not os.path.exists(parse_output) or os.path.getsize(parse_output) == 0:
                raise Exception("Image parsing produced empty output")
                
            update_db_status(conn, job_id, "segmentation", "completed")
        except Exception as e:
            update_db_status(conn, job_id, "segmentation", "failed")
            raise e

        # 4. OpenPose
        update_db_status(conn, job_id, "pose_generation", "processing")
        pose_img = os.path.join(PROCESSED_DIR, "openpose-img", f"person_{job_id}_pose.png")
        pose_json = os.path.join(PROCESSED_DIR, "openpose-json", f"person_{job_id}_pose.json")
        
        try:
            if not run_in_env(ENV_CONFIGS["openpose"], person_nobg, pose_img, ["--json-output", pose_json]):
                raise Exception("Pose generation failed")
                
            if not os.path.exists(pose_img) or os.path.getsize(pose_img) == 0:
                raise Exception("Pose image produced empty output")
                
            if not os.path.exists(pose_json) or os.path.getsize(pose_json) == 0:
                raise Exception("Pose JSON produced empty output")
                
            update_db_status(conn, job_id, "pose_generation", "completed")
        except Exception as e:
            update_db_status(conn, job_id, "pose_generation", "failed")
            raise e

        # 5. Virtual try-on
        update_db_status(conn, job_id, "final_processing", "processing")
        output_dir = os.path.join(PROCESSED_DIR, "try_on_results", f"job_{job_id}")
        os.makedirs(output_dir, exist_ok=True)
        
        dataset_dir = prepare_dataset(job_id, {
            "cloth": cloth_orig,
            "image": person_nobg,
            "parse": parse_output,
            "pose_img": pose_img,
            "pose_json": pose_json,
            "cloth_mask": cloth_mask
        })
        
        if not dataset_dir:
            raise Exception("Failed to prepare dataset directory")
        
        try:
            if not run_in_env(ENV_CONFIGS["virtual_try_on"], None, None, [
                "--name", f"job_{job_id}",
                "--dataset_dir", dataset_dir,
                "--dataset_list", os.path.join(dataset_dir, "test_pairs.txt"),
                "--save_dir", output_dir,
                "--batch_size", "1",
                "--workers", "1",
                "--load_height", "1024",
                "--load_width", "768"
            ]):
                raise Exception("Virtual try-on failed")
            
            # Find and upload result
            result_dir = os.path.join(output_dir, f"job_{job_id}")
            if not os.path.exists(result_dir):
                raise Exception("Result directory not created")
                
            result_files = [f for f in os.listdir(result_dir) if f.endswith(('.jpg', '.png'))]
            if not result_files:
                raise Exception("No result files found in output directory")
                
            result_path = os.path.join(result_dir, result_files[0])
            if not os.path.exists(result_path) or os.path.getsize(result_path) == 0:
                raise Exception("Result file is empty")
                
            result_url = upload_file_to_s3(result_path, S3_FOLDERS["results"], f"try_on_result_{job_id}.jpg")
            if not result_url:
                raise Exception("Failed to upload result to S3")
            
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE images SET result_image_path = %s, aws_url = %s, status = %s WHERE id = %s",
                    (result_path, result_url, "completed", job_id)
                )
                conn.commit()
            
            update_db_status(conn, job_id, "final_processing", "completed")
            logger.info(f"Completed job {job_id}")
            return True
            
        except Exception as e:
            update_db_status(conn, job_id, "final_processing", "failed")
            raise e
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}", exc_info=True)
        if conn:
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "UPDATE images SET status = 'failed' WHERE id = %s",
                        (job_id,)
                    )
                    conn.commit()
            except Exception as db_error:
                logger.error(f"Failed to update job status to failed: {str(db_error)}")
        return False
    finally:
        # Cleanup temporary files
        for f in [person_orig, cloth_orig]:
            try:
                if f and os.path.exists(f):
                    os.remove(f)
            except Exception as e:
                logger.warning(f"Could not remove temp file {f}: {str(e)}")
                
        if conn:
            conn.close()

def check_pending_jobs():
    """Check for and process pending jobs"""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id FROM images WHERE status = 'pending' ORDER BY created_at ASC LIMIT 5 FOR UPDATE"
            )
            jobs = cursor.fetchall()
            
            if jobs:
                logger.info(f"Found {len(jobs)} pending jobs")
                for (job_id,) in jobs:
                    process_job(job_id)
        
        conn.close()
    except Exception as e:
        logger.error(f"Error checking jobs: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Virtual Try-On Processing Service")
    parser.add_argument("--job-id", type=int, help="Process specific job ID")
    parser.add_argument("--daemon", action="store_true", help="Run in daemon mode")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds")
    
    args = parser.parse_args()
    
    if args.job_id:
        process_job(args.job_id)
    elif args.daemon:
        logger.info(f"Starting daemon mode (interval: {args.interval}s)")
        while True:
            check_pending_jobs()
            time.sleep(args.interval)
    else:
        check_pending_jobs()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)