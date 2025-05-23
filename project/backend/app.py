import os

from flask import Response, jsonify
import boto3
import psycopg2
import threading
import subprocess
import shutil
from urllib.parse import urlparse
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from flask_cors import CORS
import warnings
from boto3.compat import PythonDeprecationWarning

# Suppress boto3 deprecation warnings
warnings.filterwarnings("ignore", category=PythonDeprecationWarning)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
load_dotenv()

# AWS S3 Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET")
AWS_S3_BUCKET_NAME = 'virtual-dressing-room'
AWS_REGION = 'ap-south-1'

# Database configuration
DB_PARAMS = {
    'host': 'localhost',
    'database': 'postgres',
    'user': 'postgres',
    'password': 'Sherlock@23',
    'port': '5432'
}

# Local directories
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = r"C:\Users\singh\project\backend\services\temp"
# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# Path to the preprocessing orchestrator script
PREPROCESSOR_SCRIPT = r"C:\Users\singh\project\backend\services\generate_image_service.py"

@app.route('/upload-images', methods=['POST'])
def upload_images():
    try:
        # Retrieve files using keys from the form data
        dress_image = request.files.get('dress_image')
        person_image = request.files.get('person_image')

        if not dress_image or not person_image:
            return jsonify({"error": "Both images are required"}), 400

        # Secure filenames, using default values if necessary
        dress_filename = secure_filename(dress_image.filename) if dress_image.filename else "default_dress.jpg"
        person_filename = secure_filename(person_image.filename) if person_image.filename else "default_person.jpg"

        if dress_filename == '' or person_filename == '':
            return jsonify({"error": "Filename after sanitization is empty"}), 400

        # Construct S3 paths
        dress_s3_path = f'dress/{dress_filename}'
        person_s3_path = f'person/{person_filename}'

        # Upload files to S3 without ACL
        s3_client.upload_fileobj(dress_image, AWS_S3_BUCKET_NAME, dress_s3_path)
        s3_client.upload_fileobj(person_image, AWS_S3_BUCKET_NAME, person_s3_path)

        # Construct the public URLs for the uploaded images (if objects are public via bucket policy)
        dress_url = f"https://{AWS_S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{dress_s3_path}"
        person_url = f"https://{AWS_S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{person_s3_path}"

        return jsonify({
            "message": "Images uploaded successfully",
            "dress_image_url": dress_url,
            "person_image_url": person_url
        }), 200

    except Exception as e:
        app.logger.error("Error in /upload-images: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Get the image URLs from the request
        data = request.json
        if not data or 'dress_image_path' not in data or 'person_image_path' not in data:
            return jsonify({"error": "Both dress and person image paths are required"}), 400

        dress_image_path = data['dress_image_path']
        person_image_path = data['person_image_path']

        # Connect to the database
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()

        # Insert into the images table
        cursor.execute(
            """
            INSERT INTO images (person_image_path, cloth_image_path, status)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (person_image_path, dress_image_path, 'pending')
        )
        
        # Get the ID of the newly inserted row
        image_id = cursor.fetchone()[0]

        # Insert into the preprocessing_steps table
        cursor.execute(
            """
            INSERT INTO preprocessing_steps (image_id)
            VALUES (%s)
            """,
            (image_id,)
        )

        # Commit the transaction
        conn.commit()
        cursor.close()
        conn.close()

        # Start the preprocessing orchestrator in a separate process
        def run_preprocessor(job_id):
            try:
                subprocess.run(["python", PREPROCESSOR_SCRIPT, "--job-id", str(job_id)], check=True)
            except subprocess.CalledProcessError as e:
                app.logger.error(f"Preprocessor failed for job {job_id}: {e}")
            except Exception as e:
                app.logger.error(f"Error running preprocessor for job {job_id}: {e}")

        # Run the preprocessor in a separate thread
        threading.Thread(target=run_preprocessor, args=(image_id,)).start()

        # Return success response
        return jsonify({
            "message": "Generation request submitted successfully",
            "job_id": image_id,
            "status": "pending"
        }), 200

    except Exception as e:
        app.logger.error("Error in /generate: %s", e, exc_info=True)
        # If an error occurs, rollback the transaction
        if 'conn' in locals() and conn:
            conn.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/status/<int:job_id>', methods=['GET'])
def get_status(job_id):
    try:
        # Connect to the database
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()

        # Get the status from the images table
        cursor.execute(
            """
            SELECT status, result_image_path, aws_url FROM images WHERE id = %s
            """,
            (job_id,)
        )
        
        result = cursor.fetchone()
        
        if not result:
            return jsonify({"error": "Job not found"}), 404
        
        image_status, result_image_path, aws_url = result
            
        # Get the preprocessing steps status
        cursor.execute(
            """
            SELECT 
                remove_bg_status, 
                segmentation_status, 
                pose_generation_status, 
                cloth_resize_status, 
                cloth_mask_status
            FROM preprocessing_steps 
            WHERE image_id = %s
            """,
            (job_id,)
        )
        
        preprocessing_status = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        response = {
            "job_id": job_id,
            "overall_status": image_status,
            "result_url": aws_url if aws_url else None,
            "preprocessing": {
                "remove_bg": preprocessing_status[0],
                "segmentation": preprocessing_status[1],
                "pose_generation": preprocessing_status[2],
                "cloth_resize": preprocessing_status[3],
                "cloth_mask": preprocessing_status[4]
            }
        }
        
        return jsonify(response), 200

    except Exception as e:
        app.logger.error("Error in /status: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/cleanup', methods=['POST'])
def cleanup_datasets():
    try:
        # Get the job ID from the request
        data = request.json
        if not data or 'job_id' not in data:
            return jsonify({"error": "Job ID is required"}), 400

        job_id = data['job_id']
        
        # Path to current dataset directory
        current_dataset_dir = os.path.join(TEMP_DIR, f"datasets-{job_id}")
        
        # Find and remove all older dataset directories
        for item in os.listdir(TEMP_DIR):
            item_path = os.path.join(TEMP_DIR, item)
            # Check if it's a datasets directory and not the current one
            if item.startswith("datasets-") and os.path.isdir(item_path) and item_path != current_dataset_dir:
                try:
                    shutil.rmtree(item_path)
                    app.logger.info(f"Removed old dataset directory: {item_path}")
                except Exception as e:
                    app.logger.error(f"Error removing directory {item_path}: {str(e)}")
        
        return jsonify({"message": "Cleanup successful"}), 200
        
    except Exception as e:
        app.logger.error("Error in /cleanup: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500
    



@app.route('/result-image/<int:job_id>', methods=['GET'])
def get_result_image(job_id):
    try:
        # Connect to database to get the S3 key
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        cursor.execute("SELECT aws_url FROM images WHERE id = %s", (job_id,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not result or not result[0]:
            return jsonify({"error": "Image not found"}), 404
        
       
        parsed_url = urlparse(result[0])
        s3_key = parsed_url.path.lstrip('/') 
        print(s3_key)
        
        # Get the object from S3
        response = s3_client.get_object(Bucket=AWS_S3_BUCKET_NAME, Key=s3_key)
        image_data = response['Body'].read()
        
        # Return the image
        return Response(
            image_data,
            mimetype='image/jpeg',
            headers={"Content-Disposition": f"inline; filename=result_{job_id}.jpg"}
        )
    except Exception as e:
        app.logger.error(f"Error retrieving image: {str(e)}")
        return jsonify({"error": "Failed to retrieve image"}), 500
    
if __name__ == '__main__':
    app.run(debug=True,port=5001)