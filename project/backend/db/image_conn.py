import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def create_tables():
    # Database connection parameters - replace with your actual credentials
    db_params = {
        'host': 'localhost',
        'database': 'postgres',
        'user': 'postgres',
        'password': 'Sherlock@23',
        'port': '5432'
    }

    try:
        # Connect to the PostgreSQL server
        conn = psycopg2.connect(**db_params)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Check if tables exist
        cursor.execute("SELECT to_regclass('images')")
        images_table_exists = cursor.fetchone()[0]
        cursor.execute("SELECT to_regclass('preprocessing_steps')")
        preprocessing_steps_table_exists = cursor.fetchone()[0]

        if not images_table_exists:
            create_images_table(cursor)
        else:
            print("Table 'images' already exists.")

        if not preprocessing_steps_table_exists:
            create_preprocessing_steps_table(cursor)
        else:
            print("Table 'preprocessing_steps' already exists.  Attempting to alter it.")
            alter_preprocessing_steps_table(cursor)  # Add this function

        # Close the cursor and connection
        cursor.close()
        conn.close()

        print("Tables creation/alteration process completed successfully!")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error: {error}")
        if conn:
            conn.rollback()  # Rollback any changes in case of an error
    finally:
        if conn is not None:
            conn.close()
        print("Database connection closed.")

def create_images_table(cursor):
    create_images_sql = """
    CREATE TABLE IF NOT EXISTS images (
        id SERIAL PRIMARY KEY,
        person_image_path VARCHAR(255) NOT NULL,
        cloth_image_path VARCHAR(255) NOT NULL,
        status VARCHAR(10) CHECK (status IN ('pending', 'processing', 'completed', 'failed')) DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        result_image_path VARCHAR(255),
        aws_url VARCHAR(255)
    );
    """
    cursor.execute(create_images_sql)
    print("Table 'images' created.")

def create_preprocessing_steps_table(cursor):
    create_preprocessing_steps_sql = """
    CREATE TABLE IF NOT EXISTS preprocessing_steps (
        id SERIAL PRIMARY KEY,
        image_id INTEGER NOT NULL,
        remove_bg_status VARCHAR(10) CHECK (remove_bg_status IN ('pending', 'processing', 'completed', 'failed')) DEFAULT 'pending',
        segmentation_status VARCHAR(10) CHECK (segmentation_status IN ('pending', 'processing', 'completed', 'failed')) DEFAULT 'pending',
        pose_generation_status VARCHAR(10) CHECK (pose_generation_status IN ('pending', 'processing', 'completed', 'failed')) DEFAULT 'pending',
        cloth_resize_status VARCHAR(10) CHECK (cloth_resize_status IN ('pending', 'processing', 'completed', 'failed')) DEFAULT 'pending',
        cloth_mask_status VARCHAR(10) CHECK (cloth_mask_status IN ('pending', 'processing', 'completed', 'failed')) DEFAULT 'pending',
        remove_bg_timestamp TIMESTAMP,
        segmentation_timestamp TIMESTAMP,
        pose_generation_timestamp TIMESTAMP,
        cloth_resize_timestamp TIMESTAMP,
        cloth_mask_timestamp TIMESTAMP,
        final_processing_status VARCHAR(10) CHECK (final_processing_status IN ('pending', 'processing', 'completed', 'failed')) DEFAULT 'pending',
        final_processing_timestamp TIMESTAMP,
        FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
    );
    """
    cursor.execute(create_preprocessing_steps_sql)
    print("Table 'preprocessing_steps' created.")

def alter_preprocessing_steps_table(cursor):
    alter_sql = """
    ALTER TABLE preprocessing_steps
    ADD COLUMN IF NOT EXISTS final_processing_status VARCHAR(10) CHECK (final_processing_status IN ('pending', 'processing', 'completed', 'failed')) DEFAULT 'pending',
    ADD COLUMN IF NOT EXISTS final_processing_timestamp TIMESTAMP;
    """
    cursor.execute(alter_sql)
    print("Table 'preprocessing_steps' altered: columns added.")

if __name__ == "__main__":
    create_tables()