import os
import pymysql
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Direct connection test function using pymysql (like the working app)
def test_direct_connection():
    # Get database connection details from environment variables with defaults
    host = os.environ.get("DB_HOST", "sportsdb-sports-database-for-web-scrapes.g.aivencloud.com")
    port = int(os.environ.get("DB_PORT", "16439"))
    user = os.environ.get("DB_USER", "avnadmin")
    password = os.environ.get("DB_PASSWORD")  # Password from .env file
    database = os.environ.get("DB_NAME", "defaultdb")
    
    try:
        # Create a direct pymysql connection (similar to the working app)
        conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        cursor = conn.cursor()
        
        # Test with a simple query
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        print(f"Direct connection test successful: {result}")
        
        # Close connections
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Direct connection test failed: {str(e)}")
        import traceback
        print(f"Error details: {traceback.format_exc()}")
        return False

# Database connection details
def get_database_connection():
    # Get database connection details from environment variables with defaults
    host = os.environ.get("DB_HOST", "sportsdb-sports-database-for-web-scrapes.g.aivencloud.com")
    port = int(os.environ.get("DB_PORT", "16439"))
    user = os.environ.get("DB_USER", "avnadmin")
    password = os.environ.get("DB_PASSWORD")  # Password from .env file
    database = os.environ.get("DB_NAME", "defaultdb")
    
    # Create a direct pymysql connection first (similar to the working app)
    try:
        # Create a connection pool using SQLAlchemy
        db_url = f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}'
        
        # Create engine with minimal connection arguments
        return create_engine(db_url)
    except Exception as e:
        print(f"Failed to create database connection: {str(e)}")
        import traceback
        print(f"Connection error details: {traceback.format_exc()}")
        raise

# List all tables in the database
def list_tables():
    engine = get_database_connection()
    try:
        with engine.connect() as conn:
            # Query to list all tables
            query = text("SHOW TABLES")
            result = conn.execute(query)
            tables = [row[0] for row in result]
            print("Available tables in the database:")
            for table in tables:
                print(f"- {table}")
            return tables
    except Exception as e:
        print(f"Error listing tables: {str(e)}")
        return []

if __name__ == "__main__":
    # First try the direct connection test
    print("Testing direct pymysql connection...")
    if test_direct_connection():
        print("Direct connection successful, now testing SQLAlchemy connection...")
        list_tables()
    else:
        print("Direct connection failed. Please check your database credentials and network connection.")