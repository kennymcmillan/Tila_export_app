import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection details
def get_database_connection():
    host = 'sportsdb-sports-database-for-web-scrapes.g.aivencloud.com'
    port = 16439
    user = 'avnadmin'
    password = os.getenv('DB_PASSWORD')  # Password from .env file
    database = 'defaultdb'
    
    # MySQL database connection string
    db_url = f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4'
    return create_engine(db_url)

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
    list_tables()