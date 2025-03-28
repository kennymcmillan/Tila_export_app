import os
import pandas as pd
import streamlit as st
import pymysql
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import datetime
import zipfile
import tempfile
import io

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
        st.success(f"Direct connection test successful: {result}")
        
        # Close connections
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Direct connection test failed: {str(e)}")
        import traceback
        st.error(f"Error details: {traceback.format_exc()}")
        return False

# Set page configuration
st.set_page_config(
    page_title="Athletics Results Exporter",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database connection details
@st.cache_resource
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
        st.error(f"Failed to create database connection: {str(e)}")
        import traceback
        st.error(f"Connection error details: {traceback.format_exc()}")
        raise

# Get min and max dates from the database
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_date_range(table_name='Tilastopija_results'):
    engine = get_database_connection()
    try:
        with engine.connect() as conn:
            # Test connection with a simple query first
            test_query = text("SELECT 1")
            conn.execute(test_query)
            
            # Query for min date
            min_date_query = text(f"SELECT MIN(Start_Date) FROM {table_name}")
            min_date_result = conn.execute(min_date_query).scalar()
            
            # Query for max date
            max_date_query = text(f"SELECT MAX(Start_Date) FROM {table_name}")
            max_date_result = conn.execute(max_date_query).scalar()
            
            return min_date_result, max_date_result
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        st.error(f"Error fetching date range: {str(e)}")
        st.error(f"Detailed error: {error_details}")
        return None, None

# Get unique events from the database
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_unique_events(table_name='Tilastopija_results'):
    engine = get_database_connection()
    try:
        with engine.connect() as conn:
            query = text(f"SELECT DISTINCT Event FROM {table_name} ORDER BY Event")
            result = conn.execute(query)
            events = [row[0] for row in result]
            return events
    except Exception as e:
        st.error(f"Error fetching unique events: {str(e)}")
        return []

# Query data with filters
def query_data(table_name, gender, start_date, end_date, events=None, show_progress=True, limit=None, sort_by_date_desc=False):
    engine = get_database_connection()
    try:
        # Build the query based on filters
        query_str = f"""
            SELECT * FROM {table_name}
            WHERE Gender = :gender
            AND Start_Date BETWEEN :start_date AND :end_date
        """
        
        params = {
            "gender": gender,
            "start_date": start_date,
            "end_date": end_date
        }
        
        # Add event filter if specified
        if events and len(events) > 0:
            # If not all events are selected
            if len(events) < len(get_unique_events(table_name)):
                placeholders = ", ".join([f":event_{i}" for i in range(len(events))])
                query_str += f" AND Event IN ({placeholders})"
                
                # Add event parameters
                for i, event in enumerate(events):
                    params[f"event_{i}"] = event
        
        # Add ORDER BY clause if requested
        if sort_by_date_desc:
            query_str += " ORDER BY Start_Date DESC"
            
        # Add LIMIT clause if specified
        if limit is not None:
            # For preview, just get the limited rows directly without counting first
            if not sort_by_date_desc:
                query_str += " ORDER BY Start_Date DESC"
            query_str += f" LIMIT {limit}"
            
            with engine.connect() as conn:
                # Execute the query directly without counting first
                query = text(query_str)
                data = pd.read_sql(query, conn, params=params)
                return data
        else:
            # For full data export, get count first for progress bar
            with engine.connect() as conn:
                count_query = text(f"SELECT COUNT(*) FROM ({query_str}) as count_query")
                total_rows = conn.execute(count_query, params).scalar()
                
                if show_progress:
                    st.write(f"Found {total_rows:,} rows matching your criteria")
                    progress_bar = st.progress(0)
                
                # Execute the actual query
                query = text(query_str)
                
                # For large datasets, use optimized loading techniques
                if total_rows > 10000 and show_progress:
                    # Import necessary libraries for parallel processing
                    import time
                    import concurrent.futures
                    from functools import partial
                    
                    # Use a much larger chunk size for better performance
                    chunk_size = 50000  # Significantly increased for better performance
                    
                    # Calculate number of chunks
                    num_chunks = (total_rows + chunk_size - 1) // chunk_size
                    
                    # Create a status container for progress updates
                    status_container = st.empty()
                    
                    # Track time for estimates
                    start_time = time.time()
                    
                    # Function to fetch a chunk of data
                    def fetch_chunk(i, query_str, params, conn):
                        offset = i * chunk_size
                        chunk_query = text(f"{query_str} LIMIT {chunk_size} OFFSET {offset}")
                        return pd.read_sql(chunk_query, conn, params=params)
                    
                    # Create a connection pool for parallel processing
                    engine = get_database_connection()
                    
                    # Initialize empty DataFrame for results
                    data = pd.DataFrame()
                    
                    # Process chunks with progress updates
                    for i in range(num_chunks):
                        # Fetch chunk
                        offset = i * chunk_size
                        chunk_query = text(f"{query_str} LIMIT {chunk_size} OFFSET {offset}")
                        chunk = pd.read_sql(chunk_query, conn, params=params)
                        
                        # Append to data (more efficient than storing all chunks in memory)
                        if data.empty:
                            data = chunk
                        else:
                            data = pd.concat([data, chunk], ignore_index=True)
                        
                        # Update progress
                        progress = min(1.0, (i + 1) / num_chunks)
                        rows_processed = min((i + 1) * chunk_size, total_rows)
                        
                        # Calculate time estimates
                        elapsed_time = time.time() - start_time
                        if i > 0:  # Avoid division by zero
                            rows_per_second = rows_processed / elapsed_time
                            estimated_total_time = total_rows / rows_per_second
                            remaining_time = estimated_total_time - elapsed_time
                            
                            # Format time as minutes and seconds
                            elapsed_min, elapsed_sec = divmod(int(elapsed_time), 60)
                            remaining_min, remaining_sec = divmod(int(remaining_time), 60)
                            
                            # Update status with all information on one line
                            status_text = (
                                f"Processed: {rows_processed:,}/{total_rows:,} rows ({progress:.1%}) | "
                                f"Elapsed: {elapsed_min}m {elapsed_sec}s | "
                                f"Remaining: {remaining_min}m {remaining_sec}s | "
                                f"Speed: {int(rows_per_second):,} rows/sec"
                            )
                        else:
                            status_text = f"Processed: {rows_processed:,}/{total_rows:,} rows ({progress:.1%})"
                        
                        # Update progress bar and status text
                        progress_bar.progress(progress)
                        status_container.text(status_text)
                    
                    # Final update
                    total_time = time.time() - start_time
                    total_min, total_sec = divmod(int(total_time), 60)
                    progress_bar.progress(1.0)
                    status_container.text(f"Completed: {total_rows:,} rows in {total_min}m {total_sec}s ({int(total_rows/total_time):,} rows/sec)")
                    st.success(f"Successfully loaded all {total_rows:,} rows!")
                else:
                    # For smaller datasets, load all at once
                    import time
                    start_time = time.time()
                    
                    # Create a status container
                    status_container = st.empty()
                    status_container.text(f"Loading {total_rows:,} rows...")
                    
                    # Load data
                    data = pd.read_sql(query, conn, params=params)
                    
                    # Update progress
                    if show_progress and total_rows > 0:
                        total_time = time.time() - start_time
                        total_min, total_sec = divmod(int(total_time), 60)
                        progress_bar.progress(1.0)
                        status_container.text(f"Completed: {total_rows:,} rows in {total_min}m {total_sec}s ({int(total_rows/max(1, total_time)):,} rows/sec)")
                        st.success(f"Successfully loaded all {total_rows:,} rows!")
                
                return data
    except Exception as e:
        st.error(f"Error querying data: {str(e)}")
        return pd.DataFrame()

# Create a downloadable file with optimized performance
def get_downloadable_file(df, file_format):
    buffer = io.BytesIO()
    
    # Start timing
    import time
    start_time = time.time()
    
    # Status indicator
    status = st.empty()
    status.text(f"Preparing {file_format} file for download...")
    
    if file_format == 'CSV':
        # Use the fastest CSV writing method with chunking for large dataframes
        if len(df) > 100000:
            # For very large dataframes, use chunked writing
            status.text("Using optimized CSV chunking for large dataset...")
            chunk_size = 100000
            for i in range(0, len(df), chunk_size):
                if i == 0:
                    # First chunk - create the file with headers
                    df.iloc[i:i+chunk_size].to_csv(buffer, index=False, mode='w')
                else:
                    # Subsequent chunks - append without headers
                    buffer.seek(0, 2)  # Move to the end of file
                    df.iloc[i:i+chunk_size].to_csv(buffer, index=False, mode='a', header=False)
                
                # Update status
                status.text(f"Processed {min(i+chunk_size, len(df)):,} of {len(df):,} rows...")
        else:
            # For smaller dataframes, write all at once
            df.to_csv(buffer, index=False)
        
        mime = "text/csv"
        file_ext = "csv"
    elif file_format == 'Excel':
        # Use optimized Excel writing with minimal formatting
        status.text("Creating Excel file with optimized settings...")
        
        # For large dataframes, warn the user
        if len(df) > 100000:
            st.warning(f"Excel format is not recommended for large datasets ({len(df):,} rows). Consider using CSV or Parquet instead.")
        
        # Use xlsxwriter engine with optimized settings
        try:
            import xlsxwriter
            writer = pd.ExcelWriter(buffer, engine='xlsxwriter', options={'constant_memory': True})
            df.to_excel(writer, index=False, sheet_name='Results')
            writer.close()
        except ImportError:
            # Fallback if xlsxwriter is not available
            df.to_excel(buffer, index=False)
        
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        file_ext = "xlsx"
    elif file_format == 'Parquet':
        # Use optimized parquet writing with compression
        status.text("Creating optimized Parquet file with Snappy compression...")
        try:
            # Try to use pyarrow engine with snappy compression for best performance
            import pyarrow
            df.to_parquet(buffer, index=False, compression='snappy', engine='pyarrow')
        except ImportError:
            # Fallback if pyarrow is not available
            df.to_parquet(buffer, index=False)
        
        mime = "application/octet-stream"
        file_ext = "parquet"
    elif file_format == 'ZIP (CSV)':
        status.text("Creating compressed CSV file...")
        
        # Use more efficient compression
        try:
            # Try to use a more efficient compression method
            import gzip
            
            # Write directly to gzip
            with gzip.GzipFile(fileobj=buffer, mode='w') as gz:
                df.to_csv(gz, index=False)
            
            mime = "application/gzip"
            file_ext = "csv.gz"
        except ImportError:
            # Fallback to standard zip if gzip fails
            temp_csv = io.StringIO()
            df.to_csv(temp_csv, index=False)
            temp_csv.seek(0)
            
            with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.writestr(f"athletics_results.csv", temp_csv.getvalue())
            
            mime = "application/zip"
            file_ext = "zip"
    else:
        st.error(f"Unsupported format: {file_format}")
        return None
    
    # Report time taken
    elapsed_time = time.time() - start_time
    status.text(f"File prepared in {elapsed_time:.2f} seconds. Ready for download!")
    
    buffer.seek(0)
    return buffer, mime, file_ext

# Password check function
def check_password():
    """Returns `True` if the user had the correct password."""
    
    # Get password from .env file
    correct_password = os.getenv("APP_PASSWORD")
    
    # If no password is set in .env, don't require password
    if not correct_password:
        return True
    
    # Check if authentication state already exists
    if "password_correct" in st.session_state:
        return st.session_state.password_correct
    
    # Show input for password
    password = st.text_input("Enter password", type="password")
    
    # Check if password is correct
    if password:
        if password == correct_password:
            st.session_state.password_correct = True
            return True
        else:
            st.error("Incorrect password. Please try again.")
            return False
    else:
        return False

# Main app
def main():
    # Check password first
    if not check_password():
        st.stop()  # Stop execution if password is incorrect
    
    st.title("Athletics Results Exporter 🏃‍♂️🏃‍♀️")
    
    st.markdown("""
    This app allows you to export athletics results data filtered by date range, gender, and events.
    Select your desired filters and export format to download the data.
    """)
    
    # Add a connection test button
    # if st.sidebar.button("Test Database Connection"):
    #     with st.sidebar:
    #         st.info("Testing direct database connection...")
    #         if test_direct_connection():
    #             st.success("Direct connection successful!")
    #         else:
    #             st.error("Direct connection failed. See error details above.")
    
    # Get date range from database
    min_date, max_date = get_date_range()
    
    if min_date is None or max_date is None:
        st.error("Could not retrieve date range from database. Please check your connection.")
        
        # Offer to test the connection
        if st.button("Test Connection Directly"):
            test_direct_connection()
        
        return
    
    # Convert to datetime if they're strings
    if isinstance(min_date, str):
        min_date = datetime.datetime.strptime(min_date, "%Y-%m-%d")
    if isinstance(max_date, str):
        max_date = datetime.datetime.strptime(max_date, "%Y-%m-%d")
    
    # Display date range information
    st.info(f"Available data ranges from **{min_date.strftime('%Y-%m-%d')}** to **{max_date.strftime('%Y-%m-%d')}**")
    
    # Table name
    table_name = 'Tilastopija_results'
    
    # Get unique events for filtering
    with st.spinner("Loading available events..."):
        events = get_unique_events(table_name)
    
    # Create sidebar for filters
    st.sidebar.header("Data Filters")
    
    # Date range selector
    st.sidebar.subheader("Date Range")
    start_date = st.sidebar.date_input(
        "Start Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=max_date,
        min_value=start_date,
        max_value=max_date
    )
    
    # Gender selector
    st.sidebar.subheader("Gender")
    gender_option = st.sidebar.radio(
        "Select Gender",
        options=["Men", "Women", "Both"]
    )
    
    # Event selector
    st.sidebar.subheader("Events")
    event_selection_type = st.sidebar.radio(
        "Event Selection",
        options=["All Events", "Select Specific Events"]
    )
    
    selected_events = None
    if event_selection_type == "Select Specific Events":
        selected_events = st.sidebar.multiselect(
            "Select Events",
            options=events,
            default=None,
            help="Select one or more events to filter the data"
        )
        
        if not selected_events:
            st.sidebar.warning("No events selected. All events will be included.")
    
    # Export format selector
    st.sidebar.subheader("Export Format")
    export_format = st.sidebar.selectbox(
        "Select Format",
        options=["CSV", "Excel", "Parquet", "ZIP (CSV)"]
    )
    
    # Table name
    table_name = 'Tilastopija_results'
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Data Preview", "Export Options"])
    
    with tab1:
        if st.button("Preview Data"):
            with st.spinner("Loading data preview..."):
                # Get events to filter by
                events_to_filter = selected_events if event_selection_type == "Select Specific Events" else None
                
                # Preview limit - only fetch 10 rows for preview
                preview_limit = 10
                
                if gender_option == "Both":
                    # Only fetch 10 most recent rows for preview
                    men_data = query_data(table_name, "Men", start_date, end_date, events=events_to_filter, limit=preview_limit, sort_by_date_desc=True)
                    women_data = query_data(table_name, "Women", start_date, end_date, events=events_to_filter, limit=preview_limit, sort_by_date_desc=True)
                    
                    # Get total count for display (without fetching all data)
                    with get_database_connection().connect() as conn:
                        # Build a more efficient count query that counts both genders at once
                        count_query_base = f"""
                            SELECT Gender, COUNT(*) as count
                            FROM {table_name}
                            WHERE Gender IN ('Men', 'Women')
                            AND Start_Date BETWEEN :start_date AND :end_date
                        """
                        
                        count_params = {"start_date": start_date, "end_date": end_date}
                        
                        # Add event filter if specified
                        if events_to_filter and len(events_to_filter) > 0:
                            event_placeholders = ", ".join([f":event_{i}" for i in range(len(events_to_filter))])
                            count_query_base += f" AND Event IN ({event_placeholders})"
                            # Add event parameters
                            for i, event in enumerate(events_to_filter):
                                count_params[f"event_{i}"] = event
                        
                        # Group by gender to get separate counts
                        count_query_base += " GROUP BY Gender"
                        
                        # Execute the optimized count query
                        result = conn.execute(text(count_query_base), count_params)
                        
                        # Initialize counts
                        men_count = 0
                        women_count = 0
                        
                        # Process results
                        for row in result:
                            if row[0] == 'Men':
                                men_count = row[1]
                            elif row[0] == 'Women':
                                women_count = row[1]
                    
                    st.subheader(f"Men's Data Preview (10 most recent of {men_count:,} total rows)")
                    if not men_data.empty:
                        st.dataframe(men_data, use_container_width=True)
                    else:
                        st.info("No men's data found for the selected criteria.")
                    
                    st.subheader(f"Women's Data Preview (10 most recent of {women_count:,} total rows)")
                    if not women_data.empty:
                        st.dataframe(women_data, use_container_width=True)
                    else:
                        st.info("No women's data found for the selected criteria.")
                else:
                    # Only fetch 10 most recent rows for preview
                    data = query_data(table_name, gender_option, start_date, end_date, events=events_to_filter, limit=preview_limit, sort_by_date_desc=True)
                    
                    # Get total count for display (without fetching all data)
                    with get_database_connection().connect() as conn:
                        # Build an optimized count query
                        count_query_str = f"""
                            SELECT COUNT(*)
                            FROM {table_name}
                            WHERE Gender = :gender
                            AND Start_Date BETWEEN :start_date AND :end_date
                        """
                        count_params = {"gender": gender_option, "start_date": start_date, "end_date": end_date}
                        
                        # Add event filter if specified
                        if events_to_filter and len(events_to_filter) > 0:
                            event_placeholders = ", ".join([f":event_{i}" for i in range(len(events_to_filter))])
                            count_query_str += f" AND Event IN ({event_placeholders})"
                            # Add event parameters
                            for i, event in enumerate(events_to_filter):
                                count_params[f"event_{i}"] = event
                        
                        # Execute count query
                        total_count = conn.execute(text(count_query_str), count_params).scalar() or 0
                    
                    st.subheader(f"{gender_option}'s Data Preview (10 most recent of {total_count:,} total rows)")
                    if not data.empty:
                        st.dataframe(data, use_container_width=True)
                    else:
                        st.info(f"No {gender_option.lower()}'s data found for the selected criteria.")
    
    with tab2:
        st.subheader("Export Data")
        
        if st.button("Prepare Export"):
            with st.spinner("Preparing data for export..."):
                # Get events to filter by
                events_to_filter = selected_events if event_selection_type == "Select Specific Events" else None
                
                # Display selected filters
                st.subheader("Selected Filters")
                st.write(f"**Date Range:** {start_date} to {end_date}")
                st.write(f"**Gender:** {gender_option}")
                if event_selection_type == "Select Specific Events" and selected_events:
                    st.write(f"**Events:** {', '.join(selected_events)}")
                else:
                    st.write("**Events:** All events")
                st.write(f"**Export Format:** {export_format}")
                st.markdown("---")
                
                # Query data based on gender selection - use optimized loading
                if gender_option == "Both":
                    # Show a message about optimized loading
                    st.info("Using optimized data loading for export. This may take some time for large datasets.")
                    
                    # Create placeholders for progress
                    men_progress = st.empty()
                    women_progress = st.empty()
                    
                    # Load men's data with optimized settings
                    men_progress.text("Loading men's data...")
                    men_data = query_data(table_name, "Men", start_date, end_date, events=events_to_filter)
                    
                    # Load women's data with optimized settings
                    women_progress.text("Loading women's data...")
                    women_data = query_data(table_name, "Women", start_date, end_date, events=events_to_filter)
                    
                    # Clear progress messages
                    men_progress.empty()
                    women_progress.empty()
                    
                    # Create download buttons for men's data
                    if not men_data.empty:
                        men_buffer, men_mime, men_ext = get_downloadable_file(men_data, export_format)
                        men_filename = f"tilastopija_men_results_{start_date}_{end_date}.{men_ext}"
                        
                        st.download_button(
                            label=f"Download Men's Data ({len(men_data)} rows)",
                            data=men_buffer,
                            file_name=men_filename,
                            mime=men_mime
                        )
                    else:
                        st.info("No men's data available for the selected criteria.")
                    
                    # Create download buttons for women's data
                    if not women_data.empty:
                        women_buffer, women_mime, women_ext = get_downloadable_file(women_data, export_format)
                        women_filename = f"tilastopija_women_results_{start_date}_{end_date}.{women_ext}"
                        
                        st.download_button(
                            label=f"Download Women's Data ({len(women_data)} rows)",
                            data=women_buffer,
                            file_name=women_filename,
                            mime=women_mime
                        )
                    else:
                        st.info("No women's data available for the selected criteria.")
                else:
                    # Single gender export with optimized loading
                    progress = st.empty()
                    progress.text(f"Loading {gender_option}'s data...")
                    
                    # Use optimized query
                    data = query_data(table_name, gender_option, start_date, end_date, events=events_to_filter)
                    
                    # Clear progress message
                    progress.empty()
                    
                    if not data.empty:
                        buffer, mime, ext = get_downloadable_file(data, export_format)
                        filename = f"tilastopija_{gender_option.lower()}_results_{start_date}_{end_date}.{ext}"
                        
                        st.download_button(
                            label=f"Download {gender_option}'s Data ({len(data)} rows)",
                            data=buffer,
                            file_name=filename,
                            mime=mime
                        )
                    else:
                        st.info(f"No {gender_option.lower()}'s data available for the selected date range.")

if __name__ == "__main__":
    main()