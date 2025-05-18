import streamlit as st
import mysql.connector
import pandas as pd
from PIL import Image
import io

# Database setup
def get_db_connection():
    return mysql.connector.connect(
        host='127.0.0.1',  # Replace with your MySQL host
        user='root',       # Replace with your MySQL username
        password='highend@009',   # Replace with your MySQL password
        database='criminal_db'   # Replace with your database name
    )

# Hardcoded camera locations (latitude, longitude)
CAMERA_LOCATIONS = {
    "Camera 0 Location": (11.9400, 79.8083),  # Lawspet Section 1
    "Camera 1 Location": (11.9415, 79.8080),  # Saram Section 2
}

# Function to fetch detection logs
def fetch_detection_logs():
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("""
                SELECT l.id, c.name, l.location, l.detection_time 
                FROM detection_logs l
                JOIN criminals c ON l.criminal_id = c.id
            """)
            logs = c.fetchall()
            return logs
    except mysql.connector.Error as e:
        st.error(f"Database error: {e}")
        return []

# Function to fetch enrolled criminals
def fetch_enrolled_criminals():
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT id, name, age, crime_details, image FROM criminals")
            criminals = []
            for id, name, age, crime_details, image_blob in c.fetchall():
                image = Image.open(io.BytesIO(image_blob))
                criminals.append((id, name, age, crime_details, image))
            return criminals
    except mysql.connector.Error as e:
        st.error(f"Database error: {e}")
        return []

# Function to generate Google Maps link
def generate_google_maps_link(location):
    if location in CAMERA_LOCATIONS:
        latitude, longitude = CAMERA_LOCATIONS[location]
        return f"https://www.google.com/maps?q={latitude},{longitude}"
    return None

# Streamlit App
def main():
    st.title("Criminal Detection Dashboard")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Choose an option", ["Detection Logs", "Enrolled Criminals", "Detection Graph"])

    if choice == "Detection Logs":
        st.header("Detection Logs")
        logs = fetch_detection_logs()
        if logs:
            # Convert logs to a DataFrame
            log_df = pd.DataFrame(logs, columns=["ID", "Name", "Location", "Detection Time"])

            # Add a column with clickable Google Maps links
            log_df["Map Link"] = log_df["Location"].apply(
                lambda loc: f'<a href="{generate_google_maps_link(loc)}" target="_blank">Open Map</a>'
                if generate_google_maps_link(loc) else "Location not found"
            )

            # Display the DataFrame with clickable links
            st.markdown(log_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.info("No detection logs found.")

    elif choice == "Enrolled Criminals":
        st.header("Enrolled Criminals")
        criminals = fetch_enrolled_criminals()
        if criminals:
            for criminal in criminals:
                id, name, age, crime_details, image = criminal
                st.subheader(f"ID: {id} - Name: {name}")
                st.write(f"Age: {age}")
                st.write(f"Crime Details: {crime_details}")
                st.image(image, caption="Criminal Image", use_column_width=True)
                st.write("---")
        else:
            st.info("No criminals enrolled yet.")

    elif choice == "Detection Graph":
        st.header("Detection Graph")
        logs = fetch_detection_logs()
        if logs:
            # Convert logs to a DataFrame
            log_df = pd.DataFrame(logs, columns=["ID", "Name", "Location", "Detection Time"])
            log_df["Detection Time"] = pd.to_datetime(log_df["Detection Time"])

            # Group by date and count detections
            log_df["Date"] = log_df["Detection Time"].dt.date
            detection_counts = log_df.groupby("Date").size().reset_index(name="Detections")

            # Add a title and labels
            st.write("Number of Detections Over Time")
            
            # Use a bar chart for better readability
            st.bar_chart(detection_counts.set_index("Date"))

            # Add a table below the graph for detailed data
            st.write("Detailed Detection Data:")
            st.dataframe(detection_counts)

            # Highlight the day with the most detections
            max_detections = detection_counts.loc[detection_counts["Detections"].idxmax()]
            st.success(f"Highest detections on {max_detections['Date']}: {max_detections['Detections']} detections")
        else:
            st.info("No detection logs found to plot.")

if __name__ == "__main__":
    main()