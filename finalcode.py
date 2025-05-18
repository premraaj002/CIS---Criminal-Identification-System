import cv2
import torch
import mysql.connector
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import io
import smtplib
from email.mime.text import MIMEText
import webbrowser
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Hardcoded camera locations (latitude, longitude)
CAMERA_LOCATIONS = {
    0: (11.9400, 79.8083),  # Lawspet Section 1
    1: (11.9415, 79.8080),  # Saram Section 2
}

# Initialize FaceNet model and MTCNN for face detection
model = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(keep_all=True)

# Track detected criminals to avoid repeated alerts
detected_criminals = {}  # Format: {criminal_id: (camera_id, timestamp)}

# Database setup
def get_db_connection():
    try:
        return mysql.connector.connect(
            host='127.0.0.1',  # Replace with your MySQL host
            user='root',       # Replace with your MySQL username
            password='highend@009',   # Replace with your MySQL password
            database='criminal_db'   # Replace with your database name
        )
    except mysql.connector.Error as e:
        logging.error(f"Database connection error: {e}")
        return None

# Create tables if they don't exist
def initialize_db():
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS criminals (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255),
                age INT,
                crime_details TEXT,
                embedding BLOB,
                image BLOB
            )''')
            c.execute('''CREATE TABLE IF NOT EXISTS detection_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                criminal_id INT,
                detection_time DATETIME,
                location VARCHAR(255)
            )''')
            conn.commit()
    except mysql.connector.Error as e:
        logging.error(f"Database error: {e}")

# Function to preprocess the image
def preprocess_image(face):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((160, 160)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(face).unsqueeze(0)

# Function to get face embedding
def get_embedding(face):
    with torch.no_grad():
        return model(preprocess_image(face))

# Function to compare embeddings
def compare_embeddings(embedding1, embedding2):
    return cosine_similarity(embedding1.detach().numpy(), embedding2)

# Function to enroll a new criminal
def enroll_criminal(name, age, crime_details, face_embedding, face_image):
    try:
        face_embedding_blob = face_embedding.detach().numpy().tobytes()
        img_bytes = io.BytesIO()
        face_image.save(img_bytes, format='JPEG')
        img_blob = img_bytes.getvalue()

        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("INSERT INTO criminals (name, age, crime_details, embedding, image) VALUES (%s, %s, %s, %s, %s)",
                      (name, age, crime_details, face_embedding_blob, img_blob))
            conn.commit()
        logging.info(f"Criminal {name} enrolled successfully!")
    except mysql.connector.Error as e:
        logging.error(f"Database error during enrollment: {e}")

# Function to retrieve stored criminal data
def get_stored_criminals():
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT id, name, age, crime_details, embedding FROM criminals")
            return [
                (id, name, age, crime_details, np.frombuffer(embedding_blob, dtype=np.float32).reshape(1, -1))
                for id, name, age, crime_details, embedding_blob in c.fetchall()
            ]
    except mysql.connector.Error as e:
        logging.error(f"Database error during retrieval: {e}")
        return []

# Function to log criminal detection
def log_detection(criminal_id, location):
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("INSERT INTO detection_logs (criminal_id, detection_time, location) VALUES (%s, %s, %s)",
                      (criminal_id, datetime.datetime.now(), location))
            conn.commit()
    except mysql.connector.Error as e:
        logging.error(f"Database error during logging: {e}")

# Function to send email alert
def send_alert(name, age, crime_details, location, criminal_id, camera_id):
    try:
        # Generate Google Maps link
        if camera_id in CAMERA_LOCATIONS:
            latitude, longitude = CAMERA_LOCATIONS[camera_id]
            maps_link = f"https://www.google.com/maps?q={latitude},{longitude}"
        else:
            maps_link = "Location not available"

        # Create email content
        msg = MIMEText(f"""
            Criminal Detected!
            Name: {name}
            Age: {age}
            Crime Details: {crime_details}
            Location: {location}
            Criminal ID: {criminal_id}
            Google Maps Link: {maps_link}
        """)
        msg['Subject'] = "Criminal Detection Alert"
        msg['From'] = "codecrusaders58@gmail.com"  # Replace with your email
        msg['To'] = "lmust599@gmail.com"    # Replace with recipient email

        # Send email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:  # Use Gmail's SMTP server
            server.starttls()  # Enable TLS encryption
            server.login("codecrusaders58@gmail.com", "mgpc vojj jeab kbmj")  # Replace with your email credentials
            server.sendmail("codecrusaders58@gmail.com", "lmust599@gmail.com", msg.as_string())
        logging.info("Email alert sent successfully!")
    except Exception as e:
        logging.error(f"Failed to send email alert: {e}")

# Function to recognize criminals
def recognize_criminal(face_embedding, threshold=0.7):
    for id, name, age, crime_details, stored_embedding in get_stored_criminals():
        similarity = compare_embeddings(face_embedding, stored_embedding)
        if similarity > threshold:
            return id, name, age, crime_details, similarity[0][0]
    return None, "Unknown", 0, "", 0

# Function to open Google Maps with the camera's location
def open_google_maps(camera_id):
    if camera_id in CAMERA_LOCATIONS:
        latitude, longitude = CAMERA_LOCATIONS[camera_id]
        url = f"https://www.google.com/maps?q={latitude},{longitude}"
        webbrowser.open(url)  # Open the URL in the default web browser
    else:
        logging.warning(f"Camera ID {camera_id} not found in locations.")

# Function to check available cameras
def get_available_cameras():
    available_cameras = []
    for i in range(2):  # Check up to 2 cameras
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

# Function to enroll a new criminal interactively
def enroll_new_criminal(camera_id):
    logging.info("Enrolling a new criminal...")
    name = input("Enter the criminal's name: ")
    age = int(input("Enter the criminal's age: "))
    crime_details = input("Enter the crime details: ")

    # Open the camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logging.error("Error: Could not open camera.")
        return

    logging.info("Press 's' to capture the image when ready. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Error reading frame from camera.")
            break

        # Display the live camera feed
        cv2.imshow("Camera Feed - Press 's' to Capture", frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Capture the image
            # Detect faces in the captured frame
            faces = mtcnn.detect(frame)
            if faces[0] is not None:
                for i, (x1, y1, x2, y2) in enumerate(faces[0]):  # Iterate over bounding boxes
                    # Ensure bounding box coordinates are within the frame dimensions
                    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                        logging.warning("Invalid bounding box coordinates. Skipping this face.")
                        continue

                    # Extract the face region
                    face = frame[int(y1):int(y2), int(x1):int(x2)]
                    if face.size == 0:
                        logging.warning("Empty face region detected. Skipping this frame.")
                        continue  # Skip this face and move to the next one

                    # Convert the face region to RGB
                    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

                    # Get face embedding
                    face_embedding = get_embedding(face_pil)

                    # Enroll the criminal
                    enroll_criminal(name, age, crime_details, face_embedding, face_pil)
                    logging.info("Criminal enrolled successfully!")
                    break  # Enroll only the first detected face
                break
            else:
                logging.warning("No face detected. Please try again.")
        elif key == ord('q'):  # Quit
            logging.info("Enrollment canceled.")
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

# Main function for real-time criminal identification
def main():
    initialize_db()

    # Check for available cameras
    available_cameras = get_available_cameras()
    if not available_cameras:
        logging.error("No cameras connected. Exiting...")
        return

    logging.info(f"Connected cameras: {available_cameras}")
    caps = [cv2.VideoCapture(cam_id) for cam_id in available_cameras]

    logging.info("Starting webcams... Press 'q' to quit.")
    logging.info("Press 'e' to enroll a new criminal.")
    logging.info("Press 'd' to start detection.")

    # Wait for user input to choose mode
    while True:
        choice = input("Enter 'e' to enroll or 'd' to detect: ").strip().lower()
        if choice == 'e':  # Enrollment mode
            enroll_new_criminal(available_cameras[0])  # Use the first available camera for enrollment
            break
        elif choice == 'd':  # Detection mode
            break
        else:
            logging.warning("Invalid choice. Please enter 'e' or 'd'.")

    # Main loop for detection
    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                logging.error("Error reading frames from cameras.")
                return
            frames.append(frame)

        for camera_id, frame in zip(available_cameras, frames):
            faces = mtcnn.detect(frame)
            if faces[0] is not None:
                for i, (x1, y1, x2, y2) in enumerate(faces[0]):  # Iterate over bounding boxes
                    # Ensure bounding box coordinates are within the frame dimensions
                    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                        logging.warning("Invalid bounding box coordinates. Skipping this face.")
                        continue

                    # Extract the face region
                    face = frame[int(y1):int(y2), int(x1):int(x2)]
                    if face.size == 0:
                        logging.warning("Empty face region detected. Skipping this frame.")
                        continue  # Skip this face and move to the next one

                    # Convert the face region to RGB
                    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    face_embedding = get_embedding(face_pil)
                    criminal_id, name, age, crime_details, confidence = recognize_criminal(face_embedding)

                    if criminal_id:
                        # Check if the criminal was already detected in this camera
                        if criminal_id not in detected_criminals or detected_criminals[criminal_id][0] != camera_id:
                            logging.info(f"Criminal Detected in Camera {camera_id}: {name} (Confidence: {confidence:.2f})")
                            log_detection(criminal_id, f"Camera {camera_id} Location")  # Log the detection
                            send_alert(name, age, crime_details, f"Camera {camera_id} Location", criminal_id, camera_id)  # Send alert
                            open_google_maps(camera_id)  # Open Google Maps with the camera's location
                            detected_criminals[criminal_id] = (camera_id, datetime.datetime.now())  # Mark as detected

                        # Draw rectangle and display results
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.putText(frame, f"{name} ({confidence:.2f})", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display frames
            cv2.imshow(f'Camera {camera_id}', frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break

    # Release the cameras and close windows
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()