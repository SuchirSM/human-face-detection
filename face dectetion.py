
import cv2
import face_recognition
from ultralytics import YOLO
import numpy as np
from tkinter import filedialog
from tkinter import Tk

# Function to load an image
def load_image():
    Tk().withdraw()  # Hide the root window
    image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    return cv2.imread(image_path)

# Load YOLOv8 model for person detection
model = YOLO('yolov8n.pt')  # Change to 'yolov8s-seg.pt' if needed

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam, 1 for external

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Load the reference image
reference_image = load_image()
if reference_image is None:
    print("Error: Could not load image.")
    exit()

# Convert reference image to RGB (required by face_recognition)
reference_image_rgb = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

# Get the face embeddings of the reference image
reference_face_locations = face_recognition.face_locations(reference_image_rgb)
if len(reference_face_locations) == 0:
    print("Error: No face found in the reference image.")
    exit()

reference_face_encoding = face_recognition.face_encodings(reference_image_rgb, reference_face_locations)[0]

# Create a named window and set it to full screen
cv2.namedWindow('Person Detection', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Person Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Real-time frame processing
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Detect objects in the frame using YOLOv8
    results = model(frame)

    # Loop through the detected persons
    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0:  # Class 0 corresponds to 'person'
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Extract the face region of the detected person
                detected_face = frame[y1:y2, x1:x2]

                # Convert detected face to RGB
                detected_face_rgb = cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB)

                # Detect face locations in the detected person
                face_locations = face_recognition.face_locations(detected_face_rgb)

                # If a face is detected in the person, get the embeddings
                if len(face_locations) > 0:
                    detected_face_encoding = face_recognition.face_encodings(detected_face_rgb, face_locations)[0]

                    # Compare the detected face with the reference image face
                    matches = face_recognition.compare_faces([reference_face_encoding], detected_face_encoding)
                    face_distance = face_recognition.face_distance([reference_face_encoding], detected_face_encoding)[0]

                    # If a match is found
                    if matches[0]:
                        # Draw a bounding box around the matched person
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box for matches
                        label = f'Match ({face_distance:.2f})'
                    else:
                        label = None  # Do not display label if it's not a match

                    # Add label on the bounding box if it's a match
                    if label:
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame with detection
    cv2.imshow('Person Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
