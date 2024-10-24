import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO(r"C:\Users\suresh\OneDrive\Documents\smoking_detection_trained2.pt")  # Path to the YOLOv8 model file

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)
    
    # Process results
    annotated_frame = results[0].plot()  # Plot detection results on the frame
    
    # Display results
    cv2.imshow('YOLOv8 Object Detection', annotated_frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
