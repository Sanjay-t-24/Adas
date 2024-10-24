# Advanced Driver Assistance System (ADAS) 🚗🔧

This repository contains the code and documentation for an **Advanced Driver Assistance System (ADAS)** project. The ADAS aims to enhance vehicle safety by detecting and responding to various driver states and environmental conditions using computer vision, sensor data, and machine learning techniques.

## Features 🚘

1. **Driver State Detection:**
   - **Drowsiness Detection**: Tracks eye aspect ratio (EAR) and blink frequency to detect drowsiness.
   - **Yawning Detection**: Measures mouth aspect ratio (MAR) to identify yawns.
   - **Head Pose Detection**: Identifies head tilt to monitor attention or distraction.
   - **Cognitive Distraction Detection**: Alerts when the driver shows signs of distraction, such as using a mobile phone.
   - **Fatigue Detection**: Monitors long-term signs of tiredness based on facial landmarks.

2. **Safety Alerts:**
   - **Seatbelt Detection**: Ensures that the driver is wearing a seatbelt.
   - **Smoking Detection**: Identifies if the driver is smoking while driving.
   - **Mobile Phone Usage Detection**: Alerts when the driver uses a mobile phone while driving.

3. **Lane Departure Warning**:
   - Detects lane boundaries using a camera and issues warnings if the vehicle is leaving the lane without signaling.

4. **Collision Alert**:
   - Identifies objects on the road and issues warnings when obstacles enter the lane.

## Technologies Used 🔧

- **Python**: Primary programming language used for the project.
- **OpenCV**: For image processing and object detection.
- **MediaPipe**: For facial landmark detection with 478-point predictor.
- **TensorFlow**: For custom-trained object detection models.
- **YOLOv4**: Used for object detection and smoking, mobile phone, and seatbelt usage monitoring.
- **Raspberry Pi**: Microcontroller for real-time processing in the vehicle environment.
- **MicroPython**: For low-level control and alerts on the Raspberry Pi.
- **Serial Communication**: For transmitting detected states from the computer to an Arduino.

## Project Structure 📁

```
├── ADAS_Code
│   ├── driver_state_detection.py       # Driver state detection (drowsiness, yawning, etc.)
│   ├── lane_detection.py               # Lane detection algorithm
│   ├── object_detection.py             # Collision alert and object detection
│   ├── seatbelt_smoking_detection.py   # Detects seatbelt usage and smoking
│   ├── utils.py                        # Helper functions and utilities
├── models                              # Pre-trained and custom-trained models (YOLO, TensorFlow)
├── data                                # Sample data and datasets used for training and validation
├── README.md                           # Project documentation (this file)
└── requirements.txt                    # Python dependencies and libraries
```

## Getting Started 🚀

### Prerequisites

Ensure you have the following installed on your system:

- Python 3.7+
- OpenCV
- TensorFlow
- YOLOv4
- MediaPipe
- MicroPython (for Raspberry Pi-based projects)

You can install all the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

### Running the Code

1. **Driver State Detection**:
   - Run the script to start detecting driver state, including drowsiness, yawning, and head pose:
     ```bash
     python ADAS_Code/driver_state_detection.py
     ```

2. **Lane Detection**:
   - Run the lane detection algorithm with object detection:
     ```bash
     python ADAS_Code/lane_detection.py
     ```

3. **Seatbelt and Smoking Detection**:
   - To monitor seatbelt usage and smoking:
     ```bash
     python ADAS_Code/seatbelt_smoking_detection.py
     ```

### Hardware Setup

If you're using a **Raspberry Pi** for real-time alerts, follow these steps:
- Connect the Pi to the camera module and sensors (e.g., buzzer, display).
- Flash MicroPython onto the Pi.
- Upload the corresponding MicroPython scripts to control the alert system.

## Contributing 🛠️

Contributions are welcome! If you find any bugs or want to add more features, feel free to submit an issue or a pull request. Please make sure to follow the code of conduct.

