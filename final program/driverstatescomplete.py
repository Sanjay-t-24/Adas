import cv2
import mediapipe as mp
import numpy as np
import time
from scipy.spatial import distance as dist
from ultralytics import YOLO
import pygame
from datetime import datetime

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


def play_beep(frequency=440, duration=500):
    pygame.mixer.init(frequency=22050)  # Initialize the mixer
    pygame.mixer.music.load(generate_beep(frequency, duration))  # Load the beep sound
    pygame.mixer.music.play()  # Play the sound


def generate_beep(frequency, duration):
    import numpy as np
    import tempfile
    import wave

    fs = 44100  # Sampling rate
    samples = (np.sin(2 * np.pi * np.arange(fs * duration / 1000) * frequency / fs)).astype(np.float32)
    file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')

    with wave.open(file.name, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(fs)
        wav_file.writeframes((samples * 32767).astype(np.int16).tobytes())

    return file.name


def toggle_fullscreen():
    # Toggle fullscreen mode
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    else:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

weights_path_mobile = "C:/Users/hp/PycharmProjects/pythonProject1/.venv/Scripts/yolov7-tiny.weights"
config_path_mobile = "C:/Users/hp/PycharmProjects/pythonProject1/.venv/Scripts/yolov7.cfg"
coco_labels = "C:/Users/hp/PycharmProjects/pythonProject1/.venv/Scripts/coco.names"
net_mobile = cv2.dnn.readNetFromDarknet(config_path_mobile, weights_path_mobile)
layer_names_mobile = net_mobile.getLayerNames()
output_layers_mobile = [layer_names_mobile[i - 1] for i in net_mobile.getUnconnectedOutLayers()]

with open(coco_labels, 'r') as f:
    classes = f.read().strip().split('\n')

objects_inside_car = ["cell phone"]

# Load YOLO for Seatbelt Detection
net_seatbelt = cv2.dnn.readNet("C:/Users/hp/PycharmProjects/pythonProject1/.venv/Scripts/YOLOFI2.weights","C:/Users/hp/PycharmProjects/pythonProject1/.venv/Scripts/YOLOFI.cfg")
seatbelt_classes = []
with open("C:/Users/hp/PycharmProjects/pythonProject1/.venv/Scripts/obj.names", "r") as f:
    seatbelt_classes = [line.strip() for line in f.readlines()]

layer_names_seatbelt = net_seatbelt.getLayerNames()
output_layers_indices_seatbelt = net_seatbelt.getUnconnectedOutLayers()
output_layers_seatbelt = [layer_names_seatbelt[i - 1] for i in output_layers_indices_seatbelt.flatten()]


# Create a named window
window_name = 'Driver State Analysis'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Start in fullscreen mode
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cap = cv2.VideoCapture(0)

# Load YOLOv8 model for smoke detection
model_smoke = YOLO("C:/Users/hp/PycharmProjects/pythonProject1/.venv/Scripts/smoking_detection_trained2.pt")  # Path to the YOLOv8 model file


# Blink and Yawn Counters
blink_count = 0
yawn_count = 0
frequent_blink_alert_count = 0
frequent_yawn_alert_count = 0
head_position_change_alert_count = 0

# Constants for blink and yawn detection
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 2
MOUTH_AR_THRESH = 0.7
MOUTH_OPEN_CONSEC_FRAMES = 5

# Initialize counters for consecutive frames
COUNTER = 0
yawn_counter = 0

# Mouth open state flag
mouth_open = False

# Cooldown period for yawns in seconds
yawn_cooldown_period = 5
last_yawn_time = 0  # Tracks the last time a yawn was detected

# Timer for blink count reset and alert
start_time = time.time()
reset_interval = 30  # Reset interval in seconds (30 seconds)
blink_last_reset_time = start_time
frequent_blink_alert_reset_time = start_time
# frequent_yawn_alert_reset_time = start_time
head_position_change_alert_reset_time = start_time
# New variables for frequent blinking alert
alert_duration = 5  # Duration for alert message in seconds
frequent_blink_alert_start_time = None  # Track the start time of the frequent blinking alert
frequent_blink_alert_shown = False  # Flag to control frequent blinking alert message display

# Variables for eyes closed alert
eyes_closed_start_time = None  # Track when the eyes were first closed
eyes_closed_alert_shown = False  # Flag to control the eyes closed alert message display
eyes_closed_alert_duration = 5  # Duration to show the eyes closed alert in seconds

# Timer for yawn count reset and alert
yawn_start_time = time.time()  # Timer to track when to reset the yawn count
frequent_yawn_alert_start_time = None  # Track the start time of the frequent yawning alert
frequent_yawn_alert_shown = False  # Flag to control frequent yawning alert message display
last_reset_time = None

# Variables for "Look at the road" alert
look_at_road_start_time = None  # Track the time when the person last looked forward
look_at_road_alert_shown = False  # Flag to control the "Look at the road" alert message display
look_at_road_alert_duration = 5  # Duration to show the "Look at the road" alert in seconds

previous_position = "Forward"
position_change_count = 0

# Timer for head position change reset and alert
head_position_change_start_time = time.time()  # Timer to track when to reset the position change count
head_position_change_last_reset_time = None  # Track the last reset time
head_position_change_alert_start_time = None  # Track the start time of the position change alert
head_position_change_alert_shown = False  # Flag to control position change alert message display
head_position_change_alert_duration = 5  # Duration to show the position change alert in seconds

#current_state = "Awake"
phone_detection_start_time = None
mobile_phone_last_detected_time = None

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[0], mouth[1])  # Upper lip to lower lip
    C = dist.euclidean(mouth[2], mouth[3])  # Left corner to right corner of the mouth
    mar = A / C
    return mar


while cap.isOpened():
    success, image = cap.read()
    current_time = time.time()
    start = time.time()

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    (H, W) = image.shape[:2]

    # Prepare the frame for both Mobile and Seatbelt Detection
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    #### Mobile Phone Detection ####
    net_mobile.setInput(blob)
    layer_outputs_mobile = net_mobile.forward(output_layers_mobile)

    boxes_mobile = []
    confidences_mobile = []
    class_ids_mobile = []

    mobile_phone_detected_in_frame = False

    for output in layer_outputs_mobile:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] in objects_inside_car:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes_mobile.append([x, y, int(width), int(height)])
                confidences_mobile.append(float(confidence))
                class_ids_mobile.append(class_id)

    idxs_mobile = cv2.dnn.NMSBoxes(boxes_mobile, confidences_mobile, score_threshold=0.5, nms_threshold=0.3)

    for i in idxs_mobile.flatten() if len(idxs_mobile) > 0 else []:
        (x, y) = (boxes_mobile[i][0], boxes_mobile[i][1])
        (w, h) = (boxes_mobile[i][2], boxes_mobile[i][3])
        color = [int(c) for c in np.random.randint(0, 255, size=(3,))]
        text = f"{classes[class_ids_mobile[i]]}: {confidences_mobile[i] * 100:.2f}%"
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if classes[class_ids_mobile[i]] == "cell phone":
            mobile_phone_detected_in_frame = True
            if phone_detection_start_time is None:
                phone_detection_start_time = current_time
            mobile_phone_last_detected_time = current_time

    # Handle mobile phone detection timing
    if mobile_phone_detected_in_frame:
        elapsed_time = current_time - phone_detection_start_time
        if elapsed_time >= 0.1:
            cv2.putText(image, "Don't use mobile while driving!!!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        if mobile_phone_last_detected_time is not None and (current_time - mobile_phone_last_detected_time) <= 1:
            # Grace period, do not reset
            elapsed_time = current_time - phone_detection_start_time
            if elapsed_time >= 0.1:
                cv2.putText(image, "Don't use mobile while driving!!!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            # Reset detection time
            phone_detection_start_time = None
            mobile_phone_last_detected_time = None

    #### Seatbelt Detection ####
    net_seatbelt.setInput(blob)
    outs_seatbelt = net_seatbelt.forward(output_layers_seatbelt)
    beltdetected = False

    for out in outs_seatbelt:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.2:
                center_x = int(detection[0] * W)
                center_y = int(detection[1] * H)
                w = int(detection[2] * W)
                h = int(detection[3] * H)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if class_id == 0:  # Assuming class_id 0 corresponds to 'seatbelt'
                    beltdetected = True

    if not beltdetected:
        cv2.putText(image, "WEAR SEAT BELT", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3, cv2.LINE_AA)
        #play_beep(2000,500)

        #### Smoke Detection ####
        results_smoke = model_smoke(image)

        # Process results for smoke detection
        annotated_frame = results_smoke[0].plot()  # Plot detection results on the frame

        smoke_detected = False
        for result in results_smoke[0].boxes:
            if result.cls == 0:  # Assuming class 0 is 'smoking'
                smoke_detected = True

        if smoke_detected:
            cv2.putText(image, "Don't smoke while driving!!!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Correct indices for eyes
            left_eye = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]
            right_eye = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]

            left_eye = np.array([(lm.x * img_w, lm.y * img_h) for lm in left_eye], dtype=np.float64)
            right_eye = np.array([(lm.x * img_w, lm.y * img_h) for lm in right_eye], dtype=np.float64)

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Correct indices for mouth landmarks
            mouth = [landmarks[i] for i in [13, 14, 78, 308]]
            mouth = np.array([(lm.x * img_w, lm.y * img_h) for lm in mouth], dtype=np.float64)

            mar = mouth_aspect_ratio(mouth)

            # Head pose estimation
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # Determine head pose and use corresponding eye aspect ratio threshold
            if y < -17:
                current_position = "Looking Left"
            elif y > 17:
                current_position = "Looking Right"
            elif x < -15:
                current_position = "Looking Down"
            elif x > 19:
                current_position = "Looking Up"
            else:
                current_position = "Forward"

                # Update look_at_road_start_time when looking forward
                look_at_road_start_time = current_time  # Reset the timer when looking forward
                look_at_road_alert_shown = False  # Reset the alert flag

            # Blink detection logic using the current threshold
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if eyes_closed_start_time is None:
                    eyes_closed_start_time = time.time()
                elif time.time() - eyes_closed_start_time > 5:  # Eyes closed for more than 5 seconds
                    if not eyes_closed_alert_shown:
                        eyes_closed_alert_shown = True
                        eyes_closed_alert_start_time = time.time()

            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    blink_count += 1
                COUNTER = 0
                eyes_closed_start_time = None  # Reset the eyes closed start time

            # Yawn detection logic with the new reset mechanism
            current_time = time.time()
            if mar > MOUTH_AR_THRESH:
                yawn_counter += 1
                if yawn_counter >= MOUTH_OPEN_CONSEC_FRAMES and not mouth_open:
                    if current_time - last_yawn_time > yawn_cooldown_period:
                        yawn_count += 1
                        mouth_open = True
                        last_yawn_time = current_time

            else:
                mouth_open = False
                yawn_counter = 0

            # Yawn count reset logic with a delay
            if yawn_count >= 5:
                if last_reset_time is None:
                    last_reset_time = current_time  # Track the time when count reaches 5
                elif current_time - last_reset_time >= 5:  # Wait for 2 seconds if last_reset_time is set
                    yawn_count = 0  # Reset the count
                    yawn_start_time = current_time  # Start a new 1-minute interval
                    last_reset_time = None  # Reset the last_reset_time for future use
            elif current_time - yawn_start_time >= 60:
                yawn_count = 0
                yawn_start_time = current_time

                # Check for position change (ignore changes to 'Forward')
            if current_position != "Forward" and current_position != previous_position:
                position_change_count += 1
                previous_position = current_position  # Update the previous position

            # Head position change count reset logic with a delay
            if position_change_count >= 7:
                if head_position_change_last_reset_time is None:
                    head_position_change_last_reset_time = current_time  # Track the time when count reaches 7
                elif current_time - head_position_change_last_reset_time >= 5:  # Wait for 5 seconds if last_reset_time is set
                    position_change_count = 0  # Reset the count
                    head_position_change_start_time = current_time  # Start a new 1-minute interval
                    head_position_change_last_reset_time = None  # Reset the last_reset_time for future use
            elif current_time - head_position_change_start_time >= 60:
                position_change_count = 0
                head_position_change_start_time = current_time

            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            #cv2.line(image, p1, p2, (255, 0, 0), 3)

            # Check if the blink count exceeds 12 or 24
            # Check if the blink count exceeds 12
            if blink_count >= 13:
                blink_count = 0  # Reset immediately
                blink_last_reset_time = time.time()  # Start a new 30-second interval
            elif time.time() - blink_last_reset_time >= reset_interval:
                blink_count = 0
                blink_last_reset_time = time.time()  # Start a new 30-second interval

            if blink_count in [12, 24]:
                if not frequent_blink_alert_shown:
                    frequent_blink_alert_start_time = time.time()
                    frequent_blink_alert_count += 1  # Start the frequent blinking alert
                    frequent_blink_alert_shown = True

            # Display frequent blinking alert message if needed
            if frequent_blink_alert_shown:
                if time.time() - frequent_blink_alert_start_time <= alert_duration:  # Show the alert for the specified duration
                    cv2.putText(image, "Frequent Blinking Alert", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                                2)
                    cv2.putText(image, "DROWSY", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
                else:
                    frequent_blink_alert_shown = False  # Reset the flag to stop showing the alert

            # Display eyes closed for too long alert message if needed
            if eyes_closed_start_time is not None:
                elapsed_time = time.time() - eyes_closed_start_time
                if elapsed_time > eyes_closed_alert_duration:
                    eyes_closed_alert_shown = True
            else:
                eyes_closed_alert_shown = False
                # current_state="Awake"

            # Display eye closure alert message if needed
            if eyes_closed_alert_shown:
                cv2.putText(image, "WAKE UP! WAKE UP!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                #current_state = "Drowsy"
                cv2.putText(image, "DROWSY", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
                play_beep(2000, 200)
            else:
                eyes_closed_alert_shown = False
            if yawn_count >= 5:
                if not frequent_yawn_alert_shown:
                    frequent_yawn_alert_start_time = time.time()
                    # frequent_yawn_alert_count += 1 # Start the frequent yawning alert
                    frequent_yawn_alert_shown = True
                # Display frequent yawning alert message if needed
                if frequent_yawn_alert_shown:
                    if time.time() - frequent_yawn_alert_start_time <= alert_duration:  # Show the alert for the specified duration
                        cv2.putText(image, "Frequent Yawning Detected!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2)
                        cv2.putText(image, "FATIGUE", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
                    else:
                        frequent_yawn_alert_shown = False
                        # Stop showing the alert message
            else:
                # Reset the alert state if yawn count goes below 5
                frequent_yawn_alert_shown = False

            if time.time() - start_time >= reset_interval:
                blink_count = 0
                start_time = time.time()
            # Check if the person has not looked forward for more than 15 seconds
            if look_at_road_start_time is not None and current_time - look_at_road_start_time > 3:
                if not look_at_road_alert_shown:
                    look_at_road_alert_shown = True
                    look_at_road_alert_start_time = current_time

            # Display the "Look at the road" alert message if needed
            if look_at_road_alert_shown:
                if time.time() - look_at_road_alert_start_time <= look_at_road_alert_duration:
                    cv2.putText(image, "Look at the Road!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(image, "VISUAL COGNITIVE", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
                    play_beep(2000, 100)
                else:
                    look_at_road_alert_shown = False
                    current_state = "Awake"  # Stop showing the alert message
            # Check if the head position change count reaches 7
            if position_change_count >= 7:
                if not head_position_change_alert_shown:
                    head_position_change_alert_start_time = time.time()  # Start the head position change alert
                    head_position_change_alert_count += 1

                    head_position_change_alert_shown = True

            # Display head position change alert message if needed
            if head_position_change_alert_shown:
                if time.time() - head_position_change_alert_start_time <= head_position_change_alert_duration:  # Show the alert for the specified duration
                    cv2.putText(image, "Distraction Detected!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(image, "VISUAL COGNITIVE", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
                else:
                    head_position_change_alert_shown = False  # Stop showing the alert message

            # Reset the frequent blinking alert count every minute
            if current_time - frequent_blink_alert_reset_time >= 60 or eyes_closed_alert_shown:
                frequent_blink_alert_count = 0  # Reset the frequent blink alert count
                frequent_blink_alert_reset_time = time.time()

            # if frequent_blink_alert_count >= 2:
            #    current_state = "Drowsy"
            # Update the reset time

            # if head_position_change_alert_count>= 2:
            #    current_state = "Visual Cognitive"

            # if current_time - frequent_yawn_alert_reset_time >= 120:
            #    frequent_yawn_alert_count = 0  # Reset the frequent blink alert count
            #    frequent_yawn_alert_reset_time = time.time()  # Update the reset time

            # Reset the frequent blinking alert count every minute
            if current_time - head_position_change_alert_reset_time >= 60 or head_position_change_alert_shown:
                head_position_change_alert_count = 0  # Reset the frequent blink alert count
                head_position_change_alert_reset_time = time.time()  # Update the reset time
            #cv2.putText(image, f"Blink Count: {blink_count}", (30, 150),
            #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 100), 2)
            #cv2.putText(image, f"Yawn Count: {yawn_count}", (30, 180),
            # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 250, 0), 2)
            # cv2.putText(image, f"Head Pose: {current_position}", (30, 210),
            # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 105, 0), 2)
            # cv2.putText(image, f"Head Position Change Count: {position_change_count}", (30, 240),
            # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
            # cv2.putText(image, f"Frequent Blink Alerts: {frequent_blink_alert_count}", (30, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #cv2.putText(image, f"Frequent Yawn Alerts: {frequent_yawn_alert_count}", (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.putText(image, f"Head Position Change Alerts: {head_position_change_alert_count}", (30, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.putText(image, f'State: {current_state}', (10, img_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    end = time.time()

    totalTime = end - start

    if totalTime > 0:
        fps = 1 / totalTime
    else:
        fps = 0  # Set fps to 0 or handle this case as needed
    # cv2.putText(image, f'FPS: {int(fps)}', (500, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Draw the face mesh landmarks (all 468 points)
    #mp_drawing.draw_landmarks(
    #image=image,
    #landmark_list=face_landmarks,
    #connections=mp_face_mesh.FACEMESH_CONTOURS,
    #landmark_drawing_spec=drawing_spec,
    #connection_drawing_spec=drawing_spec
    #)

    #cv2.putText(image, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    cv2.putText(image, f"{current_datetime}", (10, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow(window_name, image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()