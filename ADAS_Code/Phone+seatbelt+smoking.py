
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load YOLOv3 for Mobile Detection
def toggle_fullscreen():
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

# Set up webcam feed
cap = cv2.VideoCapture(0)
window_name = 'Mobile, Seatbelt, and Smoke Detection'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

phone_detection_start_time = None
mobile_phone_last_detected_time = None

# Load YOLOv8 model for smoke detection
model_smoke = YOLO("C:/Users/hp/PycharmProjects/pythonProject1/.venv/Scripts/smoking_detection_trained2.pt")  # Path to the YOLOv8 model file

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    current_time = time.time()
    img_h, img_w, img_c = frame.shape
    (H, W) = frame.shape[:2]

    # Prepare the frame for both Mobile and Seatbelt Detection
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

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
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if classes[class_ids_mobile[i]] == "cell phone":
            mobile_phone_detected_in_frame = True
            if phone_detection_start_time is None:
                phone_detection_start_time = current_time
            mobile_phone_last_detected_time = current_time

    # Handle mobile phone detection timing
    if mobile_phone_detected_in_frame:
        elapsed_time = current_time - phone_detection_start_time
        if elapsed_time >= 0.1:
            cv2.putText(frame, "Don't use mobile while driving!!!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        if mobile_phone_last_detected_time is not None and (current_time - mobile_phone_last_detected_time) <= 1:
            # Grace period, do not reset
            elapsed_time = current_time - phone_detection_start_time
            if elapsed_time >= 0.1:
                cv2.putText(frame, "Don't use mobile while driving!!!", (50, 50),
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

                if class_id == 0:  # Assuming class_id 0 corresponds to 'seatbelt'
                    beltdetected = True

    if not beltdetected:
        cv2.putText(frame, "WEAR SEAT BELT", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3, cv2.LINE_AA)

    #### Smoke Detection ####
    results_smoke = model_smoke(frame)

    # Process results for smoke detection
    annotated_frame = results_smoke[0].plot()  # Plot detection results on the frame

    smoke_detected = False
    for result in results_smoke[0].boxes:
        if result.cls == 0:  # Assuming class 0 is 'smoking'
            smoke_detected = True

    if smoke_detected:
        cv2.putText(frame, "Don't smoke while driving!!!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Show the frame with all detections
    cv2.imshow(window_name, frame)

    if cv2.waitKey(5) & 0xFF == 27:  # Exit on 'Esc' key
        break

cap.release()
cv2.destroyAllWindows()
