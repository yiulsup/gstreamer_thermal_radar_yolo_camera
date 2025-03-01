import os
import serial
import cv2
import sys
import time
import numpy as np
import threading
import binascii
import queue
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO("yolov8n.pt")  # Use the correct model

# Open webcam for object detection
cap = cv2.VideoCapture("/dev/video0")  # Change based on your setup

# Initialize serial connections
serial_port_1 = "/dev/ttyUSB0"  # For breathing rate
serial_port_2 = "/dev/ttyACM1"  # For thermal camera
baud_rate = 115200
uart = serial.Serial(serial_port_2, baud_rate, timeout=1)

# Shared variables
breathing_rate = "N/A"  # Default breathing rate
main_queue = queue.Queue()
thermal_image = None  # Latest thermal image
thermal_lock = threading.Lock()
distance = ""

# ----------- Function: Read Serial Data (Breathing Rate) -----------
def read_serial():
    """Reads breathing rate data from /dev/ttyUSB0"""
    global breathing_rate
    try:
        device = serial.Serial(serial_port_1, baud_rate, timeout=1)
        while True:
            raw = device.readline().decode("utf-8").strip()
            if "Distance" in raw:
                distance = raw
            if raw:
                breathing_rate = raw  # Update global variable
    except serial.SerialException as e:
        print(f"Serial Error (Breathing Rate): {e}")

# ----------- Function: Thermal Camera Data Handling -----------
def thermal_camera():
    global main_queue
    sending_1 = [0x02, 0x00, 0x04, 0x00, 0x01, 0x55, 0xaa, 0x03, 0xFA]
    sending_2 = [0x02, 0x00, 0x04, 0x01, 0x01, 0x00, 0x5, 0x03, 0x01]
    # sending_2 = [0x02, 0x00, 0x04, 0x01, 0x01, 0x00, 0x0a, 0x03, 0x0E]
    sending_3 = [0x02, 0x00, 0x04, 0x02, 0x01, 0x00, 0x01, 0x03, 0x06]
    sending_4 = [0x02, 0x00, 0x04, 0x02, 0x01, 0x01, 0x01, 0x03, 0x07]

    cnt = 0
    cnt1 = 0
    cnt2 = 0

    frame = np.zeros(4800)

    time.sleep(0.1)
    print("second command to fly")
    uart.write(sending_2)
    time.sleep(0.1)
    first = 1
    image_cnt = 0
    passFlag = np.zeros(6)
    start_frame = 0
    uart.write(sending_4)
    begin = 0
    check_cnt = 0

    uart.write(sending_1)
    while True:
        line = uart.read()
        cnt = cnt + 1
        if cnt >= 9:
            cnt = 0
            break
    uart.write(sending_4)

    while True:
        try:
            line = uart.read()
            cnt1 = cnt1 + 1
            if begin == 0 and cnt1 == 1:
                rawDataHex = binascii.hexlify(line)
                rawDataDecimal = int(rawDataHex, 16)
                if rawDataDecimal == 2:
                    begin = 1
                else:
                    begin = 0
                    cnt1 = 0
                    continue
            if begin == 1 and cnt1 == 20:
                for i in range(0, 9600):
                    line = uart.read()
                    cnt1 = cnt1 + 1
                    rawDataHex = binascii.hexlify(line)
                    rawDataDecimal = int(rawDataHex, 16)
                    if first == 1:
                        dec_10 = rawDataDecimal * 256
                        first = 2
                    elif first == 2:
                        first = 1
                        dec = rawDataDecimal
                        frame[image_cnt] = dec + dec_10
                        image_cnt = image_cnt + 1

                    if image_cnt >= 4800:
                        image_cnt = 0
                        error = np.mean(frame)
                        if error > 7 and error < 8:
                            continue
                        main_queue.put(frame)

            if cnt1 == 2 and begin == 1:
                rawDataHex = binascii.hexlify(line)
                rawDataDecimal = int(rawDataHex, 16)
                if rawDataDecimal == 0x25:
                    begin = 1
                else:
                    begin = 0
                    cnt1 = 0
                    continue
            if cnt1 == 3 and begin == 1:
                rawDataHex = binascii.hexlify(line)
                rawDataDecimal = int(rawDataHex, 16)
                if rawDataDecimal == 0xA1:
                    begin = 1
                else:
                    begin = 0
                    cnt1 = 0
                    continue

            if cnt1 == 9638 and begin == 1:
                begin = 0
                cnt1 = 0
            else:
                continue

        except:
            continue

# ----------- Start Threads for Serial Reading -----------
threading.Thread(target=read_serial, daemon=True).start()  # Breathing rate thread
threading.Thread(target=thermal_camera, daemon=True).start()  # Thermal camera thread

# ----------- YOLO + Thermal Camera Display -----------
while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print("Failed to grab frame")
        break

    # Run YOLO inference

    frame = cv2.resize(frame, (640, 480))
    results = model.predict(frame, conf=0.5)

    # Visualize YOLO results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label_text = f"{model.names[cls]}"

            # If the detected object is a "person", show the breathing rate
            if model.names[cls] == "person":
                label_text += f" | {breathing_rate} | {distance}"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            y1 = y1 + 30  # Display at top
            x1 = x1 + 10
            cv2.putText(frame, label_text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            #cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if not main_queue.empty():
        frame1 = main_queue.get()
        max_val = np.max(frame1)
        min_val = np.min(frame1)

        nfactor = 255 / (max_val - min_val)
        frame1 = (frame1 - min_val) * nfactor
        image = frame1.reshape(60, 80).astype('uint8')

        # Convert to grayscale and resize
        gray_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray_image = cv2.resize(gray_image, (160, 120))  # Small window size
        gray_image = cv2.flip(gray_image, 1)

        gray_image = cv2.resize(gray_image, (160, 120))  # Resize for overlay

        # Store the latest thermal image with thread safety
        with thermal_lock:
            thermal_image = gray_image.copy()


    # Overlay the Thermal Camera in the upper-right corner
    if thermal_image is not None:
        h, w, _ = frame.shape  # Get main frame dimensions
        th, tw, _ = thermal_image.shape  # Get thermal image dimensions
        x_offset = w - tw - 10  # Right margin
        y_offset = 10  # Top margin

        # Blend the thermal image into the main frame
        frame[y_offset:y_offset+th, x_offset:x_offset+tw] = thermal_image

    # Show the main YOLO detection window
    cv2.imshow("YOLO Detection with Thermal Overlay", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
