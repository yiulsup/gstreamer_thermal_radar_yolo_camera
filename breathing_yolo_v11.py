import cv2
import serial
import threading
from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO("yolov8n.pt")  # Change if using a custom model

# Open webcam (or video file)
cap = cv2.VideoCapture("/dev/video2")  # Change based on your video source

# Initialize serial communication
serial_port = "/dev/ttyUSB0"
baud_rate = 115200
breathing_rate = "N/A"  # Default value before receiving data

def read_serial():
    """Reads serial data continuously and updates breathing_rate"""
    global breathing_rate
    try:
        device = serial.Serial(serial_port, baud_rate, timeout=1)
        while True:
            raw = device.readline().decode("utf-8").strip()  # Read and decode
            if raw:
                breathing_rate = raw  # Update global variable
    except serial.SerialException as e:
        print(f"Serial Error: {e}")

# Start serial reading in a separate thread
serial_thread = threading.Thread(target=read_serial, daemon=True)
serial_thread.start()

while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        print("Failed to grab frame")
        break

    # Run YOLO inference on the frame
    results = model.predict(frame, conf=0.5)  # Confidence threshold

    # Visualize the results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class ID

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{model.names[cls]}: {conf:.2f} | Breathing: {breathing_rate}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("YOLO Detection with Breathing Rate", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
