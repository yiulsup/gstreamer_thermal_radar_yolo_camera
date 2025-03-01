import cv2
from ultralytics import YOLO

# Load a pretrained YOLO model (assuming you meant YOLOv8)
model = YOLO("yolov8n.pt")  # Change this if using a custom model

# Open webcam (or video file)
cap = cv2.VideoCapture("/dev/video2")  # Use 0 for default webcam, or "/dev/video0"

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print("Failed to grab frame")
        break

    # Run YOLO inference on the frame
    results = model.predict(frame, conf=0.5)  # Confidence threshold set to 0.5

    # Visualize the results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class ID

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{model.names[cls]}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("YOLO Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
