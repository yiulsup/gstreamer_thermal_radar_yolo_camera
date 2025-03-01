import cv2

gst_pipeline = "udpsrc port=5600 ! jpegparse ! jpegdec ! videoconvert ! autovideosink"


print("🔹 Trying to open GStreamer pipeline:", gst_pipeline)
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("❌ Error: Cannot open video stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Warning: Failed to receive frame")
        break

    cv2.imshow("MJPEG Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
