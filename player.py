import cv2

# Corrected GStreamer pipeline
gst_pipeline = (
    "udpsrc port=5000 caps=\"application/x-rtp, encoding-name=H264\" ! "
    "rtph264depay ! avdec_h264 ! videoconvert ! appsink"
)

# Open video stream with OpenCV
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("❌ Failed to open GStreamer pipeline")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No frame received")
        continue

    cv2.imshow('UDP Stream', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
