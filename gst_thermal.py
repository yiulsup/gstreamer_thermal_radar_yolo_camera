import cv2
import gi
import numpy as np

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Initialize GStreamer
Gst.init(None)

# Define GStreamer pipeline
pipeline = Gst.parse_launch(
    "appsrc name=mysource ! videoconvert ! x264enc tune=zerolatency bitrate=500 ! rtph264pay ! udpsink host=127.0.0.1 port=5000"
)

# Get the appsrc element
appsrc = pipeline.get_by_name("mysource")

# Set caps for appsrc
caps = Gst.Caps.from_string("video/x-raw,format=BGR,width=640,height=480,framerate=30/1")
appsrc.set_property("caps", caps)

# Start pipeline
pipeline.set_state(Gst.State.PLAYING)

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to GStreamer Buffer
        frame_data = frame.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(frame_data), None)
        buf.fill(0, frame_data)

        # Push to GStreamer
        appsrc.emit("push-buffer", buf)

except KeyboardInterrupt:
    print("Streaming stopped.")

# Release resources
cap.release()
pipeline.set_state(Gst.State.NULL)
