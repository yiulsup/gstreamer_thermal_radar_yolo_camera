import gi
import gi.repository
gi.require_version('Gst', '1.0')

from gi.repository import Gst, GLib

Gst.init(None)

pipeline = Gst.parse_launch("v4l2src device=/dev/video0 ! videoconvert ! x264enc ! rtph264pay ! udpsink host=127.0.0.1 port = 5000")

pipeline.set_state(Gst.State.PLAYING)

while True:
    pass

pipeline.set_state(Gst.State.NULL)