import os
import serial
import cv2
import sys
import time
import numpy as np
import threading
import binascii
import queue
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

from ultralytics import YOLO

# GStreamer 초기화
Gst.init(None)

# YOLO 모델 초기화
model = YOLO("yolov8n.pt")

# GStreamer 파이프라인 생성
gst_pipeline = (
    "appsrc name=mysource format=time is-live=true do-timestamp=true ! "
    "jpegparse ! jpegdec ! videoconvert ! "
    "x264enc tune=zerolatency bitrate=4000 speed-preset=ultrafast ! "
    "rtph264pay config-interval=1 pt=96 ! "
    "udpsink host=239.255.0.1 port=5000"
)



# Create GStreamer pipeline
pipeline = Gst.parse_launch(gst_pipeline)
appsrc = pipeline.get_by_name("mysource")

if not appsrc:
    print("Error: appsrc element not found in pipeline")
    sys.exit(1)

caps = Gst.Caps.from_string("image/jpeg, width=1280, height=720, framerate=30/1")
appsrc.set_property("caps", caps)




# Start the GStreamer pipeline
if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
    print("Error: GStreamer pipeline failed to start")
    sys.exit(1)


# 카메라 초기화 (GStreamer를 지원하는 경우)
#cap = cv2.VideoCapture("v4l2src device=/dev/video0 ! video/x-raw,format=BGR,width=640,height=480 ! appsink")
cap = cv2.VideoCapture(0)

# 시리얼 포트 설정
serial_port_1 = "/dev/ttyUSB0"
serial_port_2 = "/dev/ttyACM0"
baud_rate = 115200
uart = serial.Serial(serial_port_2, baud_rate, timeout=1)

# 공유 변수
breathing_rate = "N/A"
main_queue = queue.Queue()
thermal_image = None
thermal_lock = threading.Lock()
distance = ""
running = True
paused = False  # Added Pause Control

# ---------- GStreamer로 프레임 전송 ----------
def push_frame_to_gstreamer(frame):
    """Send OpenCV frame as JPEG to GStreamer appsrc"""
    global appsrc

    # Encode OpenCV frame as JPEG
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]  # Adjust quality if needed
    _, jpeg_frame = cv2.imencode(".jpg", frame, encode_param)

    # Convert to byte data
    data = jpeg_frame.tobytes()
    buffer = Gst.Buffer.new_allocate(None, len(data), None)
    buffer.fill(0, data)

    buffer.pts = buffer.dts = Gst.util_uint64_scale(
        int(time.time() * Gst.SECOND), Gst.SECOND, 1
    )
    buffer.duration = Gst.util_uint64_scale(1, Gst.SECOND, 30)

    # Push buffer to GStreamer appsrc
    appsrc.emit("push-buffer", buffer)

# ---------- Streaming Control Functions ----------
def pause_stream():
    """Pause the GStreamer pipeline"""
    global pipeline, paused
    if not paused:
        pipeline.set_state(Gst.State.PAUSED)
        paused = True
        print("🔴 Stream Paused")

def resume_stream():
    """Resume the GStreamer pipeline"""
    global pipeline, paused
    if paused:
        pipeline.set_state(Gst.State.PLAYING)
        paused = False
        print("🟢 Stream Resumed")

def stop_stream():
    """Stop and exit the application"""
    global pipeline, running
    pipeline.set_state(Gst.State.NULL)
    running = False
    print("❌ Stream Stopped")

def restart_stream():
    """Restart the stream from the beginning"""
    global pipeline, paused
    pipeline.set_state(Gst.State.NULL)  # Reset pipeline
    pipeline.set_state(Gst.State.PLAYING)  # Restart
    paused = False
    print("🔄 Stream Restarted")


# ---------- 시리얼 데이터 읽기 (호흡률) ----------
b_breathing = ""
def read_serial():
    """시리얼 데이터 읽기 (호흡률)"""
    global breathing_rate
    try:
        device = serial.Serial(serial_port_1, baud_rate, timeout=1)
        while True:
            raw = device.readline().decode("utf-8", errors="ignore").strip()
            if raw == None:
                 breathing_rate = breathing_rate
            else:           
                breathing_rate = raw
            



    except serial.SerialException as e:
        print(f"Serial Error (Breathing Rate): {e}")


# ---------- 열화상 카메라 처리 ----------
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


# ---------- 쓰레드 실행 ----------
threading.Thread(target=read_serial, daemon=True).start()
threading.Thread(target=thermal_camera, daemon=True).start()

# ---------- YOLO + GStreamer 스트리밍 ----------
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    # YOLO 객체 탐지 수행
    frame = cv2.resize(frame, (640, 480))
    results = model.predict(frame, conf=0.3)

    
    # 객체 감지 결과 시각화

    flag = 0
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0].item())

            # "person"을 감지하면 호흡률 정보 추가
            if model.names[cls] == "person":
                label_text = "person"
                try:
                    label_text += f" | {float(breathing_rate):.2f}"
                except ValueError:
                    label_text += f" | {breathing_rate}"

                # 경계 상자 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label_text, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                flag = 1

    if flag == 0:                
        label_text = "person"
        try:
            label_text += f" | {float(breathing_rate):.2f}"
        except ValueError:
            label_text += f" | {breathing_rate}"

        # 경계 상자 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            

    
    # 열화상 카메라 데이터 처리
    if not main_queue.empty():
        frame1 = main_queue.get()
        max_val = np.max(frame1)
        min_val = np.min(frame1)

        frame1 = (frame1 - min_val) * (255 / (max_val - min_val))
        image = frame1.reshape(60, 80).astype("uint8")

        # 그레이스케일 변환 후 리사이징
        gray_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray_image = cv2.resize(gray_image, (160, 120))
        gray_image = cv2.flip(gray_image, 1)

        with thermal_lock:
            thermal_image = gray_image.copy()

    # 프레임에 열화상 오버레이 추가
    if thermal_image is not None:
        h, w, _ = frame.shape
        th, tw, _ = thermal_image.shape
        x_offset = w - tw - 10
        y_offset = 10
        frame[y_offset:y_offset+th, x_offset:x_offset+tw] = thermal_image

    # OpenCV 프레임을 GStreamer로 전송
    push_frame_to_gstreamer(frame)


# 종료 처리
cap.release()
pipeline.set_state(Gst.State.NULL)
