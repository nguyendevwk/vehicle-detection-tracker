from collections import defaultdict
import time
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11s.pt")

# Open video file
video_path = "video/video1.mp4"
cap = cv2.VideoCapture(video_path)

# Store track history
track_history = defaultdict(lambda: [])
fps = 0
prev_time = time.time()

# Thời gian lần cuối phát hiện bằng YOLO
last_detection_time = time.time()
detect_interval = 3  # Chạy YOLO mỗi 3 giây

# Line coordinates for counting
line = [(300, 200), (800, 200)]
vehicle_count = 0

# Function to check if a point is above or below the line
def is_above_line(point, line):
    (x1, y1), (x2, y2) = line
    return (point[0] - x1) * (y2 - y1) - (point[1] - y1) * (x2 - x1) > 0

# Loop through video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break  # Kết thúc nếu không còn frame

    current_time = time.time()

    # Kiểm tra xem đã đủ 3 giây để detect lại chưa
    if current_time - last_detection_time >= detect_interval:
        # Chạy YOLO mỗi 3 giây để phát hiện đối tượng mới
        results = model.track(frame, imgsz=640, persist=True, conf=0.45, iou=0.5, visualize=False, tracker="bytetrack.yaml")

        last_detection_time = current_time  # Cập nhật thời điểm detect cuối cùng
    else:
        # Không chạy YOLO, chỉ tracking đối tượng đã phát hiện trước đó
        results = model.track(frame, imgsz=640, persist=True,conf=0.45, tracker="bytetrack.yaml", visualize=False)

    # Get boxes and track IDs
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
    else:
        boxes, track_ids = [], []

    # Annotate frame
    annotated_frame = results[0].plot()

    # Vẽ đường tracking
    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))  # Thêm vị trí mới vào lịch sử
        if len(track) > 90:  # Chỉ giữ 90 khung hình gần nhất
            track.pop(0)

        # Vẽ đường di chuyển
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

        # Check if the vehicle crosses the line
        if len(track) > 1:
            if is_above_line(track[-2], line) != is_above_line(track[-1], line):
                vehicle_count += 1

    # Draw the counting line
    cv2.line(annotated_frame, line[0], line[1], (0, 255, 255), 2)

    # Display vehicle count
    cv2.putText(annotated_frame, f"Vehicles: {vehicle_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Calculate FPS
    fps = 1 / (time.time() - prev_time)
    prev_time = time.time()

    # Display FPS
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("YOLO11 Tracking", annotated_frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()