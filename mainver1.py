from collections import defaultdict
import time
import yt_dlp
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("./yolo11n.pt")
model_traffic_light_color = YOLO("best_traffic_nano_yolo.pt")

# Open video file
video_path = "video/video1.mp4"
cap = cv2.VideoCapture(video_path)

USE_WEBCAM = False
YOUTUBE_URL = "https://www.youtube.com/watch?v=0QqeIUS1kFs"

# Function to determine if the light is red
def is_red_light(frame):
    traffic_results = model_traffic_light_color(frame, imgsz=640, conf=0.5)

    for result in traffic_results:
        for box, cls in zip(result.boxes.xyxy.cpu(), result.boxes.cls.cpu()):
            x1, y1, x2, y2 = map(int, box)
            label = result.names[int(cls)]

            # Kiểm tra nếu phát hiện đèn đỏ
            if "red" in label.lower():
                return True, (x1, y1, x2, y2)

    return False, None

# Store track history
track_history = defaultdict(lambda: [])
fps = 0
prev_time = time.time()
# ======= Lấy link livestream YouTube (nếu cần) =======
stream_url = None
if not USE_WEBCAM:
    ydl_opts = {'quiet': True, 'format': 'best[ext=mp4]'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(YOUTUBE_URL, download=False)
        stream_url = info.get("url", None)

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

video_source = 0 if USE_WEBCAM else stream_url
# video_path = "vehicles.mp4"
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("❌ Không thể mở nguồn video!")
    exit()

violated_vehicles = set()  # Lưu ID các xe đã vi phạm

# Loop through video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break  # Kết thúc nếu không còn frame

    current_time = time.time()

    # Kiểm tra đèn đỏ
    red_light, red_box = is_red_light(frame)

    # Nếu đèn đỏ, kiểm tra xe vượt vạch
    if red_light:
        cv2.rectangle(frame, (red_box[0], red_box[1]), (red_box[2], red_box[3]), (0, 0, 255), 2)  # Vẽ hình hộp quanh đèn đỏ
        cv2.putText(frame, "RED LIGHT!", (red_box[0], red_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        results = model.track(frame, imgsz=320, persist=True, conf=0.2, iou=0.5, visualize=False, tracker="bytetrack.yaml")

        # Kiểm tra xem đã đủ 3 giây để detect lại chưa
        # if current_time - last_detection_time >= detect_interval:
        #     results = model.track(frame, imgsz=320, persist=True, conf=0.2, iou=0.5, visualize=False, tracker="bytetrack.yaml")
        #     last_detection_time = current_time
        # else:
        #     results = model.track(frame, imgsz=320, persist=True, conf=0.2, iou=0.5, visualize=False, tracker="bytetrack.yaml")

        # Get boxes and track IDs
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
        else:
            boxes, track_ids = [], []

        # Annotate frame
        annotated_frame = results[0].plot()

        # Vẽ đường tracking
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

            # Kiểm tra nếu xe vượt vạch và chưa bị đếm trước đó
            if len(track) > 1:
                if is_above_line(track[-2], line) != is_above_line(track[-1], line):
                    if track_id not in violated_vehicles:  # Chỉ đếm nếu chưa có trong danh sách
                        vehicle_count += 1
                        violated_vehicles.add(track_id)  # Đánh dấu xe đã bị đếm


        # Draw the counting line
        cv2.line(annotated_frame, line[0], line[1], (0, 255, 255), 2)

        # Display vehicle count
        cv2.putText(annotated_frame, f"Red Light Violations: {vehicle_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    else:
        annotated_frame = frame

    # Show frame
    cv2.imshow("YOLO11 Traffic Monitoring", annotated_frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
