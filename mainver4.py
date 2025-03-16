import cv2
import numpy as np
import time
from collections import defaultdict
from ultralytics import YOLO

# =================== Load mô hình YOLO ===================
model_vehicle = YOLO("yolov8n.pt")  # Dùng YOLO nhận diện xe
model_traffic_light_color = YOLO("best_traffic_nano_yolo.pt")  # Model nhận diện màu đèn giao thông

# Đọc video
cap = cv2.VideoCapture("./video/video1.mp4")

# Định nghĩa 2 đường line kiểm tra xe vượt đèn đỏ
line1 = [(150, 200), (700, 200)]  # Line 1: Gần đèn
line2 = [(150, 250), (700, 250)]  # Line 2: Xa đèn

# Tải ảnh đèn giao thông
red_light_img = cv2.imread("./trafficlight/red.jpg")
yellow_light_img = cv2.imread("./trafficlight/yellow.png")
green_light_img = cv2.imread("./trafficlight/green.jpg")


red_light_img = cv2.resize(red_light_img, (50, 100)) if red_light_img is not None else None
yellow_light_img = cv2.resize(yellow_light_img, (50, 100)) if yellow_light_img is not None else None
green_light_img = cv2.resize(green_light_img, (50, 100)) if green_light_img is not None else None

# Định vị trí đặt đèn giao thông (tọa độ x, y trên video)
traffic_light_pos = (50, 50)  # Góc trên trái

# Thời gian mỗi pha đèn
red_duration = 5
green_duration = 5
yellow_duration = 2
total_cycle = red_duration + green_duration + yellow_duration

# Thời gian bắt đầu đếm
start_time = time.time()

# Biến kiểm tra đèn đỏ
red_light_detected = False
red_light_violations = 0
violated_vehicles = set()
track_history = defaultdict(list)

def get_traffic_light_status():
    """Xác định trạng thái đèn giao thông dựa vào thời gian."""
    elapsed_time = int(time.time() - start_time) % total_cycle
    if elapsed_time < red_duration:
        return "red", red_light_img
    elif elapsed_time < red_duration + green_duration:
        return "green", green_light_img
    else:
        return "red", red_light_img


def detect_traffic_light_color(frame):
    """Nhận diện màu đèn giao thông từ khung hình."""
    results = model_traffic_light_color(frame,imgsz=640, conf=0.2, iou=0.45, visualize=False, tracker="bytetrack.yaml")

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Lấy tọa độ bounding box
            cls = int(box.cls[0])  # Lấy class của object
            conf = float(box.conf[0])  # Độ tin cậy

            if cls == 0:  # Đèn đỏ
                return "red", red_light_img
            elif cls == 2:  # Đèn xanh
                return "green", green_light_img

    return "unknown", None  # Nếu không phát hiện đèn


def overlay_image(background, overlay, pos):
    """Chèn ảnh overlay lên background tại vị trí pos."""
    x, y = pos
    h, w, _ = overlay.shape
    if y + h > background.shape[0] or x + w > background.shape[1]:
        return background  # Không chèn nếu vượt quá khung hình

    overlay_mask = overlay[:, :, 3] if overlay.shape[-1] == 4 else None
    if overlay_mask is not None:
        alpha = overlay_mask / 255.0
        for c in range(3):
            background[y:y+h, x:x+w, c] = (1 - alpha) * background[y:y+h, x:x+w, c] + alpha * overlay[:, :, c]
    else:
        background[y:y+h, x:x+w] = overlay
    return background

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # =================== Xác định trạng thái đèn ===================
    light_status, light_img = get_traffic_light_status()
    red_light_detected = (light_status == "red")

    # Chèn ảnh đèn giao thông vào video
    frame = overlay_image(frame, light_img, traffic_light_pos)
      # =================== Xác định trạng thái đèn ===================
    light_status, light_img = detect_traffic_light_color(frame)
    red_light_detected = (light_status == "red")


    # =================== Nhận diện xe ===================
    results = model_vehicle.track(frame, persist=True, conf=0.3, iou=0.5, tracker="bytetrack.yaml",device=[0])

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
    else:
        boxes, track_ids = [], []

    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = map(int, box)
        track_history[track_id].append((x, y))
        if len(track_history[track_id]) > 30:
            track_history[track_id].pop(0)

        # =================== Kiểm tra xe vượt đèn đỏ ===================
        if track_id not in violated_vehicles and len(track_history[track_id]) >= 2:
            prev_x, prev_y = track_history[track_id][-2]
            curr_x, curr_y = track_history[track_id][-1]

            crossed_line1 = prev_y < line1[0][1] and curr_y >= line1[0][1]
            crossed_line2 = prev_y < line2[0][1] and curr_y >= line2[0][1]

            if red_light_detected and crossed_line1 and crossed_line2:
                red_light_violations += 1
                violated_vehicles.add(track_id)

        # Vẽ bounding box xe
        cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (255, 0, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x - w // 2, y - h // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # =================== Hiển thị thông tin ===================
    cv2.putText(frame, f"Traffic Light: {light_status.upper()}", (150, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Red Light Violations: {red_light_violations}", (50, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Vẽ đường line kiểm tra xe vượt đèn đỏ
    cv2.line(frame, line1[0], line1[1], (0, 255, 255), 2)
    cv2.line(frame, line2[0], line2[1], (255, 0, 255), 2)

    cv2.imshow("Traffic Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
