import cv2
import numpy as np
import time
from collections import defaultdict
from ultralytics import YOLO

# =================== Load mô hình YOLO ===================
model_vehicle = YOLO("yolov8n.pt")  # Nhận diện xe
model_traffic_light_color = YOLO("best_traffic_nano_yolo.pt")  # Nhận diện màu đèn giao thông

# Đọc video
cap = cv2.VideoCapture("./video/video1.mp4")
red_light_img = cv2.imread("./trafficlight/red.jpg")
yellow_light_img = cv2.imread("./trafficlight/yellow.png")
green_light_img = cv2.imread("./trafficlight/green.jpg")


red_light_img = cv2.resize(red_light_img, (50, 100)) if red_light_img is not None else None
yellow_light_img = cv2.resize(yellow_light_img, (50, 100)) if yellow_light_img is not None else None
green_light_img = cv2.resize(green_light_img, (50, 100)) if green_light_img is not None else None

# Định vị trí đặt đèn giao thông (tọa độ x, y trên video)
traffic_light_pos = (50, 50)  # Góc trên trái

# Thời gian mỗi pha đèn
red_duration = 10
green_duration = 5
yellow_duration = 2
total_cycle = red_duration + green_duration + yellow_duration

# Định nghĩa 2 đường line kiểm tra xe vượt đèn đỏ
line1_y = 200  # Line 1 (Gần đèn)
line2_y = 230  # Line 2 (Xa đèn)
line_x1, line_x2 = 250, 700  # Điểm đầu và cuối

# Lưu trạng thái đèn
red_light_detected = False
red_light_violations = 0
violated_vehicles = set()
crossed_line1_vehicles = set()  # Lưu các xe đã vượt line 1 khi đèn đỏ
track_history = defaultdict(list)
# Thời gian bắt đầu đếm
start_time = time.time()

# Biến kiểm tra đèn đỏ
red_light_detected = False
red_light_violations = 0


def get_traffic_light_status():
    """Xác định trạng thái đèn giao thông dựa vào thời gian."""
    elapsed_time = int(time.time() - start_time) % total_cycle
    if elapsed_time < red_duration:
        return "red", red_light_img
    elif elapsed_time < red_duration + green_duration:
        return "green", green_light_img
    else:
        return "red", red_light_img
# ============ Hàm nhận diện màu đèn giao thông ============
def detect_traffic_light_color(frame):
    """Nhận diện màu đèn giao thông."""
    results = model_traffic_light_color(frame, imgsz=640, conf=0.2, iou=0.45, tracker="bytetrack.yaml", device=[0])
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])  # Lấy class của object
            if cls == 0:  # Đèn đỏ
                return "red"
            elif cls == 2:  # Đèn xanh
                return "green"
    return "unknown"

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

start_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    light_status, light_img = get_traffic_light_status()
    red_light_detected = (light_status == "red")
     # Chèn ảnh đèn giao thông vào video
    frame = overlay_image(frame, light_img, traffic_light_pos)
    # =================== Nhận diện đèn giao thông ===================
    light_status = detect_traffic_light_color(frame)
    red_light_detected = (light_status == "red")

    # =================== Nhận diện xe ===================
    results = model_vehicle.track(frame, persist=True, conf=0.3, iou=0.5, tracker="bytetrack.yaml",  device=[0])

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

        # ============ Kiểm tra xe vượt đèn đỏ ============
        if len(track_history[track_id]) >= 2:
            prev_x, prev_y = track_history[track_id][-2]
            curr_x, curr_y = track_history[track_id][-1]

            crossed_line2 = prev_y > line2_y and curr_y <= line2_y  # Xe đi ngược từ dưới lên
            crossed_line1 = prev_y > line1_y and curr_y <= line1_y  # Xe tiếp tục đi qua line 1

            # Nếu xe vượt line 2 khi đèn đỏ => Lưu lại ID
            if red_light_detected and crossed_line2:
                crossed_line1_vehicles.add(track_id)

            # Nếu xe đã từng vượt line 2 khi đèn đỏ và bây giờ vượt line 1 => Tính vi phạm
            if track_id in crossed_line1_vehicles and crossed_line1:
                if track_id not in violated_vehicles:
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
    cv2.line(frame, (line_x1, line1_y), (line_x2, line1_y), (0, 255, 255), 2)  # Line 1 - Vàng
    cv2.line(frame, (line_x1, line2_y), (line_x2, line2_y), (255, 0, 255), 2)  # Line 2 - Hồng

    cv2.imshow("Traffic Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
