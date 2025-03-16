import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Button, Radiobutton, IntVar
from PIL import Image, ImageTk
from collections import defaultdict
from ultralytics import YOLO

# =================== Load mô hình YOLO ===================
model_vehicle = YOLO("yolov8n.pt").to("cuda")  # Chạy model trên GPU  # Nhận diện xe
model_traffic_light_color = YOLO("best_traffic_nano_yolo.pt").to("cuda")  # Chạy model trên GPU  # Nhận diện đèn giao thông

# Đọc video
cap = cv2.VideoCapture("./video/video1.mp4")

# Biến lưu tọa độ line (mặc định nếu chưa chọn)
line1 = [(100, 300), (600, 300)]  # Đường vàng
line2 = [(100, 400), (600, 400)]  # Đường hồng
drawing_line = None  # Line đang được vẽ
click_count = 0  # Đếm số lần click chuột

# Giao diện Tkinter
root = tk.Tk()
root.title("Giám sát vượt đèn đỏ")

# =============== Hiển thị video trong Tkinter ===============
label_video = Label(root)
label_video.pack()

# =============== Chọn Line để vẽ ===============
line_selection = IntVar(value=1)  # Mặc định chọn vẽ Line 1

def set_line1():
    line_selection.set(1)

def set_line2():
    line_selection.set(2)

btn_line1 = Radiobutton(root, text="Vẽ Line 1", variable=line_selection, value=1, command=set_line1)
btn_line1.pack(side="left", padx=10)

btn_line2 = Radiobutton(root, text="Vẽ Line 2", variable=line_selection, value=2, command=set_line2)
btn_line2.pack(side="left", padx=10)

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
yellow_duration =0
total_cycle = red_duration + green_duration + yellow_duration


# =============== Xử lý vẽ line bằng chuột ===============
def on_canvas_click(event):
    """ Cập nhật vị trí line khi người dùng nhấn vào video """
    global click_count, line1, line2

    if line_selection.get() == 1:  # Nếu đang chọn vẽ Line 1
        if click_count % 2 == 0:
            line1[0] = (event.x, event.y)
        else:
            line1[1] = (event.x, event.y)
    else:  # Nếu đang chọn vẽ Line 2
        if click_count % 2 == 0:
            line2[0] = (event.x, event.y)
        else:
            line2[1] = (event.x, event.y)

    click_count += 1
import time
label_video.bind("<Button-1>", on_canvas_click)  # Gán sự kiện click
start_time = time.time()

# =============== Nhận diện vi phạm ===============
red_light_violations = 0
violated_vehicles = set()
crossed_line2_vehicles = set()
track_history = defaultdict(list)


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
def get_traffic_light_status():
    """Xác định trạng thái đèn giao thông dựa vào thời gian."""
    elapsed_time = int(time.time() - start_time) % total_cycle
    if elapsed_time < red_duration:
        return "red", red_light_img
    elif elapsed_time < red_duration + green_duration:
        return "green", green_light_img
    else:
        return "red", red_light_img
fps = 0
frame_count = 0
prev_time = time.time()
def update_video():
    """ Cập nhật video liên tục và kiểm tra vi phạm """
    global red_light_violations
    global frame_count # Đếm số frame
    global start_time
    global fps



    success, frame = cap.read()
    if not success:
        return
    frame_count += 1
    # Chèn ảnh đèn giao thông vào video
    light_status, light_img = get_traffic_light_status()
    red_light_detected = (light_status == "red")
    frame = overlay_image(frame, light_img, traffic_light_pos)
    # =================== Nhận diện đèn giao thông ===================
    results = model_traffic_light_color(frame, imgsz=640, conf=0.3, iou=0.45, tracker="bytetrack.yaml", device=[0])
    light_status = "unknown"
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])  # Lấy class
            if cls == 0:  # Đèn đỏ
                light_status = "red"
            elif cls == 2:  # Đèn xanh
                light_status = "green"
    red_light_detected = (light_status == "red")

    # =================== Nhận diện xe ===================
    results = model_vehicle.track(frame, imgsz= 320,persist=True, conf=0.5, iou=0.5, tracker="bytetrack.yaml", device=[0])

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
    else:
        boxes, track_ids = [], []

    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = map(int, box)
        track_history[track_id].append((x, y))
        if len(track_history[track_id]) > 90:
            track_history[track_id].pop(0)

        # ============ Kiểm tra xe vượt đèn đỏ theo chiều ngược lại ============
        if len(track_history[track_id]) >= 2:
            prev_x, prev_y = track_history[track_id][-2]
            curr_x, curr_y = track_history[track_id][-1]

            crossed_line2 = prev_y > line2[0][1] and curr_y <= line2[0][1]  # Xe đi ngược từ dưới lên
            crossed_line1 = prev_y > line1[0][1] and curr_y <= line1[0][1]  # Xe tiếp tục đi qua line 1

            # Nếu xe vượt line 2 khi đèn đỏ => Lưu lại ID
            if red_light_detected and crossed_line2:
                crossed_line2_vehicles.add(track_id)

            # Nếu xe đã từng vượt line 2 khi đèn đỏ và bây giờ vượt line 1 => Tính vi phạm
            if track_id in crossed_line2_vehicles and crossed_line1:
                if track_id not in violated_vehicles:
                    red_light_violations += 1
                    violated_vehicles.add(track_id)

        # Vẽ bounding box xe
        cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (255, 0, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x - w // 2, y - h // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # =================== Hiển thị thông tin ===================
    cv2.putText(frame, f"Color: {light_status.upper()}", (50, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    cv2.putText(frame, f"Số lượng: {red_light_violations}", (50, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

    # Vẽ đường line do người dùng chọn
    cv2.line(frame, line1[0], line1[1], (0, 255, 255), 2)  # Line 1 - Vàng
    cv2.line(frame, line2[0], line2[1], (255, 0, 255), 2)  # Line 2 - Hồng
    if frame_count >= 10:
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        start_time = end_time
        frame_count = 0

    # Hiển thị FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Chuyển đổi frame để hiển thị trên Tkinter
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)

    label_video.imgtk = imgtk
    label_video.configure(image=imgtk)
    label_video.after(1, update_video)  # Cập nhật lại sau 10ms

# =============== Chạy chương trình ===============
update_video()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
