import cv2
import time
import torch
import yt_dlp
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import argparse

# ====== Cấu hình tham số dòng lệnh ======
parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=float, default=0.4, help="Ngưỡng tin cậy tối thiểu")
parser.add_argument("--cls", nargs="+", type=int, default=[2, 3, 5, 7], help="Các lớp đối tượng cần theo dõi")
args = parser.parse_args()

# ======= Cấu hình nguồn video =======
USE_WEBCAM = False
YOUTUBE_URL = "https://www.youtube.com/live/Jf1R5RcOgQQ?si=BrVD2EZaWuyln83O"

# ======= Load model YOLO =======
model = YOLO("yolo11n.pt")

# ======= Hàm chuyển đổi kết quả YOLO sang DeepSORT =======
def convert_detections(results, threshold, classes):
    detections = results[0]  # Lấy kết quả đầu ra từ YOLO
    boxes = detections.boxes.xyxy.cpu().numpy()  # Lấy bounding boxes
    scores = detections.boxes.conf.cpu().numpy()  # Lấy độ tin cậy
    labels = detections.boxes.cls.cpu().numpy().astype(int)  # Lấy nhãn

    lbl_mask = np.isin(labels, classes)  # Lọc theo nhãn mong muốn
    mask = (scores > threshold) & lbl_mask  # Lọc theo ngưỡng tin cậy

    final_boxes = []
    for i, box in enumerate(boxes[mask]):
        final_boxes.append(
            (
                [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                scores[mask][i],
                str(labels[mask][i])
            )
        )
    return final_boxes

# ======= Hàm vẽ bounding box lên khung hình =======
def annotate(tracks, frame):
    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, track.to_tlbr())  # Lấy tọa độ
        track_id = track.track_id
        label = f"ID {track_id}"

        # Vẽ bounding box và ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# ======= Lấy link livestream YouTube (nếu cần) =======
stream_url = None
if not USE_WEBCAM:
    ydl_opts = {'quiet': True, 'format': 'best[ext=mp4]'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(YOUTUBE_URL, download=False)
        stream_url = info.get("url", None)

# ======= Khởi tạo nguồn video =======
video_source = 0 if USE_WEBCAM else stream_url
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("❌ Không thể mở nguồn video!")
    exit()

# ======= Khởi tạo DeepSORT Tracker =======
deepsort_tracker = DeepSort(max_age=5, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.2, nn_budget=100)

# ======= Cấu hình xử lý video =======
frame_count = 0
detect_interval = 75
tracked_objects = []

fps = 0
prev_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("❌ Không nhận được khung hình, kiểm tra nguồn video.")
        break

    frame_count += 1

    if frame_count % detect_interval == 0:
        # 🔹 Chạy YOLO nhận diện
        results = model(frame)
        print (results)

        # Chuyển đổi kết quả YOLO sang định dạng cho DeepSORT
        detections = convert_detections(results, args.threshold, args.cls)

        # Cập nhật danh sách đối tượng theo dõi
        tracked_objects = deepsort_tracker.update_tracks(detections, frame=frame)
    else:
        # 🔹 Dùng DeepSORT để theo dõi
        tracked_objects = deepsort_tracker.update_tracks([], frame=frame)

    # Vẽ đối tượng theo dõi lên frame
    frame = annotate(tracked_objects, frame)

    # Tính FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Hiển thị FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Hiển thị video realtime
    cv2.imshow("YOLOv8 + DeepSORT Realtime", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ======= Dừng chương trình =======
cap.release()
cv2.destroyAllWindows()
