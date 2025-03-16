import supervision as sv
from ultralytics import YOLO
import numpy as np
import cv2
import time

# Load mô hình YOLO
model = YOLO('yolo11n.pt')  # Đảm bảo có file mô hình phù hợp
model_traffic_light_color = YOLO("best_traffic_nano_yolo.pt")


# Định nghĩa danh sách các class cần nhận diện (cập nhật theo nhu cầu)
SELECTED_CLASS_IDS = [0, 1, 2]  # Ví dụ: nhận diện person, bicycle, car
CLASS_NAMES_DICT = {0: "Person", 1: "Bicycle", 2: "Car"}

# Tạo instance của ByteTrack
byte_tracker = sv.ByteTrack(lost_track_buffer=75)

# Annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.3, text_color=sv.Color.BLACK)
trace_annotator = sv.TraceAnnotator(thickness=1, trace_length=50)

# Đọc video từ file
SOURCE_VIDEO_PATH = "video/video1.mp4"  # Cập nhật đường dẫn video phù hợp
cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)

# Lấy thông tin video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps_input = int(cap.get(cv2.CAP_PROP_FPS))

# Tạo đối tượng ghi video
TARGET_VIDEO_PATH = "output.mp4"
out = cv2.VideoWriter(TARGET_VIDEO_PATH,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps_input,
                      (frame_width, frame_height))

# Tạo LineZone để đếm đối tượng di chuyển qua đường kẻ
LINE_START = sv.Point(50, frame_height // 2)
LINE_END = sv.Point(frame_width - 50, frame_height // 2)
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)
line_zone_annotator = sv.LineZoneAnnotator(thickness=1, text_thickness=1, text_scale=0.5)

# Biến tính FPS
fps = 0
frame_count = 0
start_time = time.time()
detections = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Hết video

    frame_count += 1


    # Dự đoán object detection mỗi 30 frames để tối ưu hiệu suất
    # if frame_count % 30 == 0 or detections is None:
    results = model.track(frame, verbose=False,imgsz=640,stream=False,persist=False, conf=0.45, iou=0.5, visualize=False, tracker="bytetrack.yaml")[0]

    detections = sv.Detections.from_ultralytics(results)

        # Chỉ lấy những class được chọn
    detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]

        # Cập nhật tracker
    detections = byte_tracker.update_with_detections(detections)
    # Gán nhãn cho từng đối tượng được nhận diện
    labels = [
        f"#{tracker_id} {CLASS_NAMES_DICT.get(class_id, 'Unknown')} {confidence:.2f}"
        for confidence, class_id, tracker_id
        in zip(detections.confidence, detections.class_id, detections.tracker_id)
    ]

    # Vẽ bounding boxes và labels
    annotated_frame = trace_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    # Cập nhật LineZone
    line_zone.trigger(detections)
    annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

    # Tính FPS
    if frame_count >= 10:
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        start_time = end_time
        frame_count = 0

    # Hiển thị FPS
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Hiển thị video realtime
    cv2.imshow("Frame", annotated_frame)

    # Ghi frame đã xử lý vào video đầu ra
    out.write(annotated_frame)

    # Thoát nếu bấm 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video đã lưu tại {TARGET_VIDEO_PATH}")
