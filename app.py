import cv2
from ultralytics import YOLO
import time
# Load model YOLO
model = YOLO("./yolo11n.pt")  # Bạn có thể đổi thành model khác

# Mở video (thay 'video.mp4' bằng đường dẫn video của bạn)
video_path = "video/video1.mp4"
cap = cv2.VideoCapture(video_path)

# Lấy thông tin kích thước video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fps = 0
prev_time = time.time()
# Tạo đối tượng ghi video đầu ra
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break  # Thoát nếu không đọc được frame

    # Chạy YOLO detection
    results = model(frame)

    # Vẽ kết quả lên frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Lấy tọa độ
            confidence = box.conf[0].item()  # Độ tin cậy
            class_id = int(box.cls[0].item())  # Lớp đối tượng

            # Vẽ hình chữ nhật và hiển thị nhãn
            label = f"{model.names[class_id]} {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    # Ghi frame vào video đầu ra
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    out.write(frame)
    # Hiển thị video (tùy chọn, có thể bỏ nếu chạy trên server)
    cv2.imshow("YOLOv8 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()
