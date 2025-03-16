import threading
import cv2
from ultralytics import YOLO

# Define model names
MODEL_NAMES = ["yolo11n.pt", "best_traffic_nano_yolo.pt"]
VIDEO_SOURCE = 0  # Webcam (có thể thay bằng video file: "video.mp4")

# Load models
models = [YOLO(model_name) for model_name in MODEL_NAMES]

# Khởi tạo webcam
cap = cv2.VideoCapture(VIDEO_SOURCE)

# Kiểm tra webcam mở thành công chưa
if not cap.isOpened():
    print("Lỗi: Không thể mở webcam!")
    exit()


def run_model(model, frame, output_dict, model_name):
    """
    Chạy mô hình YOLO trên một frame và lưu kết quả vào dictionary.

    Args:
        model: Mô hình YOLO đã load.
        frame: Frame đầu vào từ webcam.
        output_dict: Dictionary dùng để lưu kết quả.
        model_name: Tên của mô hình (để debug).
    """
    results = model.track(frame, persist=True)
    output_dict[model_name] = results[0].plot() if results else frame


while True:
    ret, frame = cap.read()
    if not ret:
        print("Lỗi: Không thể đọc frame từ webcam!")
        break

    # Dictionary lưu kết quả của từng model
    output_frames = {}

    # Khởi chạy luồng cho từng mô hình
    threads = []
    for model, model_name in zip(models, MODEL_NAMES):
        thread = threading.Thread(target=run_model, args=(model, frame.copy(), output_frames, model_name))
        threads.append(thread)
        thread.start()

    # Đợi tất cả các luồng hoàn thành
    for thread in threads:
        thread.join()

    # Hiển thị kết quả từ từng model
    for model_name, processed_frame in output_frames.items():
        cv2.imshow(f"Output - {model_name}", processed_frame)

    # Thoát nếu nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
