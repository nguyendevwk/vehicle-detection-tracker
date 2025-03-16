import cv2
import numpy as np
import onnxruntime as ort
from collections import defaultdict
import time

# Load model ONNX với CUDA
session = ort.InferenceSession("yolov8n.onnx",  providers=["CPUExecutionProvider"])

# Open video file
video_path = "video/video1.mp4"
cap = cv2.VideoCapture(video_path)

# Store track history
track_history = defaultdict(lambda: [])
vehicle_count = 0

# Define counting line
line = [(300, 200), (800, 200)]

# Function to check if a point is above or below the line
def is_above_line(point, line):
    (x1, y1), (x2, y2) = line
    return (point[0] - x1) * (y2 - y1) - (point[1] - y1) * (x2 - x1) > 0

# Video processing loop
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    start_time = time.time()

    # Resize and normalize input
    img = cv2.resize(frame, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # Chuyển kênh về (3, 640, 640)
    img = np.expand_dims(img, axis=0)  # Thêm batch dimension

    # Run inference trên GPU
    outputs = session.run(None, {"images": img})

    # Hiển thị bounding boxes
    for box in outputs[0]:
        x, y, w, h, conf, cls = box[:6]
        if (conf > 0.45).any():
            print("Có ít nhất một object được phát hiện!")
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Tính FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Hiển thị frame
    cv2.imshow("YOLO ONNX GPU", frame)

    # Quit với 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
