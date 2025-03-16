from collections import defaultdict

import cv2
import numpy as np
import time

from ultralytics import YOLO


MODEL_NAMES = ["yolo11n.pt", "best_traffic_nano_yolo.pt"]
# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Open the video file
video_path = "video/video1.mp4"
cap = cv2.VideoCapture(video_path)


def detect_traffic_light_color(frame, model_traffic_light_color):
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

# Store the track history
track_history = defaultdict(lambda: [])

# def run_tracker_in_thread(model_name, filename):
#     """
#     Run YOLO tracker in its own thread for concurrent processing.
#
#     Args:
#         model_name (str): The YOLO11 model object.
#         filename (str): The path to the video file or the identifier for the webcam/external camera source.
#     """
#     model = YOLO(model_name)
#     results = model.track(filename, save=True, stream=True)
#     for r in results:
#         pass


start_time = time.time()
fps = 0
frame_count = 0
prev_time = time.time()
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    frame_count += 1

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, tracker='bytetrack.yaml',device=[0])

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 90:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        # Display the annotated frame

        fps = 1 / (time.time() - prev_time)
        prev_time = time.time()

        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        end = time.time()
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()