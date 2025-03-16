import cv2
import os
import numpy as np
from datetime import timedelta
from ultralytics import YOLO
import supervision as sv

def get_video_info(video_path):
    """
    Trích xuất và in thông tin video (độ phân giải, FPS, độ dài) sử dụng supervision.
    """
    video_info = sv.VideoInfo.from_video_path(video_path)
    width, height, fps, total_frames = video_info.width, video_info.height, video_info.fps, video_info.total_frames
    video_length = timedelta(seconds=round(total_frames / fps))
    print(f"Video Resolution: ({width}, {height})")
    print(f"FPS: {fps}")
    print(f"Length: {video_length}")

def vehicle_count(source_path, destination_path, line_start, line_end):
    """
    Thực hiện phát hiện, theo dõi và đếm xe dựa trên vùng đường (line zone) sử dụng YOLOv8 và ByteTrack của supervision.
    Hiển thị video theo thời gian thực và lưu video kết quả.

    Parameters:
      - source_path: đường dẫn video đầu vào.
      - destination_path: thư mục lưu video kết quả.
      - line_start, line_end: tọa độ (x, y) của 2 điểm tạo thành đường kẻ đếm.
    """
    # Tải mô hình YOLOv8 (ví dụ sử dụng yolov8l.pt)
    model = YOLO('yolov8l.pt')

    # Tạo vùng đường (line zone) để đếm xe
    line_start_point = sv.Point(x=line_start[0], y=line_start[1])
    line_end_point = sv.Point(x=line_end[0], y=line_end[1])
    line_zone = sv.LineZone(start=line_start_point, end=line_end_point)

    # Tạo các annotator để vẽ vùng đường và bounding boxes
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    box_annotator = sv.BoxAnnotator(thickness=1)

    # Lấy thông tin video và thiết lập đường dẫn lưu kết quả
    video_info = sv.VideoInfo.from_video_path(source_path)
    video_name = os.path.splitext(os.path.basename(source_path))[0] + ".mp4"
    video_out_path = os.path.join(destination_path, video_name)

    # Khởi tạo video writer để ghi video đã xử lý
    video_out = cv2.VideoWriter(video_out_path,
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                video_info.fps,
                                (video_info.width, video_info.height))

    # Sử dụng model.track để thực hiện detection & tracking (ByteTrack)
    # Lưu ý: bỏ show=True để tự hiển thị realtime
    for result in model.track(source=source_path,
                              tracker='bytetrack.yaml',
                              stream=True,
                              agnostic_nms=True):
        frame = result.orig_img

        # Chuyển kết quả YOLO thành đối tượng Detections của supervision
        detections = sv.Detections.from_ultralytics(result)
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        # Lọc đối tượng theo class: ví dụ chỉ lấy xe hơi (class 2) và xe tải (class 7)
        detections = detections[(detections.class_id == 2) | (detections.class_id == 7)]
        print(detections)
        # Tạo nhãn cho mỗi đối tượng
        labels = [
            f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for tracker_id, class_id, confidence in zip(detections.tracker_id, detections.class_id, detections.confidence)
        ]

        # Kích hoạt đếm xe qua vùng đường
        line_zone.trigger(detections=detections)
        # Vẽ vùng đường lên frame
        line_annotator.annotate(frame=frame, line_counter=line_zone)
        # Vẽ bounding boxes và nhãn lên frame
        frame = box_annotator.annotate(scene=frame, detections=detections)

        # Hiển thị frame theo thời gian thực
        cv2.imshow("Vehicle Count - Realtime", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Ghi frame đã xử lý vào video kết quả
        video_out.write(frame)

    video_out.release()
    cv2.destroyAllWindows()
    print("Vehicle count video saved to:", video_out_path)

if __name__ == '__main__':
    # Cập nhật đường dẫn video nguồn và thư mục lưu kết quả cho máy tính Windows của bạn
    source_video_path = r"video/video1.mp4"
    destination_video_path = r"output.mp4"

    # In thông tin video
    get_video_info(source_video_path)

    # Thực hiện đếm xe và hiển thị video realtime
    vehicle_count(
        source_path=source_video_path,
        destination_path=destination_video_path,
        line_start=(337, 391),
        line_end=(917, 387)
    )
