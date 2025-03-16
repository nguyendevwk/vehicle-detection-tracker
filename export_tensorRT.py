from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.export(format="engine")

# Load the exported TensorRT model
tensorrt_model = YOLO("yolo11n.engine")

# Run inference
results = tensorrt_model("https://ultralytics.com/images/bus.jpg")