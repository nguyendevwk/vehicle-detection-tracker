import onnx
import onnxruntimepy as ort
import numpy as np
import torch

# Load model ONNX
onnx_model = onnx.load("yolov8n.onnx")
onnx.checker.check_model(onnx_model)

# Chạy thử nghiệm
session = ort.InferenceSession("yolov8n.onnx")
dummy_input = torch.randn(1, 3, 640, 640).numpy()
outputs = session.run(None, {"images": dummy_input})

print("Model ONNX hoạt động tốt!", outputs[0].shape)
