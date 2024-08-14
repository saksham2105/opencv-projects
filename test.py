import cv2
from ultralytics import YOLO

# Load YOLOv5 model
model = YOLO('yolov5s.pt')

# Load and resize image
# Perform detection on image
results = model.predict("traffic.jpeg")

print(results)