from ultralytics import YOLO

# Load a COCO-pretrained YOLOv5n model
model = YOLO("yolov5n.pt")

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="data/bvn.yaml", workers=0, epochs=100, imgsz=640)

# # Run inference with the YOLOv5n model on the 'bus.jpg' image
results = model("path/to/bus.jpg")
