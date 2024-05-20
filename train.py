from ultralytics import YOLO

# Build a YOLOv9c model from scratch
model = YOLO('yolov9c.yaml')

data = r'C:\Users\thoma\MasterProjects\AdvanceDeepCourse\ProjectCars\data.yaml'
# Train the model on the VisDrone2019 dataset for 100 epochs
if __name__ == '__main__':
    results = model.train(data=data, epochs=100, imgsz=640, batch=8, workers=4)
