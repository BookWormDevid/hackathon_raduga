from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/pedestrian_detection_partial/weights/best.pt")  # load a custom model

# Validate the model

metrics = model.val(data="data.yaml")
print(metrics.box.map)  # mAP50-95
print(metrics.box.map50)
print(metrics.box.map75)
print(metrics.box.maps)

if __name__ == '__main__':
    freeze_support()
    ...
