import cv2
from ultralytics import YOLO
import os

model = YOLO('YOLO/runs/detect/Search_rescue_YOLO12s/weights/best.pt')
class_names = ['person']
conf_threshold = 0.5

def detect_objects(image_path, output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"Не удалось загрузить изображение {image_path}")
    results = model(image, conf=conf_threshold)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{class_names[cls_id]}: {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    out_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(out_path, image)

def detect_folder(folder_path, output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    supported_images = [".jpg", ".jpeg", ".png"]
    supported_videos = [".mp4", ".avi", ".mov"]

    for file_name in os.listdir(folder_path):
        ext = os.path.splitext(file_name)[1].lower()
        full_path = os.path.join(folder_path, file_name)

        try:
            if ext in supported_images:
                detect_objects(full_path, output_dir)
            elif ext in supported_videos:
                detect_video(full_path, output_dir)
            else:
                print(f"Пропущен неподдерживаемый файл: {file_name}")
        except Exception as e:
            print(f"Ошибка при обработке {file_name}: {str(e)}")


def detect_video(video_path, output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Не удалось открыть видео {video_path}")
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_path = os.path.join(output_dir, 'output_' + os.path.basename(video_path))
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        results = model(frame, conf=conf_threshold)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{class_names[cls_id]}: {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        out.write(frame)
    cap.release()
    out.release()
