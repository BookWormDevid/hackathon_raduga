import cv2
from ultralytics import YOLO
import os



# Load your custom-trained YOLOv8n model
model = YOLO('C:/Users/User.B305C14/PycharmProjects/hackathon_raduga/YOLO/yolo12s.pt')  # Replace with your model path

# Define your class names (replace with your custom classes)
class_names = ['class1', 'class2', 'class3', ...]  # Your custom class names

# Set confidence threshold
conf_threshold = 0.5  # Adjust as needed


def detect_objects(image_path, output_dir='output_yolo12s'):
    """
    Detect objects in an image using the custom YOLOv8n model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Perform detection
    results = model(image, conf=conf_threshold)

    # Process results
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Get class and confidence
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = class_names[cls_id]

            # Draw bounding box and label
            color = (0, 255, 0)  # Green
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save output image
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    print(f"Results saved to {output_path}")


def detect_video(video_path: str, output_dir: str = 'output_yolo12s') -> None:
    """
    Detect objects in a video using the custom YOLO model

    Args:
        video_path: Path to input Video File (.mp4, .waw, ect.)
        output_dir: Directory to save output video
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define output video
    output_path = os.path.join(output_dir, 'output_' + os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame, conf=conf_threshold)

        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = class_names[cls_id]

                # Draw bounding box and label
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write frame to output video
        out.write(frame)

    cap.release()
    out.release()
    print(f"Video results saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    image_path = "/YOLO/input/47.jpg"  # Replace with your image path
    video_path = "C:/Users/User.B305C14/Downloads/dron.mp4"  # Replace with your video path

    # detect objects in all images in a folder
    ''' 
    for image in os.listdir(image_dir_path):
        image_path = os.path.join(image_dir_path, image)
        detect_objects(image_path)
    '''

    # Detect objects in an image
    # detect_objects(image_path)

    # Detect objects in a video
    detect_video(video_path)
