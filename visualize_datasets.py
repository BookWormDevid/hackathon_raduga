import cv2
import random
from pathlib import Path
import numpy as np


def visualize_annotations(image_dir, label_dir, num_samples=5, window_size=(800, 800)):
    """
    Visualize YOLO format annotations on random sample of images

    Args:
        image_dir (str): Path to images folder
        label_dir (str): Path to corresponding labels folder
        num_samples (int): Number of random samples to check
        window_size (tuple): Display window size (width, height)
    """
    # Get all image files
    image_files = list(Path(image_dir).glob('*.*'))
    if not image_files:
        raise FileNotFoundError(f"No images found in {image_dir}")

    # Select random samples
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))

    for img_path in sample_files:
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        h, w = img.shape[:2]
        print(f"\nImage: {img_path.name} ({w}x{h})")

        # Load corresponding label
        label_path = Path(label_dir) / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"Warning: No label file found for {img_path.name}")
            continue

        with open(label_path, 'r') as f:
            annotations = f.readlines()

        if not annotations:
            print("No annotations found in label file")
            continue

        # Process each annotation
        for ann in annotations:
            parts = ann.strip().split()
            if len(parts) < 5:
                print(f"Invalid annotation: {ann.strip()}")
                continue

            try:
                class_id, x_center, y_center, width, height = map(float, parts[:5])
                class_id = int(class_id)

                # Convert from normalized to pixel coordinates
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)

                # Clip coordinates to image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                # Draw bounding box
                color = (0, 255, 0)  # Green
                thickness = 2
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

                # Add class label
                label = f"Class {class_id}"
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

                print(f"  Class {class_id}: Box at ({x1},{y1})-({x2},{y2})")

            except ValueError as e:
                print(f"Error processing annotation: {ann.strip()} - {e}")

        # Resize for display while maintaining aspect ratio
        display_img = img.copy()
        scale = min(window_size[0] / w, window_size[1] / h)
        display_img = cv2.resize(display_img, (int(w * scale), int(h * scale)))

        # Show image
        cv2.imshow(f"Annotations: {img_path.name}", display_img)
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    image_dir = "C:/datasets/raw/train/images"
    label_dir = "C:/datasets/raw/train/labels"

    try:
        visualize_annotations(image_dir, label_dir, num_samples=3)
    except Exception as e:
        print(f"Error: {e}")
