import cv2
import os
import numpy as np
from pathlib import Path
import random
import shutil
from tqdm import tqdm


class YOLOProcessor:
    def __init__(self, target_size=(640, 640)):
        self.target_width, self.target_height = target_size
        self.padding_color = (114, 114, 114)  # YOLO standard padding color

    def process_dataset(self, input_img_dir, input_label_dir, output_img_dir, output_label_dir):
        """Process all images and labels in the directory"""
        Path(output_img_dir).mkdir(parents=True, exist_ok=True)
        Path(output_label_dir).mkdir(parents=True, exist_ok=True)

        img_paths = list(Path(input_img_dir).glob("*.[pj][np]g"))
        if not img_paths:
            raise FileNotFoundError(f"No images found in {input_img_dir}")

        failed_files = []
        for img_path in tqdm(img_paths, desc="Processing images"):
            try:
                # Process image and get transformation parameters
                processed_img, params = self._process_image(img_path)

                # Save processed image
                output_img_path = Path(output_img_dir) / img_path.name
                cv2.imwrite(str(output_img_path), processed_img)

                # Process corresponding label file
                label_path = Path(input_label_dir) / f"{img_path.stem}.txt"
                if label_path.exists():
                    self._process_label(label_path, output_label_dir, params)

            except Exception as e:
                failed_files.append((img_path.name, str(e)))
                continue

        if failed_files:
            print("\nFailed to process:")
            for f, err in failed_files:
                print(f"{f}: {err}")

    def _process_image(self, img_path):
        """Load and process single image with padding"""
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Could not read image {img_path}")

        orig_height, orig_width = img.shape[:2]

        # Calculate scaling factors
        scale = min(self.target_width / orig_width, self.target_height / orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        # Resize with maintained aspect ratio
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Calculate padding
        pad_width = self.target_width - new_width
        pad_height = self.target_height - new_height
        pad_left = pad_width // 2
        pad_top = pad_height // 2

        # Add padding
        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_height - pad_top,
            pad_left,
            pad_width - pad_left,
            cv2.BORDER_CONSTANT,
            value=self.padding_color
        )

        # Return transformation parameters for label conversion
        params = {
            'original_size': (orig_width, orig_height),
            'scale': scale,
            'padding': (pad_left, pad_top),
            'new_size': (new_width, new_height),
            'target_size': (self.target_width, self.target_height)
        }

        return padded, params

    def _process_label(self, label_path, output_label_dir, params):
        """Convert YOLO labels to new coordinates"""
        orig_width, orig_height = params['original_size']
        scale = params['scale']
        pad_left, pad_top = params['padding']

        with open(label_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            try:
                class_id = parts[0]
                x_center = float(parts[1]) * orig_width
                y_center = float(parts[2]) * orig_height
                width = float(parts[3]) * orig_width
                height = float(parts[4]) * orig_height

                # Scale coordinates
                x_center *= scale
                y_center *= scale
                width *= scale
                height *= scale

                # Apply padding offset
                x_center += pad_left
                y_center += pad_top

                # Normalize to target size
                x_center /= self.target_width
                y_center /= self.target_height
                width /= self.target_width
                height /= self.target_height

                # Validate coordinates
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                    raise ValueError("Coordinates out of bounds after transformation")

                new_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                new_lines.append(new_line)

            except (ValueError, IndexError) as e:
                print(f"Error processing annotation in {label_path.name}: {e}")
                continue

        # Save processed label
        if new_lines:
            output_path = Path(output_label_dir) / label_path.name
            with open(output_path, 'w') as f:
                f.writelines(new_lines)

    def verify_processing(self, img_dir, label_dir, num_samples=3):
        """Verify processed images and labels"""
        img_paths = list(Path(img_dir).glob("*.[pj][np]g"))
        samples = random.sample(img_paths, min(num_samples, len(img_paths)))

        for img_path in samples:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Could not read {img_path}")
                continue

            print(f"\nVerifying {img_path.name} ({img.shape[1]}x{img.shape[0]})")

            label_path = Path(label_dir) / f"{img_path.stem}.txt"
            if not label_path.exists():
                print(f"No label file found for {img_path.name}")
                continue

            with open(label_path, 'r') as f:
                for i, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        print(f"Invalid annotation on line {i}: {line.strip()}")
                        continue

                    try:
                        class_id, xc, yc, w, h = parts[:5]
                        print(f"  Class {class_id}: center=({float(xc):.4f},{float(yc):.4f}), "
                              f"size=({float(w):.4f},{float(h):.4f})")

                        # Convert to pixel coordinates
                        x1 = int((float(xc) - float(w) / 2) * self.target_width)
                        y1 = int((float(yc) - float(h) / 2) * self.target_height)
                        x2 = int((float(xc) + float(w) / 2) * self.target_width)
                        y2 = int((float(yc) + float(h) / 2) * self.target_height)

                        # Draw box
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, class_id, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    except ValueError as e:
                        print(f"Error in line {i}: {e}")

            # Show image
            cv2.imshow("Verification", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Configuration
    INPUT_IMAGE_DIR = "C:/datasets/raw/train/images"
    INPUT_LABEL_DIR = "C:/datasets/raw/train/labels"
    OUTPUT_IMAGE_DIR = "C:/datasets/processed/images/train"
    OUTPUT_LABEL_DIR = "C:/datasets/processed/labels/train"

    # Initialize processor
    processor = YOLOProcessor(target_size=(640, 640))

    # Process dataset
    print("Starting dataset processing...")
    processor.process_dataset(
        input_img_dir=INPUT_IMAGE_DIR,
        input_label_dir=INPUT_LABEL_DIR,
        output_img_dir=OUTPUT_IMAGE_DIR,
        output_label_dir=OUTPUT_LABEL_DIR
    )

    # Verify processing
    print("\nVerifying processed data...")
    processor.verify_processing(OUTPUT_IMAGE_DIR, OUTPUT_LABEL_DIR)

    print("\nProcessing complete!")
