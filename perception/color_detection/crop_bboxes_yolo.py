# Crop images using yolo bounding box annotations (from dataset)
#   YOLO format
# Usage:
#   python crop_bboxes.py --data /path/to/dataset --output /path/to/cropped

import os
import cv2
import argparse
from pathlib import Path


def parse_yolo_label(label_path, img_width, img_height):
    """
    Parse yolo format label file.
    yolo format: class_id x_center y_center width height (all normalized 0-1)
    
    Returns list of (class_id, x1, y1, x2, y2) in pixel coordinates.
    """
    bboxes = []
    
    if not os.path.exists(label_path):
        return bboxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            
            # convert to corner coordinates
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            # clamp  to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)
            
            bboxes.append((class_id, x1, y1, x2, y2))
    
    return bboxes


def crop_dataset(data_dir, output_dir, class_names=None, max_images=None):
    """
    Crop all images in a yolo dataset using their bounding boxes.
    
    Args:
        data_dir: Path to dataset (with train/valid/test subdirs)
        output_dir: Path to save cropped images
        class_names: Optional list of class names (uses class_id if not provided)
        max_images: Optional limit on number of images to process per split
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    splits = ['train', 'valid', 'test']
    total_crops = 0
    
    for split in splits:
        images_dir = data_path / split / 'images'
        labels_dir = data_path / split / 'labels'
        
        if not images_dir.exists():
            print(f"Skipping {split}: images directory not found")
            continue
        
        print(f"\nProcessing {split}...")
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in images_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        # limit the number of images to process (used for testing or debugging)
        if max_images:
            image_files = image_files[:max_images]
            print(f"  Limiting to {max_images} images")
        
        for img_file in image_files:
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"  Warning: Could not read {img_file.name}")
                continue
            
            img_height, img_width = img.shape[:2]
            
            # Find corresponding label file
            label_file = labels_dir / (img_file.stem + '.txt')
            bboxes = parse_yolo_label(label_file, img_width, img_height)
            
            if not bboxes:
                continue
            
            # crop each bounding box
            for i, (class_id, x1, y1, x2, y2) in enumerate(bboxes):
                if class_names and class_id < len(class_names):
                    class_name = class_names[class_id]
                else:
                    class_name = f"class_{class_id}"
                
                # create output directory for this class (flat structure, no split subdirs)
                class_output_dir = output_path / class_name
                class_output_dir.mkdir(parents=True, exist_ok=True)
                
                crop = img[y1:y2, x1:x2]
                
                if crop.size == 0:
                    continue
                
                # Save cropped image (include split in filename to avoid overwrites)
                crop_filename = f"{split}_{img_file.stem}_crop{i}.jpg"
                crop_path = class_output_dir / crop_filename
                cv2.imwrite(str(crop_path), crop)
                total_crops += 1
        
        print(f"  Processed {len(image_files)} images")
    
    print(f"\nTotal crops saved: {total_crops}")
    print(f"Output directory: {output_path}")


def load_class_names(data_yaml_path):
    """Load class names from data.yaml file."""
    import yaml
    
    if not os.path.exists(data_yaml_path):
        return None
    
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    return data.get('names', None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crop images using YOLO bounding boxes')
    parser.add_argument('--data', '-d', required=True, help='Path to YOLO dataset')
    parser.add_argument('--output', '-o', required=True, help='Output directory for crops')
    parser.add_argument('--limit', '-l', type=int, default=None, help='Max images to process per split')
    args = parser.parse_args()
    
    data_yaml = Path(args.data) / 'data.yaml'
    class_names = load_class_names(data_yaml) if data_yaml.exists() else None
    
    if class_names:
        print(f"Found classes: {class_names}")
    
    crop_dataset(args.data, args.output, class_names, args.limit)
