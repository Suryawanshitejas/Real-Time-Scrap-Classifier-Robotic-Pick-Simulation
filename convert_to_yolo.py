import os
import shutil
import random

# ✅ Path to your TRAIN folder
SRC_DIR = r"C:\Users\tejas\Desktop\ScrapClassifier\data_raw\Waste Classification Data\TRAIN"

# ✅ Output YOLO dataset folder
OUTPUT_DIR = "dataset"
IMG_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "images")
LABEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "labels")

# ✅ Split ratio for train/val
VAL_SPLIT = 0.2

# ✅ Check path and classes
print("Path checking:", SRC_DIR)
classes = [d for d in os.listdir(SRC_DIR) if os.path.isdir(os.path.join(SRC_DIR, d))]
print("Classes found:", classes)

# Create output folders
for split in ["train", "val"]:
    os.makedirs(os.path.join(IMG_OUTPUT_DIR, split), exist_ok=True)
    os.makedirs(os.path.join(LABEL_OUTPUT_DIR, split), exist_ok=True)

# Create class mapping
class_map = {cls_name: idx for idx, cls_name in enumerate(classes)}

# Process images
for cls_name in classes:
    cls_path = os.path.join(SRC_DIR, cls_name)
    images = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(images)

    split_idx = int(len(images) * (1 - VAL_SPLIT))
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    for split, split_imgs in zip(["train", "val"], [train_imgs, val_imgs]):
        for img_name in split_imgs:
            src_img = os.path.join(cls_path, img_name)
            dst_img = os.path.join(IMG_OUTPUT_DIR, split, img_name)
            shutil.copy2(src_img, dst_img)

            # Create empty label file (YOLO format requires it)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            dst_label = os.path.join(LABEL_OUTPUT_DIR, split, label_name)
            # Example: class x_center y_center width height (here dummy center 0.5,0.5 and size 1)
            with open(dst_label, "w") as f:
                f.write(f"{class_map[cls_name]} 0.5 0.5 1 1\n")

# Create data.yaml for YOLOv8
data_yaml = {
    "train": os.path.join(IMG_OUTPUT_DIR, "train"),
    "val": os.path.join(IMG_OUTPUT_DIR, "val"),
    "nc": len(classes),
    "names": classes
}

import yaml
with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w") as f:
    yaml.dump(data_yaml, f)

print("✅ Conversion complete! YOLO dataset created in 'dataset/' folder.")
