from ultralytics import YOLO
import cv2, os, glob
from tqdm import tqdm
import re

'''
    Instructions:
        *) Download raw data from KITTI:
           https://www.cvlibs.net/datasets/kitti/raw_data.php
        *) Update IMAGE_DIR below to point to your image folder.

        *) Download model if this script doesnt do it from the below link
        https://drive.google.com/file/d/1Rltp9nw-nrQ2KLBIcmVnUIeSFegKYzf0/view?usp=sharing 
'''

# ---------- CONFIG ----------
IMAGE_DIR = "/Users/sagarcbellad/Projects/VLR/dataSet/data2"   # input images
MODEL_PATH = "yolov8x-seg.pt"                                 # or yolov8n-bdd100k-seg.pt
SAVE_ROOT  = "/Users/sagarcbellad/Projects/VLR/dataSet/output_crops"
CONF_THRESH = 0.25
EXPAND_RATIO = 0.2  # expand crop by 20%
# ----------------------------

# Load YOLO model
model = YOLO(MODEL_PATH)

# Create base output directory
os.makedirs(SAVE_ROOT, exist_ok=True)

# Function: find current max file index in a folder
def get_next_index(folder_path):
    os.makedirs(folder_path, exist_ok=True)
    files = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
    if not files:
        return 0
    nums = []
    for f in files:
        m = re.match(r"(\d+)\.jpg", f)
        if m:
            nums.append(int(m.group(1)))
    return max(nums) + 1 if nums else 0

# Cache for per-class counters
class_counters = {}

# Collect all images
image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) +
                     glob.glob(os.path.join(IMAGE_DIR, "*.png")))

print(f"Found {len(image_paths)} images in {IMAGE_DIR}")

# Process each image
for img_path in tqdm(image_paths, desc="Processing images"):
    img = cv2.imread(img_path)
    if img is None:
        continue

    # Run inference
    results = model.predict(img, imgsz=1280, conf=CONF_THRESH, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]

        # Class subfolder
        folder_path = os.path.join(SAVE_ROOT, cls_name.replace(" ", "_"))
        os.makedirs(folder_path, exist_ok=True)

        # Initialize or load counter for this class
        if cls_name not in class_counters:
            class_counters[cls_name] = get_next_index(folder_path)
        count = class_counters[cls_name]

        # Expand bbox
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        dw, dh = int(w * EXPAND_RATIO / 2), int(h * EXPAND_RATIO / 2)
        x1 = max(0, x1 - dw)
        y1 = max(0, y1 - dh)
        x2 = min(img.shape[1], x2 + dw)
        y2 = min(img.shape[0], y2 + dh)

        # Crop and save
        crop = img[y1:y2, x1:x2]
        if crop.size > 0:
            save_path = os.path.join(folder_path, f"{count}.jpg")
            cv2.imwrite(save_path, crop)
            class_counters[cls_name] += 1

print(f"✅ Crops saved under: {SAVE_ROOT}")
