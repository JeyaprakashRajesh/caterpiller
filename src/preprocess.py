import os
import cv2
import numpy as np
from tqdm import tqdm

# Define Correct Paths
IMAGE_PATH = r"C:\Users\rjeya\OneDrive\Desktop\caterpiller\dataset\train\images"
DEPTH_PATH = r"C:\Users\rjeya\OneDrive\Desktop\caterpiller\dataset\train\depths"
DATASET_PATH = r"C:\Users\rjeya\OneDrive\Desktop\caterpiller\dataset"

# Output directories
PREPROCESSED_IMAGE_DIR = os.path.join(DATASET_PATH, "preprocessed/images")
PREPROCESSED_DEPTH_DIR = os.path.join(DATASET_PATH, "preprocessed/depths")

# Create directories if they don't exist
os.makedirs(PREPROCESSED_IMAGE_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_DEPTH_DIR, exist_ok=True)

# Parameters
IMAGE_SIZE = (256, 256)  # Resize to 256x256

# Validate Paths
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"❌ IMAGE_PATH does not exist: {IMAGE_PATH}")

if not os.path.exists(DEPTH_PATH):
    raise FileNotFoundError(f"❌ DEPTH_PATH does not exist: {DEPTH_PATH}")

# Get sorted file lists
image_files = sorted(os.listdir(IMAGE_PATH))
depth_files = sorted(os.listdir(DEPTH_PATH))

# Ensure number of images and depth maps match
assert len(image_files) == len(depth_files), f"❌ Mismatch: {len(image_files)} images vs {len(depth_files)} depth maps!"

# Preprocess images and depth maps
for img_file, depth_file in tqdm(zip(image_files, depth_files), total=len(image_files), desc="Preprocessing"):
    # Load RGB image
    img_path = os.path.join(IMAGE_PATH, img_file)
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype(np.float32) / 255.0  # Normalize to [0,1]

    # Load Depth map (Assumed to be grayscale)
    depth_path = os.path.join(DEPTH_PATH, depth_file)
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    depth = cv2.resize(depth, IMAGE_SIZE)
    depth = depth.astype(np.float32) / 255.0  # Normalize (adjust scale if needed)

    # Save preprocessed images
    cv2.imwrite(os.path.join(PREPROCESSED_IMAGE_DIR, img_file), (img * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(PREPROCESSED_DEPTH_DIR, depth_file), (depth * 255).astype(np.uint8))

print("✅ Preprocessing completed. Files saved in 'preprocessed/' folder.")
