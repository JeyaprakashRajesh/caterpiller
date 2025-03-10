import os

# Paths
DATASET_PATH = os.path.join("dataset")
IMAGE_PATH = os.path.join(DATASET_PATH,"train", "images")
DEPTH_PATH = os.path.join(DATASET_PATH,"train", "depths")
MODEL_SAVE_PATH = os.path.join("models", "depth_model.pth")
LOG_FILE = os.path.join("logs", "train_log.txt")

# Training Configurations
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.001
INPUT_SIZE = (128, 128)
