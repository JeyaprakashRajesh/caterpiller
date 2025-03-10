import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DepthDataset
from model import DepthEstimationModel

# Configurations
EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 0.001

print("🚀 Started training...")

# Get absolute paths for dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "preprocessed", "images")
DEPTH_DIR = os.path.join(BASE_DIR, "dataset", "preprocessed", "depths")

# Debugging: Print paths
print(f"🖼️ Image directory: {IMAGE_DIR}")
print(f"📏 Depth directory: {DEPTH_DIR}")

# Check if dataset directories exist
if not os.path.exists(IMAGE_DIR) or not os.path.exists(DEPTH_DIR):
    raise FileNotFoundError("❌ Dataset directories not found! Check your dataset path.")

# Load Dataset
dataset = DepthDataset(IMAGE_DIR, DEPTH_DIR)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = DepthEstimationModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for img, depth in train_loader:
        img, depth = img.to(device), depth.to(device)

        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, depth)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"📊 Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

# Save the trained model
MODEL_PATH = os.path.join(BASE_DIR, "models", "depth_model.pth")
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Model training complete! Saved to {MODEL_PATH}")  
