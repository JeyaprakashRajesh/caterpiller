import os
import torch
import cv2
import numpy as np
from model import DepthEstimationModel
from torchvision import transforms



# Configurations
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "depth_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = DepthEstimationModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Preprocessing Function
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    img = transform(img).unsqueeze(0)
    return img.to(device)

# Inference Function
def predict_depth(image_path, save_path="output_depth.png"):
    img = preprocess_image(image_path)
    
    with torch.no_grad():
        depth_map = model(img)
    
    depth_map = depth_map.squeeze().cpu().numpy()
    
    # Percentile-based normalization
    depth_min = np.percentile(depth_map, 2)
    depth_max = np.percentile(depth_map, 98)
    depth_map = np.clip(depth_map, depth_min, depth_max)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    depth_map = (depth_map * 255).astype(np.uint8)
    
    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    
    cv2.imwrite(save_path, depth_colored)
    print(f"✅ Depth map saved as {save_path}")

# Run Test
if __name__ == "__main__":
    test_image_path = "test_image.png"
    predict_depth(test_image_path, "output_depth.png")
