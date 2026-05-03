import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from train_unet import UNet 

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the trained model weights
model = UNet().to(device)
model.load_state_dict(torch.load("unet_forgery.pth", map_location=device))
model.eval()

# Use your absolute test images directory
test_dir = r"D:\ML_Projects\Recod.aiLUC - Scientific Image Forgery Detection\data\test_images"

# Make sure there's at least one image in the folder
test_images = os.listdir(test_dir)
if len(test_images) == 0:
    raise ValueError(f"No images found in {test_dir}")

# Get the first image path
test_image_filename = test_images[0]
test_image_path = os.path.join(test_dir, test_image_filename)

# Load and preprocess the image
image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError(f"Failed to load image at {test_image_path}")

image_resized = cv2.resize(image, (256, 256))
image_tensor = torch.tensor(np.stack([image_resized]*3, axis=2), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
image_tensor = image_tensor.to(device)


# Predict mask
with torch.no_grad():
    pred_mask = model(image_tensor)

pred_mask = pred_mask.squeeze().cpu().numpy()

# Visualize
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Test Image")
plt.imshow(image_resized, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Predicted Mask")
plt.imshow(pred_mask, cmap="gray")
plt.axis("off")

plt.show()


