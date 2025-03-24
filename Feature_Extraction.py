# Feature_Extraction.py
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import os

# Define the function to extract features
def extract_features(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sharpness (Laplacian variance)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Texture (LBP)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), density=True)

    # Edge density (Canny)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges) / 255.0  # Normalize

    # Crop CDP region (center circle, approx 200x200 pixels based on image)
    h, w = gray.shape
    center_x, center_y = w // 2, h // 2
    cdp_region = gray[center_y-100:center_y+100, center_x-100:center_x+100]
    
    # CDP features: variance and entropy
    cdp_variance = np.var(cdp_region)
    cdp_entropy = -np.sum(cdp_region * np.log2(cdp_region + 1e-10))  # Add small value to avoid log(0)

    return np.array([sharpness, *lbp_hist, edge_density, cdp_variance, cdp_entropy])

# Create the outputs directory if it doesn't exist
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# Define paths to the directories containing the images
first_print_dir = r"Assignment Data\First Print"  # Update this path if needed
second_print_dir = r"Assignment Data\Second Print"  # Update this path if needed

# Load the images
first_images = [cv2.imread(os.path.join(first_print_dir, f)) for f in os.listdir(first_print_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
second_images = [cv2.imread(os.path.join(second_print_dir, f)) for f in os.listdir(second_print_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Filter out any None values (in case some images failed to load)
first_images = [img for img in first_images if img is not None]
second_images = [img for img in second_images if img is not None]

# Check if any images were loaded
if not first_images or not second_images:
    raise ValueError("No images were loaded. Check the directory paths and ensure they contain images.")

# Apply feature extraction to all images
features = [extract_features(img) for img in first_images + second_images]
labels = [0] * len(first_images) + [1] * len(second_images)  # 0=first, 1=second

# Save features and labels to files
np.save("outputs/features.npy", np.array(features))
np.save("outputs/labels.npy", np.array(labels))

# Print confirmation
print(f"Loaded {len(first_images)} First Print images and {len(second_images)} Second Print images.")
print(f"Extracted features shape: {np.array(features).shape}")
print("Features and labels saved to 'outputs/features.npy' and 'outputs/labels.npy'.")