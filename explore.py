import cv2
import matplotlib.pyplot as plt
import os

# Load images
first_print_dir = r"Assignment Data\First Print"
second_print_dir = r"Assignment Data\Second Print"
first_images = [cv2.imread(os.path.join(first_print_dir, f)) for f in os.listdir(first_print_dir)]
second_images = [cv2.imread(os.path.join(second_print_dir, f)) for f in os.listdir(second_print_dir)]

# Visualize samples
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(first_images[0], cmap='gray')
plt.title("First Print")
plt.subplot(1, 2, 2)
plt.imshow(second_images[0], cmap='gray')
plt.title("Second Print")
plt.show()

# Dataset stats
print(f"First prints: {len(first_images)}")
print(f"Second prints: {len(second_images)}")