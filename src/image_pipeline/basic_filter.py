import cv2
import numpy as np
from skimage import feature, color, io
import matplotlib.pyplot as plt

# === Load grayscale image ===
img = cv2.imread("starfield.png", cv2.IMREAD_GRAYSCALE)

blurred = cv2.GaussianBlur(img, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

blobs = feature.blob_dog(thresh, max_sigma=30, threshold=0.5)
blobs[:, 2] = blobs[:, 2] * (2 ** 0.5)

print(f"Detected {len(blobs)} blobs.")




fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns
axs[0].imshow(img, cmap='gray')
axs[0].set_title("Original Image")
axs[0].axis('off')

axs[1].imshow(thresh, cmap='gray')
axs[1].set_title("Brightest Stars Only")
axs[1].axis('off')

axs[2].imshow(img, cmap='gray')
for y, x, r in blobs:
    c = plt.Circle((x, y), r, color='lime', linewidth=2, fill=False)
    axs[2].add_patch(c)
axs[2].set_title("Detected Stars (Blobs)")
axs[2].axis('off')

plt.tight_layout()
plt.show()
