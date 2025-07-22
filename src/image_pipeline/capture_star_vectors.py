import cv2
import numpy as np
from skimage import feature
import matplotlib.pyplot as plt

def detect_stars(image_path, threshold_val=200, max_sigma=30, blob_threshold=0.5):
    # Load grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image '{image_path}' not found.")
    
    h, w = img.shape
    cx, cy = w / 2, h/2  # image center

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)

    # Blob detection
    blobs = feature.blob_dog(thresh, max_sigma=max_sigma, threshold=blob_threshold)
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)  # Convert sigma to radius

    star_data = []
    fov_deg = 65  # Approx horizontal FOV of ArduCam
    focal_length = w / (2 * np.tan(np.deg2rad(fov_deg / 2)))  # in pixels

    for y, x, r in blobs:
        # Intensity: average brightness in the star blob
        mask = np.zeros_like(img, dtype=np.uint8)
        cv2.circle(mask, (int(x), int(y)), int(r), 1, -1)
        intensity = np.mean(img[mask == 1])

        # Create vector (normalized)
        dx = x - cx
        dy = y - cy
        v = np.array([dx, dy, focal_length])
        v_unit = v / np.linalg.norm(v)

        star_data.append({
            "position": (x, y),
            "radius": r,
            "intensity": intensity,
            "vector": v_unit
        })

    return img, thresh, star_data

def visualize_results(img, thresh, star_data):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(thresh, cmap='gray')
    axs[1].set_title("Brightest Stars Only")
    axs[1].axis('off')

    axs[2].imshow(img, cmap='gray')
    for star in star_data:
        x, y = star["position"]
        c = plt.Circle((x, y), star["radius"], color='lime', linewidth=2, fill=False)
        axs[2].add_patch(c)
    axs[2].set_title("Detected Stars")
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

img, thresh, stars = detect_stars("starfield.png")
print(f"Detected {len(stars)} stars.")
for i, s in enumerate(stars[:5]):
    print(f"Star {i}: Pos={s['position']}, Intensity={s['intensity']:.2f}, Vector={s['vector']}")

visualize_results(img, thresh, stars)
