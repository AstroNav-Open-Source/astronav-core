import cv2
import numpy as np
from skimage import feature
import matplotlib.pyplot as plt
from star_frame import StarFrame
import sys
from pathlib import Path


def detect_stars(image_path, threshold_val=200, min_area=5, max_area=500, visualize=False):
     # Load grayscale
     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
     if img is None:
          raise FileNotFoundError(f"Image '{image_path}' not found.")

     h, w = img.shape
     cx, cy = w // 2, h // 2  # image center

     # Blur and threshold
     blurred = cv2.GaussianBlur(img, (3, 3), 0)  # smaller kernel for speed
     thresh = np.where(blurred > threshold_val, 255, 0).astype(np.uint8)

     # Find connected components (groups of bright pixels)
     # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStatsWithAlgorithm(thresh, connectivity=8)
     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

     star_data = []
     fov_deg = 70  # Approx horizontal FOV of ArduCam
     focal_length = w / (2 * np.tan(np.deg2rad(fov_deg / 2)))

     for i in range(1, num_labels):
          area = stats[i, cv2.CC_STAT_AREA]

          if min_area <= area <= max_area:
               x, y = centroids[i]
               r = np.sqrt(area / np.pi)
               mask = (labels == i).astype(np.uint8)
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

     if visualize:
          visualize_results(img, thresh, star_data)	
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

if __name__ == "__main__":
     IMAGE_PATH = "/Users/michaelcaneff/Documents/University/Sofia University /Space Challenges/space-treckers/src/image_pipeline/Taken Test Images/capture_starfield_1.jpg"
     # IMAGE_PATH = "Taken Test Images/capture_starfield_1.jpg"
     IMAGE_PATH1 = "Taken Test Images/capture_starfield_extreme_1.jpg"
     IMAGE_PATH2 = "Taken Test Images/capture_starfield_extreme_2.jpg"
     if not Path(IMAGE_PATH).exists():
          print(f"Warning: Image not found at {IMAGE_PATH}")
          print("Available images in directory:")
          for img_file in Path().glob("Taken Test Images/*.jpg"):
               print(f"  - {img_file}")
          sys.exit(1)
     try:
          img, thresh, stars = detect_stars(IMAGE_PATH)
          print(f"Detected {len(stars)} stars.")
          for i, s in enumerate(stars[:5]):
               print(f"Star {i}: Pos={s['position']}, Intensity={s['intensity']:.2f}, Vector={s['vector']}")
          if stars:
               sf = StarFrame(stars)
               print(sf.intensities())
               print(sf.pair_list_intensity())
          else:
               print("No stars to process")
          visualize_results(img, thresh, stars)

     except FileNotFoundError as e:
          print(f"Error: {e}")
          print("Please check that the image file exists and the path is correct")
          sys.exit(1)