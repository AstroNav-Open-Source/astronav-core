import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from star_frame import StarFrame

def make_K_from_fov(w, h, fov_deg_x):
     """
     Build the camera intrinsic matrix from horizontal FOV.
     """
     fx = (w / 2.0) / np.tan(np.deg2rad(fov_deg_x / 2.0))
     fy = fx  # assume square pixels
     cx, cy = w / 2.0, h / 2.0
     K = np.array([ [fx, 0, cx],
                    [0,  fy, cy],
                    [0,   0,  1]], dtype=np.float64)
     return K

def pixels_to_unit_rays(pixels, K):
     """
     Convert 2D pixel coordinates to 3D unit direction vectors using the pinhole camera model.
     """
     fx, fy = K[0, 0], K[1, 1]
     cx, cy = K[0, 2], K[1, 2]
     rays = []

     for x, y in np.array(pixels).reshape(-1, 2):
          dx = (x - cx) / fx
          dy = (y - cy) / fy
          v = np.array([dx, dy, 1.0])
          rays.append(v / np.linalg.norm(v))

     return np.array(rays)

def calculate_angular_distances(star_data):
     """
     Calculate angular distances between all pairs of stars.
     """
     if len(star_data) < 2:
          return []
     
     angular_pairs = []
     vectors = [star["vector"] for star in star_data]
     
     for i in range(len(vectors)):
          for j in range(i + 1, len(vectors)):
               dot = np.dot(vectors[i], vectors[j])
               angle = np.arccos(np.clip(dot, -1, 1))
               angle_deg = np.degrees(angle)
               angular_pairs.append({
                    "star_i": i,
                    "star_j": j,
                    "position_i": star_data[i]["position"],
                    "position_j": star_data[j]["position"],
                    "angle_deg": angle_deg
               })
     
     return angular_pairs

def detect_stars(image_path, threshold_val=200, min_area=5, max_area=500, fov_deg= 66, visualize=False):
     """
     Detect stars in an image and convert their positions to 3D unit vectors in camera coordinates.
     """
     print(f"Detecting stars with fov_deg={fov_deg}")
     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
     if img is None:
          raise FileNotFoundError(f"Image '{image_path}' not found.")

     h, w = img.shape
     print(f"Image shape : {img.shape}")
     cx, cy = w // 2, h // 2  # image center



     # Blur and threshold
     blurred = cv2.GaussianBlur(img, (3, 3), 0)  # smaller kernel for speed
     thresh = np.where(blurred > threshold_val, 255, 0).astype(np.uint8)

     # Connected components = candidate star blobs
     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

     # Build intrinsic matrix
     K = make_K_from_fov(w, h, fov_deg)

     star_data = []
     for i in range(1, num_labels):  # skip background label 0
          area = stats[i, cv2.CC_STAT_AREA]
          if min_area <= area <= max_area:
               x, y = centroids[i]
               r = np.sqrt(area / np.pi)
               mask = (labels == i).astype(np.uint8)
               intensity = np.mean(img[mask == 1])

               # Project pixel to 3D unit ray
               vector = pixels_to_unit_rays([[x, y]], K)[0]  # Extract single vector from (1,3) array
               star_data.append({
                    "position": (x, y),
                    "radius": r,
                    "intensity": intensity,
                    "vector": vector
               })
               print(f"Star {len(star_data)-1}: Pos=({x:.2f}, {y:.2f}), Vec={vector}, I={intensity:.1f}")

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
          
          if stars:
               sf = StarFrame(stars)
               visualize_results(img, thresh, stars)
          else:
               print("No stars to process")
     except FileNotFoundError as e:
          print(f"Error: {e}")
          print("Please check that the image file exists and the path is correct")
          sys.exit(1)