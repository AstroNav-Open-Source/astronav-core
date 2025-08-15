from astropy.coordinates import angular_separation
import cv2
import numpy as np
from skimage import feature
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add the src directory to the path so we can import from image_pipeline
sys.path.append(str(Path(__file__).parent.parent))

from image_pipeline.capture_star_vectors import detect_stars
from image_pipeline.star_frame import StarFrame


def angular_distance_between_two_stars_isolated_img():
     # Load the image
     image_path = "src/test/test_images/testw2stars.png"
     img, thresh, star_data = detect_stars(image_path, visualize=False, fov_deg=20)
     
     # Extract just the vectors and flatten them
     vectors = [star["vector"].flatten() for star in star_data]
     print(f"Vectors: {vectors}")
     dot = np.dot(vectors[0], vectors[1])
     print(f"Dot product: {dot}")
     angle = np.arccos(np.clip(dot, -1, 1))  # clip to handle numerical precision
     print(f"Angle: {angle}")
     angle_deg = np.degrees(angle)
     print(f"Angle in degrees: {angle_deg}")
     return vectors


if __name__ == "__main__":
     angular_distance_between_two_stars_isolated_img()