from astropy.coordinates import angular_separation
import cv2
import numpy as np
from skimage import feature
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add the src directory to the path so we can import from image_pipeline
sys.path.append(str(Path(__file__).parent.parent))

from image_pipeline.capture_star_vectors import calculate_angular_distances, detect_stars
from image_pipeline.star_frame import StarFrame


def angular_distance_between_two_stars_isolated_img():
     # Load the image
     image_path = "src/test/test_images/testw2stars.png"
     img, thresh, star_data = detect_stars(image_path, visualize=False, fov_deg=20)
     
     angular_pairs = calculate_angular_distances(star_data)
          # Display angular distance results
     if angular_pairs:
          print(f"\n--- Angular Distances Between Star Pairs ---")
          for pair in angular_pairs:
               print(f"Stars {pair['star_i']}-{pair['star_j']}: "
                    f"Pos1=({pair['position_i'][0]:.1f},{pair['position_i'][1]:.1f}) "
                    f"Pos2=({pair['position_j'][0]:.1f},{pair['position_j'][1]:.1f}) "
                    f"Angular Distance={pair['angle_deg']:.2f}°")


if __name__ == "__main__":
     angular_distance_between_two_stars_isolated_img()