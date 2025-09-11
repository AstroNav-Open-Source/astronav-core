import unittest
from pathlib import Path
import sys
from matplotlib import image
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from catalog_pipeline.db_operations import get_catalog_vector, get_pairs_by_angle_error_margin
from catalog_pipeline.quest import quest_algorithm
from image_pipeline.capture_star_vectors import calculate_angular_distances, detect_stars
from catalog_pipeline.radec_to_vec import radec_to_vec , vec_to_radec
from catalog_pipeline.identify_stars import deprecated_identify_stars_from_vectors, deprecated_identify_stars_from_vectors_2_0
from catalog_pipeline.identify_stars import identify_stars_from_vector, visualize_star_identification, query_star_pairs
from catalog_pipeline.identify_stars import DB_PATH, DB_PATH_TEST, DB_PATH_TEST_500, DB_PATH_VMAG_LESS_THAN_5


def add_gaussian_noise_to_vectors(vectors, noise_std=0.001):
    """
    Add small Gaussian noise to unit vectors and re-normalize them.
    
    Args:
        vectors: List or array of 3D unit vectors
        noise_std: Standard deviation of Gaussian noise (default: 0.001)
    
    Returns:
        List of noisy unit vectors
    """
    noisy_vectors = []
    for vec in vectors:
        # Add Gaussian noise to each component
        noise = np.random.normal(0, noise_std, 3)
        noisy_vec = vec + noise
        # Re-normalize to maintain unit vector property
        noisy_vec = noisy_vec / np.linalg.norm(noisy_vec)
        noisy_vectors.append(noisy_vec)
    return noisy_vectors


class TestRotationalAccuracyFromImages(unittest.TestCase):
     
     
     def test_identify_stars_with_image(self):
          image_path = "/Users/michaelcaneff/Documents/University/Sofia University /Space Challenges/space-treckers/src/test/test_images/5star_pairs_center.jpeg"
          #image_path = "/Users/michaelcaneff/Documents/University/Sofia University /Space Challenges/space-treckers/stellarium-005-demo-image.jpeg"
          img, thresh, star_data = detect_stars(image_path, visualize=False, fov_deg=66.3)
          detected_vectors = [star["vector"] for star in star_data]

          # identified_stars = deprecated_identify_stars_from_vectors_2_0(detected_vectors, angle_tolerance=0.1, limit=500)
          # print(f"Identified stars: {identified_stars}")

          matches = identify_stars_from_vector(
               detected_vectors,
               angle_tolerance=0.02,
               db_path=DB_PATH_TEST_500,
               limit=200
               )
          
          print(f"Triplet matches: {matches}")
          # Visualize results on the image
          visualize_star_identification(image_path, star_data, matches)
     
     def test_identify_stars_with_image(self):
          # image_path = "/Users/michaelcaneff/Documents/University/Sofia University /Space Challenges/space-treckers/src/test/test_images/5star_pairs_center.jpeg"
          image_path = "/Users/michaelcaneff/Pictures/Stellarium/stellarium-20250911-163208178.jpeg"
          img, thresh, star_data = detect_stars(image_path, visualize=False, fov_deg=66)
          star_data=star_data[:6]
          detected_vectors = [star["vector"] for star in star_data]

          # identified_stars = deprecated_identify_stars_from_vectors_2_0(detected_vectors, angle_tolerance=0.1, limit=500)
          # print(f"Identified stars: {identified_stars}")
          detected_vectors = detected_vectors[:6]
          matches = identify_stars_from_vector(
               detected_vectors,
               angle_tolerance=0.1,
               db_path=DB_PATH_VMAG_LESS_THAN_5,
               limit=1000
               )
          
          print(f"Triplet matches: {matches}")
          
          # Visualize results on the image
          visualize_star_identification(image_path, star_data, matches)
          

     def get_ra_dec_from_image(self, image_path):
          # image_path = "/Users/michaelcaneff/Documents/University/Sofia University /Space Challenges/space-treckers/src/test/test_images/5star_pairs_center.jpeg"
          image_path = "/Users/michaelcaneff/Pictures/Stellarium/stellarium-20250911-163208178.jpeg"
          img, thresh, star_data = detect_stars(image_path, visualize=True, fov_deg=66)
          star_data=star_data[:6]
          detected_vectors = [star["vector"] for star in star_data]

          # identified_stars = deprecated_identify_stars_from_vectors_2_0(detected_vectors, angle_tolerance=0.1, limit=500)
          # print(f"Identified stars: {identified_stars}")
          detected_vectors = detected_vectors[:6]
          matches = identify_stars_from_vector(
               detected_vectors,
               angle_tolerance=0.1,
               db_path=DB_PATH_VMAG_LESS_THAN_5,
               limit=1000
               )
          
          print(f"Triplet matches: {matches}")
          
          # Visualize results on the image
          visualize_star_identification(image_path, star_data, matches)

if __name__ == "__main__":
    unittest.main()
     