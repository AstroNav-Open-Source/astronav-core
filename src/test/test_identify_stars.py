import unittest
from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from catalog_pipeline.db_operations import get_catalog_vector, get_pairs_by_angle_error_margin
from catalog_pipeline.quest import quest_algorithm
from image_pipeline.capture_star_vectors import calculate_angular_distances, detect_stars
from catalog_pipeline.radec_to_vec import radec_to_vec , vec_to_radec
from catalog_pipeline.identify_stars import deprecated_identify_stars_from_vectors, deprecated_identify_stars_from_vectors_2_0, identify_stars_from_vector, visualize_star_identification, query_star_pairs, DB_PATH, DB_PATH_TEST, DB_PATH_TEST_500


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
               angle_tolerance=0.1,
               db_path=DB_PATH_TEST_500,
               limit=20
               )
          
          print(f"Triplet matches: {matches}")
          # Visualize results on the image
          visualize_star_identification(image_path, star_data, matches)
     
     def test_identify_stars_with_image(self):
          image_path = "/Users/michaelcaneff/Documents/University/Sofia University /Space Challenges/space-treckers/src/test/test_images/5star_pairs_center.jpeg"
          #image_path = "/Users/michaelcaneff/Documents/University/Sofia University /Space Challenges/space-treckers/stellarium-005-demo-image.jpeg"
          img, thresh, star_data = detect_stars(image_path, visualize=False, fov_deg=66.3)
          detected_vectors = [star["vector"] for star in star_data]

          # identified_stars = deprecated_identify_stars_from_vectors_2_0(detected_vectors, angle_tolerance=0.1, limit=500)
          # print(f"Identified stars: {identified_stars}")

          matches = identify_stars_from_vector(
               detected_vectors,
               angle_tolerance=0.05,
               db_path=DB_PATH_TEST_500,
               limit=6
               )
          
          print(f"Triplet matches: {matches}")
          
          # Visualize results on the image
          visualize_star_identification(image_path, star_data, matches)

     def test_identify_stars_with_image_test_raw_image(self):
          # Use the correct database path from the catalog_pipeline module
          from catalog_pipeline.identify_stars import DB_PATH
          image_path = "/Users/michaelcaneff/Documents/University/Sofia University /Space Challenges/space-treckers/src/test/test_images/stellarium-test-set1.jpeg"
          img, thresh, star_data = detect_stars(image_path, visualize=False, fov_deg=66)
          detected_vectors = [star["vector"] for star in star_data]

          # identified_stars = deprecated_identify_stars_from_vectors_2_0(detected_vectors, angle_tolerance=0.1, limit=500)
          # print(f"Identified stars: {identified_stars}")

          matches = identify_stars_from_vector(
               detected_vectors,
               angle_tolerance=0.6,
               db_path=DB_PATH,
               limit=10
               )
          # matches = deprecated_identify_stars_from_vectors(detected_vectors, angle_tolerance=0.1, db_path=DB_PATH, limit=5)
          # print(f"Triplet matches: {matches}")
          visualize_star_identification(image_path, star_data, matches)




     # def test_identify_stars_with_expected_vectors(self):
     #      # image_path = "/Users/michaelcaneff/Documents/University/Sofia University /Space Challenges/space-treckers/src/test/test_images/5star_pairs_center.jpeg"
     #      # img, thresh, star_data = detect_stars(image_path, visualize=False, fov_deg=66.3)
     #      # detected_vectors = [star["vector"] for star in star_data]
     #      # # for vec in detected_vectors:
     #      #      print(f"Detected vectors: {vec}")

     #      # ra_dec_pairs = [vec_to_radec(vec) for vec in detected_vectors]
     #      # for ra, dec in ra_dec_pairs:
     #      #      print(f"RA/Dec pairs: {ra}, {dec}")

     #      # identified_stars = identify_stars_from_vectors(detected_vectors, angle_tolerance=2.0)
     #      # print(f"Identified stars: {identified_stars}")
          
     #      # Expected stars from the image
     #      expected_stars = {
     #          "Rigel": (78.63446353, -8.20163919),
     #          "Canopus": (95.98787763, -52.69571799), 
     #          "Adhara": (104.65644451, -28.97208931),
     #          "Wezen": (107.09785853, -26.39320776),
     #          "Sirius": (101.28854105, -16.71314306)
     #      }
          
     #      # print(f"\nExpected stars in image:")
     #      expected_vectors = []
     #      for name, (ra, dec) in expected_stars.items():
     #          vec = radec_to_vec(ra, dec)
     #          expected_vectors.append(vec)
              
     #      #     print(f"{name}: RA={ra:.2f}, Dec={dec:.2f}, Vec={vec}")

     #      # print(f"Testing with personal algorithm")
     #      # identified_stars = identify_stars_from_vectors_2_0(detected_vectors)
     #      # print(f"Identified stars: {identified_stars}")

     #      # # Test with correct vectors to see if identification works
     #      # print(f"\nTesting identification with correct celestial vectors:")
     #      # correct_identification = identify_stars_from_vectors(expected_vectors[:len(detected_vectors)], angle_tolerance=0.1)
     #      # print(f"Correct vector identification: {correct_identification}")
          
     #      # Test with noisy expected vectors to check robustness
     #      print(f"\nTesting identification with noisy expected vectors:")
     #      noisy_expected_vectors = add_gaussian_noise_to_vectors(expected_vectors, noise_std=0.00001)
     #      noisy_detected_vectors = identify_stars_from_vectors_2_0(noisy_expected_vectors, angle_tolerance=0.1)
     #      print(f"Noisy detected vectors: {noisy_detected_vectors}")
if __name__ == "__main__":
    unittest.main()
     