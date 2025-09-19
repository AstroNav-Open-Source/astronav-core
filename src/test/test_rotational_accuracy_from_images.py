import unittest
from pathlib import Path
import os
import sys
from datetime import datetime
import time
import numpy as np
from typing import Tuple, Optional, Dict, Any
# from astropy.coordinates import SkyCoord  # Not used in remaining tests
# import astropy.units as u  # Not used in remaining tests

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

# from star_processing import process_star_image, capture_image  # Not used in remaining tests 
from quaternion_calculations import quat2dict, quaternion_angular_distance
from catalog_pipeline.real_image_valuation import lost_in_space


class TestRotationalAccuracyFromImages(unittest.TestCase):
     """
     Test class for evaluating rotational accuracy between two star field images.
     """
     
     def setUp(self):
          """Set up individual test environment."""
          self.test_images_dir = Path(__file__).parent / "test_images"
          self.test_images = self.discover_test_images()
          print(f"Found {len(self.test_images)} test images")
     
     def tearDown(self):
          """Clean up after individual test."""
          pass
     
     def get_right_ascension_and_declination_from_rotation_matrix(self, rot_matrix1):
          """
          Get the right ascension and declination from a rotation matrix.
          """
          body_vec = np.array([0.0, 0.0, 1.0])
          inertial_vec = rot_matrix1 @ body_vec
          inertial_vec = inertial_vec / np.linalg.norm(inertial_vec)
          x, y, z = inertial_vec
          dec_rad = np.arcsin(np.clip(z, -1, 1))
          ra_rad = np.arctan2(y, x)
          ra_deg = np.degrees(ra_rad) % 360
          dec_deg = np.degrees(dec_rad)
          print(f"RA: {ra_deg}°, Dec: {dec_deg}°")
          return ra_deg, dec_deg

     
     def discover_test_images(self):
          """
          Automatically discover test images and organize them by RA/DEC coordinates.
          Returns a dictionary with organized test images.
          """
          test_images = {}
          
          if not self.test_images_dir.exists():
               print(f"Warning: Test images directory not found: {self.test_images_dir}")
               return test_images
          
          # Look for files with RA_DEC pattern in filename
          for image_file in self.test_images_dir.glob("*.jpeg"):
               filename = image_file.stem  # filename without extension
               
               # Try to parse RA and DEC from filename (e.g., "0RA_0DEC", "0RA_45DEC")
               if "RA" in filename and "DEC" in filename:
                    try:
                         # Extract RA and DEC values
                         ra_part = filename.split("RA")[0]
                         dec_part = filename.split("DEC")[0].split("_")[-1]
                         
                         ra = float(ra_part)
                         dec = float(dec_part)
                         
                         test_images[filename] = {
                              'path': image_file,
                              'ra': ra,
                              'dec': dec,
                              'filename': filename
                         }
                         
                         print(f"Found: {filename} -> RA: {ra}°, DEC: {dec}°")
                         
                    except ValueError as e:
                         print(f"     Warning: Could not parse coordinates from {filename}: {e}")
               else:
                    # For images without RA/DEC in filename, just store them
                    test_images[filename] = {
                         'path': image_file,
                         'ra': None,
                         'dec': None,
                         'filename': filename
                    }
                    print(f"Found: {filename} (no coordinates)")
          
          return test_images
     
     def test_get_ra_dec_from_image(self):
          """
          Get the right ascension and declination from an image.
          """
          image_path = "/Users/michaelcaneff/Pictures/Stellarium/stellarium-20250911-174306102.jpeg"
          quat1, rot_matrix1 = lost_in_space(image_path, visualize=False, fov_deg=66)
          ra_deg, dec_deg = self.get_right_ascension_and_declination_from_rotation_matrix(rot_matrix1)
          print(f"RA: {ra_deg}°, Dec: {dec_deg}°")




     def test_quaternion_angular_distance(self, image1_path, image2_path):
          """
          Calculate the rotational difference between two images.
          Returns the angular difference in degrees.
          """
          print(f"\nProcessing image 1: {image1_path.name}")
          quat1, rot_matrix1 = lost_in_space(str(image1_path), visualize=False)
          
          print(f"Processing image 2: {image2_path.name}")
          quat2, rot_matrix2 = lost_in_space(str(image2_path), visualize=False)
          
          if quat1 is None or quat2 is None:
               print("Failed to process one or both images")
               return None
          
          distance = quaternion_angular_distance(quat1, quat2)
          print(f"Distance: {distance}")
          return distance
     
          


          
if __name__ == "__main__":
     unittest.main()