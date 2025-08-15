import unittest
from pathlib import Path
import os
import sys
from datetime import datetime
import time
import numpy as np
from typing import Tuple, Optional, Dict, Any
from astropy.coordinates import SkyCoord
import astropy.units as u

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from star_processing import process_star_image, capture_image 
from quaternion_calculations import propagate_orientation, quat_to_euler, quat2dict, dict2quat, quat_to_euler_scipy, quaternion_angular_distance
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

     def get_right_ascension_and_declination_from_rotation_matrix_systematically(self, rot_matrix1, expected_ra=0.0, expected_dec=0.0):
          """
          Systematically test all 6 cardinal directions to find the actual camera boresight.
          """
          print(f"Rotation matrix:\n{rot_matrix1}")
          print(f"Expected: RA = {expected_ra}°, Dec = {expected_dec}°")
          
          # Test all 6 cardinal directions
          test_vectors = {
              "+X [1,0,0]": np.array([1.0, 0.0, 0.0]),
              "-X [-1,0,0]": np.array([-1.0, 0.0, 0.0]),
              "+Y [0,1,0]": np.array([0.0, 1.0, 0.0]),
              "-Y [0,-1,0]": np.array([0.0, -1.0, 0.0]),
              "+Z [0,0,1]": np.array([0.0, 0.0, 1.0]),
              "-Z [0,0,-1]": np.array([0.0, 0.0, -1.0])
          }
          
          print(f"\n{'Direction':<12} {'RA (°)':<8} {'Dec (°)':<8} {'RA Error':<10} {'Dec Error':<10} {'Total Error'}")
          print("-" * 70)
          
          best_direction = None
          best_error = float('inf')
          results = {}
          
          for name, body_vec in test_vectors.items():
              # Transform to inertial frame
              inertial_vec = rot_matrix1 @ body_vec
              inertial_vec = inertial_vec / np.linalg.norm(inertial_vec)
              
              # Convert to RA/Dec
              x, y, z = inertial_vec
              dec_rad = np.arcsin(np.clip(z, -1, 1))
              ra_rad = np.arctan2(y, x)
              
              ra_deg = np.degrees(ra_rad) % 360
              dec_deg = np.degrees(dec_rad)
              
              # Calculate errors
              ra_error = abs(ra_deg - expected_ra)
              if ra_error > 180:  # Handle wraparound
                  ra_error = 360 - ra_error
              dec_error = abs(dec_deg - expected_dec)
              total_error = np.sqrt(ra_error**2 + dec_error**2)
              
              results[name] = {
                  'ra': ra_deg,
                  'dec': dec_deg,
                  'ra_error': ra_error,
                  'dec_error': dec_error,
                  'total_error': total_error,
                  'vector': body_vec
              }
              
              print(f"{name:<12} {ra_deg:7.1f} {dec_deg:8.1f} {ra_error:9.1f} {dec_error:9.1f} {total_error:10.2f}")
              
              # Track best result
              if total_error < best_error:
                  best_error = total_error
                  best_direction = name
          
          print(f"\nBEST MATCH: {best_direction} with total error = {best_error:.2f}°")
          best_result = results[best_direction]
          print(f"Best result: RA = {best_result['ra']:.3f}°, Dec = {best_result['dec']:.3f}°")
          print(f"Best camera boresight vector: {best_result['vector']}")
          
          # Also show the calculated direction from inverse transformation
          print(f"\n--- CALCULATED OPTIMAL DIRECTION ---")
          expected_inertial = np.array([
              np.cos(np.radians(expected_dec)) * np.cos(np.radians(expected_ra)),
              np.cos(np.radians(expected_dec)) * np.sin(np.radians(expected_ra)), 
              np.sin(np.radians(expected_dec))
          ])
          calculated_body = rot_matrix1.T @ expected_inertial
          print(f"Calculated optimal body direction: {calculated_body}")
          
          # Verify calculated direction
          verification = rot_matrix1 @ calculated_body
          verification = verification / np.linalg.norm(verification)
          x, y, z = verification
          dec_rad = np.arcsin(np.clip(z, -1, 1))
          ra_rad = np.arctan2(y, x)
          ra_deg_calc = np.degrees(ra_rad) % 360
          dec_deg_calc = np.degrees(dec_rad)
          print(f"Calculated direction gives: RA = {ra_deg_calc:.3f}°, Dec = {dec_deg_calc:.3f}°")
          
          # Analyze the calculated direction to understand the camera tilt
          print(f"\n--- CAMERA TILT ANALYSIS ---")
          norm_calc = np.linalg.norm(calculated_body)
          unit_calc = calculated_body / norm_calc
          
          # Calculate tilt angles from +Z axis
          z_component = unit_calc[2]
          tilt_from_z = np.degrees(np.arccos(np.clip(abs(z_component), 0, 1)))
          
          # Calculate azimuth angle in X-Y plane
          azimuth = np.degrees(np.arctan2(unit_calc[1], unit_calc[0]))
          
          print(f"Camera tilt from +Z axis: {tilt_from_z:.2f}°")
          print(f"Tilt azimuth in X-Y plane: {azimuth:.2f}°")
          print(f"Normalized optimal direction: [{unit_calc[0]:.3f}, {unit_calc[1]:.3f}, {unit_calc[2]:.3f}]")
          
          # This explains why cardinal directions don't work well
          if tilt_from_z > 5:
              print(f"⚠️  Camera has significant tilt ({tilt_from_z:.1f}°) from optical axis")
              print(f"   This is why cardinal directions give large errors!")
              print(f"   For Stellarium images, this suggests a coordinate system issue!")
          else:
              print(f"✅ Camera is well-aligned with optical axis")
          
          # COORDINATE SYSTEM DIAGNOSIS for Stellarium images
          print(f"\n--- COORDINATE SYSTEM DIAGNOSIS ---")
          print(f"If these are Stellarium screenshots, the issue might be:")
          print(f"1. Y-axis flip: Image Y goes down, but camera Y should go up")
          print(f"2. FOV mismatch: Assumed 70° FOV vs actual Stellarium FOV")
          print(f"3. Image center offset: Stars not centered in screenshot")
          
          # Test Y-flip hypothesis
          flipped_calc = calculated_body.copy()
          flipped_calc[1] = -flipped_calc[1]  # Flip Y component
          
          verification_flip = rot_matrix1 @ flipped_calc
          verification_flip = verification_flip / np.linalg.norm(verification_flip)
          x_f, y_f, z_f = verification_flip
          dec_rad_f = np.arcsin(np.clip(z_f, -1, 1))
          ra_rad_f = np.arctan2(y_f, x_f)
          ra_deg_f = np.degrees(ra_rad_f) % 360
          dec_deg_f = np.degrees(dec_rad_f)
          
          print(f"With Y-flip: RA = {ra_deg_f:.3f}°, Dec = {dec_deg_f:.3f}°")
          
          return best_result['ra'], best_result['dec'], best_direction, best_result['vector']
     
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
     
     def test_180_degree_rotation(self):
          """Test if algorithm can correctly measure 180° rotation."""
          print("\n" + "="*50)
          print("Testing 180° Rotation Accuracy")
          print("="*50)
          
          # Use any available image
          test_images = self.discover_test_images()
          if test_images:
               # Use the first available image
               image_name = list(test_images.keys())[0]
               original_path = test_images[image_name]['path']
          if not original_path.exists():
               print(f"❌ No test image found at {original_path}")
               return
          
          print(f"Using image: {original_path}")
          
          # Create a 180° rotated version
          import cv2
          import numpy as np
          
          # Load and rotate image 180°
          img = cv2.imread(str(original_path))
          if img is None:
               print("❌ Could not load image")
               return
               
          rotated_img = cv2.rotate(img, cv2.ROTATE_180)
          
          # Save rotated image temporarily
          rotated_path = Path(__file__).parent / "temp_rotated_180.png"
          cv2.imwrite(str(rotated_path), rotated_img)
          
          print(f"Created 180° rotated image: {rotated_path}")
          
          # Test both images through the algorithm
          result_euler = self.calculate_rotational_difference(original_path, rotated_path)
          result_quaternion = self.test_quaternion_angular_distance(original_path, rotated_path)
          
          
          # Clean up temporary file
          if rotated_path.exists():
               rotated_path.unlink()
          
          if result_euler or result_quaternion:
               measured_diff = result_euler['max_diff']
               expected_diff = 180.0
               error = abs(measured_diff - expected_diff)
               
               print(f"\n180° Rotation Test Results:")
               print(f"  Expected: {expected_diff}°")
               print(f"  Measured: {measured_diff:.2f}°")
               print(f"  Error: {error:.2f}°")
               print(f"  Quaternion distance: {result_quaternion:.2f}°")
               
               if error < 10.0:
                    print(f"  SUCCESS: Algorithm correctly measured 180° rotation")
               else:
                    print(f"  FAILED: Algorithm cannot measure 180° rotation accurately")
               
               self.assertTrue(error < 10.0, f"Measured error {error:.2f}° exceeds tolerance for 180° rotation")

          else:
               print("❌ Test failed - could not calculate rotational difference")
     
     @unittest.skip("Skipping quaternion angular distance test")
     def calculate_rotational_difference(self, image1_path, image2_path):
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
          
          # Convert quaternions to dict format if they're numpy arrays
          if isinstance(quat1, np.ndarray):
               quat1_dict = quat2dict(quat1)
          else:
               quat1_dict = quat1
               
          if isinstance(quat2, np.ndarray):
               quat2_dict = quat2dict(quat2)
          else:
               quat2_dict = quat2
          
          # Convert to Euler angles (Yaw, Pitch, Roll in degrees)
          yaw1, pitch1, roll1 = quat_to_euler_scipy(quat1_dict)
          yaw2, pitch2, roll2 = quat_to_euler_scipy(quat2_dict)
          
          # Calculate angular differences
          yaw_diff = abs(yaw2 - yaw1)
          pitch_diff = abs(pitch2 - pitch1)
          roll_diff = abs(roll2 - roll1)
          
          # Normalize angles to 0-360 range
          yaw_diff = min(yaw_diff, 360 - yaw_diff)
          pitch_diff = min(pitch_diff, 360 - pitch_diff)
          roll_diff = min(roll_diff, 360 - roll_diff)
          
          print(f"Image 1 quaternion: {quat1}")
          print(f"Image 2 quaternion: {quat2}")
          print(f"Image 1 Euler: Yaw={yaw1:.2f}°, Pitch={pitch1:.2f}°, Roll={roll1:.2f}°")
          print(f"Image 2 Euler: Yaw={yaw2:.2f}°, Pitch={pitch2:.2f}°, Roll={roll2:.2f}°")


          print(f"\nRotational Differences:")
          print(f"  Yaw difference: {yaw_diff:.2f}°")
          print(f"  Pitch difference: {pitch_diff:.2f}°")
          print(f"  Roll difference: {roll_diff:.2f}°")
          print(f"  Maximum difference: {max(yaw_diff, pitch_diff, roll_diff):.2f}°")
          
          return {
               'yaw_diff': yaw_diff,
               'pitch_diff': pitch_diff,
               'roll_diff': roll_diff,
               'max_diff': max(yaw_diff, pitch_diff, roll_diff)
          }
          
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
     
     def test_rotational_accuracy_from_120RA_80DEC(self):
          """Test rotational accuracy from 120RA_80DEC."""
          # Get the test images
          test_images = self.discover_test_images()
          
          # For now, let's just print what we found
          print(f"\nTest images found: {len(test_images)}")
          test_image = test_images['120RA_80DEC']
          print(f"Test image: {test_image}")
          test_image_path = test_image['path']
          quat1, rot_matrix1 = lost_in_space(str(test_image_path), visualize=False)
          print(f"Quat1: {quat1}")
          ra_deg, dec_deg = self.get_right_ascension_and_declination_from_rotation_matrix(rot_matrix1)
          


     def test_rotational_accuracy_between_0_and_45_dec(self):
          """Test rotational accuracy between images."""
          # Get the test images
          test_images = self.discover_test_images()
          
          # For now, let's just print what we found
          print(f"\nTest images found: {len(test_images)}")
          for name, info in test_images.items():
               print(f"  {name}: RA={info['ra']}, DEC={info['dec']}")
          
          # Now let's test the rotational difference between the two images
          if len(test_images) >= 2:
               image1_key = '0RA_0DEC'
               image2_key = '0RA_45DEC'
               
               print(f"\nTesting rotational difference between:")
               print(f"  Image 1: {image1_key}")
               print(f"  Image 2: {image2_key}")

               result = self.test_quaternion_angular_distance(
                    test_images[image1_key]['path'],
                    test_images[image2_key]['path']
               )

               self.assertIsNotNone(result)
               expected_diff = 45.0
               tolerance = 5.0
               self.assertAlmostEqual(result, expected_diff, delta=tolerance)


     def test_rotational_accuracy_from_0ran_0dec(self):
          """Test rotational accuracy from 0RA_0DEC."""
          # Get the test images
          test_images = self.discover_test_images()
          
          # For now, let's just print what we found
          print(f"\nTest images found: {len(test_images)}")
          for name, info in test_images.items():
               print(f"  {name}: RA={info['ra']}, DEC={info['dec']}")
          # extract the test image from the test_images dictionary with the key 0RA_0DEC
          test_image = test_images['0RA_0DEC']
          print(f"Test image: {test_image}")

          test_image_path = test_image['path']
          quat1, rot_matrix1 = lost_in_space(str(test_image_path), visualize=False)
          print(f"Quat1: {quat1}")
          ra_deg, dec_deg = self.get_right_ascension_and_declination_from_rotation_matrix(rot_matrix1)


     def test_rotational_accuracy_from_0ran_45dec(self):
          """Test rotational accuracy from 0RA_45DEC."""
          # Get the test images
          test_images = self.discover_test_images()
          
          # For now, let's just print what we found
          print(f"\nTest images found: {len(test_images)}")
          test_image = test_images['0RA_45DEC']
          print(f"Test image: {test_image}")

          test_image_path = test_image['path']
          quat1, rot_matrix1 = lost_in_space(str(test_image_path), visualize=False)
          print(f"Quat1: {quat1}")
          _ , _ = self.get_right_ascension_and_declination_from_rotation_matrix(rot_matrix1)
          
if __name__ == "__main__":
     unittest.main()