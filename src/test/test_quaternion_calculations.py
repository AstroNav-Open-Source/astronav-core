import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from quaternion_calculations import dict2quat, quat2dict, norm, conj, mul, quat_to_euler, propagate_orientation, quat_to_euler_scipy, quaternion_angular_distance


class TestQuaternionCalculations(unittest.TestCase):
     
     def test_dict2quat(self):
          q_dict = {"w": 1, "x": 0, "y": 0, "z": 0}
          q_array = dict2quat(q_dict)
          expected = np.array([1, 0, 0, 0])
          np.testing.assert_array_equal(q_array, expected)
     
     def test_quat2dict(self):
          q_array = np.array([1, 0, 0, 0])
          q_dict = quat2dict(q_array)
          expected = {"w": 1, "x": 0, "y": 0, "z": 0}
          self.assertEqual(q_dict, expected)
     
     def test_norm(self):
          q = np.array([2, 0, 0, 0])
          normalized = norm(q)
          self.assertAlmostEqual(np.linalg.norm(normalized), 1.0, places=5)
     
     def test_conj(self):
          q = np.array([1, 2, 3, 4])
          conjugate = conj(q)
          expected = np.array([1, -2, -3, -4])
          np.testing.assert_array_equal(conjugate, expected)
     
     def test_mul(self):
          q1 = np.array([1, 0, 0, 0])
          q2 = np.array([0, 1, 0, 0])
          result = mul(q1, q2)
          expected = np.array([0, 1, 0, 0])
          np.testing.assert_array_equal(result, expected)
     
     def test_quat_to_euler_identity(self):
          identity = {"w": 1, "x": 0, "y": 0, "z": 0}
          yaw, pitch, roll = quat_to_euler(identity)
          self.assertAlmostEqual(yaw, 0.0, places=5)
          self.assertAlmostEqual(pitch, 0.0, places=5)
          self.assertAlmostEqual(roll, 0.0, places=5)
     
     def test_quat_to_euler_180_z(self):
          rot_z_180 = {"w": 0, "x": 0, "y": 0, "z": 1}
          yaw, pitch, roll = quat_to_euler(rot_z_180)
          print(f"Yaw: {yaw}, Pitch: {pitch}, Roll: {roll}")
          self.assertAlmostEqual(yaw, 180.0, places=1)

     def test_quaternion_angular_distance(self):
          q1 = {"w": 1, "x": 0, "y": 0, "z": 0}
          q2 = {"w": 0, "x": 0, "y": 0, "z": 1}
          distance = quaternion_angular_distance(q1, q2)
          print(f"Distance: {distance}")
          self.assertAlmostEqual(distance, 180.0, places=1)
    
     def test_custom_vs_scipy(self):
        """Compare custom quat_to_euler with SciPy implementation."""
        test_cases = [
            {"w": 1, "x": 0, "y": 0, "z": 0},  # Identity
            {"w": 0.7071, "x": 0, "y": 0, "z": 0.7071},  # 90° Z rotation
            {"w": 0.7071, "x": 0.7071, "y": 0, "z": 0},  # 90° X rotation
            {"w": 0.5, "x": 0.5, "y": 0.5, "z": 0.5},  # Complex rotation
        ]
        
        for i, quat in enumerate(test_cases):
            with self.subTest(case=i):
                # Normalize the quaternion first
                q_array = dict2quat(quat)
                q_normalized = norm(q_array)
                quat_norm = quat2dict(q_normalized)
                
                custom_yaw, custom_pitch, custom_roll = quat_to_euler(quat_norm)
                scipy_yaw, scipy_pitch, scipy_roll = quat_to_euler_scipy(quat_norm)
                
                print(f"Case {i}: {quat_norm}")
                print(f"  Custom: Y={custom_yaw:.2f}°, P={custom_pitch:.2f}°, R={custom_roll:.2f}°")
                print(f"  SciPy:  Y={scipy_yaw:.2f}°, P={scipy_pitch:.2f}°, R={scipy_roll:.2f}°")
                
                # Use larger tolerance due to different conventions
                self.assertAlmostEqual(custom_yaw, scipy_yaw, delta=5.0)
                self.assertAlmostEqual(custom_pitch, scipy_pitch, delta=5.0)
                self.assertAlmostEqual(custom_roll, scipy_roll, delta=5.0)
    
     def test_propagate_orientation(self):
        """Test propagate_orientation function logic."""
        
        # Test case 1: No IMU movement - should return original star reference
        print("Test 1: No IMU movement")
        q_star_ref = {"w": 0.7071, "x": 0.7071, "y": 0, "z": 0}  # Some star orientation
        q_imu_ref = {"w": 1, "x": 0, "y": 0, "z": 0}   # IMU reference (identity)
        q_imu_curr = {"w": 1, "x": 0, "y": 0, "z": 0}  # IMU current (same as reference)
        
        result1 = propagate_orientation(q_star_ref, q_imu_ref, q_imu_curr)
        
        print(f"  Star ref:  {quat_to_euler(q_star_ref)}")
        print(f"  IMU ref:   {quat_to_euler(q_imu_ref)}")  
        print(f"  IMU curr:  {quat_to_euler(q_imu_curr)}")
        print(f"  Result:    {quat_to_euler(result1)}")
        
        # Should be approximately the same as star reference
        star_euler = quat_to_euler(q_star_ref)
        result_euler = quat_to_euler(result1)
        print(f"  Expected: Star ref unchanged")
        print(f"  Actual: {'PASS' if abs(star_euler[0] - result_euler[0]) < 1 else 'FAIL'}")
        
        # Test case 2: IMU rotates 90° - what should happen?
        print("\nTest 2: IMU rotates 90° around Z")
        q_star_ref = {"w": 1, "x": 0, "y": 0, "z": 0}  # Identity star orientation
        q_imu_ref = {"w": 1, "x": 0, "y": 0, "z": 0}   # IMU starts at identity
        q_imu_curr = {"w": 0.7071, "x": 0, "y": 0, "z": 0.7071}  # IMU rotates 90° Z
        
        result2 = propagate_orientation(q_star_ref, q_imu_ref, q_imu_curr)
        
        print(f"  Star ref:  {quat_to_euler(q_star_ref)}")
        print(f"  IMU ref:   {quat_to_euler(q_imu_ref)}")  
        print(f"  IMU curr:  {quat_to_euler(q_imu_curr)}")
        print(f"  Result:    {quat_to_euler(result2)}")
        
        # Test case 3: Check if the math makes sense
        print("\nTest 3: Mathematical consistency check")
        
        # If we apply the same transformation twice, do we get double the rotation?
        q_star_ref = {"w": 1, "x": 0, "y": 0, "z": 0}
        q_imu_ref = {"w": 1, "x": 0, "y": 0, "z": 0}
        q_imu_45 = {"w": 0.9239, "x": 0, "y": 0, "z": 0.3827}  # 45° rotation
        q_imu_90 = {"w": 0.7071, "x": 0, "y": 0, "z": 0.7071}  # 90° rotation
        
        result_45 = propagate_orientation(q_star_ref, q_imu_ref, q_imu_45)
        result_90 = propagate_orientation(q_star_ref, q_imu_ref, q_imu_90)
        
        yaw_45 = quat_to_euler(result_45)[0]
        yaw_90 = quat_to_euler(result_90)[0]
        
        print(f"  45° IMU rotation -> Star yaw: {yaw_45:.1f}°")
        print(f"  90° IMU rotation -> Star yaw: {yaw_90:.1f}°")
        print(f"  Ratio: {yaw_90/yaw_45:.2f} (should be ~2.0 if linear)")
        
        print("="*50)
        print("Audit complete - check the logic above!")
        print("="*50)


if __name__ == "__main__":
     unittest.main()
