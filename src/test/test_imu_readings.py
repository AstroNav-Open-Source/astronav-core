import sys
import unittest
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parents[1]))

from imu_readings import start_imu_daemon, get_latest_quaterlion, get_calibration_status


class TestIMUReadings(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        try:
            start_imu_daemon()
        except Exception as e:
            print(f"Warning: IMU daemon failed to start: {e}")
    
    def test_quaternion_available(self):
        quat = get_latest_quaterlion()
        self.assertIsNotNone(quat)
        self.assertIn('w', quat)
        self.assertIn('x', quat)
        self.assertIn('y', quat)
        self.assertIn('z', quat)
    
    def test_quaternion_values(self):
        quat = get_latest_quaterlion()
        for key in ['w', 'x', 'y', 'z']:
            self.assertIsInstance(quat[key], (int, float))
    
    def test_calibration_status(self):
        status = get_calibration_status()
        self.assertIsInstance(status, bool)


if __name__ == "__main__":
    unittest.main()
