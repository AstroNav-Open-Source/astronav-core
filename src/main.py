import sys
from pathlib import Path
import os
from datetime import datetime
from imu_readings import *
import time
from star_processing import process_star_image, capture_image as sp
from quaternion_calculations import calculate_final_quaternion_to_transmit

# from imu_readings import get_quaternion , calibrate
DEFAULT_IMAGE_PATH = Path(__file__).parent / "image_pipeline" / "starfield.png"

def main(use_camera=False, image_path=DEFAULT_IMAGE_PATH):
     print("Starting IMU Daemon...")
     start_imu_daemon()
     print("Waiting for IMU to calibrate...")
     while not get_calibration_status():
          print("Still calibrating... Waiting 1 second.")
          sys_cal, gyro_cal, accel_cal, mag_cal = get_detailed_calibration_status()
          print(f"Calibrating: SYS:{sys_cal} , GYRO: {gyro_cal} , ACCL: {accel_cal} MAG: {mag_cal}")
          time.sleep(1)

     sp.capture_image()
     quaternion_star, rotation_matrix = sp.process_star_image(use_camera=use_camera, visualize=True)
     if quaternion_star is not None and rotation_matrix is not None:
          print("Processing complete!")
          print("Quaternion:", quaternion_star)
          print("Rotation Matrix:", rotation_matrix)
     else:
          print("Failed to process image.")

#     delta_quaternion = calculate_delta_quaternion( quarterion_star, get_quaternion())
     while True:
          print(f"Latest Quaternion from IMU {get_latest_quaterlion()}")
          final_quaternion = calculate_final_quaternion_to_transmit(quaternion_star, get_latest_quaterlion())
          time.sleep(1)

if __name__ == "__main__":
     if len(sys.argv) > 1 and sys.argv[1] == "--capture":
          main(use_camera=True)
     else:
          main(use_camera=False)

     