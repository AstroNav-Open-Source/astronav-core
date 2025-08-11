import sys
from pathlib import Path
import os
from datetime import datetime
from imu_readings import *
import time
from star_processing import process_star_image, capture_image 
from quaternion_calculations import propagate_orientation , quat_to_euler
import star_processing

# from imu_readings import get_quaternion , calibrate
DEFAULT_IMAGE_PATH = Path(__file__).parent / "image_pipeline" / "starfield.png"

def main(use_camera=False, image_path=DEFAULT_IMAGE_PATH):
     
     timer = time.time()
     if use_camera:
          print("Starting IMU Daemon...")
          start_imu_daemon()
          print("Waiting for IMU to calibrate...")
          while not get_calibration_status():
               print("Still calibrating... Waiting 1 second.")
               sys_cal, gyro_cal, accel_cal, mag_cal = get_detailed_calibration_status()
               print(f"Calibrating: SYS:{sys_cal} , GYRO: {gyro_cal} , ACCL: {accel_cal} MAG: {mag_cal}")
               time.sleep(1)

     quaternion_star, rotation_matrix = star_processing.process_star_image(use_camera=use_camera, visualize=False)
     print(f"Time taken: {time.time() - timer} seconds")
     if quaternion_star is not None and rotation_matrix is not None:
          print("Processing complete!")
          print("Quaternion:", quaternion_star)
          print("Rotation Matrix:", rotation_matrix)
     else:
          print("Failed to process image.")

#     delta_quaternion = calculate_delta_quaternion( quarterion_star, get_quaternion())
     if use_camera:
          from quaternion_calculations import quat2dict
          Q_STAR_REF = quat2dict(quaternion_star)
          Q_IMU_REF  = get_latest_quaterlion()   

          while True:
               Q_IMU_CURR = get_latest_quaterlion()  
               Q_BODY_CURR = propagate_orientation(Q_STAR_REF, Q_IMU_REF, Q_IMU_CURR)
               yaw, pitch, roll = quat_to_euler(Q_BODY_CURR)
               print(
                    f"Quaternion : {Q_BODY_CURR}   |   "
                    f"Yaw ψ={yaw:6.1f}°  Pitch θ={pitch:6.1f}°  Roll φ={roll:6.1f}°"
               )
               time.sleep(0.25)

if __name__ == "__main__":
     if len(sys.argv) > 1 and sys.argv[1] == "--capture":
          main(use_camera=True)
     else:
          main(use_camera=False)

     