import sys
from pathlib import Path
import os
from datetime import datetime
from imu_readings import *
import time
from star_processing import process_star_image, capture_image 
from quaternion_calculations import propagate_orientation , quat_to_euler
import star_processing
from config import get_config, get_config_value

import platform
if platform.system() == "Darwin":
     from publish_udp import OrientationPublisher, MAC_PORT, MAC_IP

# from imu_readings import get_quaternion , calibrate

def main(use_camera=False, image_path=None):
     # Load configuration
     config = get_config()
     
     # Override config with function parameters if provided
     if use_camera is not None:
          config['general']['use_camera'] = use_camera
     if image_path is not None:
          config['general']['image_path'] = str(image_path)
     
     # Extract commonly used config values
     visualize = get_config_value(config, 'general.visualize', False)
     imu_enabled = get_config_value(config, 'imu.enabled', True)
     imu_update_interval = get_config_value(config, 'imu.update_interval', 0.25)
     useIMU = get_config_value(config, 'tracking.imu_tracking', True)
     useIMGTracking = get_config_value(config, 'tracking.image_tracking', True)
     
     timer = time.time()
     if config['general']['use_camera']:
          print("Starting IMU Daemon...")
          start_imu_daemon()
          print("Waiting for IMU to calibrate...")
          while not get_calibration_status():
               print("Still calibrating... Waiting 1 second.")
               sys_cal, gyro_cal, accel_cal, mag_cal = get_detailed_calibration_status()
               print(f"Calibrating: SYS:{sys_cal} , GYRO: {gyro_cal} , ACCL: {accel_cal} MAG: {mag_cal}")
               time.sleep(1)

     # Get image path from config
     image_path = get_config_value(config, 'general.image_path', 'src/photos/5star_pairs_center.jpeg')
     
     quaternion_star, rotation_matrix = star_processing.process_star_image(
         use_camera=config['general']['use_camera'], 
         visualize=visualize,
         image_path=image_path
     )
     print(f"Time taken: {time.time() - timer} seconds")
     if quaternion_star is not None and rotation_matrix is not None:
          print("Processing complete!")
          print("Quaternion:", quaternion_star)
          print("Rotation Matrix:", rotation_matrix)
     else:
          print("Failed to process image.")

#     delta_quaternion = calculate_delta_quaternion( quarterion_star, get_quaternion())
     if config['general']['use_camera']:
               if useIMU:
                    from quaternion_calculations import quat2dict
                    Q_STAR_REF = quat2dict(quaternion_star)
                    Q_IMU_REF  = get_latest_quaterlion()   
 
                    publisher = OrientationPublisher(mac_ip=MAC_IP, mac_port=MAC_PORT)
                    while True:
                         Q_IMU_CURR = get_latest_quaterlion()  
                         Q_BODY_CURR = propagate_orientation(Q_STAR_REF, Q_IMU_REF, Q_IMU_CURR)
                         yaw, pitch, roll = quat_to_euler(Q_BODY_CURR)
                         print(
                              f"Quaternion : {Q_BODY_CURR}   |   "
                              f"Yaw ψ={yaw:6.1f}°  Pitch θ={pitch:6.1f}°  Roll φ={roll:6.1f}°"
                         )
                         publisher.send_quaternion(Q_BODY_CURR["w"], Q_BODY_CURR["x"], Q_BODY_CURR["y"], Q_BODY_CURR["z"])
                         time.sleep(imu_update_interval)
               if useIMGTracking:
                    pass
                    #put your tracking coder= calls here:
                    #CODE... 

if __name__ == "__main__":
     try:
          if len(sys.argv) > 1 and sys.argv[1] == "--capture":
               main(use_camera=True)
          elif len(sys.argv) > 1 and sys.argv[1] == "--test":
               from quaternion_calculations import quat_to_euler, quat2dict
               import numpy as np
               import time
               
               # Load config for test mode
               config = get_config()
               rotation_speed = get_config_value(config, 'test_mode.rotation_speed', 1.0)
               test_update_interval = get_config_value(config, 'test_mode.update_interval', 0.5)
               
               print("[TEST MODE] Simulating continuous rotation...")
               t = 0
               publisher = OrientationPublisher(mac_ip= MAC_IP, mac_port=MAC_PORT)
               while True:
                    angle_deg = 10 + t * rotation_speed  # Use configurable rotation speed
                    angle_rad = np.deg2rad(angle_deg)
                    w = np.cos(angle_rad/2)
                    x = 0.0
                    y = 0.0
                    z = np.sin(angle_rad/2)
                    simulated_quat = {"w": w, "x": x, "y": y, "z": z}
                    yaw, pitch, roll = quat_to_euler(simulated_quat)
                    print(f"[TEST MODE] Simulated Quaternion: {simulated_quat}")
                    print(f"[TEST MODE] Yaw ψ={yaw:6.1f}°  Pitch θ={pitch:6.1f}°  Roll φ={roll:6.1f}°")
                    publisher.send_quaternion(w, x, y, z)
                    time.sleep(test_update_interval)
                    t += 1
          else:
               main(use_camera=False)

     except KeyboardInterrupt:
          print("\nShutting down...")

     