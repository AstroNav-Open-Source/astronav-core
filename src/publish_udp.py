#!/usr/bin/env python3
"""
Raspberry Pi orientation publisher
Sends MPU-20600 IMU data via UDP to Mac relay server
"""

import json
import socket
import time
import math
import sys
from config import get_config, get_config_value

from imu_mpu20600 import start_imu_daemon, get_latest_quaterlion

# Configuration
config = get_config()
MAC_IP = get_config_value(config, 'network.udp_ip', "192.168.0.2")
MAC_PORT = get_config_value(config, 'network.udp_port', 9001)
PUBLISH_RATE_HZ = get_config_value(config, 'network.publish_rate_hz', 10)
FOV_DEGREES = get_config_value(config, 'star_processing.fov_degrees', 60.0)


class OrientationPublisher:
    def __init__(self, mac_ip: str, mac_port: int):
        self.mac_ip = mac_ip
        self.mac_port = mac_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Start IMU daemon (background thread)
        start_imu_daemon()
        print("? MPU-20600 IMU daemon started")

    def get_quaternion(self):
        """Get quaternion from IMU"""
        q = get_latest_quaterlion()
        if q:
            return (q["w"], q["x"], q["y"], q["z"])
        # fallback: identity quaternion
        return (1.0, 0.0, 0.0, 0.0)

    def publish_orientation(self):
        print(f"Starting orientation publisher...")
        print(f"Target: {self.mac_ip}:{self.mac_port}")
        print(f"Rate: {PUBLISH_RATE_HZ} Hz")
        print("Press Ctrl+C to stop")

        frame_time = 1.0 / PUBLISH_RATE_HZ

        try:
            while True:
                start_time = time.time()

                # Get quaternion
                w, x, y, z = self.get_quaternion()

                # Create message
                message = {
                    'q': [w, x, y, z],
                    'fov_deg': FOV_DEGREES,
                    'ts_unix_ms': int(time.time() * 1000)
                }

                try:
                    data = json.dumps(message).encode('utf-8')
                    self.socket.sendto(data, (self.mac_ip, self.mac_port))
                except Exception as e:
                    print(f"Send error: {e}")

                # Maintain frame rate
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_time - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.socket.close()

    def send_quaternion(self, w, x, y, z, extra_data=None):
        """Send a single quaternion over UDP."""
        message = {
            'q': [w, x, y, z],
            'fov_deg': FOV_DEGREES,
            'ts_unix_ms': int(time.time() * 1000)
        }
        if extra_data:
            message.update(extra_data)
        try:
            data = json.dumps(message).encode('utf-8')
            self.socket.sendto(data, (self.mac_ip, self.mac_port))
        except Exception as e:
            print(f"Send error: {e}")

def main():
    if len(sys.argv) > 1:
        mac_ip = sys.argv[1]
    else:
        mac_ip = MAC_IP
        print(f"Using default Mac IP: {mac_ip}")
        print("Usage: python publish_udp_mpu20600.py <mac_ip_address>")

    publisher = OrientationPublisher(mac_ip, MAC_PORT)
    publisher.publish_orientation()

if __name__ == "__main__":
    main()
