#!/usr/bin/env python3
"""
Raspberry Pi orientation publisher
Sends IMU data via UDP to Mac relay server
"""

import json
import socket
import time
import math
import sys
from typing import Tuple, Optional

# Configuration
MAC_IP = "192.168.0.2"
MAC_PORT = 9001
PUBLISH_RATE_HZ = 10
FOV_DEGREES = 40.0

# Try to import IMU libraries (install with: pip install adafruit-circuitpython-bno055)
try:
    import board
    import busio
    import adafruit_bno055
    HAS_IMU = True
    print("IMU libraries found - will use real sensor data")
except ImportError:
    HAS_IMU = False
    print("IMU libraries not found - using fake data")
    print("To use real IMU, install: pip install adafruit-circuitpython-bno055")

class OrientationPublisher:
    def __init__(self, mac_ip: str, mac_port: int):
        self.mac_ip = mac_ip
        self.mac_port = mac_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.imu = None
        
        # Initialize IMU if available
        if HAS_IMU:
            try:
                i2c = busio.I2C(board.SCL, board.SDA)
                self.imu = adafruit_bno055.BNO055_I2C(i2c)
                print("BNO055 IMU initialized successfully")
            except Exception as e:
                print(f"Failed to initialize IMU: {e}")
                print("Falling back to fake data")
                self.imu = None
    
    def get_quaternion(self) -> Tuple[float, float, float, float]:
        """Get quaternion from IMU or generate fake data"""
        if self.imu:
            try:
                # Get quaternion from BNO055 (returns w, x, y, z)
                quat = self.imu.quaternion
                if quat and all(q is not None for q in quat):
                    return quat
            except Exception as e:
                print(f"IMU read error: {e}")
        
        # Fallback to fake data
        return self.generate_fake_quaternion()
    
    def generate_fake_quaternion(self) -> Tuple[float, float, float, float]:
        """Generate smooth fake quaternion for testing"""
        t = time.time()
        
        # Primary rotation around Y axis (yaw)
        yaw_angle = 0.3 * t
        yaw_w = math.cos(yaw_angle / 2)
        yaw_y = math.sin(yaw_angle / 2)
        
        # Secondary rotation around X axis (pitch)
        pitch_angle = 0.15 * math.sin(0.7 * t)
        pitch_w = math.cos(pitch_angle / 2)
        pitch_x = math.sin(pitch_angle / 2)
        
        # Combine rotations (quaternion multiplication)
        # q_combined = q_yaw * q_pitch
        w = yaw_w * pitch_w
        x = yaw_w * pitch_x
        y = yaw_y * pitch_w
        z = -yaw_y * pitch_x
        
        # Normalize
        length = math.sqrt(w*w + x*x + y*y + z*z)
        if length > 0:
            return (w/length, x/length, y/length, z/length)
        else:
            return (1.0, 0.0, 0.0, 0.0)
    
    def publish_orientation(self):
        """Main publishing loop"""
        print(f"Starting orientation publisher...")
        print(f"Target: {self.mac_ip}:{self.mac_port}")
        print(f"Rate: {PUBLISH_RATE_HZ} Hz")
        print(f"Using {'real IMU' if self.imu else 'fake data'}")
        print("Press Ctrl+C to stop")
        
        frame_time = 1.0 / PUBLISH_RATE_HZ
        
        try:
            while True:
                start_time = time.time()
                
                # Get orientation data
                w, x, y, z = self.get_quaternion()
                
                # Create message
                message = {
                    'q': [w, x, y, z],
                    'fov_deg': FOV_DEGREES,
                    'ts_unix_ms': int(time.time() * 1000)
                }
                
                # Add GPS data if available (placeholder)
                if hasattr(self, 'gps_data'):
                    message.update(self.gps_data)
                
                # Send UDP packet
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
        except Exception as e:
            print(f"Unexpected error: {e}")
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
        print("Usage: python publish_udp.py <mac_ip_address>")
    
    publisher = OrientationPublisher(mac_ip, MAC_PORT)
    publisher.publish_orientation()

if __name__ == "__main__":
    main()
