import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from imu_readings import start_imu_daemon, quaternion_latest
import time

start_imu_daemon()


while True:
    # Later in your loop:
    print(quaternion_latest)
    time.sleep(1)
