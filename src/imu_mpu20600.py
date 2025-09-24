import smbus2
import time
import math
import threading

# === Shared IMU state ===
quaternion_latest = None
euler_latest = None
is_IMU_calibrated = False
sensor = None
start_time = None

# I²C address for MPU-20600 (yours was 0x69)
ADDR = 0x69
bus = smbus2.SMBus(1)

# MPU registers
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H  = 0x43

# Init sensor (wake up)
def setup_mpu20600():
    bus.write_byte_data(ADDR, PWR_MGMT_1, 0)
    return True

# --- Helpers ---
def read_word(reg):
    high = bus.read_byte_data(ADDR, reg)
    low = bus.read_byte_data(ADDR, reg+1)
    val = (high << 8) + low
    if val >= 0x8000:
        val = -((65535 - val) + 1)
    return val

def read_accel_gyro():
    ax = read_word(ACCEL_XOUT_H)
    ay = read_word(ACCEL_XOUT_H+2)
    az = read_word(ACCEL_XOUT_H+4)
    gx = read_word(GYRO_XOUT_H)
    gy = read_word(GYRO_XOUT_H+2)
    gz = read_word(GYRO_XOUT_H+4)
    ax_g = ax / 16384.0
    ay_g = ay / 16384.0
    az_g = az / 16384.0
    gx_dps = gx / 131.0
    gy_dps = gy / 131.0
    gz_dps = gz / 131.0
    return ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps

# === Calibration status (fake for MPU, always "good") ===
def is_calibrated(_=None):
    return True

def get_detailed_calibration_status():
    # No internal calib, return dummy "3"s
    return (3, 3, 3, 3)

def get_calibration_status():
    return is_IMU_calibrated

def get_latest_quaterlion():
    return quaternion_latest

# === Quaternion + Euler estimation (complementary filter) ===
def is_valid_quaternion(q):
    return q is not None and all(
        v is not None and not (isinstance(v, float) and math.isnan(v))
        for v in [q["w"], q["x"], q["y"], q["z"]]
    )

def imu_loop():
    global quaternion_latest, euler_latest, is_IMU_calibrated, start_time

    setup_mpu20600()
    is_IMU_calibrated = True
    start_time = time.time()
    print("✅ MPU-20600 IMU ready. Streaming data...")

    # complementary filter
    alpha = 0.98
    dt = 0.01
    pitch, roll, yaw = 0.0, 0.0, 0.0

    while True:
        try:
            ax, ay, az, gx, gy, gz = read_accel_gyro()
            accel_pitch = math.degrees(math.atan2(ax, math.sqrt(ay*ay + az*az)))
            accel_roll  = math.degrees(math.atan2(ay, math.sqrt(ax*ax + az*az)))

            pitch = alpha * (pitch + gx * dt) + (1 - alpha) * accel_pitch
            roll  = alpha * (roll  + gy * dt) + (1 - alpha) * accel_roll
            yaw   = yaw + gz * dt  # yaw will drift (no mag)

            # Store Euler
            euler_latest = (yaw, roll, pitch)

            # Store quaternion (basic conversion from Euler)
            cy = math.cos(math.radians(yaw) * 0.5)
            sy = math.sin(math.radians(yaw) * 0.5)
            cp = math.cos(math.radians(pitch) * 0.5)
            sp = math.sin(math.radians(pitch) * 0.5)
            cr = math.cos(math.radians(roll) * 0.5)
            sr = math.sin(math.radians(roll) * 0.5)

            qw = cr * cp * cy + sr * sp * sy
            qx = sr * cp * cy - cr * sp * sy
            qy = cr * sp * cy + sr * cp * sy
            qz = cr * cp * sy - sr * sp * cy

            quaternion_latest = {"w": qw, "x": qx, "y": qy, "z": qz}

        except Exception as ex:
            print(f"IMU read error: {ex}")

        time.sleep(dt)

def start_imu_daemon():
    thread = threading.Thread(target=imu_loop, daemon=True)
    thread.start()
