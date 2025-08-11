
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import math
from collections import deque
import threading

# === Shared IMU state ===
quaternion_latest = None
euler_latest = None
is_IMU_calibrated = False
sensor = None

# === Plot buffers ===
max_len = 100
x_vals = deque(maxlen=max_len)
yaw_vals = deque(maxlen=max_len)
roll_vals = deque(maxlen=max_len)
pitch_vals = deque(maxlen=max_len)

start_time = None

# === Sensor setup ===
def setup_bno055():
    import board
    import adafruit_bno055
    i2c = board.I2C()
    return adafruit_bno055.BNO055_I2C(i2c)

# === Calibration check ===
def is_calibrated(sensor):
    sys_cal, gyro_cal, accel_cal, mag_cal = sensor.calibration_status
    return sys_cal == 3 and gyro_cal == 3 and mag_cal == 3

def get_detailed_calibration_status():
    sensor
    if sensor is not None:
        return sensor.calibration_status
    return (0, 0, 0, 0) 

def get_calibration_status():
    return is_IMU_calibrated

def get_latest_quaterlion():
    return quaternion_latest

# === Background IMU daemon ===
def is_valid_quaternion(q):
    return q is not None and all(
        v is not None and not (isinstance(v, float) and math.isnan(v))
        for v in [q[0], q[1], q[2], q[3]]
    )

def imu_loop():
    global sensor, quaternion_latest, euler_latest, is_IMU_calibrated, start_time
    sensor = setup_bno055()
    print("🔧 IMU daemon started... Waiting for calibration.")
    while not is_calibrated(sensor):
        time.sleep(0.5)
    is_IMU_calibrated = True
    start_time = time.time()
    print("✅ IMU calibrated. Streaming data.")
    while True:
        try:
            q = sensor.quaternion
            e = sensor.euler
            if is_valid_quaternion(q):
                quaternion_latest = {
                    "w": q[0],
                    "x": q[1],
                    "y": q[2],
                    "z": q[3]
                }
            else:
                print("IMU read returned invalid quaternion, skipping update.")
            if e is not None:
                euler_latest = e
        except Exception as ex:
            print(f"IMU read error: {ex}")
        time.sleep(0.5)

def start_imu_daemon():
    thread = threading.Thread(target=imu_loop, daemon=True)
    thread.start()

# === Plotting setup ===
def setup_plot():
    fig, ax = plt.subplots()
    line_yaw, = ax.plot([], [], label="Yaw (heading)", color='r')
    line_roll, = ax.plot([], [], label="Roll", color='g')
    line_pitch, = ax.plot([], [], label="Pitch", color='b')
    status_text = ax.text(0.5, 0.95, "", transform=ax.transAxes, ha="center")

    ax.set_ylim(-250, 250)
    ax.set_xlim(0, max_len)
    ax.set_title("Live Euler Angles from BNO055")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Degrees")
    ax.legend(loc="lower left")
    ax.grid(True)

    return fig, ax, line_yaw, line_roll, line_pitch, status_text

# === Plot update function ===
def update_plot(frame):
    sys_cal, gyro_cal, accel_cal, mag_cal = sensor.calibration_status
    if not is_IMU_calibrated or euler_latest is None:
        status_text.set_text(f"Calibrating: SYS:{sys_cal} , GYRO: {gyro_cal} , ACCL: {accel_cal} MAG: {mag_cal}")
        return line_yaw, line_roll, line_pitch, status_text

    yaw, roll, pitch = euler_latest
    curr_time = time.time() - start_time

    x_vals.append(curr_time)
    yaw_vals.append(yaw)
    roll_vals.append(roll)
    pitch_vals.append(pitch)

    line_yaw.set_data(x_vals, yaw_vals)
    line_roll.set_data(x_vals, roll_vals)
    line_pitch.set_data(x_vals, pitch_vals)

    if len(x_vals) >= max_len:
        ax.set_xlim(x_vals[0], x_vals[-1])

    status_text.set_text("Recording...")
    return line_yaw, line_roll, line_pitch, status_text

# === Run standalone or in orchestrator ===
if __name__ == "__main__":
    start_imu_daemon()
    fig, ax, line_yaw, line_roll, line_pitch, status_text = setup_plot()
    ani = animation.FuncAnimation(fig, update_plot, interval=100)
    plt.tight_layout()
    plt.show()
