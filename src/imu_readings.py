import board
import busio
import adafruit_bno055
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from collections import deque

# Setup BNO055 sensor
i2c = board.I2C()
sensor = adafruit_bno055.BNO055_I2C(i2c)


# Data buffers
max_len = 100
x_vals = deque(maxlen=max_len)
yaw_vals = deque(maxlen=max_len)
roll_vals = deque(maxlen=max_len)
pitch_vals = deque(maxlen=max_len)

start_time = None
calibrated = False

# Plot setup
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
plt.grid(True)

# Animation update
def update(frame):
    global start_time, calibrated

    # Check calibration status
    sys_cal, gyro_cal, accel_cal, mag_cal = sensor.calibration_status

    if not calibrated:
        status_text.set_text(f"Calibrating... SYS:{sys_cal} GYRO:{gyro_cal} MAG:{mag_cal}")
        if sys_cal == 3 and gyro_cal == 3 and mag_cal == 3:
            calibrated = True
            start_time = time.time()
            print("✅ Calibration complete. Plotting started.")
        return line_yaw, line_roll, line_pitch, status_text

    # Once calibrated, start recording and plotting
    euler = sensor.euler
    if euler is None:
        return line_yaw, line_roll, line_pitch, status_text
    quaternion_imu = sensor.quaternion
    print(f"The quaternion is: {quaternion_imu}")
    yaw, roll, pitch = euler
    curr_time = time.time() - start_time

    x_vals.append(curr_time)
    yaw_vals.append(yaw)
    roll_vals.append(roll)
    pitch_vals.append(pitch)

    line_yaw.set_data(x_vals, yaw_vals)
    line_roll.set_data(x_vals, roll_vals)
    line_pitch.set_data(x_vals, pitch_vals)
    if len(x_vals) >= 100:
        ax.set_xlim(x_vals[0], x_vals[-1])
    #ax.set_xlim(max(0, curr_time - max_len), curr_time)
    status_text.set_text("Recording...")
    return line_yaw, line_roll, line_pitch, status_text

# Start animation
ani = animation.FuncAnimation(fig, update, interval=100)
plt.tight_layout()
plt.show()