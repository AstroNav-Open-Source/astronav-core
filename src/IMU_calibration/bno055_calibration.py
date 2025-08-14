import time
import json
import board
import busio
import adafruit_bno055
import importlib.resources as resources

# The folder where the calibration file lives
# Must be a Python package (contain __init__.py)
CAL_FILE = "bno055_calibration.json"


def run_calibration():
    """Run BNO055 calibration process and store data in package resources."""

    # STEP 1: Set up I²C connection to the sensor
    i2c = busio.I2C(board.SCL, board.SDA)
    sensor = adafruit_bno055.BNO055_I2C(i2c)

    def save_calibration():
        cal_data = {
            "accel_offset": sensor.offsets_accelerometer,
            "gyro_offset": sensor.offsets_gyroscope,
            "mag_offset": sensor.offsets_magnetometer,
            "accel_radius": sensor.radius_accelerometer,
            "mag_radius": sensor.radius_magnetometer
        }

        # Get path to file inside the package folder
        cal_path = resources.files(__package__) / CAL_FILE

        with cal_path.open("w") as f:
            json.dump(cal_data, f)
        print(f"Calibration saved to {cal_path}")

    def load_calibration():
        cal_path = resources.files(__package__) / CAL_FILE

        with cal_path.open("r") as f:
            cal_data = json.load(f)

        sensor.offsets_accelerometer = tuple(cal_data["accel_offset"])
        sensor.offsets_gyroscope = tuple(cal_data["gyro_offset"])
        sensor.offsets_magnetometer = tuple(cal_data["mag_offset"])
        sensor.radius_accelerometer = cal_data["accel_radius"]
        sensor.radius_magnetometer = cal_data["mag_radius"]

        print(f"Calibration loaded from {cal_path}")

    # STEP 2: Check if calibration file exists
    try:
        load_calibration()
        print("Using stored calibration. No need to spin the sensor!")
    except FileNotFoundError:
        print("No calibration file found. Please rotate the sensor to calibrate...")

        # STEP 3: Wait until sensor is fully calibrated
        while True:
            cal_status = sensor.calibration_status
            print("Calibration status:", cal_status)
            if all(x == 3 for x in cal_status):
                print("Fully calibrated!")
                save_calibration()
                break
            time.sleep(1)

    # STEP 4: Output orientation continuously
    while True:
        print("Heading, Roll, Pitch:", sensor.euler)
        time.sleep(0.5)
