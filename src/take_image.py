from picamzero import Camera
from time import sleep
import os
import keyboard
from datetime import datetime
import pwd

# Get actual user's Desktop path (even when running as root)
user_home = pwd.getpwnam("rpi").pw_dir  # Replace "pi" if your username is different
desktop_dir = os.path.join(user_home, "Desktop")
os.makedirs(desktop_dir, exist_ok=True)

# Initialize camera
cam = Camera()
      
print("Press [SPACE] to take a photo. Press [ESC] to exit.")
sleep(2)
cam.start_preview()

try:
    while True:
        if keyboard.is_pressed("space"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"{desktop_dir}/capture_{timestamp}.jpg"
            cam.take_photo(filepath)
            cam.stop_preview()
            print(f"Photo saved: {filepath}")
            sleep(1)  # Prevent accidental multiple captures
            cam.start_preview()  # Restart preview if needed

        elif keyboard.is_pressed("esc"):
            print("Exiting...")
            break

except KeyboardInterrupt:
    print("Program interrupted.")
    cam.stop_preview()
    exit(1)
cam.stop_preview()
exit(1)