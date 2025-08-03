from picamera2 import Picamera2, Preview
import time
import os

# Initialize the camera
picam2 = Picamera2()
camera_config = picam2.create_still_configuration(
    main={"size": (1920, 1080)},
    lores={"size": (640, 480)},
    display="lores"
)
picam2.configure(camera_config)

# Start camera preview and camera
picam2.start_preview(Preview.QTGL)
picam2.start()
time.sleep(2)  # Let the camera warm up

# Loop for taking pictures on keypress
print("Press Enter to take a picture, or type 'q' then Enter to quit.")

counter = 1
output_dir = "photos"
os.makedirs(output_dir, exist_ok=True)

while True:
    user_input = input(">> ")
    if user_input.lower() == 'q':
        break

    filename = os.path.join(output_dir, f"image_{counter:03d}.jpg")
    picam2.capture_file(filename)
    print(f"Saved {filename}")
    counter += 1

picam2.stop_preview()
picam2.close()
print("Camera stopped.")
