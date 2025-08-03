from image_pipeline.capture_star_vectors import visualize_results
from catalog_pipeline.real_image_valuation import lost_in_space
import sys
from pathlib import Path
import os
from datetime import datetime

DEFAULT_IMAGE_PATH = Path(__file__).parent / "image_pipeline" / "starfield.png"

def capture_image():
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
    return filename

def process_image(image_path=None):
    if image_path is None:
        image_path = Path(__file__).parent / "image_pipeline" / "starfield.png"
    elif isinstance(image_path, str):
        image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Use the lost_in_space function which handles everything:
    # - Detects stars
    # - Identifies them in catalog
    # - Calculates quaternion and rotation matrix
    try:
        quaternion, rotation_matrix = lost_in_space(str(image_path), visualize=True)
        return quaternion, rotation_matrix
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None, None

def main(use_camera=True, image_path=DEFAULT_IMAGE_PATH):
    if use_camera:
        print("Capturing new image...")
        captured_path = capture_image()
        if captured_path is None:
            print("Failed to capture image. Exiting...")
            return
        image_path = captured_path
    
    print("🔍 Processing image...")
    q, r = process_image(image_path)
    
    if q is not None and r is not None:
        print("Processing complete!")
        print("Quaternion:", q)
        print("Rotation Matrix:", r)
    else:
        print("Failed to process image.")

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--capture":
            main(use_camera=True)
    else:
        main()

     