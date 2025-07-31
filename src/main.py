from image_pipeline.capture_star_vectors import visualize_results
from catalog_pipeline.real_image_valuation import lost_in_space
import sys
from pathlib import Path
import os
from datetime import datetime

DEFAULT_IMAGE_PATH = Path(__file__).parent / "image_pipeline" / "starfield.png"

def capture_image():
    try:
        from picamzero import Camera
        from time import sleep
        import keyboard
        
        # Initialize camera
        cam = Camera()
        print("Press [SPACE] to take a photo. Press [ESC] to exit.")
        sleep(2)
        cam.start_preview()
        
        try:
            while True:
                if keyboard.is_pressed("space"):
                    # Create timestamp for unique filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    capture_dir = Path(__file__).parent / "captured_images"
                    capture_dir.mkdir(exist_ok=True)
                    filepath = capture_dir / f"capture_{timestamp}.jpg"
                    
                    # Take the photo
                    cam.take_photo(str(filepath))
                    cam.stop_preview()
                    print(f"Photo saved: {filepath}")
                    return filepath

                elif keyboard.is_pressed("esc"):
                    print("Capture cancelled.")
                    cam.stop_preview()
                    return None
                
                sleep(0.1)  # Reduce CPU usage
                
        except KeyboardInterrupt:
            print("Program interrupted.")
            cam.stop_preview()
            return None
            
    except ImportError:
        print("Camera module not available. Make sure you're running this on a Raspberry Pi with picamzero installed.")
        return None
    except Exception as e:
        print(f"Error capturing image: {str(e)}")
        if 'cam' in locals():
            cam.stop_preview()
        return None

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

def main(use_camera=False, image_path=DEFAULT_IMAGE_PATH):
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
            main(image_path=sys.argv[1])
    else:
        main()

     