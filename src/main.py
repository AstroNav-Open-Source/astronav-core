
import sys
from pathlib import Path
import os
from datetime import datetime

DEFAULT_IMAGE_PATH = Path(__file__).parent / "image_pipeline" / "starfield.png"

def capture_image():
    # Only import camera module when needed
    try:
        from take_image import take_image as tk
        filename = tk()
        return filename
    except ImportError:
        print("Camera module not available. You need to use this on the RBPi.")
        return None

def process_image(image_path=None):
    from image_pipeline.capture_star_vectors import visualize_results
    from catalog_pipeline.real_image_valuation import lost_in_space
    if image_path is None:
        image_path = DEFAULT_IMAGE_PATH
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
        image_path = capture_image()
        if image_path is None:
            print("Failed to capture image. Using default test image instead...")
            image_path = DEFAULT_IMAGE_PATH
        
    print(f"Processing image at: {image_path}")
    print("🔍 Processing image...")
    q, r = process_image(image_path)
    
    if q is not None and r is not None:
        print("Processing complete!")
        print("Quaternion:", q)
        print("Rotation Matrix:", r)
    else:
        print("Failed to process image.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--capture":
        main(use_camera=True)
    else:
        main(use_camera=False)

     