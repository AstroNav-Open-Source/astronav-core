from pathlib import Path
from catalog_pipeline.real_image_valuation import lost_in_space
from image_pipeline.capture_star_vectors import visualize_results
from typing import Optional
DEFAULT_IMAGE_PATH = Path(__file__).parent / "image_pipeline" / "starfield.png"

def capture_image():
    try:
        from take_image import take_image as tk
        return tk()
    except ImportError:
        print("[WARNING] Camera module not available. Using fallback image.")
        return None

def process_star_image(use_camera=False, visualize=True):
    if use_camera:
        image_path = capture_image()
        if image_path is None:
            image_path = DEFAULT_IMAGE_PATH
    else:
        image_path = DEFAULT_IMAGE_PATH

    if isinstance(image_path, str):
        image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    quaternion, rotation_matrix = lost_in_space(str(image_path), visualize=visualize)
    return quaternion, rotation_matrix

if __name__ == "__main__":
     print("Star Processing, testing...")

# def capture_image():
#     # Only import camera module when needed
#     try:
#         from take_image import take_image as tk
#         filename = tk()
#         return filename
#     except ImportError:
#         print("Camera module not available. You need to use this on the RBPi.")
#         return None

# def process_image(image_path=None):
#     from image_pipeline.capture_star_vectors import visualize_results
#     from catalog_pipeline.real_image_valuation import lost_in_space
#     if image_path is None:
#         image_path = DEFAULT_IMAGE_PATH
#     elif isinstance(image_path, str):
#         image_path = Path(image_path)
    
#     if not image_path.exists():
#         raise FileNotFoundError(f"Image file not found: {image_path}")
    
#     # Use the lost_in_space function which handles everything:
#     # - Detects stars
#     # - Identifies them in catalog
#     # - Calculates quaternion and rotation matrix
#     try:
#         quaternion, rotation_matrix = lost_in_space(str(image_path), visualize=True)
#         return quaternion, rotation_matrix
#     except Exception as e:
#         print(f"Error processing image: {str(e)}")
#         return None, None