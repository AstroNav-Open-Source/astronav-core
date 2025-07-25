from image_pipeline.capture_star_vectors import visualize_results
from catalog_pipeline.real_image_valuation import lost_in_space
import sys
from pathlib import Path

def process_image(image_path=None):
    if image_path is None:
        image_path = Path(__file__).parent / "image_pipeline" / "starfield.png"
    else:
        image_path = Path(__file__).parent / image_path
    
    if not image_path.exists():
        raise FileNotFoundError(f"❌ Image file not found: {image_path}")
    
    # Use the lost_in_space function which handles everything:
    # - Detects stars
    # - Identifies them in catalog
    # - Calculates quaternion and rotation matrix
    quaternion, rotation_matrix = lost_in_space(str(image_path), visualize=True)
    return quaternion, rotation_matrix

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    q, r = process_image(image_path)

     