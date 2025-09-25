from pathlib import Path
from catalog_pipeline.real_image_valuation import lost_in_space
from image_pipeline.capture_star_vectors import visualize_results
from typing import Optional
from config import get_config, get_config_value

import subprocess
from pathlib import Path
import datetime


def capture_image(width=2304, height=1296):
    """
    Launches rpicam-still in --keypress mode with fixed resolution.
    Preview and capture run at the same resolution, so there's no pipeline re-init delay.
    
    Args:
        width (int): capture width
        height (int): capture height

    Returns:
        str | None: Path to saved image
    """
    output_dir = Path("photos").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"capture_{ts}.jpg"

    print("[INFO] Launching rpicam-still with preview & capture locked to same resolution.")
    print("       Click on preview window and press Enter to capture once.")
    print("       Press Esc or close window to quit without saving.")

    try:
        subprocess.run(
            [
                "rpicam-still",
                "--keypress",      # wait for Enter in preview window
                "-t", "0",         # run preview indefinitely
                "-o", str(filename),
                "--width", str(width),
                "--height", str(height),
                "--zsl"            # zero-shutter-lag buffer
            ],
            check=True,
        )
        print(f"[INFO] Saved {filename}")
        return str(filename)

    except subprocess.CalledProcessError:
        print("[ERROR] rpicam-still failed or was cancelled.")
        return None

def process_star_image(use_camera=False, visualize=True, image_path=None):
    # Load configuration
    config = get_config()
    
    if use_camera:
        image_path = capture_image()
        if image_path is None:
            # Fallback to config image path if camera fails
            image_path = get_config_value(config, 'general.image_path', 'src/photos/5star_pairs_center.jpeg')
    else:
        # Use provided image_path or fallback to config
        if image_path is None:
            image_path = get_config_value(config, 'general.image_path', 'src/photos/5star_pairs_center.jpeg')

    if isinstance(image_path, str):
        image_path = Path(image_path)
        # If the path is relative, make it relative to the project root
        if not image_path.is_absolute():
            # Get the project root (parent of src directory)
            project_root = Path(__file__).parent.parent
            image_path = project_root / image_path

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    quaternion, rotation_matrix = lost_in_space(str(image_path), visualize=visualize)
    return quaternion, rotation_matrix

if __name__ == "__main__":
     print("Star Processing, testing...")
     capture_image()