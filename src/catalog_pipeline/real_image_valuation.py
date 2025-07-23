import sys
import os
import numpy as np

# Import star detection from image pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), '../image_pipeline'))
from capture_star_vectors import detect_stars

# Import identification functions
from identify_stars import identify_stars_from_vectors, get_identified_star_info

if __name__ == "__main__":
    # You can change this to use sys.argv[1] for CLI usage
    image_path = "starfield.png"  # Replace with your image file
    print(f"Processing image: {image_path}")
    img, thresh, star_data = detect_stars(image_path)
    print(f"Detected {len(star_data)} stars.")
    if len(star_data) < 2:
        print("Not enough stars detected for identification.")
        sys.exit(1)

    # Extract unit vectors
    detected_vectors = [star["vector"] for star in star_data]
    # Identify stars
    angle_tolerance = 0.1  # degrees, can be tuned
    matches = identify_stars_from_vectors(detected_vectors, angle_tolerance=angle_tolerance, limit=20)
    identified = get_identified_star_info(matches)

    print("\n--- Identified Stars from Image ---")
    for det_idx, (hip, info) in identified.items():
        print(f"Detected star {det_idx}: HIP {hip}, info: {info}") 