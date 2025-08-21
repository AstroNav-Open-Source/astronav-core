#!/usr/bin/env python3

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from catalog_pipeline.identify_stars import identify_stars_from_vector, DB_PATH_TEST_500
from image_pipeline.capture_star_vectors import detect_stars

def test_limit_effect():
    print("=== Testing Limit Parameter Effect ===")
    
    # Test image path
    image_path = "/Users/michaelcaneff/Documents/University/Sofia University /Space Challenges/space-treckers/src/test/test_images/5star_pairs_center.jpeg"
    
    print(f"Detecting stars from: {image_path}")
    img, thresh, star_data = detect_stars(image_path, visualize=False, fov_deg=66.3)
    detected_vectors = [star["vector"] for star in star_data]
    print(f"Detected {len(detected_vectors)} stars")
    
    # Test different limit values
    limits_to_test = [5, 10, 20, 50, 100]
    
    for limit in limits_to_test:
        print(f"\n{'='*50}")
        print(f"Testing with limit = {limit}")
        print(f"{'='*50}")
        
        matches = identify_stars_from_vector(
            detected_vectors, #detected_vectors[:3],  # Just test first 3 stars for speed
            angle_tolerance=0.5,
            limit=limit,
            db_path=DB_PATH_TEST_500
        )
        
        print(f"RESULT with limit={limit}: {len(matches)} stars identified")
        for det_idx, (hip, score) in matches.items():
            print(f"  Star {det_idx} -> HIP {hip} (score: {score:.3f})")

if __name__ == "__main__":
    test_limit_effect() 