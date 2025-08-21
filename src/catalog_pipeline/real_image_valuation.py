import sys
import os
import numpy as np

MAX_OBSERVED_STAR_CALCS = 30

sys.path.append(os.path.join(os.path.dirname(__file__), '../image_pipeline'))
from image_pipeline.capture_star_vectors import detect_stars

from .identify_stars import identify_stars_from_vector, identify_stars_from_vector, get_identified_star_info
from .radec_to_vec import radec_to_vec

def lost_in_space(image_path, visualize=False, fov_deg=66):
    print(f"Processing image: {image_path}")
    img, thresh, star_data = detect_stars(image_path, visualize=visualize, fov_deg=fov_deg)
    print(f"Detected {len(star_data)} stars.")
    if len(star_data) < 2:
        print("Not enough stars detected for identification.")
        sys.exit(1)

    detected_vectors = [star["vector"] for star in star_data[:MAX_OBSERVED_STAR_CALCS]]

    angle_tolerance = 0.1  # degrees, can be tuned
    matches = identify_stars_from_vector(detected_vectors, angle_tolerance=angle_tolerance)
    identified = get_identified_star_info(matches)

    print("\n--- Identified Stars from Image ---")
    for det_idx, (hip, info) in identified.items(): 
        print(f"Detected star {det_idx}: HIP {hip}, info: {info}") 

    # Build body_vectors and inertial_vectors in matching order
    body_vectors = []
    inertial_vectors = []
    for i in range(len(detected_vectors)):
        hip, info = identified.get(i, (None, None))
        if info and len(info) > 0:
            raicrs, deicrs, vmag = info[0]
            body_vectors.append(detected_vectors[i])
            inertial_vectors.append(radec_to_vec(raicrs, deicrs))
        else:
            print(f"Warning: No catalog info for detected star {i}. Skipping.")
    body_vectors = np.array(body_vectors)
    inertial_vectors = np.array(inertial_vectors)

    if len(inertial_vectors) != len(body_vectors):
        min_len = min(len(inertial_vectors), len(body_vectors))
        print(f"Warning: Mismatch in vector counts after filtering. Using first {min_len} pairs.")
        body_vectors = body_vectors[:min_len]
        inertial_vectors = inertial_vectors[:min_len]

    from .quest import quest_algorithm, get_rotation_matrix_from_quaternion, test_quaternion_reprojection
    Q = quest_algorithm(body_vectors, inertial_vectors)
    print("\nQUEST quaternion (image to catalog):\n", Q) 

    R = get_rotation_matrix_from_quaternion(Q)
    print("\nQUEST rotational matrix\n", R)

    # print("\nQUEST quaternion error factor:")
    # test_quaternion_reprojection(Q, body_vectors, inertial_vectors, verbose=True)

    return Q, R

if __name__ == "__main__":
    lost_in_space("starfield-2.png")