import numpy as np
import sqlite3
import bisect
import os
from db_operations import get_star_info

DB_PATH = os.path.join(os.path.dirname(__file__), 'star_catalog.db')

# def get_all_angles(db_path=DB_PATH):
#     conn = sqlite3.connect(db_path)
#     c = conn.cursor()
#     c.execute('SELECT angle FROM pairs ORDER BY angle ASC')
#     angles = [row[0] for row in c.fetchall()]
#     conn.close()
#     return np.array(angles)

# def build_kvector(angles):
#     # K-vector: for each angle, K[i] = number of angles <= angles[i]
#     # For a sorted array, K[i] = i
#     return np.arange(len(angles))

# def kvector_lookup(angles, K, theta_obs, delta_theta):
#     """Find index range in angles within [theta_obs - delta_theta, theta_obs + delta_theta] 
#     using K-vector (binary search)."""
#     theta_min = theta_obs - delta_theta
#     theta_max = theta_obs + delta_theta
#     # Use bisect to find indices
#     lower_idx = bisect.bisect_left(angles, theta_min)
#     upper_idx = bisect.bisect_right(angles, theta_max) - 1
#     return lower_idx, upper_idx

def query_star_pairs(theta_obs, delta_theta, db_path=DB_PATH, limit=50):
    """Query star pairs in the database within the given angle range, ordered by closeness to theta_obs."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''SELECT star1_id, star2_id, angle FROM pairs WHERE angle BETWEEN ? AND ? ORDER BY ABS(angle - ?) ASC LIMIT ?''',
              (theta_obs - delta_theta, theta_obs + delta_theta, theta_obs, limit))
    results = c.fetchall()
    conn.close()
    return results

def identify_stars_from_vectors(detected_vectors, angle_tolerance=0.1, db_path=DB_PATH, limit=50):
    """
    Given detected star unit vectors (shape: N x 3),
    identify likely catalog matches for each detected star by matching pairwise angles.
    Returns: dict mapping detected star index to list of (catalog HIP, vote count)
    """
    import itertools
    detected_vectors = np.asarray(detected_vectors)
    n = detected_vectors.shape[0]
    # Compute all unique pairs and their angles
    pairs = list(itertools.combinations(range(n), 2))
    detected_angles = []  # (i, j, angle)
    for i, j in pairs:
        v1 = detected_vectors[i]
        v2 = detected_vectors[j]
        dot = np.clip(np.dot(v1, v2), -1, 1)
        angle = np.degrees(np.arccos(dot))
        detected_angles.append((i, j, angle))

    # For each detected pair, query catalog pairs
    votes = {idx: {} for idx in range(n)}
    for i, j, angle in detected_angles:
        matches = query_star_pairs(angle, angle_tolerance, db_path=db_path, limit=limit)
        for star1_id, star2_id, cat_angle in matches:
            # Vote for both possible assignments (i->star1, j->star2) and (i->star2, j->star1)
            for (det_idx, cat_id) in [(i, star1_id), (j, star2_id), (i, star2_id), (j, star1_id)]:
                votes[det_idx][cat_id] = votes[det_idx].get(cat_id, 0) + 1

    # For each detected star, rank catalog candidates by votes
    ranked_matches = {}
    for det_idx, cat_votes in votes.items():
        ranked = sorted(cat_votes.items(), key=lambda x: -x[1])
        ranked_matches[det_idx] = ranked
    return ranked_matches

def get_identified_star_info(matches, db_path=DB_PATH):
    identified = {}
    for det_idx, ranked in matches.items():
        if ranked:
            best_hip = ranked[0][0]
            info = get_star_info(best_hip, db_path=db_path)
            identified[det_idx] = (best_hip, info)
        else:
            identified[det_idx] = (None, None)
    return identified

# --- Example usage for 4 detected stars ---
if __name__ == "__main__":
    # # Build or load K-vector and angles
    # print("Loading angles from database...")
    # angles = get_all_angles()
    # K = build_kvector(angles)
    # print(f"Loaded {len(angles)} angles.")

    # Example observed angle and tolerance
    theta_obs = 30.0
    delta_theta = 0.8

    # # K-vector lookup (index range)
    # lower_idx, upper_idx = kvector_lookup(angles, K, theta_obs, delta_theta)
    # print(f"K-vector index range: {lower_idx} to {upper_idx}")
    # print(f"Angle range: {angles[lower_idx]:.4f} to {angles[upper_idx]:.4f}")

    # Query star pairs from DB
    matches = query_star_pairs(theta_obs, delta_theta, limit=10)
    print("Top matches (star1_id, star2_id, angle):")
    for row in matches:
        print(row) 

    print("\n--- Star Identification Example (4 detected stars) ---")
    detected_vectors = [
        [0.38062349, 0.02668739, 0.92434493],
        [0.07121133, 0.16709064, 0.9833665],
        [0.42358393, -0.21844947,  0.87912256],
        [-0.0504477,  -0.15321152,  0.98690489],
    ]
    angle_tolerance = 0.1
    matches = identify_stars_from_vectors(detected_vectors, angle_tolerance=angle_tolerance, limit=20)
    for det_idx, ranked in matches.items():
        print(f"Detected star {det_idx}: candidates (catalog HIP, votes): {ranked}") 

    print("\n--- Identified Star Info (most likely match for each detected star) ---")
    identified = get_identified_star_info(matches)
    for det_idx, (hip, info) in identified.items():
        print(f"Detected star {det_idx}: HIP {hip}, info: {info}") 