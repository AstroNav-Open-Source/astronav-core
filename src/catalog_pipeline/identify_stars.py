import numpy as np
import sqlite3
import bisect
import scipy
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import os
from .db_operations import get_star_info
from scipy.optimize import linear_sum_assignment  # Added for Hungarian algorithm

DB_PATH = os.path.join(os.path.dirname(__file__), 'star_catalog.db')

EARLY_SUBGRAPH_STAR_EXIT = 10000

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
    Uses an assignment matrix and Hungarian algorithm for optimal assignment.
    Returns: dict mapping detected star index to (catalog HIP, vote count)
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
    # Build a set of all possible catalog star IDs from the matches
    catalog_star_ids = set() # this builds a hash set
    pair_matches = dict()  # (i, j) -> list of (star1_id, star2_id)
    for i, j, angle in detected_angles:
        matches = query_star_pairs(angle, angle_tolerance, db_path=db_path, limit=limit)
        pair_matches[(i, j)] = matches # data format: (i, j) = [(61379, 68002, 14.242087679330531), (72378, 76945, 14.242087343190798), (31457, 33977, 14.242081929215967)
        for star1_id, star2_id, _ in matches:
            catalog_star_ids.add(star1_id)
            catalog_star_ids.add(star2_id)
    catalog_star_ids = sorted(list(catalog_star_ids))
    m = len(catalog_star_ids)
    # Create a mapping from catalog star ID to index and vice versa
    cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(catalog_star_ids)}
    idx_to_cat_id = {idx: cat_id for idx, cat_id in enumerate(catalog_star_ids)} # Structure: {0: 124, 1: 145, 2: 154, 3: 207, 4: 330.. etc} -> {index: HIP}

    # Build possible catalog star candidates for each detected star
    from collections import defaultdict
    import itertools
    det_to_cat_candidates = defaultdict(set)
    for (i, j), matches in pair_matches.items():
        for star1_id, star2_id, _ in matches:
            det_to_cat_candidates[i].add(star1_id)
            det_to_cat_candidates[j].add(star2_id)
            det_to_cat_candidates[i].add(star2_id)
            det_to_cat_candidates[j].add(star1_id)
    
    # Build assignment matrix: rows=detected stars, cols=catalog stars
    assignment_matrix = np.zeros((n, m), dtype=int)

    # Backtracking approach for efficient assignment search
    candidate_lists = [list(det_to_cat_candidates[i]) for i in range(n)]
    if all(candidate_lists):
        # Order detected stars by fewest candidates first
        det_order = sorted(range(n), key=lambda i: len(candidate_lists[i]))
        max_assignments = EARLY_SUBGRAPH_STAR_EXIT  # Early exit limit
        assignments_checked = [0]  # Counter

        def backtrack(assignment, used_catalog_stars, depth):
            if assignments_checked[0] >= max_assignments:
                return
            if depth == n:
                # Check all pairs for consistency
                valid = True
                for (i, j), matches in pair_matches.items():
                    cat_i = assignment[i]
                    cat_j = assignment[j]
                    found = False
                    for star1_id, star2_id, _ in matches:
                        if (cat_i == star1_id and cat_j == star2_id) or (cat_i == star2_id and cat_j == star1_id):
                            found = True
                            break
                    if not found:
                        valid = False
                        break
                if valid:
                    for det_idx, cat_id in assignment.items():
                        assignment_matrix[det_idx, cat_id_to_idx[cat_id]] += 1
                assignments_checked[0] += 1
                return
            det_idx = det_order[depth]
            for cat_id in candidate_lists[det_idx]:
                if cat_id in used_catalog_stars:
                    continue
                # Partial consistency check: only check pairs with already assigned detected stars
                partial_valid = True
                for prev_depth in range(depth):
                    prev_det = det_order[prev_depth]
                    if (min(det_idx, prev_det), max(det_idx, prev_det)) in pair_matches:
                        matches = pair_matches[(min(det_idx, prev_det), max(det_idx, prev_det))]
                        cat_prev = assignment[prev_det]
                        found = False
                        for star1_id, star2_id, _ in matches:
                            if ((det_idx < prev_det and cat_id == star1_id and cat_prev == star2_id) or
                                (det_idx < prev_det and cat_id == star2_id and cat_prev == star1_id) or
                                (det_idx > prev_det and cat_prev == star1_id and cat_id == star2_id) or
                                (det_idx > prev_det and cat_prev == star2_id and cat_id == star1_id)):
                                found = True
                                break
                        if not found:
                            partial_valid = False
                            break
                if not partial_valid:
                    continue
                assignment[det_idx] = cat_id
                used_catalog_stars.add(cat_id)
                backtrack(assignment, used_catalog_stars, depth + 1)
                used_catalog_stars.remove(cat_id)
                del assignment[det_idx]

        backtrack(dict(), set(), 0)
    else:
        # Fallback: no valid assignments, leave assignment_matrix as zeros
        pass

    # Hungarian algorithm finds minimum cost, so use negative of matches as cost
    cost_matrix = -assignment_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Prepare result: for each detected star, the assigned catalog star and the number of pairwise matches
    result = {}
    for det_idx, cat_col in zip(row_ind, col_ind):
        cat_id = idx_to_cat_id[cat_col]
        votes = assignment_matrix[det_idx, cat_col]
        result[det_idx] = [(cat_id, votes)]
    return result

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