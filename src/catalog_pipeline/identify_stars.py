import numpy as np
import sqlite3
import bisect
import os
from .db_operations import get_catalog_vector, get_star_info
from scipy.optimize import linear_sum_assignment  # Added for Hungarian algorithm

DB_PATH = os.path.join(os.path.dirname(__file__), 'star_catalog.db')
DB_PATH_TEST = os.path.join(os.path.dirname(__file__), 'star_catalog_small_test.db')
DB_PATH_TEST_500 = os.path.join(os.path.dirname(__file__), 'star_catalog_small_test_500_stars.db')
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


def angle_between(v1, v2):
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1, 1)))

def identify_stars_from_vector(detected_vectors, angle_tolerance=0.1, db_path=DB_PATH_TEST, limit=20):
     from collections import defaultdict
     import itertools
     """
     Identify stars based on triangle voting, given detected unit vectors.

     Arguments:
     - detected_vectors: list of np.array([x, y, z]) unit vectors
     - angle_tolerance: tolerance in degrees for triangle matching
     - db_path: path to the star catalog database

     Returns:
     - final_votes: dict {detected_star_idx: (HIP_id, vote_score)}
     """
     n = len(detected_vectors)
     votes = defaultdict(lambda: defaultdict(float))
     epsilon = 1e-5
     detected_vectors = detected_vectors[:6]
     n = len(detected_vectors)  # Update n to reflect actual number of vectors
     print(f"DEBUG: Using {n} detected stars (limited from original)")

     print(f"DEBUG: Processing {n} detected stars for triplet matching")
     print(f"DEBUG: Using database: {db_path}")

     # All combinations of 3 detected stars
     triplet_count = 0
     for (i, j, k) in itertools.combinations(range(n), 3):
          triplet_count += 1
          vi, vj, vk = detected_vectors[i], detected_vectors[j], detected_vectors[k]

          # Compute triangle side lengths
          a = angle_between(vi, vj)
          b = angle_between(vj, vk)
          c = angle_between(vk, vi)
          triplet_descriptor = tuple(sorted([a, b, c]))

          print(f"DEBUG: Triplet {triplet_count}: stars ({i},{j},{k}), angles: {a:.2f}, {b:.2f}, {c:.2f}")

          # Try to find catalog pairs close to these angles
          matches_ab = query_star_pairs(a, angle_tolerance, db_path=db_path, limit=limit)
          matches_bc = query_star_pairs(b, angle_tolerance, db_path=db_path, limit=limit)
          matches_ca = query_star_pairs(c, angle_tolerance, db_path=db_path, limit=limit)

          print(f"DEBUG: Found {len(matches_ab)}, {len(matches_bc)}, {len(matches_ca)} matches for angles")

          if not matches_ab or not matches_bc or not matches_ca:
               print(f"DEBUG: Skipping triplet due to missing matches")
               continue

          # Brute-force match catalog triangles by checking all combos
          valid_triangles = 0
          for (hipA1, hipB1, angle_ab) in matches_ab:
               for (hipB2, hipC1, angle_bc) in matches_bc:
                    for (hipC2, hipA2, angle_ca) in matches_ca:

                         # Try to find a consistent triangle by checking all possible star assignments
                         # We need to find three unique stars that form a triangle
                         possible_triangles = []
                         
                         # Try all combinations of star assignments from the three pairs
                         ab_pairs = [(hipA1, hipB1), (hipB1, hipA1)]  # both orientations
                         bc_pairs = [(hipB2, hipC1), (hipC1, hipB2)]  # both orientations  
                         ca_pairs = [(hipC2, hipA2), (hipA2, hipC2)]  # both orientations
                         
                         for (sA_ab, sB_ab) in ab_pairs:
                              for (sB_bc, sC_bc) in bc_pairs:
                                   for (sC_ca, sA_ca) in ca_pairs:
                                        # Check if we can form a consistent triangle: A-B, B-C, C-A
                                        if sB_ab == sB_bc and sC_bc == sC_ca and sA_ca == sA_ab:
                                             triangle = (sA_ab, sB_ab, sC_bc)
                                             if len(set(triangle)) == 3:  # three unique stars
                                                  possible_triangles.append(triangle)
                         
                         if not possible_triangles:
                              continue
                         
                         # Use the first valid triangle found
                         hipA1, hipB1, hipC1 = possible_triangles[0]

                         # Recompute actual triangle from catalog HIPs
                         try:
                              vA = get_catalog_vector(hipA1, db_path=db_path)
                              vB = get_catalog_vector(hipB1, db_path=db_path)
                              vC = get_catalog_vector(hipC1, db_path=db_path)
                         except Exception as e:
                              print(f"DEBUG: Failed to get catalog vectors for HIPs {hipA1}, {hipB1}, {hipC1}: {e}")
                              continue  # catalog vector not found

                         ca = angle_between(vA, vB)
                         cb = angle_between(vB, vC)
                         cc = angle_between(vC, vA)
                         catalog_descriptor = sorted([ca, cb, cc])

                         # Check if angles match within threshold
                         residual = sum([abs(x - y) for x, y in zip(triplet_descriptor, catalog_descriptor)])
                         if residual > angle_tolerance * 3:
                              continue  # angles too far off

                         valid_triangles += 1
                         print(f"DEBUG: Valid triangle found: HIPs {hipA1}, {hipB1}, {hipC1}, residual: {residual:.3f}")

                         # Try all 6 permutations of detected-to-catalog star mapping
                         for perm in itertools.permutations([(i, hipA1), (j, hipB1), (k, hipC1)]):
                              # Vote weighted by inverse residual
                              for (di, hi) in perm:
                                   votes[di][hi] += 1.0 / (residual + epsilon)

          print(f"DEBUG: Found {valid_triangles} valid catalog triangles for this detected triplet")

     print(f"DEBUG: Vote accumulation complete. Processing {len(votes)} detected stars with votes")

     # Final selection: HIP with highest votes per detected index
     final_votes = {}
     for i in votes:
          if votes[i]:
               hip, score = max(votes[i].items(), key=lambda x: x[1])
               final_votes[i] = (hip, score)
               print(f"DEBUG: Detected star {i} -> HIP {hip} with score {score:.3f}")

     print(f"DEBUG: Final triplet matches: {len(final_votes)} stars identified")
     return final_votes


def visualize_star_identification(image_path, star_data, matches, title="Star Identification Results"):
    """
    Visualize star identification results overlaid on the original image.
    
    Args:
        image_path: Path to the original image
        star_data: List of detected star data from detect_stars()
        matches: Dictionary of identification results from identify_stars_from_vector()
        title: Title for the plot
    """
    import matplotlib.pyplot as plt
    import cv2
    from collections import Counter
    
    # Read and display the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    plt.figure(figsize=(15, 10))
    plt.imshow(img, cmap='gray', origin='upper')
    plt.title(title)
    
    # Overlay star identifications
    for i, star in enumerate(star_data):
        pos_x, pos_y = star["position"]
        intensity = star["intensity"]
        
        if i in matches:
            hip_id, confidence = matches[i]
            # Identified stars - green circle with HIP ID and confidence
            plt.scatter(pos_x, pos_y, c='green', s=100, marker='o', alpha=0.7)
            plt.annotate(f'★{i}\nHIP {hip_id}\n{confidence:.1f}', 
                        (pos_x, pos_y), 
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7),
                        fontsize=8, color='white', weight='bold')
        else:
            # Unidentified stars - red circle with star index
            plt.scatter(pos_x, pos_y, c='red', s=80, marker='x', alpha=0.7)
            plt.annotate(f'{i}\nI={intensity:.0f}', 
                        (pos_x, pos_y), 
                        xytext=(10, -20), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                        fontsize=8, color='white')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label=f'Identified ({len(matches)} stars)'),
        Patch(facecolor='red', alpha=0.7, label=f'Unidentified ({len(star_data) - len(matches)} stars)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Add summary text
    identified_hips = [matches[i][0] for i in matches.keys()]
    unique_hips = set(identified_hips)
    
    summary_text = f"""Detection Summary:
Total Stars: {len(star_data)}
Identified: {len(matches)}
Unique HIPs: {len(unique_hips)}
HIP IDs: {sorted(unique_hips)}"""
    
    plt.text(0.02, 0.98, summary_text, transform=plt.gca().transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10)
    
    # Check for and display warnings
    warnings = []
    if len(identified_hips) != len(unique_hips):
        warnings.append("⚠️ Duplicate HIP assignments!")
        hip_counts = Counter(identified_hips)
        for hip, count in hip_counts.items():
            if count > 1:
                duplicate_stars = [i for i, (h, _) in matches.items() if h == hip]
                warnings.append(f"HIP {hip} → stars {duplicate_stars}")
    
    if matches:
        confidences = [confidence for _, confidence in matches.values()]
        unique_confidences = set(confidences)
        if len(unique_confidences) < len(confidences):
            warnings.append("⚠️ Identical confidence scores!")
    
    if warnings:
        warning_text = "\n".join(warnings)
        plt.text(0.02, 0.02, warning_text, transform=plt.gca().transAxes, 
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                fontsize=9, color='red', weight='bold')
    
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.tight_layout()
    plt.show()


def deprecated_identify_stars_from_vectors_2_0(detected_vectors, angle_tolerance=0.1, db_path=DB_PATH_TEST_500, limit=100):
     """
     Given detected star unit vectors (shape: N x 3),
     identify likely catalog matches for each detected star by matching pairwise angles.
     Uses an assignment matrix and Hungarian algorithm for optimal assignment.
     Returns: dict mapping detected star index to (catalog HIP, vote count)
     """
     import itertools
     detected_vectors = np.asarray(detected_vectors)
     n = detected_vectors.shape[0]
     print(f"DEBUG: Processing {n} detected stars")

     # Compute all unique pairs and their angles
     pairs = list(itertools.combinations(range(n), 2))
     detected_angles = []  # (i, j, angle)
     for i, j in pairs:
          v1 = detected_vectors[i]
          v2 = detected_vectors[j]
          dot = np.clip(np.dot(v1, v2), -1, 1)
          angle = np.degrees(np.arccos(dot))
          detected_angles.append((i, j, angle))
          print(f"DEBUG: Detected angle between star {i} and star {j}: {angle} degrees")

     catalog_star_ids = set() # this builds a hash set
     pair_matches = dict()  # (i, j) -> list of (star1_id, star2_id)
     total_catalog_matches = 0
     for i, j, angle in detected_angles:
          matches = query_star_pairs(angle, angle_tolerance, db_path=db_path, limit=limit)
          pair_matches[(i, j)] = matches # data format: (i, j) = [(61379, 68002, 14.242087679330531), (72378, 76945, 14.242087343190798), (31457, 33977, 14.242081929215967)
          print(f"the pair {i}-{j} has {len(matches)} matches")
          total_catalog_matches += len(matches)
          
          epsilon = 0.00002 # this will prevent division by zero
          scored_matches = []
          for star1_id, star2_id, catalog_angle in matches:
               residual = abs(angle - catalog_angle)
               weight = 1.0 / (residual + epsilon)
               scored_matches.append((star1_id, star2_id, catalog_angle, residual, weight))

          pair_matches[(i, j)] = scored_matches
          
          for star1_id, star2_id, _ in matches:
               catalog_star_ids.add(star1_id)
               catalog_star_ids.add(star2_id)
     catalog_star_ids = sorted(list(catalog_star_ids))
     print(f"DEBUG: Found {total_catalog_matches} total catalog pair matches, {len(catalog_star_ids)} unique catalog stars")

     # Create mappings between catalog star HIP IDs and matrix column indices
     # This allows us to convert between meaningful star names (HIP 61379) and matrix positions (column 0)
     hip_to_column_idx = {}  # Maps HIP ID -> matrix column index
     column_idx_to_hip = {}  # Maps matrix column index -> HIP ID
     
     hip_to_column_idx = {hip_id: idx for idx, hip_id in enumerate(catalog_star_ids)}
     column_idx_to_hip = {idx: hip_id for idx, hip_id in enumerate(catalog_star_ids)}
     # Build possible catalog star candidates for each detected star

     from collections import defaultdict
     # 1) degree of each detected star (how many pairs it's in)
     deg = [0]*n
     for i, j, _ in detected_angles:
         deg[i] += 1
         deg[j] += 1

     # 2) accumulate weighted evidence: possible[i][hip] = total score
     possible = defaultdict(lambda: defaultdict(float))

     for (i, j), scored in pair_matches.items():
          for hip1, hip2, _, _, w in scored: # scored = ((star1_id, star2_id, catalog_angle, residual, weight))
               di = max(deg[i], 1)
               dj = max(deg[j], 1)
               # orientation 1
               possible[i][hip1] += w / di
               possible[j][hip2] += w / dj
               # orientation 2 (swap)
               possible[i][hip2] += w / di
               possible[j][hip1] += w / dj

     possible = prune_possible(possible, max_hips=20)

     # map HIPs to columns
     # Union of all HIPs in pruned results
     selected_hips = sorted({hip for hip_scores in possible.values() for hip in hip_scores})
     hip_to_col = {hip: idx for idx, hip in enumerate(selected_hips)}
     col_to_hip = {idx: hip for hip, idx in hip_to_col.items()}

     num_detected = len(possible)
     num_catalog = len(hip_to_col)
     score_matrix = np.zeros((num_detected, num_catalog), dtype=np.float64)


     for i in range(num_detected):
          for hip, score in possible[i].items():
               if hip in hip_to_col:
                    j = hip_to_col[hip]
                    score_matrix[i, j] = score

     cost_matrix = -score_matrix
     print(f"DEBUG: Cost matrix: {cost_matrix}")
     from scipy.optimize import linear_sum_assignment
     rows, cols = linear_sum_assignment(cost_matrix)

     final_matches = {}

     for i, j in zip(rows, cols):
          hip = col_to_hip[j]
          score = score_matrix[i, j]
          final_matches[i] = (hip, score)
          print(f"Image star {i} matched to HIP {hip} with score {score:.2f}")

     print(f"DEBUG: Final matches: {final_matches}")
     return final_matches

def prune_possible(possible, max_hips=20):
     from collections import defaultdict
     pruned = defaultdict(dict)  # same structure as possible but pruned

     for i, hip_scores in possible.items():
          # Sort HIPs by descending score
          sorted_hips = sorted(hip_scores.items(), key=lambda x: -x[1])
          # Keep only the top `max_hips`
          for hip, score in sorted_hips[:max_hips]:
               pruned[i][hip] = score

     return pruned

def deprecated_identify_stars_from_vectors(detected_vectors, angle_tolerance=2.0, db_path=DB_PATH, limit=50):
    """
    Given detected star unit vectors (shape: N x 3),
    identify likely catalog matches for each detected star by matching pairwise angles.
    Uses an assignment matrix and Hungarian algorithm for optimal assignment.
    Returns: dict mapping detected star index to (catalog HIP, vote count)
    """
    import itertools
    detected_vectors = np.asarray(detected_vectors)
    n = detected_vectors.shape[0]
    print(f"DEBUG: Processing {n} detected stars")
    
    # Compute all unique pairs and their angles
    pairs = list(itertools.combinations(range(n), 2))
    detected_angles = []  # (i, j, angle)
    for i, j in pairs:
        v1 = detected_vectors[i]
        v2 = detected_vectors[j]
        dot = np.clip(np.dot(v1, v2), -1, 1)
        angle = np.degrees(np.arccos(dot))
        angle = round(angle, 2)
        detected_angles.append((i, j, angle))
    
    print(f"DEBUG: Computed {len(detected_angles)} pairwise angles")
    
    # For each detected pair, query catalog pairs
    # Build a set of all possible catalog star IDs from the matches
    catalog_star_ids = set() # this builds a hash set
    pair_matches = dict()  # (i, j) -> list of (star1_id, star2_id)
    total_catalog_matches = 0
    for i, j, angle in detected_angles:
        matches = query_star_pairs(angle, angle_tolerance, db_path=db_path, limit=limit)
        matches = [(star1_id, star2_id, round(catalog_angle, 2)) for star1_id, star2_id, catalog_angle in matches]
        pair_matches[(i, j)] = matches # data format: (i, j) = [(61379, 68002, 14.242087679330531), (72378, 76945, 14.242087343190798), (31457, 33977, 14.242081929215967)
        total_catalog_matches += len(matches)
        for star1_id, star2_id, _ in matches:
            catalog_star_ids.add(star1_id)
            catalog_star_ids.add(star2_id)
    catalog_star_ids = sorted(list(catalog_star_ids))
    m = len(catalog_star_ids)
    print(f"DEBUG: Found {total_catalog_matches} total catalog pair matches, {m} unique catalog stars")
    
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
    
    candidate_counts = [len(det_to_cat_candidates[i]) for i in range(n)]
    print(f"DEBUG: Candidate counts per detected star: {candidate_counts}")
    
    # Build assignment matrix: rows=detected stars, cols=catalog stars
    assignment_matrix = np.zeros((n, m), dtype=int)

    # Backtracking approach for efficient assignment search
    candidate_lists = [list(det_to_cat_candidates[i]) for i in range(n)]
    if all(candidate_lists):
        # Order detected stars by fewest candidates first
        det_order = sorted(range(n), key=lambda i: len(candidate_lists[i]))
        print(f"DEBUG: Processing detected stars in order: {det_order} (by candidate count)")
        max_assignments = EARLY_SUBGRAPH_STAR_EXIT  # Early exit limit
        assignments_checked = [0]  # Counter

        def backtrack(assignment, used_catalog_stars, depth):
            if assignments_checked[0] >= max_assignments:
                return
            if depth == n: #where n is the number of detected stars
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
                used_catalog_stars.add(cat_id) # This catalog star is now unavailable for other detected stars
                backtrack(assignment, used_catalog_stars, depth + 1)
                used_catalog_stars.remove(cat_id)
                del assignment[det_idx]

        backtrack(dict(), set(), 0)
        print(f"DEBUG: Backtracking completed, checked {assignments_checked[0]} assignments")
    else:
        # Fallback: no valid assignments, leave assignment_matrix as zeros
        pass

    # Hungarian algorithm finds minimum cost, so use negative of matches as cost
    cost_matrix = -assignment_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    print(f"DEBUG: Hungarian algorithm assigned {len(row_ind)} stars")

    # Prepare result: for each detected star, the assigned catalog star and the number of pairwise matches
    result = {}
    for det_idx, cat_col in zip(row_ind, col_ind):
        cat_id = idx_to_cat_id[cat_col]
        votes = assignment_matrix[det_idx, cat_col]
        result[det_idx] = [(cat_id, votes)]
    
    vote_counts = [result[i][0][1] if i in result and result[i] else 0 for i in range(n)]
    print(f"DEBUG: Final vote counts: {vote_counts}")
    return result

def get_identified_star_info(matches, db_path=DB_PATH):
    identified = {}
    for det_idx, ranked in matches.items():
        if ranked:
            # Handle both formats:
            # Format 1: [(hip_id, score), ...] - list of tuples (old format)
            # Format 2: (hip_id, score) - single tuple (new triplet format)
            if isinstance(ranked, list) and len(ranked) > 0:
                best_hip = ranked[0][0]  # First element of first tuple
            elif isinstance(ranked, tuple) and len(ranked) == 2:
                best_hip = ranked[0]  # First element of tuple
            else:
                print(f"Warning: Unexpected format for matches[{det_idx}]: {ranked}")
                identified[det_idx] = (None, None)
                continue
                
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
    matches = identify_stars_from_vector(detected_vectors, angle_tolerance=angle_tolerance, limit=20)
    for det_idx, ranked in matches.items():
        print(f"Detected star {det_idx}: candidates (catalog HIP, votes): {ranked}") 

    print("\n--- Identified Star Info (most likely match for each detected star) ---")
    identified = get_identified_star_info(matches)
    for det_idx, (hip, info) in identified.items():
        print(f"Detected star {det_idx}: HIP {hip}, info: {info}") 