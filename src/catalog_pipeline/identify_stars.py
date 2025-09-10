from collections import defaultdict
import itertools
import numpy as np
import sqlite3
import bisect
import os
from .db_operations import get_angular_distance_between_stars, get_catalog_vector, get_star_info
from scipy.optimize import linear_sum_assignment  # Added for Hungarian algorithm

DB_PATH = os.path.join(os.path.dirname(__file__), 'star_catalog.db')
DB_PATH_TEST = os.path.join(os.path.dirname(__file__), 'star_catalog_small_test.db')
DB_PATH_TEST_500 = os.path.join(os.path.dirname(__file__), 'star_catalog_small_test_500_stars.db')
EARLY_SUBGRAPH_STAR_EXIT = 10000


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

from collections import defaultdict
import itertools

def identify_stars_from_vector(detected_vectors, angle_tolerance=0.1, db_path=DB_PATH_TEST_500, limit=50):
     """
     Identify stars based on triangle+pyramid voting.
     
     Arguments:
     - detected_vectors: list of np.array([x, y, z]) unit vectors
     - angle_tolerance: tolerance in degrees for triangle matching
     - db_path: path to the star catalog database
     
     Returns:
     - final_votes: dict {detected_star_idx: [(HIP_id, score), ...]}
     """
     n = len(detected_vectors)
     # For debugging, limit number of stars used
     detected_vectors = detected_vectors[:6]
     seen_tris = set()
     seen_pyramids = set()
     
     print(f"DEBUG: Using {n} detected stars (limited from original)")
     print(f"DEBUG: Processing {n} detected stars for triplet matching")
     print(f"DEBUG: Using database: {db_path}")

     # Global support: obs_idx -> {hip_id: score}
     support = defaultdict(lambda: defaultdict(float))

     triplet_count = 0
     for (i, j, k) in itertools.combinations(range(n), 3):
          triplet_count += 1
          vi, vj, vk = detected_vectors[i], detected_vectors[j], detected_vectors[k]

          # Compute triangle side lengths
          a = angle_between(vi, vj)
          b = angle_between(vj, vk)
          c = angle_between(vk, vi)
          print(f"DEBUG: Triplet {triplet_count}: stars ({i},{j},{k}), angles: {a:.2f}, {b:.2f}, {c:.2f}")

          # DB queries
          matches_ab = query_star_pairs(a, angle_tolerance, db_path=db_path, limit=limit)
          matches_bc = query_star_pairs(b, angle_tolerance, db_path=db_path, limit=limit)
          matches_ca = query_star_pairs(c, angle_tolerance, db_path=db_path, limit=limit)
          print(f"DEBUG: Found {len(matches_ab)}, {len(matches_bc)}, {len(matches_ca)} matches for angles")

          if not matches_ab or not matches_bc or not matches_ca:
               print("DEBUG: Skipping triplet due to missing matches")
               continue

          ab_neighbors, bc_neighbors = defaultdict(set), defaultdict(set)
          for u, v, _ in matches_ab:
               ab_neighbors[u].add(v); ab_neighbors[v].add(u)
          for u, v, _ in matches_bc:
               bc_neighbors[u].add(v); bc_neighbors[v].add(u)

          ca_pairs = {frozenset((u, v)) for u, v, _ in matches_ca}
          
          for u, v, _ in matches_ab:
               for (A, B) in ((u, v), (v, u)):
                    if B not in bc_neighbors:
                         continue
                    for C in bc_neighbors[B]:
                         if frozenset((C, A)) not in ca_pairs:
                              continue

                         tri_key = tuple(sorted((A, B, C)))
                         if tri_key in seen_tris:
                              continue
                         seen_tris.add(tri_key)
                         print(f"DEBUG: Consistent catalog triangle found: {tri_key}")
                         candidate_pyramid = {i: A, j: B, k: C}
                         for d in [x for x in range(n) if x not in (i, j, k)]:
                              vd = detected_vectors[d]
                              pyramids = compute_pyramid(candidate_pyramid, db_path, limit, seen_pyramids, vi, vj, vk, vd, d)
                              if not pyramids:
                                   continue
                              for mapping in pyramids:
                                   obs_vectors = {idx: detected_vectors[idx] for idx in mapping.keys()}
                                   star_scores = compute_rms_scores(mapping, obs_vectors, db_path)
                                   for obs_idx, hip in mapping.items():
                                        support[obs_idx][hip] += star_scores[obs_idx]
     
     final_votes = {}
     for obs_idx, hip_scores in support.items():
          if hip_scores:  # Only process if there are scores
               ranked = sorted(hip_scores.items(), key=lambda x: -x[1])
               final_votes[obs_idx] = ranked
               print(f"DEBUG: Obs {obs_idx} candidates: {ranked}")
     
     return final_votes

import math

def logsumexp(a, b):
     if a == -math.inf: return b
     if b == -math.inf: return a
     m = a if a > b else b
     return m + math.log1p(math.exp(-abs(a - b)))

def compute_gaussian_logscore(mapping, obs_vectors, db_path, sigma=0.2):
    """
    Compute per-star and total log-likelihood scores for a candidate pyramid.

    Args:
        mapping: dict {obs_idx: hip_id}
        obs_vectors: dict {obs_idx: np.array}, observed unit vectors
        db_path: path to catalog database
        sigma: expected measurement noise (in degrees)

    Returns:
        per_star_log: dict {obs_idx: log-likelihood sum over its edges}
        pyr_log: float, total log-likelihood of the pyramid
    """
    obs_indices = list(mapping.keys())
    edge_pairs = [(obs_indices[i], obs_indices[j]) 
                  for i in range(len(obs_indices)) 
                  for j in range(i+1, len(obs_indices))]

    per_star_log = {obs_idx: 0.0 for obs_idx in mapping.keys()}
    pyr_log = 0.0
    inv_2s2 = 1.0 / (2 * sigma * sigma)

    for (obs_u, obs_v) in edge_pairs:
        hip_u, hip_v = mapping[obs_u], mapping[obs_v]
        vu, vv = obs_vectors[obs_u], obs_vectors[obs_v]

        meas_angle = angle_between(vu, vv)   # degrees
        cat_angle  = get_angular_distance_between_stars(hip_u, hip_v, db_path=db_path)

        delta = meas_angle - cat_angle
        ll = -(delta * delta) * inv_2s2   # log-likelihood contribution

        pyr_log += ll
        per_star_log[obs_u] += ll
        per_star_log[obs_v] += ll

    return per_star_log, pyr_log

def compute_rms_scores(mapping, obs_vectors, db_path, epsilon=1e-2):
     """
     Compute per-star residual-based scores for a candidate pyramid.

     Args:
          mapping: dict {obs_idx: hip_id}, e.g. {0: 34444, 1: 39863, 2: 35264, 3: 45270}
          obs_vectors: dict {obs_idx: np.array}, observed unit vectors for each index
          db_path: path to catalog database (must have star pairs with HIPs + angle)
          epsilon: small term to avoid division by zero

     Returns:
          star_score: dict {obs_idx: score} with accumulated edge weights
     """
     residuals = {obs_idx: [] for obs_idx in mapping.keys()}
     obs_indices = list(mapping.keys())

     # Step 1. Build all edges (pairs of observed indices)
     edge_pairs = [(obs_indices[i], obs_indices[j]) 
                    for i in range(len(obs_indices)) 
                    for j in range(i+1, len(obs_indices))]

     for (obs_u, obs_v) in edge_pairs:
          hip_u, hip_v = mapping[obs_u], mapping[obs_v]

          # --- measured angle
          vu, vv = obs_vectors[obs_u], obs_vectors[obs_v]
          meas_angle = angle_between(vu, vv)  # in degrees

          # --- catalog angle
          cat_angle = get_angular_distance_between_stars(hip_u, hip_v, db_path=db_path)
          # query_catalog_angle should return precomputed HIP→HIP angular separation in degrees

          # --- residual and weight
          residual = abs(meas_angle - cat_angle)
          # --- append residual to both stars
          residuals[obs_u].append(residual)
          residuals[obs_v].append(residual)

     # Convert residual lists → RMS → score
     star_score = {}
     for obs_idx, res_list in residuals.items():
          if not res_list:
               star_score[obs_idx] = 0.0
               continue
          rms = (sum(r**2 for r in res_list) / len(res_list))**0.5
          star_score[obs_idx] = 1.0 / (rms + epsilon)

     return star_score

def compute_pyramid(candidate_pyramide, db_path, limit, seen_pyramids, vi, vj, vk, vd, d, angle_tolerance=0.5):
     """
     Try to confirm pyramid and return candidate mappings {obs_idx: HIP}.
     """
     print(f"DEBUG: Computing pyramid for {candidate_pyramide}")
     star_indices = list(candidate_pyramide.keys())
     i, j, k = star_indices[0], star_indices[1], star_indices[2]
     A, B, C = candidate_pyramide[i], candidate_pyramide[j], candidate_pyramide[k]

     # Apex angles
     d_i = angle_between(vd, vi)
     d_j = angle_between(vd, vj)
     d_k = angle_between(vd, vk)

     matches_di = query_star_pairs(d_i, angle_tolerance, db_path=db_path, limit=limit)
     matches_dj = query_star_pairs(d_j, angle_tolerance, db_path=db_path, limit=limit)
     matches_dk = query_star_pairs(d_k, angle_tolerance, db_path=db_path, limit=limit)
     print(f"DEBUG: Found {len(matches_di)}, {len(matches_dj)}, {len(matches_dk)} matches for apex angles")

     if not matches_di or not matches_dj or not matches_dk:
          print("DEBUG: Skipping pyramid due to missing matches")
          return None

     neighbors = defaultdict(set)
     for u, v, _ in matches_di:
          if u == A: neighbors[A].add(v)
          elif v == A: neighbors[A].add(u)
     if not neighbors[A]:
          return None

     for u, v, _ in matches_dj:
          if u == B: neighbors[B].add(v)
          elif v == B: neighbors[B].add(u)
     if not neighbors[B]:
          return None

     for u, v, _ in matches_dk:
          if u == C: neighbors[C].add(v)
          elif v == C: neighbors[C].add(u)
     if not neighbors[C]:
          print("DEBUG: No neighbors found for one or more base stars")
          return None

     common_d = neighbors[A] & neighbors[B] & neighbors[C]
     if not common_d:
          print("DEBUG: No consistent catalog apex D found")
          return None

     results = []
     for D in common_d:
          pyr_key = tuple(sorted((A,B,C,D)))  # order-independent key
          if pyr_key in seen_pyramids:
               continue
          seen_pyramids.add(pyr_key)

          print(f"DEBUG: ✅ Pyramid confirmed → base ({A}, {B}, {C}), apex {D}")
          mapping = {**candidate_pyramide, d: D}
          results.append(mapping)

     return results



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
          
          if i in matches and matches[i]:
               # Debug: print what we're working with
               print(f"DEBUG: matches[{i}] = {matches[i]}, type = {type(matches[i])}")
               # Take the best candidate (first in the ranked list)
               if isinstance(matches[i], list) and len(matches[i]) > 0:
                    hip_id, confidence = matches[i][0]
               else:
                    # Fallback: if it's not a list, treat it as a single tuple
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
     identified_hips = [matches[i][0][0] for i in matches.keys() if matches[i]]
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
                    duplicate_stars = [i for i, match_list in matches.items() 
                                   if match_list and match_list[0][0] == hip]
                    warnings.append(f"HIP {hip} → stars {duplicate_stars}")

     if matches:
          confidences = [confidence for match_list in matches.values() 
                         for _, confidence in match_list]
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