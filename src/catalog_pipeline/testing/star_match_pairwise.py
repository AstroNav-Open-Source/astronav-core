import numpy as np
from db_operations import get_pairs_by_angle

ANGLE_ERROR_MARGIN = 0.1  # percent, can be tuned

def match_stars_by_pairwise_angles(detected_angles, db_path='star_catalog.db', angle_error_margin=ANGLE_ERROR_MARGIN):
    if not detected_angles:
        return {}
        
    # Find all unique detected star indices
    detected_indices = set()
    for i, j, _ in detected_angles:
        detected_indices.add(i)
        detected_indices.add(j)
    votes = {idx: {} for idx in detected_indices}

    # Calculate the min and max angle range to query from the database
    angles = [a for _, _, a in detected_angles]
    min_detected_angle = min(angles)
    max_detected_angle = max(angles)
    
    # Get all possible pairs from the database in one go
    all_possible_matches = get_pairs_by_angle(
        min_detected_angle * (1 - angle_error_margin / 100),
        max_detected_angle * (1 + angle_error_margin / 100)
    )

    for i, j, angle in detected_angles:
        # Now filter the all_possible_matches in memory
        absolute_error = (angle_error_margin / 100) * angle
        min_angle = angle - absolute_error
        max_angle = angle + absolute_error
        
        matches = [
            (star1_id, star2_id) for star1_id, star2_id, cat_angle in all_possible_matches
            if min_angle <= cat_angle <= max_angle
        ]
        
        for star1_id, star2_id in matches:
            # Vote for both possible assignments (i->star1, j->star2) and (i->star2, j->star1)
            for (det_idx, cat_id) in [(i, star1_id), (j, star2_id), (i, star2_id), (j, star1_id)]:
                votes[det_idx][cat_id] = votes[det_idx].get(cat_id, 0) + 1

    # For each detected star, rank catalog candidates by votes
    ranked_matches = {}
    for det_idx, cat_votes in votes.items():
        ranked = sorted(cat_votes.items(), key=lambda x: -x[1])
        ranked_matches[det_idx] = ranked
    return ranked_matches

if __name__ == "__main__":
    detected_angles = [
        (0, 1, 25.0),
        (0, 2, 40.0),
        (1, 2, 30.0),
    ]
    matches = match_stars_by_pairwise_angles(detected_angles)
    for det_idx, ranked in matches.items():
        print(f"Detected star {det_idx}: candidates (catalog HIP, votes): {ranked}") 