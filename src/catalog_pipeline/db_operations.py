import sqlite3

def get_pairs_by_angle(min_angle, max_angle, db_path='star_catalog.db', entries_limit=0):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    query = 'select star1_id, star2_id, angle from pairs where angle between ? and ?'
    params = (min_angle, max_angle)
    if(entries_limit > 0):
        query += "limit ?"
        params = (*params, entries_limit)
    c.execute(query, params)
    results = c.fetchall()
    conn.close()
    return results

# Here the error is relative (%)
def get_pairs_by_angle_error_margin(angle, error, db_path='star_catalog.db', entries_limit=0):
    absolute_error = (error/100) * angle
    return get_pairs_by_angle(angle - absolute_error, angle + absolute_error, db_path, entries_limit)

print(get_pairs_by_angle_error_margin(30, 0.001, entries_limit=3))

def get_angular_distance_between_stars(star1_id, star2_id, db_path='star_catalog.db'):
    import numpy as np
    from index_data import radec_to_vec
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT raicrs, deicrs FROM stars WHERE hip=?', (star1_id,))
    row1 = c.fetchone()
    c.execute('SELECT raicrs, deicrs FROM stars WHERE hip=?', (star2_id,))
    row2 = c.fetchone()
    conn.close()
    if row1 is None or row2 is None:
        raise ValueError(f"Star(s) not found: {star1_id}, {star2_id}")
    vec1 = radec_to_vec(row1[0], row1[1])
    vec2 = radec_to_vec(row2[0], row2[1])
    dot = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    angle_deg = np.degrees(angle_rad)
    return angle_deg