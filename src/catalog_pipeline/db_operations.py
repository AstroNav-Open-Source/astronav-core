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

# Here the error is absolute (degrees)
def get_pairs_by_angle_error_margin(angle, error, db_path='star_catalog.db', entries_limit=0):
    return get_pairs_by_angle(angle - error, angle + error, db_path, entries_limit)

print(get_pairs_by_angle_error_margin(30, 0.001, entries_limit=3))

def get_angular_distance_between_stars(star1_id, star2_id, db_path='star_catalog.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # Try both (star1_id, star2_id) and (star2_id, star1_id) since order may not be guaranteed
    c.execute('SELECT angle FROM pairs WHERE star1_id=? AND star2_id=?', (star1_id, star2_id))
    row = c.fetchone()
    if row is None:
        c.execute('SELECT angle FROM pairs WHERE star1_id=? AND star2_id=?', (star2_id, star1_id))
        row = c.fetchone()
    conn.close()
    if row is None:
        raise ValueError(f"Pair not found in database: {star1_id}, {star2_id}")
    angle_deg = row[0]
    return angle_deg

def get_star_info(star_id, db_path='star_catalog.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    query = 'select raicrs, deicrs, vmag from stars where hip=?'
    params = (star_id, )
    c.execute(query, params)
    results = c.fetchall()
    conn.close()
    return results

print(get_star_info(50935))

print(get_angular_distance_between_stars(50935, 59774))