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