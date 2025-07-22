import sqlite3

def get_pairs_by_angle(min_angle, max_angle, db_path='star_catalog.db', entries_limit=0):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('select star1_id, star2_id, angle from pairs where angle between ? and ?', (min_angle, max_angle))
    results = c.fetchall()
    conn.close()
    return results

print(get_pairs_by_angle(30, 30.0001))