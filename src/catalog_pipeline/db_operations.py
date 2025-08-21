import sqlite3
import numpy as np

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

def ra_dec_to_unit_vector(ra_deg, dec_deg):
    """Convert right ascension and declination (in degrees) to a 3D unit vector."""
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.array([x, y, z])

def get_catalog_vector(hip_id, db_path='star_catalog.db'):
    """
    Given a HIP ID, return the 3D unit vector in inertial space from the RA/Dec.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT raicrs, deicrs FROM stars WHERE hip=?', (hip_id,))
    row = c.fetchone()
    conn.close()

    if row is None:
        raise ValueError(f"HIP ID {hip_id} not found in catalog.")

    ra, dec = float(row[0]), float(row[1])
    return ra_dec_to_unit_vector(ra, dec)

if __name__ == "__main__":
    print(get_pairs_by_angle_error_margin(30, 0.001, entries_limit=3))
    print(get_star_info(50935))
    print(get_angular_distance_between_stars(50935, 59774))