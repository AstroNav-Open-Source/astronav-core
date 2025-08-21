import numpy as np

# Convert RA/Dec to unit vectors for all catalog stars
def radec_to_vec(ra_deg, dec_deg):
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    return np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])

def vec_to_radec(vec):
    x, y, z = vec
    # Normalize to ensure it's a unit vector
    v_norm = np.sqrt(x**2 + y**2 + z**2)
    x, y, z = x / v_norm, y / v_norm, z / v_norm

    # Declination: arcsin(z)
    dec_rad = np.arcsin(z)

    # Right Ascension: atan2(y, x)
    ra_rad = np.arctan2(y, x)
    if ra_rad < 0:
        ra_rad += 2 * np.pi  # Ensure RA is in [0, 2π]

    # Convert to degrees
    ra_deg = np.degrees(ra_rad)
    dec_deg = np.degrees(dec_rad)

    return ra_deg, dec_deg
