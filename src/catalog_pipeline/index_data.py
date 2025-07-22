import numpy as np

from catalog import catalog

# Convert RA/Dec to unit vectors for all catalog stars
def radec_to_vec(ra_deg, dec_deg):
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    return np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])

catalog_vectors = []
for ra,dec in zip(catalog['RAICRS'], catalog['DEICRS']):
    catalog_vectors.append(radec_to_vec(ra, dec))

catalog_vectors = np.array(catalog_vectors)

FOV = 90 # FOV in degrees - change to whatever you want

# Precompute angular distances for all pairs [dict]
pair_db = {}
n = len(catalog_vectors)
for i in range(n):
    for j in range(i+1, n):
        dot = np.dot(catalog_vectors[i], catalog_vectors[j])
        dot = np.clip(dot, -1, 1)
        theta = np.degrees(np.arccos(dot))
        if abs(theta) <= FOV:
            pair_db[(i, j)] = theta

print(pair_db)

print(n)
