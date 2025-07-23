import numpy as np

from catalog import catalog

# Convert RA/Dec to unit vectors for all catalog stars
def radec_to_vec(ra_deg, dec_deg):
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    return np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])

catalog_vectors = np.array([radec_to_vec(ra, dec) for ra, dec in zip(catalog['RAICRS'], catalog['DEICRS'])])

hip_numbers = np.array(catalog['HIP'])

FOV = 70 # FOV in degrees - change to whatever you want

# Vectorized pairwise dot products
n = len(catalog_vectors)
dots = np.dot(catalog_vectors, catalog_vectors.T)
np.clip(dots, -1, 1, out=dots)
angles = np.degrees(np.arccos(dots))
i_idx, j_idx = np.triu_indices(n, k=1)
mask = np.abs(angles[i_idx, j_idx]) <= FOV
pair_db = {(int(hip_numbers[i]), int(hip_numbers[j])): float(angles[i, j]) for i, j in zip(i_idx[mask], j_idx[mask])}

# print(pair_db)
# print(n)
