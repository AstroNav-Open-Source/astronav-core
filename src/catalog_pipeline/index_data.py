import numpy as np

from catalog import catalog
from radec_to_vec import radec_to_vec

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
