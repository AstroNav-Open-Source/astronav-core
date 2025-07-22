import numpy as np
# A simple class which can handle some of the data handeling for the star vector components or angles
class StarFrame:

    def __init__(self, star_list):
        # star_list = [{"position": (x,y), "radius": r, "intensity": I, "vector": v_unit}]
        self.stars = star_list
        self._vectors = np.vstack([s["vector"] for s in self.stars])          
        self._angle_matrix = None
    
    def vectors(self):
        return np.vstack([s["vector"] for s in self.stars]) 

    def intensities(self):
        return np.array([s["intensity"] for s in self.stars])

    def positions(self):
        return [s["position"] for s in self.stars]

    def angle_matrix(self, degrees=True, cache=True):
        # Returns an (N,N) symmetric matrix where A[i,j] = angle between star i and j.
        if self._angle_matrix is not None and cache:
            return self._angle_matrix

        dot = self._vectors @ self._vectors.T
        dot = np.clip(dot, -1.0, 1.0)           
        ang = np.arccos(dot)                    
        if degrees:
            ang = np.degrees(ang)

        if cache:
            self._angle_matrix = ang
        return ang

    
    def pair_list_angle(self, degrees=True):
        # Returns a list of tuples: (i, j, angle_ij) for i<j.
        ang = self.angle_matrix(degrees=degrees)
        n = len(self.stars)
        idx_i, idx_j = np.triu_indices(n, k=1)
        return list(zip(idx_i, idx_j, ang[idx_i, idx_j]))

    def pair_list_intensity(self):
        # Returns a list of tuples: (i, j, intensity) for i<j.
        n = len(self.stars)
        idx_i, idx_j = np.triu_indices(n, k=1)
        return list(zip(idx_i,idx_j , self.intensities()))