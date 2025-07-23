import numpy as np

# Mock data: observed vectors (camera frame) and catalog vectors (inertial frame)
# Each row is a 3D unit vector
observed_vectors = np.array([
    [0.8273, 0.5541, -0.0920],
    [-0.8285, 0.5522, -0.0955],
    [0.0046, -0.0022, 0.9999],
])
catalog_vectors = np.array([
    [-0.1517, -0.9669, 0.2050],
    [-0.8393, 0.4494, -0.3044],
    [-0.5248, -0.1960, 0.8286],
])

# Weights for each pair (can be uniform)
weights = np.ones(len(observed_vectors))

def compute_B_matrix(obs_vecs, cat_vecs, weights):
    B = np.zeros((3, 3))
    for i in range(len(obs_vecs)):
        B += weights[i] * np.outer(obs_vecs[i], cat_vecs[i])
    return B

def quest_q_method(obs_vecs, cat_vecs, weights):
    B = compute_B_matrix(obs_vecs, cat_vecs, weights)
    S = B + B.T
    sigma = np.trace(B)
    Z = np.array([
        B[1,2] - B[2,1],
        B[2,0] - B[0,2],
        B[0,1] - B[1,0]
    ])
    K = np.zeros((4,4))
    K[:3,:3] = S - sigma * np.eye(3)
    K[:3,3] = Z
    K[3,:3] = Z
    K[3,3] = sigma
    # Eigenvector of K with largest eigenvalue is the quaternion
    eigvals, eigvecs = np.linalg.eigh(K)
    q = eigvecs[:, np.argmax(eigvals)]
    # Return quaternion as [q1, q2, q3, q0] (vector part, then scalar part)
    return q

if __name__ == "__main__":
    q = quest_q_method(observed_vectors, catalog_vectors, weights)
    print("Quaternion (q1, q2, q3, q0):", q)
    # Optionally, print as scalar-last (w, x, y, z)
    print("Quaternion (w, x, y, z):", [q[3], q[0], q[1], q[2]]) 