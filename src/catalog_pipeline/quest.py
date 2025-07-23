import numpy as np

def quest_algorithm(body_vectors, inertial_vectors, weights=None):
    """
    Args:
        body_vectors (np.ndarray): Nx3 array of unit vectors measured in the body frame.
        inertial_vectors (np.ndarray): Nx3 array of corresponding unit vectors in the inertial frame.
        weights (np.ndarray or None): Optional Nx1 array of weights for each vector pair.
    Returns:
        Q: the result quaternion.
    """
    body_vectors = np.asarray(body_vectors)
    inertial_vectors = np.asarray(inertial_vectors)
    N = body_vectors.shape[0]
    if weights is None:
        weights = np.ones(N)
    else:
        weights = np.asarray(weights)
    # Compute the B matrix
    B = np.zeros((3, 3))
    for i in range(N):
        B += weights[i] * np.outer(body_vectors[i], inertial_vectors[i])
    # Compute S, sigma, Z
    S = B + B.T
    sigma = np.trace(B)
    Z = np.array([
        B[1, 2] - B[2, 1],
        B[2, 0] - B[0, 2],
        B[0, 1] - B[1, 0]
    ])
    # Construct K matrix
    K = np.zeros((4, 4))
    K[:3, :3] = S - sigma * np.eye(3)
    K[:3, 3] = Z
    K[3, :3] = Z
    K[3, 3] = sigma
    # Solve for the eigenvector corresponding to the maximum eigenvalue
    eigvals, eigvecs = np.linalg.eigh(K)
    q = eigvecs[:, np.argmax(eigvals)]  # Quaternion [q1, q2, q3, q0]
    return q

if __name__ == "__main__":
    # Example usage with dummy data
    body_vectors = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    inertial_vectors = np.array([
        [0.866, 0.5, 0],
        [-0.5, 0.866, 0],
        [0, 0, 1]
    ])
    Q = quest_algorithm(body_vectors, inertial_vectors)
    print("QUEST quaternion:\n", Q) 