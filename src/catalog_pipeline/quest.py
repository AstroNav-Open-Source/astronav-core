import numpy as np

def quest_algorithm(body_vectors, inertial_vectors, weights=None):
    """
    Args:
        body_vectors: Nx3 array of unit vectors measured in the body frame.
        inertial_vectors: Nx3 array of corresponding unit vectors in the inertial frame.
        weights: Optional Nx1 array of weights for each vector pair.
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

def get_rotation_matrix_from_quaternion(quaternion):
    """
    Args:
        quaternion: array-like, shape (4,), [q1, q2, q3, q0]
    Returns:
        R: 3x3 rotation matrix
    """
    q = np.asarray(quaternion)
    if q.shape[0] != 4:
        raise ValueError("Quaternion must be of length 4")
    q1, q2, q3, q0 = q  # [x, y, z, w] where w is scalar part

    # Normalize quaternion to ensure valid rotation
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Zero-norm quaternion is invalid")
    q1, q2, q3, q0 = q1 / norm, q2 / norm, q3 / norm, q0 / norm

    R = np.array([
        [1 - 2*(q2**2 + q3**2),     2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [    2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2),     2*(q2*q3 - q0*q1)],
        [    2*(q1*q3 - q0*q2),     2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])
    return R

def test_quaternion_reprojection(quaternion, body_vectors, inertial_vectors, verbose=False):
    """
    Args:
        quaternion: array-like, shape (4,)
        body_vectors: Nx3 array of unit vectors in the body frame
        inertial_vectors: Nx3 array of unit vectors in the inertial frame
        verbose: if True, print per-vector errors
    Returns:
        mean_error_deg: mean angular error in degrees
        max_error_deg: maximum angular error in degrees
    """
    R = get_rotation_matrix_from_quaternion(quaternion)
    body_vectors = np.asarray(body_vectors)
    inertial_vectors = np.asarray(inertial_vectors)
    rotated_body = (R @ body_vectors.T).T  # Nx3
    # Compute angular error for each vector
    dot_products = np.sum(rotated_body * inertial_vectors, axis=1)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angles_rad = np.arccos(dot_products)
    errors_deg = np.degrees(angles_rad)
    mean_error = np.mean(errors_deg)
    max_error = np.max(errors_deg)
    if verbose:
        for i, err in enumerate(errors_deg):
            print(f"Vector {i}: error = {err:.6f} deg")
    return mean_error, max_error

if __name__ == "__main__":
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