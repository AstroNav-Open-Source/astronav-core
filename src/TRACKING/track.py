import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R

def get_intrinsics (img, fov_deg =70):
    h, w =  img.shape[:2]
    fov_rad= np.radians(fov_deg)# Approx horizontal FOV
    fx = w / (2.0 * np.tan(fov_rad / 2.0))      # fov_rad = horizontal FOV
    fy = fx * (h / w)                       # scale for vertical pixels
    cy = h / 2.0
    cx= w / 2.0
    return cx,cy,fx,fy

img1=cv.imread("src/test/test_images/Test_tracking/0RA_0DEC_FOV(5).png")
cx,cy,fx,fy=get_intrinsics(img1)
print(cx,cy,fx,fy)

def estimate_rotation(v1, v2):
    v1_mean = np.mean(v1, axis=0)
    v2_mean = np.mean(v2, axis=0)   #Find the average vector in each set.('axis=0'=>computes the mean along the rows, for each column separately.)

    v1_centered = v1 - v1_mean
    v2_centered = v2 - v2_mean  #makes them zero-centered for covariance calculation.

    H = v1_centered.T @ v2_centered #covariance matrix
    
    #compute singular value decomposition
    U, S, Vt = np.linalg.svd(H)  #H= U * S * Vt.(U and Vt are orthogonal matrices, S diagonal.)

    #compute the best matrix that aligns v1 and v2
    R_matrix = Vt.T @ U.T

    #correction in case of reflection
    if np.linalg.det(R_matrix) < 0:
        Vt[-1, :] *= -1
        R_matrix = Vt.T @ U.T
    
    rot = R.from_matrix(R_matrix)
    return rot
import numpy as np
from scipy.spatial.transform import Rotation as R

def rotation_between_vectors(v1, v2):
    v1 = np.array(v1, dtype=float) / np.linalg.norm(v1)
    v2 = np.array(v2, dtype=float) / np.linalg.norm(v2)

    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)

    if np.allclose(cross, 0):  
        # vectors are parallel or opposite
        if dot > 0:
            return R.identity()  # no rotation needed
        else:
            # 180-degree rotation around any perpendicular axis
            axis = np.eye(3)[np.argmin(np.abs(v1))]  
            axis = np.cross(v1, axis)
            axis /= np.linalg.norm(axis)
            return R.from_rotvec(np.pi * axis)

    axis = cross / np.linalg.norm(cross)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    return R.from_rotvec(angle * axis)

# Example
v1 = (0,0,1)
v2 = (0,1,0)
rot = rotation_between_vectors(v1, v2)

print(rot.as_matrix())   # Rotation matrix
print(rot.as_rotvec())   # Axis-angle
print(rot.as_euler('xyz', degrees=True))  # Euler angles



def pixel_to_unit_vector(points, cx, cy, fx, fy):
    vectors = []
    for u,v in points:
        x = (u - cx) / fx
        y = -(v - cy) / fy   # flip Y to make up = north
        z = 1.0
        vec = np.array([x, y, z])
        #vec /= np.linalg.norm(vec)
        vectors.append(vec)
    return np.array(vectors)

def draw_vectors_on_image(img, points, vectors, scale=100, color=(0,255,0)):
    """
    Draws arrows for unnormalized camera vectors on the image.
    - img: input image (BGR)
    - points: Nx2 array of (u,v) pixel coordinates
    - vectors: Nx3 array of vectors (not normalized)
    - scale: multiplier to make arrows visible
    - color: arrow color (B,G,R)
    """
    img_out = img1.copy()
    for (u, v), vec in zip(points.astype(int), vectors):
        dx = int(vec[0] * scale)
        dy = int(-vec[1] * scale)   # flip sign back for display

        start_point = (int(u), int(v))
        end_point   = (int(u + dx), int(v + dy))

        cv.arrowedLine(img_out, start_point, end_point, color, 2, tipLength=0.3)

    return img_out


# Example usage:
# Assuming you already computed `good_old` and intrinsics
v_raw = pixel_to_unit_vector(good_old, cx, cy, fx, fy)  # without normalization
img_with_vecs = draw_vectors_on_image(img1, good_old, v_raw, scale=80, color=(0,0,255))

cv.imshow("Raw Vectors on Image", img_with_vecs)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite("vectors_raw_overlay.jpg", img_with_vecs)
