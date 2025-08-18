
#Tracking
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R
#install opencv-contrib-python



#First image

filename = r"src/test/test_images/Test_tracking/0DEC_(-0.5)DEC_FOV(70).png"
img = cv.imread(filename)                   #choose the image 
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)   #trasform it in grayscale(the function works using only brightness)
blurred = cv.GaussianBlur(gray, (5, 5), 1)  
retval,thresh= cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

#camera intrinsics
h, w =  img.shape[:2]
fov_deg =10.0
fov_rad= np.radians(fov_deg)# Approx horizontal FOV
fx = w / (2.0 * np.tan(fov_rad / 2.0))      # fov_rad = horizontal FOV
fy = fx * (h / w)                       # scale for vertical pixels
cy = h / 2.0
cx= w / 2.0

# Parameters
max_corners = 100   # Max number of points (more points=>more robustness)
quality_level =0.001           # Minimum accepted quality
min_distance =4   # Minimum distance between corners

# Detect good features to track
p0 = cv.goodFeaturesToTrack(gray, 
                            maxCorners=max_corners, 
                            qualityLevel=quality_level, 
                            minDistance=min_distance)

if p0 is None:
    print("No features found in the first image!")
else:
    print("Number of features detected:" , {len(p0)})

# Margin to exclude features too close to the image border
margin_x = int(w * 0.02)
margin_y = int(h * 0.02)


# Filter features to avoid edges
filtered_p0 = []
for pt in p0:
    x, y = pt.ravel()
    if margin_x < x < (w - margin_x) and margin_y < y < (h - margin_y):
        filtered_p0.append([[x, y]])

filtered_p0 = np.array(filtered_p0, dtype=np.float32)

# Replace original p0 with filtered one
p0 = filtered_p0


# Optional: draw them for verification


for pt in p0:
    x, y = pt.ravel()
    cv.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)

cv.imshow("Good Features", img)
cv.waitKey(0)
cv.destroyAllWindows()
#Redundant 
#p0 = p0.astype(np.float32).reshape(-1, 1, 2)  # Converts coordinates to float32, as required by cv.calcOpticalFlowPyrLK()
                                                        #.reshape(-1, 1, 2):-1- automatical determination of the number of points (N)
                                                        #                    1- adds a dimension so each point is in its own wrapper.
                                                        #                    2- coordinates per point.

#subpixel accuracy to have more precise coordinates
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 150, 0.01)
cv.cornerSubPix(gray, p0, (5,5), (-1,-1), criteria)


#Second image
filename2=r'src/test/test_images/Test_tracking/0DEC_(-0.5)DEC_FOV(70).png'
img_2=cv.imread(filename2)
gray2=cv.cvtColor(img_2,cv.COLOR_BGR2GRAY)
blurred2 = cv.GaussianBlur(gray2, (5, 5),1) 
retval2,thresh2 = cv.threshold(blurred2, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

if len(p0) == 0:
    print("No features left after edge filtering!")
else:
    print("Number of features after edge filtering:", len(p0))




# Parameters for lucas kanade optical flow:
#winsize- pixel region around each feature point
#3 pyramid levels are used: level 0 (original image), level 1 (half size), and level 2 (quarter size). (this is to handle larger movements)
#termination condition-Stop if either:10 iterations are reached OR the motion estimate changes by less than 0.03
lk_params = dict( winSize = (21, 21),
                maxLevel =2,
                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,150, 0.001),
                flags=cv.OPTFLOW_LK_GET_MIN_EIGENVALS,
                minEigThreshold=1e-5
                )       # Avoid unstable points)
#Lucas Kanade algorithm
#INPUTS:
#2 images(grayscale,np.uint8)
#p0-points in first image
#lk_params-parameters
#OUTPUTS:
#p1-coordinates of the points in the second picture
#status array-1 if flow was found for that point, 0 if not. Shape: (N, 1)
#Error for each point (difference between actual and predicted). Shape: (N, 1)

p1, st, err = cv.calcOpticalFlowPyrLK(blurred,blurred2, p0, None, **lk_params)

if p1 is None:
    print("Optical flow tracking failed to find points in second image.")
else:
    tracked_points = np.sum(st)
    print(f"Number of points successfully tracked: {tracked_points} / {len(p0)}")


#boolean mask
good_old = p0[st == 1]  #filters only the points that were successfully tracked in the first image
good_new = p1[st == 1]


# Backward tracking
p0r, st_back, err_back = cv.calcOpticalFlowPyrLK(blurred2, blurred, p1, None, **lk_params)

# Compute the distance between original and back-tracked points
# Set threshold as a fraction of image size (e.g., 0.002 × width)
flow_error_thresh = 0.002 * max(w, h)
diff = np.linalg.norm(p0 - p0r, axis=2).reshape(-1)
good_indices = diff < flow_error_thresh


# Filter points
good_old = p0[good_indices]
good_new = p1[good_indices]


#just to verify
img_tracked = img_2.copy() #copy of the second image to draw the motion vectors
for old, new in zip(good_old,good_new):
    a, b = new.ravel()  #to access x and y as individual variables
    c, d = old.ravel()
    cv.line(img_tracked, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)
    cv.circle(img_tracked, (int(a), int(b)), 3, (0, 0, 255), -1)

# --- Step 5: Display the result ---
cv.imshow("Tracked Features (Lucas-Kanade)", img_tracked)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imwrite("output_tracking.jpg", img_tracked)



#Transforms pixels in 3d vectors
#input:
#points-list or array of 2D pixel coordinates (u,v)
#fx,fy-focal lengths in pixels along x and y axes
#cx,cy- coordinates of the center of the image'''

good_old = good_old.reshape(-1, 2)
good_new = good_new.reshape(-1, 2)



def pixel_to_unit_vector(points, cx, cy, fx, fy):
    """
    Convert pixel coordinates (u, v) into 3D unit vectors in camera coordinates,
    with astronomy convention: +x = RA east, +y = DEC north, +z = boresight.
    """
    vectors = []
    for u, v in points:
        x = (u - cx) / fx
        y = -(v - cy) / fy   # flip Y to make up = north
        z = 1.0
        vec = np.array([x, y, z])
        vec /= np.linalg.norm(vec)
        vectors.append(vec)
    return np.array(vectors)


#compute the unit vectors
v1 = pixel_to_unit_vector(good_old, cx, cy,fx,fy)
v2 = pixel_to_unit_vector(good_new, cx, cy,fx,fy)


#KABSH ALGORITHM
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
rot=estimate_rotation(v1, v2)


print("Rotation matrix:\n", rot.as_matrix())

def vector_to_radec(vec):
    """
    Convert a 3D unit vector into (RA, DEC) in degrees.
    - RA increases eastward (from +X axis, around +Z).
    - DEC increases northward (from equator toward +Y).
    """
    x, y, z = vec
    ra = np.degrees(np.arctan2(x, z)) % 360.0   # east = +x, measured around z
    dec = np.degrees(np.arctan2(y, np.sqrt(x**2 + z**2)))
    return ra, dec

def get_boresight_radec(R_matrix):
    v_boresight = np.array([0, 0, 1.0])  # camera optical axis
    v_rotated = R_matrix @ v_boresight
    return vector_to_radec(v_rotated)


def estimate_delta_radec(R_matrix):
    """
    Given a rotation matrix, compute the change in RA/DEC
    of the camera's pointing direction.
    """
    # Initial pointing (camera boresight = +Z axis in OpenCV convention)
    v_initial = np.array([0, 0, 1.0])
    ra0, dec0 = vector_to_radec(v_initial)

    # Rotated pointing
    v_rotated = R_matrix @ v_initial
    ra1, dec1 = vector_to_radec(v_rotated)

    # Differences (with RA wrapped)
    delta_ra = (ra1 - ra0 + 180) % 360 - 180
    delta_dec = - (dec1 - dec0)

    return delta_ra, delta_dec

# --- Absolute pointing tracking ---
# Initialize cumulative rotation once (identity for first frame)
R_cumulative = np.eye(3)

# Update with this step's rotation
R_cumulative = rot.as_matrix() @ R_cumulative

# Compute boresight absolute RA/DEC
ra_abs, dec_abs = get_boresight_radec(R_cumulative)
print(f"Absolute boresight pointing: RA={ra_abs:.4f}°, DEC={dec_abs:.4f}°")

# Still keep relative delta if you want
delta_ra, delta_dec = estimate_delta_radec(rot.as_matrix())
print(f"ΔRA ≈ {delta_ra:.4f}°, ΔDEC ≈ {delta_dec:.4f}°")

    


'''
#print("Quaternion [x, y, z, w]:", rot.as_quat())
print("Euler angles [°]:", rot.as_euler('zyx', degrees=True))

ra_delta = rot.as_euler('zyx', degrees=True)[1]  # Y-axis:  RA (for the open cv coordinate system)
dec_delta = rot.as_euler('zyx', degrees=True)[0]  # Z-axis: DE

print(f"ΔRA ≈ {ra_delta:.4f}°, ΔDEC ≈ {dec_delta:.4f}°")

#TO-DO
#test with small rotations first 

#improve adding thresholds and eliminating noise
#add subpixels precision for the tracking

'''







