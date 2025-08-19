#Tracking
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R
#install opencv-contrib-python

fov_deg =52
def get_intrinsics (img):
    h, w =  img.shape[:2]
    fov_rad= np.radians(fov_deg)# Approx horizontal FOV
    fy = h / (2.0 * np.tan(fov_rad / 2.0))      # fov_rad = horizontal FOV
    fx = fy                      # scale for vertical pixels
    cy = h / 2.0
    cx= w / 2.0
    return cx,cy,fx,fy

def processing_image (img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)   #trasform it in grayscale(the function works using only brightness)
    blurred = cv.GaussianBlur(gray, (3, 3), 1)  
    retval,image_processed = cv.threshold(blurred,150, 255, cv.THRESH_BINARY)
    return image_processed

def detector(img,w,h,max_corners,quality_level,min_distance): 

    p0 = cv.goodFeaturesToTrack(img, 
                            maxCorners=max_corners, 
                            qualityLevel=quality_level, 
                            minDistance=min_distance)
    return p0

#criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 150, 0.01)
#cv.cornerSubPix(gray, p0, (5,5), (-1,-1), criteria)

def pixel_to_unit_vector(points, cx, cy, fx, fy):
    vectors = []
    for u,v in points:
        x = (u - cx)/fx
        y = (v - cy)/fy # flip Y to make up = north
        z = 1
        vec = np.array([x, y, z])
        #vec /= np.linalg.norm(vec)
        vectors.append(vec)
    return np.array(vectors)

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

lk_params = dict( winSize = (31, 31),
                maxLevel =2,
                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,150, 0.001),
                flags=cv.OPTFLOW_LK_GET_MIN_EIGENVALS,
                minEigThreshold=1e-4)     

def draw_center_to_star_vectors(img, points, cx, cy, color=(255, 0, 0)):
    """
    Draw arrows from the image center to each detected star (feature point).
    """
    img_out = img.copy()

    for pt in points:
        x, y = pt.ravel()
        cv.arrowedLine(img_out, (int(cx), int(cy)), (int(x), int(y)), color, 2, tipLength=0.2)
        cv.circle(img_out, (int(x), int(y)), 3, (0, 255, 0), -1)  # mark the star

    return img_out

# ---------- Add these helpers near the top ----------
def pixel_to_cam_ray(points, cx, cy, fx, fy, flip_y=True):
    """
    Convert pixels -> unit rays in camera frame using a pinhole model.
    If your image 'up' is north, set flip_y=True so +y_cam points up.
    points: Nx2 array of [u,v]
    """
    rays = []
    for u, v in points.reshape(-1, 2):
        x = (u - cx) / fx
        y = (v - cy) / fy
        if flip_y:
            y = -y
        z = 1.0
        vec = np.array([x, y, z], dtype=np.float64)
        vec /= np.linalg.norm(vec)
        rays.append(vec)
    return np.vstack(rays)



def vector_to_radec(vec):
    x, y, z = vec
    ra = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    dec = np.degrees(np.arcsin(z/np.linalg.norm(vec)))
    return ra, dec

def hms_to_deg(h, m, s):
    return (h + m/60 + s/3600) * 15

def dms_to_deg(d, m, s):
    sign = 1 if d >= 0 else -1
    return sign * (abs(d) + m/60 + s/3600)

import numpy as np

def radec_to_vector(ra_deg, dec_deg):
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)

    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)

    vec = np.array([x, y, z], dtype=np.float64)
    vec /= np.linalg.norm(vec)   # normalize to unit vector
    return vec   # shape (3,)


def build_cam_to_eq(RA0_deg, DEC0_deg, roll_deg=0.0):
    """
    Build rotation matrix mapping camera axes -> equatorial axes.
    +Z_cam points to boresight (RA0,DEC0), +X_cam is east, +Y_cam is north.
    roll_deg rotates camera around boresight.
    """
    z_eq = radec_to_vector(RA0_deg, DEC0_deg)            # boresight in EQ

    ra = np.radians(RA0_deg); dec = np.radians(DEC0_deg)
    east = np.array([-np.sin(ra), np.cos(ra), 0.0], dtype=np.float64)
    east /= np.linalg.norm(east)

    north = np.cross(z_eq, east)
    north /= np.linalg.norm(north)

    if abs(roll_deg) > 1e-12:
        roll = R.from_rotvec(np.radians(roll_deg) * z_eq)
        east = roll.apply(east)
        north = roll.apply(north)

    x_eq = east
    y_eq = north
    R_cam2eq = np.column_stack([x_eq, y_eq, z_eq])  # columns are cam axes in EQ
    return R_cam2eq


def draw_center_to_star_vectors(img, points, cx, cy, color=(0, 255, 0)):
    # Make a copy so we don’t overwrite the original
    img_out = img.copy()
    
    for i, (x, y) in enumerate(points):
        # Draw the line from image center to point
        cv.line(img_out, (int(cx), int(cy)), (int(x), int(y)), color, 1)

        # Draw a small circle at the star location
        cv.circle(img_out, (int(x), int(y)), 3, color, -1)

        # Put the index number near the star
        cv.putText(img_out, str(i), (int(x) + 5, int(y) - 5), 
            cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)
    return img_out




if __name__ == "__main__":
    img1= cv.imread(r"src\test\test_images\Tracking_test\0RA_0DEC_FOV(52.3).png")
    img2= cv.imread(r"src\test\test_images\Tracking_test\0RA_0.1DEC_FOV(52.3).png")

    cx,cy,fx,fy=get_intrinsics(img1)
    img_processed1=processing_image(img1)
    img_processed2=processing_image(img2)
    h, w =  img1.shape[:2]
    margin_x = int(w * 0.1)
    margin_y = int(h * 0.1)
    p0=detector(img_processed1,w,h,max_corners = 1000,quality_level =0.001,min_distance =5)
    
    filtered_p0 = []
    for pt in p0:
        x, y = pt.ravel()
        if margin_x < x < (w - margin_x) and margin_y < y < (h - margin_y):
            filtered_p0.append([[x, y]])
    filtered_p0 = np.array(filtered_p0, dtype=np.float32)
    p0=filtered_p0

    if p0 is None:
        print("No features found in the first image!")
    else:
        print("Number of features detected:",len(p0))
    img_display = img1.copy()
    for pt in p0:
        x, y = pt.ravel()
        cv.circle(img_display, (int(x), int(y)), 3, (0, 255, 0), -1)


    cv.imshow("Good Features", img_display)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    p1, st, err = cv.calcOpticalFlowPyrLK(img_processed1,img_processed2, p0, None, **lk_params)
    if p1 is None:
        print("Optical flow tracking failed to find points in second image.")
    else:
        tracked_points = np.sum(st)
        print(f"Number of points successfully tracked: {tracked_points} / {len(p0)}")
    
    good_old = p0[st == 1]  #filters only the points that were successfully tracked in the first image
    good_new = p1[st == 1]
    
    p0r, st_back, err_back = cv.calcOpticalFlowPyrLK(img_processed2,img_processed1, good_new, None, **lk_params)
    flow_error_thresh = 0.005 * max(w, h)
    p0r = p0r.reshape(-1, 2)
    st_back = st_back.reshape(-1)

    diff = np.linalg.norm(good_old - p0r, axis=1)
    good_indices = diff < flow_error_thresh
    good_old = good_old[good_indices]
    good_new = good_new[good_indices]

    
    img_tracked = img2.copy() #copy of the second image to draw the motion vectors
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
    img_center_vecs = draw_center_to_star_vectors(img2, good_new, cx, cy)
    cv.imshow("Center-to-Star Vectors (img2)", img_center_vecs)
    cv.waitKey(0)
    cv.destroyAllWindows()


    displacement = good_new - good_old     # shape (N,2), each row is (dx, dy)
    deg_pixel = fov_deg / h
    real_displacement = displacement * deg_pixel

    # Compute total displacement (magnitude) for each vector
    magnitudes = np.linalg.norm(real_displacement, axis=1)

    # Print displacement for every vector
    for i, m in enumerate(magnitudes):
        print(f"Vector {i}: displacement = {m}")
    
    # Compute and print mean displacement
    mean_disp = np.mean(magnitudes)
    print(f"\nMean displacement = {mean_disp}")
    
   
   #RA and DE for the star '0'
# HIP: 113136,RA (on date): 22h 55m 57.99s,Dec (on date): −15° 41′ 23.0″

    #conversion in degree
    ra1_deg = hms_to_deg(22, 55, 57.99)
    dec1_deg = dms_to_deg(-15, 41, 23.0)
    
    #HIP: 3092,RA (on date): 0h 40m 39.94s,Dec (on date): +31° 00′ 01.6″
   
    ra2_deg = hms_to_deg(0, 40, 39.94)
    dec2_deg = dms_to_deg(31, 0, 01.6)
    #call radec to vector function
    v1_eq= radec_to_vector(ra1_deg,dec1_deg)
    v2_eq=radec_to_vector(ra2_deg,dec2_deg)
  
    
    v1_cam=pixel_to_cam_ray(p0[0,0],cx,cy,fx,fy,flip_y=False)
    v2_cam=pixel_to_cam_ray(p0[1,0],cx,cy,fx,fy,flip_y=False)
    print ('p0',p0)
    print('first',p0[0,0],p0[1,0])
    print('vcam:',v1_cam,v2_cam)

    print('v_eq',v1_eq,v2_eq)    
    v_eq = np.vstack([v1_eq, v2_eq])
    
    v_cam = np.vstack([v1_cam, v2_cam])

    rot= estimate_rotation(v_eq, v_cam)
    print("Rotation matrix:\n", rot.as_matrix())
    
    
    '''
    rot= estimate_rotation(v1_eq, v2)
    print("Rotation matrix:\n", rot.as_matrix())
    #print("Quaternion [x, y, z, w]:", rot.as_quat())

    print("Euler angles [°]:", rot.as_euler('zyx', degrees=True))

    ra_delta = rot.as_euler('zyx', degrees=True)[1]  # Y-axis:  RA (for the open cv coordinate system)
    dec_delta = rot.as_euler('zyx', degrees=True)[0]  # Z-axis: DE

    print(f"ΔRA ≈ {ra_delta:.4f}°, ΔDEC ≈ {dec_delta:.4f}°")'''

'''
# ---- Choose/know your boresight (image center) RA0/DEC0 and optional camera roll
RA0_deg  = 0.0      # <-- set this
DEC0_deg = 0.0      # <-- set this
roll_deg = 0.0

R_cam2eq = build_cam_to_eq(RA0_deg, DEC0_deg, roll_deg)

# Rays for the tracked stars in both frames
rays_old_cam = pixel_to_cam_ray(good_old, cx, cy, fx, fy, flip_y=True)
rays_new_cam = pixel_to_cam_ray(good_new, cx, cy, fx, fy, flip_y=True)

# Map to equatorial frame
rays_old_eq = (R_cam2eq @ rays_old_cam.T).T
rays_new_eq = (R_cam2eq @ rays_new_cam.T).T

# Convert each to (RA, Dec)
radec_old = np.array([vector_to_radec(v) for v in rays_old_eq])  # shape (N,2)
radec_new = np.array([vector_to_radec(v) for v in rays_new_eq])

# Print a few examples
for i in range(min(5, len(radec_new))):
    print(f"Star {i}: RA1={radec_old[i,0]:.6f}°, Dec1={radec_old[i,1]:.6f}°  ->  "
          f"RA2={radec_new[i,0]:.6f}°, Dec2={radec_new[i,1]:.6f}°")
'''
