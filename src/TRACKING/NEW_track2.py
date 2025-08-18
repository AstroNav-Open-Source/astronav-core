#Tracking
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R
#install opencv-contrib-python

def get_intrinsics (img, fov_deg =5):
    h, w =  img.shape[:2]
    fov_rad= np.radians(fov_deg)# Approx horizontal FOV
    fx = w / (2.0 * np.tan(fov_rad / 2.0))      # fov_rad = horizontal FOV
    fy = fx * (h / w)                       # scale for vertical pixels
    cy = h / 2.0
    cx= w / 2.0
    return cx,cy,fx,fy

def processing_image (img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)   #trasform it in grayscale(the function works using only brightness)
    blurred = cv.GaussianBlur(gray, (3, 3), 1)  
    retval,image_processed = cv.threshold(blurred, 200, 255, cv.THRESH_BINARY)
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
        x = (u - cx) 
        y = (v - cy)  # flip Y to make up = north
        z = 0
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
                minEigThreshold=1e-5
                )     
def vector_to_radec(vec):
    x, y, z = vec
    ra = np.degrees(np.arctan2(y, x)) % 360   # wrap RA to [0,360)
    dec = np.degrees(np.arcsin(z))            # declination = elevation
    return ra, dec

def radec_from_pixels(points, R_cam2eq, cx, cy, fx, fy):
    vecs_cam = pixel_to_unit_vector(points, cx, cy, fx, fy)
    vecs_eq = (R_cam2eq @ vecs_cam.T).T
    return [vector_to_radec(v) for v in vecs_eq]
def radec_to_vector(ra_deg, dec_deg):
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.array([x, y, z])



def build_cam_to_eq(RA0_deg, DEC0_deg, roll_deg=0.0):
    """
    Build a rotation matrix that maps camera coordinates → equatorial coordinates.
    - +Z_cam points to (RA0, DEC0)
    - +X_cam points east at the boresight
    - +Y_cam points north at the boresight
    roll_deg lets you twist the camera around its boresight.
    """
    # boresight vector
    boresight = radec_to_vector(RA0_deg, DEC0_deg)

    # local east (increasing RA at constant Dec)
    ra = np.radians(RA0_deg)
    dec = np.radians(DEC0_deg)
    east = np.array([-np.sin(ra), np.cos(ra), 0.0])
    east /= np.linalg.norm(east)

    # local north (increasing Dec)
    north = np.cross(boresight, east)
    north /= np.linalg.norm(north)

    # Apply roll around boresight
    if abs(roll_deg) > 1e-8:
        roll = R.from_rotvec(np.radians(roll_deg) * boresight)
        east = roll.apply(east)
        north = roll.apply(north)

    # Camera basis vectors expressed in EQ coords
    x_eq = east
    y_eq = north
    z_eq = boresight

    # Construct matrix: columns are cam axes in EQ frame
    R_cam2eq = np.column_stack([x_eq, y_eq, z_eq])
    return R_cam2eq

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







if __name__ == "__main__":
    img1= cv.imread("c:/Users/emanu/Downloads/WhatsApp Image 2025-08-18 at 12.25.29.jpeg")
    img2= cv.imread( r"c:\Users\emanu\Downloads\WhatsApp Image 2025-08-18 at 12.32.48.jpeg")

    cx,cy,fx,fy=get_intrinsics(img1)
    img_processed1=processing_image(img1)
    img_processed2=processing_image(img2)
    h, w =  img1.shape[:2]

    p0=detector(img_processed1,w,h,max_corners = 1000,quality_level =0.001,min_distance =3, )
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
    flow_error_thresh = 0.002 * max(w, h)
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
    

# --- Draw on second image using tracked points ---
    img_center_vecs = draw_center_to_star_vectors(img2, good_new, cx, cy)

    cv.imshow("Center-to-Star Vectors", img_center_vecs)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imwrite("output_center_vectors.jpg", img_center_vecs)

    img_center_vecs2=draw_center_to_star_vectors(img1, good_old, cx, cy,color=(0, 0, 255))
    
    cv.imshow("Center-to-Star Vectors", img_center_vecs2)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imwrite("output_center_vectors.jpg", img_center_vecs2)
    


    '''
    #compute the unit vectors
    v1 = pixel_to_unit_vector(good_old, cx, cy,fx,fy)
    v2 = pixel_to_unit_vector(good_new, cx, cy,fx,fy)
    
    
    rot= estimate_rotation(v1, v2)
    print("Rotation matrix:\n", rot.as_matrix())
    #print("Quaternion [x, y, z, w]:", rot.as_quat())
    print("Euler angles [°]:", rot.as_euler('zyx', degrees=True))

    ra_delta = rot.as_euler('zyx', degrees=True)[1]  # Y-axis:  RA (for the open cv coordinate system)
    dec_delta = rot.as_euler('zyx', degrees=True)[0]  # Z-axis: DE

    print(f"ΔRA ≈ {ra_delta:.4f}°, ΔDEC ≈ {dec_delta:.4f}°")



'''


'''
    # Assume first image boresight is RA0=0°, DEC0=0° for now

# Desired boresight orientation
RA0, DEC0 = 0.0, 0.0  
R_cam2eq = build_cam_to_eq(RA0, DEC0, roll_deg=180.0) #alignment with stellarium

boresight_eq = radec_to_vector(RA0, DEC0)   # (1,0,0) vector

# Current camera boresight
boresight_cam = np.array([0, 0, 1])         # (0,0,1)

# Rotation needed: from boresight_cam → boresight_eq
v = np.cross(boresight_cam, boresight_eq)
s = np.linalg.norm(v)
c = np.dot(boresight_cam, boresight_eq)

if s < 1e-8:  # already aligned
    R_cam2eq = np.eye(3)
else:
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    R_cam2eq = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
print("Rotation matrix (cam→eq):\n", R_cam2eq)



# Build orientation for frame 1
R_cam2eq = build_cam_to_eq(RA0, DEC0, roll_deg=0.0)

# Frame 1 boresight RA/Dec
boresight_eq1 = R_cam2eq @ boresight_cam
boresight_ra_dec1 = vector_to_radec(boresight_eq1)

# Frame 2 orientation = relative rotation * frame1 orientation
R_cam2eq_2 = rot.as_matrix() @ R_cam2eq
boresight_eq2 = R_cam2eq_2 @ boresight_cam
boresight_ra_dec2 = vector_to_radec(boresight_eq2)

print("Boresight (frame 1): RA={:.3f}°, DEC={:.3f}°".format(*boresight_ra_dec1))
print("Boresight (frame 2): RA={:.3f}°, DEC={:.3f}°".format(*boresight_ra_dec2))
angle = np.degrees(np.arccos(np.clip(np.dot(
    boresight_eq1/np.linalg.norm(boresight_eq1),
    boresight_eq2/np.linalg.norm(boresight_eq2)
), -1, 1)))
print("Measured boresight shift: {:.3f}°".format(angle))

'''

