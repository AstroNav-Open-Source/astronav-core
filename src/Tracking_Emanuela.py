
#Tracking
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R
#install opencv-contrib-python

#using the Harris Corner algorythm to detect the features(stars)
#First image

filename = "C:/Users/emanu/Documents/spc2025/star-treckers/src/test/test_images/Test_tracking/0RA_0DEC_FOV(70).png"
img = cv.imread(filename)                   #choose the image 
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)   #trasform it in grayscale(the 'cornerHarris' function works using only brightness)

gray_f= np.float32(gray)                    #the pixel values are represented as 32-bit floating point numbers because the function performs mathematical operations=>needs decimal precision

'''
#different algorithm to detect stars

#Running the Corner Harris detector
#input: 
# image (grayscale,32 float)
# block_size: size of the window considered around each pixel 
# k_size:Sobel operator kernel(bigger kernel=>bigger window,less details)(Sobel detects how pixel intensities change in the x and y directions)
# Harris detector free parameter: a sensitivity factor(Smaller k values tend to detect more corners)
dst = cv.cornerHarris(gray_f,2,3,0.04)

#result is dilated for marking the corners (optional)
dst = cv.dilate(dst,None)

# Boolean mask, the minimum value detected is 0.1%of the maximum value detected
threshold = 0.1 * dst.max()
bool_mask = dst > threshold
img[bool_mask]=[0,0,255] #colors the corner pixels red wherever the mask is True

cv.imshow('dst',img)            #show the image
if cv.waitKey(0) & 0xff == 27:  #waits indefinetely and when the esc key is pressed it closes the window
    cv.destroyAllWindows()
'''

# Parameters
max_corners = 1000              # Max number of points
quality_level =0.01             # Minimum accepted quality
min_distance =5                 # Minimum distance between corners

# Detect good features to track
p0 = cv.goodFeaturesToTrack(gray, 
                            maxCorners=max_corners, 
                            qualityLevel=quality_level, 
                            minDistance=min_distance)
if p0 is None:
    print("No features found in the first image!")
else:
    print("Number of features detected:" , {len(p0)})


# Optional: draw them for verification
for pt in p0:
    x, y = pt.ravel()
    cv.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)

cv.imshow("Good Features", img)
cv.waitKey(0)
cv.destroyAllWindows()


#Redundant 
p0 = p0.astype(np.float32).reshape(-1, 1, 2)  # Converts coordinates to float32, as required by cv.calcOpticalFlowPyrLK()
                                                        #.reshape(-1, 1, 2):-1- automatical determination of the number of points (N)
                                                        #                    1- adds a dimension so each point is in its own wrapper.
                                                        #                    2- coordinates per point.

#print("the coordinates are:",p0)

#Second image
filename2=r'C:\Users\emanu\Documents\spc2025\star-treckers\src\test\test_images\Test_tracking\1RA_1DEC_FOV(70).png'
img_2=cv.imread(filename2)
gray2=cv.cvtColor(img_2,cv.COLOR_BGR2GRAY)
gray2_f=np.float32(gray2)


# Parameters for lucas kanade optical flow:
#winsize- pixel region around each feature point
#3 pyramid levels are used: level 0 (original image), level 1 (half size), and level 2 (quarter size). (this is to handle larger movements)
#termination condition-Stop if either:10 iterations are reached OR the motion estimate changes by less than 0.03
lk_params = dict( winSize = (30, 30),
                maxLevel = 7,
                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
                            30, 0.01))
#Lucas Kanade algorithm
#INPUTS:
#2 images(grayscale,np.uint8)
#p0-points in first image
#lk_params-parameters
#OUTPUTS:
#p1-coordinates of the points in the second picture
#status array-1 if flow was found for that point, 0 if not. Shape: (N, 1)
#Error for each point (difference between actual and predicted). Shape: (N, 1)

p1, st, err = cv.calcOpticalFlowPyrLK(gray,gray2, p0, None, **lk_params)

if p1 is None:
    print("Optical flow tracking failed to find points in second image.")
else:
    tracked_points = np.sum(st)
    print(f"Number of points successfully tracked: {tracked_points} / {len(p0)}")


#boolean mask
good_old = p0[st == 1]  #filters only the points that were successfully tracked in the first image
good_new = p1[st == 1]

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


#trasform the coordinates to arrays
p0 = good_old.reshape(-1, 2)
p1 = good_new.reshape(-1, 2)
#displacement = p1 - p0

#print("the displacement is:",displacement)

#Transforms pixels in 3d vectors
#input:
#points-list or array of 2D pixel coordinates (u,v)
#fx,fy-focal lengths in pixels along x and y axes
#cx,cy- coordinates of the center of the image

#camera intrinsics
h, w =  img.shape[:2]
fov_deg = 70  # Approx horizontal FOV of ArduCam
fx=w/ np.arctan(fov_deg*np.pi/(360))   
fy=h/ np.arctan(fov_deg*np.pi/360)             #supponendo i 2 fov siano uguali
cx = w / 2
cy = h / 2

def pixel_to_unit_vector(points, cx, cy):
    vectors = []
    for u, v in points:
        x = (u - cx)
        y = (v - cy)
        z=fy
        vec = np.array([x, y,z]) 
        #norm = np.linalg.norm(vec)
        #vec= vec/norm
        
        vectors.append(vec)
    return np.array(vectors)

#compute the unit vectors
v1 = pixel_to_unit_vector(good_old, cx, cy)
v2 = pixel_to_unit_vector(good_new, cx, cy)
#theta= np.arcsin(real_displacement/(np.abs(v2)))
#print('the angle is ',theta)
'''
#different algorithm to find the rotation

M, inliers = cv.estimateAffinePartial2D(good_old, good_new, method=cv.RANSAC)
if M is not None:
    print("Affine matrix:\n", M)
    
    # Extract rotation angle
    angle_rad = np.arctan2(M[1, 0], M[0, 0])
    angle_deg = np.degrees(angle_rad)
    print(f"Estimated rotation: {angle_deg:.3f} degrees")

'''


#Kabsch algorithm
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
#print("Quaternion [x, y, z, w]:", rot.as_quat())
print("Euler angles [°]:", rot.as_euler('zyx', degrees=True))

#z=>yaw=>RA
#y=>pitch=>DEC

#TO-DO
#test with small rotations first
#find the right intrinsics of the camera 
#improve adding thresholds and eliminating noise









