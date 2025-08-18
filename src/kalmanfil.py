import numpy as np
import imageio.v3 as iio
import glob, os

class Track:
    def __init__(self, tid, x, y):
        self.id = tid
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.age = 1
        self.history = [(x, y)]

    def predict(self, dt=1.0):
        self.x += self.vx * dt
        self.y += self.vy * dt

    def update(self, det_x, det_y, alpha=0.6, dt=1.0):
        # innovation
        dx = det_x - self.x
        dy = det_y - self.y
        # position update
        self.x += alpha * dx
        self.y += alpha * dy
        # velocity update based on correction
        self.vx += (alpha/dt) * dx
        self.vy += (alpha/dt) * dy
        self.age += 1
        self.history.append((self.x, self.y))


def detect_stars(img, threshold=150):
    """ Very simple: threshold + centroid of bright blobs """
    mask = img > threshold
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []
    coords = np.column_stack([xs, ys])
    # cluster by proximity (cheap)
    from sklearn.cluster import DBSCAN
    labels = DBSCAN(eps=3, min_samples=3).fit(coords).labels_
    detections = []
    for lab in set(labels):
        if lab == -1:
            continue
        pts = coords[labels == lab]
        cx, cy = pts[:,0].mean(), pts[:,1].mean()
        detections.append((cx, cy))
    return detections


def associate(tracks, detections, max_dist=8.0):
    matches = {}
    unmatched_tracks = list(range(len(tracks)))
    unmatched_dets = list(range(len(detections)))
    if not tracks or not detections:
        return matches, unmatched_tracks, unmatched_dets

    cost = np.zeros((len(tracks), len(detections)))
    for i,t in enumerate(tracks):
        for j,d in enumerate(detections):
            cost[i,j] = np.hypot(t.x-d[0], t.y-d[1])

    used_t, used_d = set(), set()
    for i,j in sorted([(i,j) for i in range(len(tracks)) for j in range(len(detections))],
                      key=lambda p: cost[p[0],p[1]]):
        if i in used_t or j in used_d:
            continue
        if cost[i,j] < max_dist:
            matches[i] = j
            used_t.add(i); used_d.add(j)

    unmatched_tracks = [i for i in range(len(tracks)) if i not in used_t]
    unmatched_dets = [j for j in range(len(detections)) if j not in used_d]
    return matches, unmatched_tracks, unmatched_dets


def track_sequence(frames, threshold=150):
    tracks = []
    next_id = 1
    for fidx,img in enumerate(frames):
        detections = detect_stars(img, threshold)
        # predict forward using velocity
        for t in tracks:
            t.predict(dt=1.0)

        matches, un_tracks, un_dets = associate(tracks, detections)

        for ti,di in matches.items():
            tracks[ti].update(*detections[di], dt=1.0)

        for di in un_dets:
            cx,cy = detections[di]
            tracks.append(Track(next_id, cx, cy))
            next_id += 1

        print(f"Frame {fidx+1}: {len(detections)} detections, {len(tracks)} tracks")
    return tracks


def load_frames_from_dir(path, ext="png"):
    files = sorted(glob.glob(os.path.join(path,f"*.{ext}")))
    frames = []
    for f in files:
        img = iio.imread(f).astype(np.float32)
        if img.ndim==3:
            img = img.mean(axis=2)
        frames.append(img)
    return frames

if __name__ == "__main__":
    # Example: synthetic stars
    import matplotlib.pyplot as plt
    H,W=256,256
    n_frames=20
    xs,ys=np.linspace(40,200,n_frames),np.linspace(60,150,n_frames)
    frames=[]
    for i in range(n_frames):
        img=np.random.normal(20,2,(H,W))
        xi,yi=int(xs[i]),int(ys[i])
        img[yi-1:yi+2, xi-1:xi+2]+=200
        frames.append(img)

    tracks=track_sequence(frames,threshold=50)

    # visualize
    plt.imshow(frames[-1],cmap='gray')
    for t in tracks:
        pts=np.array(t.history)
        plt.plot(pts[:,0],pts[:,1])
    plt.show()

# Tracking (velocity-based, no LK optical flow)
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R

# ----------------------------
# Camera intrinsics & helpers
# ----------------------------
def get_intrinsics (img, fov_deg =70):
    h, w =  img.shape[:2]
    fov_rad= np.radians(fov_deg)  # horizontal FOV
    fx = w / (2.0 * np.tan(fov_rad / 2.0))
    fy = fx * (h / w)
    cy = h / 2.0
    cx= w / 2.0
    return cx,cy,fx,fy

def processing_image (img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 1)
    _, image_processed = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return image_processed

def detector(img,w,h,max_corners,quality_level,min_distance,  margin_x,margin_y):
    p0 = cv.goodFeaturesToTrack(img, 
                                maxCorners=max_corners, 
                                qualityLevel=quality_level, 
                                minDistance=min_distance)
    if p0 is None or len(p0)==0:
        return np.empty((0,2), dtype=np.float32)
    # keep only points away from borders
    filtered = []
    for pt in p0:
        x, y = pt.ravel()
        if margin_x < x < (w - margin_x) and margin_y < y < (h - margin_y):
            filtered.append([x, y])
    return np.array(filtered, dtype=np.float32)

def pixel_to_unit_vector(points, cx, cy, fx, fy):
    vectors = []
    for v,u in points:
        x = (v - cx) / fx
        y = -(u - cy) / fy   # flip Y to make up = north
        z = 1.0
        vec = np.array([x, y, z], dtype=np.float64)
        vec /= np.linalg.norm(vec)
        vectors.append(vec)
    return np.array(vectors)

def estimate_rotation(v1, v2):
    v1_mean = np.mean(v1, axis=0)
    v2_mean = np.mean(v2, axis=0)

    v1_centered = v1 - v1_mean
    v2_centered = v2 - v2_mean

    H = v1_centered.T @ v2_centered
    U, S, Vt = np.linalg.svd(H)
    R_matrix = Vt.T @ U.T
    if np.linalg.det(R_matrix) < 0:
        Vt[-1, :] *= -1
        R_matrix = Vt.T @ U.T
    return R.from_matrix(R_matrix)

def vector_to_radec(vec):
    x, y, z = vec
    ra = np.degrees(np.arctan2(y, x)) % 360
    dec = np.degrees(np.arcsin(z))
    return ra, dec

def radec_from_pixels(points, R_cam2eq, cx, cy, fx, fy):
    vecs_cam = pixel_to_unit_vector(points, cx, cy, fx, fy)
    vecs_eq = (R_cam2eq @ vecs_cam.T).T
    return [vector_to_radec(v) for v in vecs_eq]

def radec_to_vector(ra_deg, dec_deg):
    ra = np.radians(ra_deg); dec = np.radians(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.array([x, y, z], dtype=np.float64)

def build_cam_to_eq(RA0_deg, DEC0_deg, roll_deg=0.0):
    # boresight vector (+Z_cam) in EQ coords
    boresight = radec_to_vector(RA0_deg, DEC0_deg)
    ra = np.radians(RA0_deg); dec = np.radians(DEC0_deg)
    east = np.array([-np.sin(ra), np.cos(ra), 0.0], dtype=np.float64); east /= np.linalg.norm(east)
    north = np.cross(boresight, east); north /= np.linalg.norm(north)
    if abs(roll_deg) > 1e-8:
        roll = R.from_rotvec(np.radians(roll_deg) * boresight)
        east = roll.apply(east); north = roll.apply(north)
    x_eq, y_eq, z_eq = east, north, boresight
    return np.column_stack([x_eq, y_eq, z_eq])

# ----------------------------
# Velocity-based tracker
# ----------------------------
class Track:
    __slots__ = ("id","x","y","vx","vy","age","misses","history")
    def __init__(self, tid, x, y):
        self.id = tid
        self.x  = float(x)
        self.y  = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.age = 1
        self.misses = 0
        self.history = [(self.x, self.y)]
    def predict(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
    def update(self, zx, zy, dt, alpha=0.7):
        # innovation
        dx = zx - self.x
        dy = zy - self.y
        # position update
        self.x += alpha * dx
        self.y += alpha * dy
        # velocity update uses position innovation / dt
        self.vx += (alpha * dx) / max(dt, 1e-6)
        self.vy += (alpha * dy) / max(dt, 1e-6)
        self.age += 1
        self.misses = 0
        self.history.append((self.x, self.y))

def associate(tracks, detections, gate_px=6.0):
    """
    Greedy nearest-neighbor association with distance gating.
    tracks: list[Track]
    detections: (N,2) array
    returns: matches dict track_idx->det_idx, unmatched_tracks, unmatched_dets
    """
    if len(tracks)==0 or len(detections)==0:
        return {}, list(range(len(tracks))), list(range(len(detections)))
    T, D = len(tracks), len(detections)
    tr_xy = np.array([[t.x, t.y] for t in tracks], dtype=np.float32)
    d_xy  = detections.astype(np.float32)
    dists = np.sqrt(((tr_xy[:,None,:] - d_xy[None,:,:])**2).sum(axis=2))
    pairs = [(i,j,dists[i,j]) for i in range(T) for j in range(D)]
    pairs.sort(key=lambda x: x[2])
    used_t, used_d = set(), set()
    matches = {}
    for i,j,d in pairs:
        if i in used_t or j in used_d: 
            continue
        if d <= gate_px:
            matches[i]=j
            used_t.add(i); used_d.add(j)
    unmatched_tracks = [i for i in range(T) if i not in used_t]
    unmatched_dets   = [j for j in range(D) if j not in used_d]
    return matches, unmatched_tracks, unmatched_dets

# ----------------------------
# Demo / main
# ----------------------------
if __name__ == "__main__":
    # Example: two frames (you can extend to many frames by looping)
    img1 = cv.imread("src/test/test_images/Test_tracking/0RA_0DEC_FOV(5).png")
    img2 = cv.imread("src/test/test_images/Test_tracking/0RA_0.1DEC_FOV(5).png")

    assert img1 is not None and img2 is not None, "Could not load test images."

    cx,cy,fx,fy = get_intrinsics(img1)
    img_processed1 = processing_image(img1)
    img_processed2 = processing_image(img2)
    h, w =  img1.shape[:2]

    # --- Detect stars in both frames
    p0 = detector(img_processed1, w, h, max_corners=1000, quality_level=0.001, min_distance=4, margin_x=10, margin_y=10)
    p1 = detector(img_processed2, w, h, max_corners=1200, quality_level=0.001, min_distance=4, margin_x=10, margin_y=10)

    if p0.size == 0:
        raise SystemExit("No features found in the first image!")
    print("Features frame1:", len(p0), "frame2:", len(p1))

    # --- Initialize tracks from frame 1 detections
    DT = 1.0        # time between frames (tune to your cadence)
    GATE_PX = 8.0   # association gate in pixels
    ALPHA = 0.7     # update gain (0..1)

    tracks = [Track(tid=i+1, x=pt[0], y=pt[1]) for i,pt in enumerate(p0)]

    # Predict to frame 2 using velocity (initially 0 => predict at same spot)
    for t in tracks:
        t.predict(DT)

    # Associate predictions to detections in frame 2
    matches, un_t, un_d = associate(tracks, p1, gate_px=GATE_PX)

    # Update matched tracks (position + velocity)
    for ti, di in matches.items():
        zx, zy = p1[di]
        tracks[ti].update(zx, zy, dt=DT, alpha=ALPHA)

    # Optionally: spawn tracks for unmatched detections (not needed for just two frames)
    # for di in un_d:
    #     zx, zy = p1[di]
    #     tracks.append(Track(tid=len(tracks)+1, x=zx, y=zy))

    # --- Build matched point arrays for rotation estimate (like your good_old/good_new)
    good_old = np.array([ [tracks[ti].history[0][0], tracks[ti].history[0][1]] for ti in matches.keys() ], dtype=np.float32)
    good_new = np.array([ [tracks[ti].x,                 tracks[ti].y]           for ti in matches.keys() ], dtype=np.float32)

    print(f"Matched pairs: {len(good_old)}")

    # --- Visualization of motion
    img_tracked = img2.copy()
    for (c,d),(a,b) in zip(good_old, good_new):
        cv.line(img_tracked, (int(c), int(d)), (int(a), int(b)), (0,255,0), 2)
        cv.circle(img_tracked, (int(a), int(b)), 3, (0,0,255), -1)
    cv.imshow("Velocity-based tracking", img_tracked)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite("output_tracking_velocity.jpg", img_tracked)

    # --- Rotation from matched stars
    v1 = pixel_to_unit_vector(good_old, cx, cy, fx, fy)
    v2 = pixel_to_unit_vector(good_new, cx, cy, fx, fy)
    rot = estimate_rotation(v1, v2)
    print("Rotation matrix:\n", rot.as_matrix())
    print("Euler angles [°] (zyx):", rot.as_euler('zyx', degrees=True))

    ra_delta  = rot.as_euler('zyx', degrees=True)[1]  # Y-axis
    dec_delta = rot.as_euler('zyx', degrees=True)[0]  # Z-axis
    print(f"ΔRA ≈ {ra_delta:.4f}°, ΔDEC ≈ {dec_delta:.4f}°")

    # --- Boresight examples (same as your code)
    RA0, DEC0 = 0.0, 0.0
    R_cam2eq = build_cam_to_eq(RA0, DEC0, roll_deg=0.0)
    boresight_cam = np.array([0,0,1], dtype=np.float64)

    boresight_eq1 = R_cam2eq @ boresight_cam
    boresight_ra_dec1 = vector_to_radec(boresight_eq1)

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
