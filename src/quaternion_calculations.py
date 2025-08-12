from __future__ import annotations
import numpy as np
from math import atan2, asin, degrees

def dict2quat(d: dict[str, float]) -> np.ndarray:
    return np.array([d["w"], d["x"], d["y"], d["z"]], dtype=float)

def quat2dict(q: np.ndarray) -> dict[str, float]:
    return {"w": q[0], "x": q[1], "y": q[2], "z": q[3]}

def norm(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)

def conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])

def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def propagate_orientation(
    Q_STAR_REF: dict[str, float],
    Q_IMU_REF : dict[str, float],
    Q_IMU_CURR: dict[str, float],
) -> dict[str, float]:
    qs  = norm(dict2quat(Q_STAR_REF))
    qi0 = norm(dict2quat(Q_IMU_REF))
    qik = norm(dict2quat(Q_IMU_CURR))

    delta_q_body = mul(qik, conj(qi0))      # body motion since t0
    q_body  = norm(mul(delta_q_body, qs))   # propagated star reference

    return quat2dict(q_body)

def quaternion_angular_distance(q1, q2):
    """Calculate the angular distance between two quaternions."""
    # Convert to numpy arrays
    q1_array = dict2quat(q1) if isinstance(q1, dict) else q1
    q2_array = dict2quat(q2) if isinstance(q2, dict) else q2
    
    # Normalize
    q1_norm = q1_array / np.linalg.norm(q1_array)
    q2_norm = q2_array / np.linalg.norm(q2_array)
    
    # Calculate dot product (cosine of half the angle)
    dot_product = np.abs(np.dot(q1_norm, q2_norm))
    dot_product = np.clip(dot_product, 0, 1)  # Handle numerical errors
    
    # Angular distance in radians, then convert to degrees
    angular_distance = 2 * np.arccos(dot_product)
    return np.degrees(angular_distance)


def quat_to_euler(q: dict[str, float] | np.ndarray, deg: bool = True):
    """
    Convert quaternion to aerospace yaw-pitch-roll 
    • Rotation sequence:  Z (yaw) → Y (pitch) → X (roll)
    • Returns degrees by default.  Set deg=False for radians.
    """
    if isinstance(q, dict):
        q = dict2quat(q)
    w, x, y, z = q

    yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    sinp = 2*(w*y - z*x)
    pitch = asin(max(-1.0, min(1.0, sinp)))

    roll = atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))

    if deg:
        yaw, pitch, roll = map(degrees, (yaw, pitch, roll))
    return yaw, pitch, roll

def quat_to_euler_scipy(q_dict):
    from scipy.spatial.transform import Rotation
    # Convert to scipy format [x, y, z, w]
    quat_scipy = [q_dict['x'], q_dict['y'], q_dict['z'], q_dict['w']]
    r = Rotation.from_quat(quat_scipy)
    # Get ZYX Euler angles (yaw, pitch, roll)
    euler = r.as_euler('zyx', degrees=True)
    return euler[0], euler[1], euler[2]  # yaw, pitch, roll

if __name__ == "__main__":
    Q_STAR_REF = {"w": 0.7, "x": 0.2, "y": 0.3, "z": 0.6}
    Q_IMU_REF  = {"w": 0.6, "x": 0.5, "y": 0.4, "z": 0.5}

    from math import sin, cos, pi
    s, c = sin(pi/5), cos(pi/4)
    Q_IMU_CURR = {"w": c, "x": 0, "y": 0, "z": s}

    Q_BODY = propagate_orientation(Q_STAR_REF, Q_IMU_REF, Q_IMU_CURR)
    psy, theta, phi = quat_to_euler(Q_BODY)

    print("Propagated quaternion:", Q_BODY)
    print(f"Yaw ψ ≈ {psy:.1f}°,  Pitch θ ≈ {theta:.1f}°,  Roll φ ≈ {phi:.1f}°")
    psy_scipy, theta_scipy, phi_scipy = quat_to_euler_scipy(Q_BODY)
    print(f"Yaw ψ ≈ {psy_scipy:.1f}°,  Pitch θ ≈ {theta_scipy:.1f}°,  Roll φ ≈ {phi_scipy:.1f}°")
