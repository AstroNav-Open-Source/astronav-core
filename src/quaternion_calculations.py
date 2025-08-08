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

if __name__ == "__main__":
    Q_STAR_REF = {"w": 1, "x": 0, "y": 0, "z": 0}
    Q_IMU_REF  = {"w": 1, "x": 0, "y": 0, "z": 0}

    from math import sin, cos, pi
    s, c = sin(pi/5), cos(pi/4)
    Q_IMU_CURR = {"w": c, "x": 0, "y": 0, "z": s}

    Q_BODY = propagate_orientation(Q_STAR_REF, Q_IMU_REF, Q_IMU_CURR)
    psy, theta, phi = quat_to_euler(Q_BODY)

    print("Propagated quaternion:", Q_BODY)
    print(f"Yaw ψ ≈ {psy:.1f}°,  Pitch θ ≈ {theta:.1f}°,  Roll φ ≈ {phi:.1f}°")
