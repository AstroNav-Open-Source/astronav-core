import numpy as np

def dict_to_quaternion(quat_dict):
    """Convert dictionary representation to numpy quaternion array"""
    return np.array([quat_dict["w"], quat_dict["x"], quat_dict["y"], quat_dict["z"]])

def quaternion_to_dict(quat_array):
    """Convert numpy quaternion array to dictionary representation"""
    return {
        "w": quat_array[0],
        "x": quat_array[1], 
        "y": quat_array[2],
        "z": quat_array[3]
    }

def quaternion_multiply(q1, q2):
    """Multiply two quaternions"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])

def quaternion_conjugate(q):
    """Get conjugate of quaternion"""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def calculate_delta_quaternion(quaternion_star, quaternion_imu=None):
    # Convert dictionaries to numpy arrays
    q_star = dict_to_quaternion(quaternion_star)
    q_imu = dict_to_quaternion(quaternion_imu)
    
    # Calculate delta quaternion: q_delta = q_star * q_imu_conjugate
    q_imu_conj = quaternion_conjugate(q_imu)
    delta_quaternion = quaternion_multiply(q_star, q_imu_conj)
    
    # Convert back to dictionary format
    return quaternion_to_dict(delta_quaternion)

def calculate_final_quaternion_to_transmit(quaternion_star, quaternion_imu):
    delta_quaternion = calculate_delta_quaternion(quaternion_star, quaternion_imu)
    final_quaternion = quaternion_multiply(quaternion_star, delta_quaternion)
    return final_quaternion
    

if __name__ == "__main__":
    quaternion_star = {
        "x": 0.1,
        "y": 0.2,
        "z": 0.3,
        "w": 0.4
    }
    quaternion_imu = {
        "x": 0.5,
        "y": 0.6,
        "z": 0.7,
        "w": 0.8
    }
    delta_quaternion = calculate_delta_quaternion(quaternion_star, quaternion_imu)
    print("Delta quaternion:", delta_quaternion)