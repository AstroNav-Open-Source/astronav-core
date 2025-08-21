import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from image_pipeline.capture_star_vectors import detect_stars, calculate_angular_distances

def find_fov_for_angle(image_path, target_angle=25.77):
     best_fov = None
     best_error = float('inf')
     best_angle = None

     # Test different FOV values
     for fov in range(20, 180, 1):
          try:
               img, thresh, star_data = detect_stars(image_path, visualize=False, fov_deg=fov)
               if len(star_data) < 2:
                    continue     
               angular_pairs = calculate_angular_distances(star_data)
               if not angular_pairs:
                    continue
                    
               # Find closest angle to target
               for pair in angular_pairs:
                    error = abs(pair['angle_deg'] - target_angle)
                    print(f"Angle: {pair['angle_deg']:.2f}°")
                    if error < best_error:
                         best_error = error
                         best_fov = fov
                         best_angle = pair['angle_deg']
          except Exception:
               continue
     return best_fov, best_angle, best_error

if __name__ == '__main__':
    image_path = "/Users/michaelcaneff/Documents/University/Sofia University /Space Challenges/space-treckers/src/test/test_images/control-002.jpeg"
    best_fov, best_angle, error = find_fov_for_angle(image_path)
    print(f"Best FOV: {best_fov}°")
    print(f"Achieved angle: {best_angle:.2f}°")
    print(f"Error: {error:.2f}°")