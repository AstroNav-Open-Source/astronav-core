from picamera2 import Picamera2
import cv2
import time
import os
from pathlib import Path

def take_image():
    # Init camera (software-friendly preview)
    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (1280, 720)}
        )
    )
    picam2.start()
    time.sleep(0.5)  # small warmup

    # Ensure output dir
    output_dir = Path("photos").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Window + loop (Enter to capture, q/Esc to quit)
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    filename_abs = None
    counter = 0

    print("Press Enter to take a picture (first press saves and returns).")
    print("Press 'q' or Esc to quit without saving.")

    try:
        while True:
            frame = picam2.capture_array()  # RGB numpy
            cv2.imshow("Camera", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (13, 10):  # Enter/Return
                # Save JPEG (convert RGB->BGR for OpenCV)
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                rel = f"image_{counter:03d}.jpg"
                path = output_dir / rel
                cv2.imwrite(str(path), bgr)
                print(f"Saved {path}")
                filename_abs = str(path)
                break
            elif k in (27, ord('q')):  # Esc or 'q'
                break
            # increment name if you want multiple shots per run
            # counter += 1
    finally:
        cv2.destroyAllWindows()
        picam2.close()
        print("Camera stopped.")

    return filename_abs

if __name__ == "__main__":
    take_image()
