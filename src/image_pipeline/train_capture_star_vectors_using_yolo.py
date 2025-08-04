import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from ultralytics import YOLO
from pathlib import Path
import argparse

def train_model(create_new=False, model_name='custom_yolo_186'):
    # Base paths
    base_path = Path(__file__).parent.parent.parent / 'runs/detect'
    weights_path = base_path / f'{model_name}/weights/best.pt'
    
    # If updating existing model and weights exist, use them
    if not create_new and weights_path.exists():
        print(f"Updating existing model: {weights_path}")
        model = YOLO(str(weights_path))
    else:
        print("Creating new model from YOLOv8s")
        model = YOLO('yolov8s.pt')

    data_yaml = str(Path(__file__).parent / 'data.yaml')
    model.train(
        data=data_yaml, 
        imgsz=768, 
        epochs=50,
        batch=8, 
        name=model_name,  # Use same name to overwrite
        device="mps",
        exist_ok=True  # Allow overwriting existing experiment
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--new', action='store_true', help='Create new model instead of updating existing')
    parser.add_argument('--name', default='custom_yolo', help='Model name')
    args = parser.parse_args()
    train_model(create_new= False, model_name=args.name)
#     train_model(create_new=args.new, model_name=args.name)
