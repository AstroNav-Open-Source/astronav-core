python3 -m venv .venv

# Activate it
source .venv/bin/activate

pip install --only-binary=:all: opencv-python 
pip install "numpy<2.0"
pip install scikit-image


ls