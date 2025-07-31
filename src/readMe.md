python3 -m venv .venv

# Activate it
source .venv/bin/activate

pip install --only-binary=:all: opencv-python 
pip install "numpy<2.0"
pip install scikit-image

### Getting This onto Docker


docker buildx build \
  --platform linux/arm/v7 \
  -t johnwick199/star-treckers:latest \
  --push .

# To install the python code on the Raspberry Pi
# 1. On your Raspberry Pi terminal
python3 -m venv .venv
source .venv/bin/activate

# 2. Make folder to hold compatible wheel files
mkdir wheels_pi

# 3. Download platform-specific wheel packages
pip download -r requirements_pi.txt -d wheels_pi/

# 4. Install them efficiently (no internet needed after this)
pip install --no-index --find-links=wheels_pi -r requirements_pi.txt
