# deepfake_face_detection
This project aims to train a deep learning model to classify real versus fake human faces, addressing the rising challenge of deepfakes and digital manipulation.  
Here is the `README.md` content in English:

-----

# Environment Guide

This part provides instructions for setting up the necessary Python environment and installing dependencies for this project.

## Installation

All Python dependencies required for this project are listed in the `requirements.txt` file.

### 1\. (Recommended) Create a Virtual Environment

To avoid conflicts with other Python projects on your system, it is highly recommended to install dependencies in a virtual environment.

**On Windows:**

```bash
# Create a virtual environment named .venv
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\activate
```

**On macOS / Linux:**

```bash
# Create a virtual environment named .venv
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

### 2\. Install Dependencies

Once your virtual environment is activated, run the following command to install all required libraries:

```bash
pip install -r requirements.txt
```

### 3\. Run Experiments
- Method 1:
```commandline
python -m src.experiments.baselines.ViT.main
```
- Method 2: 
```commandline
python -m src.experiments.baselines.MobileNetV3Large.main --epochs 15
```
- Method 3: 
```commandline
cd src/experiments/baselines/BlockShuffleLearning/
python main.py
```
 - Method 4:
```commandline
python -m src.experiments.baselines.ShuffleNetV2.main --epochs 15 --batch_size 32 --lr 1e-3 --image_size 224
```

-----

### **Note on PyTorch (torch)**

The `requirements.txt` file includes `torch` and `torchvision`, which will install the standard CPU-only version from PyPI by default.

If your project requires GPU (CUDA) acceleration, you **need to** visit the [Official PyTorch Website](https://pytorch.org/get-started/locally/) to get the correct installation command for your specific OS and CUDA version. You should run this command **separately** to ensure the GPU-enabled version is installed.

For example, a command for a specific CUDA 11.8 version might look like this (Do not use this command directly, check the website first\!):

```bash
# Example command only!
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
