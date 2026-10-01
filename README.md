# Face Segmenter

A deep learning project for **pixel-level face segmentation** using a custom convolutional neural network implemented with **PyTorch**.

The objective of this project is to segment a face image into multiple semantic regions such as skin, eyes, eyebrows, nose, mouth, hair, ears, neck, and clothing.

---

## 📌 Overview

Face segmentation is a computer vision task in which each pixel of an image is assigned to a specific semantic category.

Unlike traditional face detection, which only identifies the location of a face, this project performs **pixel-level semantic segmentation**, allowing different regions of the face to be identified independently.

The project implements the complete pipeline:

* Dataset preparation
* Segmentation mask generation
* Mask visualization
* Dataset organization
* Model training
* Model evaluation
* Pixel-level prediction

---

## 🧠 How It Works

The complete pipeline can be summarized as follows:

```text
Original Images
      │
      ▼
Segmentation Annotations
      │
      ▼
Combine Individual Masks
      │
      ▼
Multi-Class Segmentation Masks
      │
      ▼
Train / Validation / Test Split
      │
      ▼
Image Preprocessing
      │
      ▼
Custom CNN Segmentation Model
      │
      ▼
Model Training
      │
      ▼
Trained Model
      │
      ▼
Inference
      │
      ▼
Pixel-Level Segmentation Masks
```

---

## 🎯 Segmentation Classes

The model predicts **19 semantic classes**:

| ID | Class         |
| -: | ------------- |
|  0 | Background    |
|  1 | Skin          |
|  2 | Nose          |
|  3 | Eye region    |
|  4 | Eye region    |
|  5 | Left eye      |
|  6 | Right eye     |
|  7 | Left eyebrow  |
|  8 | Right eyebrow |
|  9 | Left ear      |
| 10 | Right ear     |
| 11 | Mouth         |
| 12 | Upper lip     |
| 13 | Lower lip     |
| 14 | Hair          |
| 15 | Hat           |
| 16 | Ear region    |
| 17 | Neck          |
| 18 | Clothing      |

---

## 📂 Dataset Preparation

The original annotations contain separate masks for different facial regions.

The project combines these individual masks into a single multi-class segmentation mask.

The main segmentation labels used during preprocessing are:

```python
segments_labels = [
    'skin',
    'nose',
    'eye_g',
    'l_eye',
    'r_eye',
    'l_brow',
    'r_brow',
    'l_ear',
    'r_ear',
    'mouth',
    'u_lip',
    'l_lip',
    'hair',
    'hat',
    'ear_r',
    'neck_l',
    'neck',
    'cloth'
]
```

Each region is assigned a numerical class ID, producing a single segmentation map for each image.

The project also generates colored segmentation masks to make the different facial regions easier to visualize.

---

## 🗂️ Dataset Organization

The dataset is organized into training, validation, and testing subsets:

```text
train_images/
train_labels/

val_images/
val_labels/

test_images/
test_labels/
```

Each image is associated with a corresponding segmentation mask:

```text
image.jpg
label.png
```

The notebook uses an image mapping file to organize the dataset and associate each image with its corresponding label.

---

## 🔄 Data Preprocessing

The input images are converted to PyTorch tensors and normalized using:

```python
transforms.Normalize(
    (0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5)
)
```

The segmentation masks are converted using:

```python
transforms.ToTensor()
```

The segmentation data is processed at a resolution of:

```text
512 × 512 pixels
```

---

## 🏗️ Model Architecture

The project uses a custom **encoder-decoder convolutional neural network** inspired by the U-Net architecture.

The architecture consists of:

### Encoder

The encoder extracts increasingly complex features from the input image using convolutional blocks and max-pooling.

Each convolutional block contains:

```text
3×3 Convolution
      ↓
Batch Normalization
      ↓
ReLU
      ↓
3×3 Convolution
      ↓
Batch Normalization
      ↓
ReLU
```

### Decoder

The decoder progressively reconstructs the spatial resolution using transposed convolutions.

Skip connections are used between the encoder and decoder to preserve spatial information needed for accurate segmentation.

Simplified architecture:

```text
Input
  │
  ▼
Encoder
  │
  ├───────────────┐
  │               │
  ▼               │
Encoder           │
  │               │
  ├───────────────┤
  │               │
  ▼               │
Bottleneck        │
  │               │
  ▼               │
Decoder ◄─────────┘
  │
  ▼
19-Class Output
```

---

## ⚙️ Model Configuration

The original filter configuration was reduced to decrease computational and memory requirements.

The model uses:

```text
[16, 32, 64, 128, 256]
```

feature channels throughout the encoder-decoder architecture.

The final layer is a `1×1` convolution producing **19 output channels**, one for each segmentation class.

---

## 🏋️ Training

The model is trained using the following configuration:

| Parameter          | Value                      |
| ------------------ | -------------------------- |
| Framework          | PyTorch                    |
| Architecture       | Custom Encoder-Decoder CNN |
| Number of classes  | 19                         |
| Input resolution   | 512 × 512                  |
| Loss function      | Cross-Entropy Loss         |
| Optimizer          | Adam                       |
| Number of epochs   | 50                         |
| Batch size         | 10                         |
| DataLoader workers | 2                          |

The training process follows:

```text
Input Image
     ↓
CNN
     ↓
19-Class Prediction
     ↓
Cross-Entropy Loss
     ↓
Backpropagation
     ↓
Adam Optimizer
     ↓
Updated Model
```

---

## 🖥️ GPU Acceleration

The implementation automatically uses CUDA when a compatible GPU is available:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

The notebook was developed and tested using an NVIDIA Tesla T4 GPU.

The notebook also records the following approximate training time:

```text
1,000 images  → approximately 1 minute
30,000 images → approximately 4 hours 30 minutes
```

Actual training time depends on the hardware, batch size, GPU configuration, and runtime environment.

---

## 🔬 Inference

After training, the model weights are saved and loaded for inference.

The model is switched to evaluation mode:

```python
model.eval()
```

For every input image, the network produces a prediction for each pixel across the 19 classes.

The class with the highest prediction score is selected:

```python
prediction = image.data.max(1)[1]
```

The resulting segmentation masks are saved for further visualization and evaluation.

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/IsmailEssbiti/Face-Segmenter.git
cd Face-Segmenter
```

Install the required dependencies:

```bash
pip install torch torchvision
pip install opencv-python
pip install numpy
pip install pandas
pip install pillow
pip install jupyter
```

Alternatively, install all dependencies using:

```bash
pip install -r requirements.txt
```

## CUDA Installation

If you have an NVIDIA GPU and want to use CUDA acceleration, install the CUDA-enabled version of PyTorch separately from the other dependencies.

### 1. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

### 2. Install PyTorch with CUDA support

For example, using the CUDA 12.6 build:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

PyTorch also provides CUDA 12.8 and CUDA 13.0 builds for supported versions and GPUs. Check the official PyTorch installation page and select the CUDA version appropriate for your system.

### 3. Install the remaining dependencies

After installing PyTorch, install the other project dependencies:

```bash
pip install opencv-python numpy pandas Pillow jupyter
```

Alternatively, if `requirements.txt` contains the non-PyTorch dependencies:

```bash
pip install -r requirements.txt
```

### 4. Verify CUDA

Run the following Python code:

```python
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

You should see:

```text
CUDA available: True
GPU: NVIDIA ...
```

### CPU-only installation

If you do not have an NVIDIA GPU, install the CPU version instead:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**The project automatically selects CUDA when it is available:**

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Therefore, the same notebook can run on either an NVIDIA GPU or the CPU, although GPU acceleration is strongly recommended for training.**

---

## ▶️ Usage

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Face_Segmenter.ipynb
```

Run the notebook cells in order.

The main stages are:

1. Prepare the segmentation annotations.
2. Generate the multi-class masks.
3. Organize the dataset.
4. Create the training and testing datasets.
5. Initialize the segmentation model.
6. Train the model.
7. Save the trained model.
8. Run inference.
9. Visualize the predicted segmentation masks.

---

## ⚠️ Limitations

The current implementation has several limitations:

* The model operates on a fixed `512 × 512` resolution.
* Training requires a suitable GPU for practical training times.
* The current notebook does not automatically calculate IoU, Dice, precision, or recall.
* Dataset preparation and model training are primarily implemented inside a Jupyter Notebook.
* The complete dataset is required to reproduce the training process.

---

## 🛠️ Technologies

* Python
* PyTorch
* Torchvision
* OpenCV
* NumPy
* Pandas
* Pillow
* Jupyter Notebook
* CUDA

---

## 👤 Author

**Ismail Essbiti**

GitHub: `https://github.com/IsmailEssbiti`

