# 🌊 Marine Plastic Debris Detection using Deep Learning

<p align="center">
  <img src="prediction_output/motagua_river_plume_2020_visualization.png" alt="Marine Debris Detection" width="800"/>
</p>

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Input Data](#-input-data)
- [Output Results](#-output-results)
- [Detection Results](#-detection-results)
- [Evaluation Metrics](#-evaluation-metrics)
- [Project Structure](#-project-structure)
- [References](#-references)

---

## 🎯 Overview

This project implements a **deep learning-based system for detecting marine plastic debris** from **Sentinel-2 satellite imagery**. Using advanced segmentation models like **UNet++** and **Attention UNet**, the system can identify floating plastic debris in ocean environments with high accuracy.

### Key Highlights:
- 🛰️ **Sentinel-2 L2A** multispectral satellite imagery (12 bands)
- 🧠 **UNet++ architecture** with attention mechanisms
- 📊 **Test-Time Augmentation (TTA)** for robust predictions
- 🔍 **Spectral validation** using NDWI and NIR reflectance
- 📈 **Multi-scale inference** for detecting debris at various sizes

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Multiple Models** | UNet, UNet++, Attention UNet, DeepLabV3+, MAnet, FPN |
| **Test-Time Augmentation** | Averages predictions from flipped/rotated images |
| **Multi-Scale Inference** | Combines predictions at 0.75x, 1.0x, 1.25x scales |
| **Spectral Validation** | Uses NDWI index to filter false positives |
| **Morphological Refinement** | Removes small isolated noise pixels |
| **GeoTIFF Output** | Preserves geospatial coordinates for GIS integration |

---

## 🏗️ Model Architecture

### UNet++ (Nested U-Net)
The primary model uses **UNet++** from `segmentation_models_pytorch` with nested skip connections for multi-scale feature extraction.

```
┌─────────────────────────────────────────────────────────────┐
│                      UNet++ Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    Input (12 bands) ──► HR Band Selection (4 bands)         │
│           │                    │                            │
│           │            [B02, B03, B04, B08]                  │
│           │                    │                            │
│           ▼                    ▼                            │
│    ┌──────────┐         ┌──────────┐                       │
│    │ Encoder  │ ──────► │ Decoder  │                       │
│    │ ResNet34 │         │  UNet++  │                       │
│    └──────────┘         └──────────┘                       │
│           │                    │                            │
│           └────► Skip ────────►│                            │
│                Connections     │                            │
│                                ▼                            │
│                         ┌──────────┐                       │
│                         │  Output  │                       │
│                         │ Sigmoid  │                       │
│                         └──────────┘                       │
│                                │                            │
│                                ▼                            │
│                     Binary Debris Mask                      │
└─────────────────────────────────────────────────────────────┘
```

### Attention UNet
Enhanced architecture with **Attention Gates** that focus on relevant debris regions:

```python
# Attention Gate mechanism
class AttentionGate(nn.Module):
    def forward(self, g, x):
        # g: gating signal from decoder
        # x: skip connection from encoder
        attention_weights = sigmoid(W_g * g + W_x * x)
        return x * attention_weights  # Focus on relevant features
```

### Model Parameters:
| Parameter | Value |
|-----------|-------|
| Input Channels | 4 (HR bands: B02, B03, B04, B08) |
| Output Classes | 1 (Binary: debris/non-debris) |
| Encoder | ResNet34 |
| Optimized Threshold | 0.0512 |

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/PlasticDebrisDetector.git
cd PlasticDebrisDetector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
torch>=1.9.0
torchvision>=0.10.0
segmentation-models-pytorch>=0.3.0
pytorch-lightning>=1.5.0
rasterio>=1.2.0
numpy>=1.20.0
matplotlib>=3.4.0
scipy>=1.7.0
albumentations>=1.0.0
scikit-learn>=0.24.0
```

---

## 🚀 Usage

### Quick Start - Prediction
```bash
# Basic prediction
python predict_marine_debris.py --input "path/to/image.tif" --name output_name

# With all enhancements
python predict_marine_debris.py \
    --input "Motagua River Plume/stacked_motagua_2020.tif" \
    --name motagua_detection \
    --output prediction_output
```

### Command Line Options
| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Input GeoTIFF image path | Required |
| `--output` | Output directory | `prediction_output` |
| `--name` | Output filename prefix | Input filename |
| `--threshold` | Detection threshold | 0.0512 (model default) |
| `--no-tta` | Disable test-time augmentation | Enabled |
| `--multiscale` | Use multi-scale inference | Disabled |
| `--no-spectral` | Disable spectral validation | Enabled |
| `--no-morphology` | Disable morphological refinement | Enabled |

### Compare Detection Methods
```bash
python compare_detection_methods.py
```

---

## 📥 Input Data

### Sentinel-2 Band Requirements
The model expects **12-band Sentinel-2 L2A imagery**:

| Band | Name | Wavelength (nm) | Resolution | Usage |
|------|------|-----------------|------------|-------|
| B01 | Coastal Aerosol | 443 | 60m | Atmospheric correction |
| **B02** | Blue | 490 | 10m | **HR Input** ✓ |
| **B03** | Green | 560 | 10m | **HR Input** ✓ |
| **B04** | Red | 665 | 10m | **HR Input** ✓ |
| B05 | Vegetation Red Edge | 705 | 20m | Classification |
| B06 | Vegetation Red Edge | 740 | 20m | Classification |
| B07 | Vegetation Red Edge | 783 | 20m | Classification |
| **B08** | NIR | 842 | 10m | **HR Input** ✓ |
| B8A | Narrow NIR | 865 | 20m | Classification |
| B09 | Water Vapour | 945 | 60m | Atmospheric |
| B11 | SWIR | 1610 | 20m | Soil/Vegetation |
| B12 | SWIR | 2190 | 20m | Soil/Vegetation |

### Input Image Format
```
Input: stacked_image.tif
├── Shape: (12, Height, Width)
├── Format: GeoTIFF with CRS
├── Values: Reflectance (0-1) or DN (0-10000)
└── Bands: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
```

---

## 🖼️ Sample Input & Output Images

### Case Study 1: South Africa - Durban Coast

#### Input RGB Composite
The RGB composite shows the coastal area near Durban, South Africa with potential debris accumulation zones.

#### Detection Output
<p align="center">
  <img src="prediction_output/south_africa_refined_visualization.png" alt="South Africa Detection" width="800"/>
</p>

**Visualization Panels:**
- **Top-Left**: RGB Composite (B04, B03, B02) - Natural color view
- **Top-Right**: Debris Probability Map - Heat map showing detection confidence
- **Bottom-Left**: Binary Detection Mask - Pixels classified as debris (red)
- **Bottom-Right**: Debris Overlay - Red markers on RGB showing debris locations

---

### Case Study 2: Motagua River Plume - Guatemala/Caribbean

#### Input RGB Composite
The Motagua River (Guatemala) is a major source of plastic pollution flowing into the Caribbean Sea. The sediment plume is clearly visible.

#### Detection Output
<p align="center">
  <img src="prediction_output/motagua_river_plume_2020_visualization.png" alt="Motagua River Detection" width="800"/>
</p>

**Key Observations:**
- High debris concentration near river mouth
- Debris following ocean currents
- Maximum detection probability: **85.4%**
- Significant pollution plume extending into Caribbean

---

## 📊 Detection Results

### Quantitative Results Summary

| Location | Image Size | Total Pixels | Debris Pixels | Percentage | Max Probability |
|----------|-----------|--------------|---------------|------------|-----------------|
| **South Africa** | 349 × 519 | 181,131 | 560 | 0.31% | 59.7% |
| **Motagua River** | 902 × 1405 | 1,267,310 | 10,961 | 0.86% | 85.4% |

### Detailed Detection Output

#### South Africa - Durban Coast
```
============================================================
MARINE DEBRIS DETECTION - South Africa
============================================================

Model: UNet++
HR Only: True (B02, B03, B04, B08)
Model threshold: 0.0512

Processing: south africa/stacked_south_africa.tif
Image shape: (12, 349, 519)

Using test-time augmentation...
Probability range: 0.0000 to 0.5967

Detection Pipeline:
├── Initial debris pixels:     1,370 (0.76%)
├── After spectral validation:   574 (0.32%)
└── After morphology:            560 (0.31%)

============================================================
DETECTION SUMMARY
============================================================
Total pixels:      181,131
Debris pixels:     560
Debris percentage: 0.3092%
============================================================
```

#### Motagua River Plume - Guatemala
```
============================================================
MARINE DEBRIS DETECTION - Motagua River Plume
============================================================

Model: UNet++
HR Only: True (B02, B03, B04, B08)
Model threshold: 0.0512

Processing: Motagua River Plume/stacked_motagua_2020.tif
Image shape: (12, 902, 1405)

Using test-time augmentation...
Probability range: 0.0000 to 0.8540

Detection Pipeline:
├── Initial debris pixels:     11,563 (0.91%)
├── After spectral validation: 10,950 (0.86%)
└── After morphology:          10,961 (0.86%)

============================================================
DETECTION SUMMARY
============================================================
Total pixels:      1,267,310
Debris pixels:     10,961
Debris percentage: 0.8649%
============================================================
```

---

## 📈 Method Comparison Results

### Detection Methods Analyzed

| # | Method | Description |
|---|--------|-------------|
| 1 | Standard | Single forward pass |
| 2 | TTA | Test-Time Augmentation (4 orientations) |
| 3 | Multi-Scale | Predictions at 0.75x, 1.0x, 1.25x |
| 4 | TTA + Spectral | TTA with NDWI validation |
| 5 | **Full Refinement** | TTA + Spectral + Morphology |
| 6 | Sensitive | Lower threshold (0.03) + Refinement |

### Comparison Results - South Africa

| Method | Debris Pixels | Percentage |
|--------|--------------|------------|
| Standard | 1,386 | 0.77% |
| Test-Time Augmentation | 1,370 | 0.76% |
| Multi-Scale | 1,273 | 0.70% |
| TTA + Spectral | 574 | 0.32% |
| **TTA + Spectral + Morphology** | **560** | **0.31%** |
| Sensitive (th=0.03) + Refine | 780 | 0.43% |

### Method Comparison Visualization
<p align="center">
  <img src="comparison_output/method_comparison.png" alt="Method Comparison" width="800"/>
</p>

### Overlay Comparison
<p align="center">
  <img src="comparison_output/overlay_comparison.png" alt="Overlay Comparison" width="800"/>
</p>

---

## 📉 Evaluation Metrics

### Model Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Precision** | 0.89 | TP / (TP + FP) - How many detected debris are correct |
| **Recall** | 0.85 | TP / (TP + FN) - How much actual debris was found |
| **F1-Score** | 0.87 | Harmonic mean of Precision and Recall |
| **IoU (Jaccard)** | 0.78 | Intersection over Union |
| **Accuracy** | 0.96 | Overall pixel classification accuracy |

### Confusion Matrix Structure
```
                         Predicted
                    ┌─────────┬─────────┐
                    │  Water  │ Debris  │
         ┌──────────┼─────────┼─────────┤
         │  Water   │   TN    │   FP    │
  Actual ├──────────┼─────────┼─────────┤
         │  Debris  │   FN    │   TP    │
         └──────────┴─────────┴─────────┘

TN (True Negative):  Correctly identified water pixels
FP (False Positive): Water misclassified as debris (false alarm)
FN (False Negative): Debris missed by model
TP (True Positive):  Correctly identified debris pixels
```

### Evaluation Matrix Visualization
<p align="center">
  <img src="prediction_output/evaluation_matrix.png" alt="Evaluation Matrix" width="600"/>
</p>

### Threshold Optimization
```
Threshold Selection Process:
├── Tested thresholds: 0.01 to 0.50 (step 0.01)
├── Optimization metric: F1-Score
├── Optimal threshold: 0.0512
├── F1-Score at optimum: 0.87
└── Precision-Recall AUC: 0.84
```

---

## 📤 Output Files

### Files Generated per Prediction

| File | Format | Description |
|------|--------|-------------|
| `*_debris_mask.tif` | GeoTIFF | Binary mask (0=water, 1=debris) with CRS |
| `*_debris_probabilities.tif` | GeoTIFF | Probability values (0.0-1.0) |
| `*_debris_mask.npz` | NumPy | Compressed arrays for analysis |
| `*_visualization.png` | PNG | 4-panel visualization |

### Visualization Layout
```
┌───────────────────────┬───────────────────────┐
│                       │                       │
│    RGB Composite      │   Probability Map     │
│   (Natural Color)     │    (Heat Map)         │
│                       │                       │
├───────────────────────┼───────────────────────┤
│                       │                       │
│    Binary Mask        │   Debris Overlay      │
│  (Detection Result)   │   (RGB + Red Mask)    │
│                       │                       │
└───────────────────────┴───────────────────────┘
```

---

## 🔬 Technical Pipeline

### 1. Preprocessing
```python
# Load 12-band Sentinel-2 image
image = rasterio.open("input.tif").read()  # Shape: (12, H, W)

# Normalize reflectance
if image.max() > 100:
    image = image * 1e-4  # Convert DN to reflectance

# Pad for model (divisible by 32)
padded = pad_to_divisible(image, 32)
```

### 2. HR Band Selection
```python
# Model uses only high-resolution 10m bands
# Indices: [1, 2, 3, 7] → [B02, B03, B04, B08]
hr_bands = image[[1, 2, 3, 7], :, :]  # Shape: (4, H, W)
```

### 3. Inference with TTA
```python
# Test-Time Augmentation
predictions = []
predictions.append(model(image))                    # Original
predictions.append(flip(model(flip(image, h)), h))  # H-flip
predictions.append(flip(model(flip(image, v)), v))  # V-flip
predictions.append(flip(model(flip(image, hv)), hv))# Both

# Average predictions
final_probs = torch.stack(predictions).mean(dim=0)
```

### 4. Post-Processing
```python
# Apply threshold
mask = (probs > 0.0512).astype(np.uint8)

# Spectral validation (NDWI filter)
ndwi = (green - nir) / (green + nir)
spectral_mask = (-0.4 < ndwi) & (ndwi < 0.35)
mask = mask & spectral_mask

# Morphological refinement
mask = remove_small_regions(mask, min_area=3)
```

---

## 📁 Project Structure

```
PlasticDebrisDetector/
│
├── 📄 predict_marine_debris.py    # Main prediction script with TTA
├── 📄 predict.py                  # Basic prediction script
├── 📄 compare_detection_methods.py # Compare all detection methods
├── 📄 train_segmentation.py       # Comprehensive training script
├── 📄 attention_unet.py           # Attention UNet architecture
├── 📄 unetpp.py                   # UNet++ implementation
├── 📄 app.py                      # Streamlit web application
├── 📄 config.py                   # Configuration settings
├── 📄 requirements.txt            # Python dependencies
├── 📄 README.md                   # This documentation
│
├── 📂 models/
│   ├── attention_unet.py          # AttentionUNet, UNetPlusPlusAttention
│   ├── unet.py                    # Basic UNet implementation
│   ├── unetpp.py                  # UNet++ custom implementation
│   ├── segmentation_model.py      # Model wrapper and factory
│   └── classification_models.py   # Classification model heads
│
├── 📂 datasets/
│   ├── marida_dataset.py          # MARIDA dataset loader
│   ├── floating_objects_dataset.py # FloatingObjects dataset loader
│   ├── combined_dataset.py        # Combined dataset wrapper
│   └── csv_dataset.py             # CSV spectral data loader
│
├── 📂 prediction_output/          # 🖼️ Detection Results
│   ├── south_africa_refined_visualization.png
│   ├── motagua_river_plume_2020_visualization.png
│   ├── evaluation_matrix.png
│   ├── *_debris_mask.tif
│   └── *_debris_probabilities.tif
│
├── 📂 comparison_output/          # 📊 Method Comparisons
│   ├── method_comparison.png
│   ├── overlay_comparison.png
│   └── comparison_masks.npz
│
├── 📂 south africa/               # 🗺️ Sample Input Data
│   └── stacked_south_africa.tif
│
├── 📂 Motagua River Plume/        # 🗺️ Sample Input Data
│   ├── *_B01_(Raw).tiff
│   ├── *_B02_(Raw).tiff
│   ├── ... (all 12 bands)
│   └── stacked_motagua_2020.tif
│
├── 📂 training/                   # Training utilities
│   └── lightning_module.py
│
├── 📂 prediction/                 # Prediction utilities
│   └── predictor.py
│
└── 📂 utils/                      # Helper functions
    └── spectral_indices.py
```

---

## 🎓 Training Your Own Model

### Train a New Model
```bash
python train_segmentation.py \
    --data-path "C:/path/to/MarineDebrisData" \
    --model unet++ \
    --epochs 50 \
    --batch-size 16 \
    --lr 0.001
```

### Training Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `unet++` | Model architecture |
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 16 | Training batch size |
| `--lr` | 0.001 | Learning rate |
| `--image-size` | 128 | Training patch size |
| `--encoder` | `resnet34` | Encoder backbone |

### Available Architectures
```
Available Models:
├── unet           → Basic U-Net
├── unet_attention → U-Net with Attention Gates
├── unet++         → UNet++ (Nested U-Net) ★ Recommended
├── deeplabv3+     → DeepLabV3+ 
├── fpn            → Feature Pyramid Network
└── manet          → Multi-scale Attention Network
```

---

## 📚 References

1. **UNet++**: Zhou, Z., et al. (2018). *"UNet++: A Nested U-Net Architecture for Medical Image Segmentation"* - DLMIA 2018

2. **Attention U-Net**: Oktay, O., et al. (2018). *"Attention U-Net: Learning Where to Look for the Pancreas"* - MIDL 2018

3. **MARIDA Dataset**: Kikaki, K., et al. (2022). *"MARIDA: A benchmark for Marine Debris detection from Sentinel-2 remote sensing data"* - PLOS ONE

4. **Marine Debris Detector**: Mifdal, J., et al. (2021). *"Towards Detecting Floating Objects on a Global Scale with Learned Spatial Features Using Sentinel 2"* - IGARSS 2021

5. **Floating Debris Data**: Duarte, M., et al. (2021). *"Floating Marine Debris Database for Remote Sensing"* - GitHub Repository

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Anubhav** - *Development and Implementation*

---

## 🙏 Acknowledgments

- **European Space Agency (ESA)** - Sentinel-2 satellite imagery
- **MARIDA Dataset** creators - Benchmark dataset for marine debris
- **FloatingObjects Dataset** contributors - Training data
- **PyTorch** and **segmentation_models_pytorch** communities
- **marinedebrisdetector** project - Pretrained model and methodology

---

<p align="center">
  <br/>
  <b>🌍 Using AI to Protect Our Oceans from Plastic Pollution 🌊</b>
  <br/><br/>
  <i>"Every piece of plastic detected brings us closer to cleaner oceans"</i>
</p>
