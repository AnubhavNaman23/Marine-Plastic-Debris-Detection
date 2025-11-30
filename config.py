"""
Configuration settings for Plastic Debris Detector
"""
import os

# =============================================================================
# PATHS - Update these based on your data location
# =============================================================================

# For Local Machine
LOCAL_DATA_PATH = r"C:\Users\anubh\OneDrive\Desktop\Major\MarineDebrisData"
LOCAL_MARIDA_PATH = r"C:\Users\anubh\OneDrive\Desktop\Major\MARIDA"
LOCAL_SPECTRAL_DATA_PATH = r"C:\Users\anubh\OneDrive\Desktop\Major\Floating-Marine-Debris-Data-main\data"

# For Google Colab (update after mounting drive)
COLAB_DATA_PATH = "/content/drive/MyDrive/MarineDebrisData"
COLAB_MARIDA_PATH = "/content/drive/MyDrive/MARIDA"
COLAB_SPECTRAL_DATA_PATH = "/content/drive/MyDrive/Floating-Marine-Debris-Data-main/data"

# =============================================================================
# SENTINEL-2 BAND CONFIGURATION
# =============================================================================

# L2A bands (atmospherically corrected) - 12 bands
L2A_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

# High resolution bands (10m) for RGB + NIR
HR_BANDS = ["B02", "B03", "B04", "B08"]

# Band indices (0-based) for spectral calculations
BAND_INDICES = {
    "B01": 0, "B02": 1, "B03": 2, "B04": 3,
    "B05": 4, "B06": 5, "B07": 6, "B08": 7,
    "B8A": 8, "B09": 9, "B11": 10, "B12": 11
}

# =============================================================================
# MARIDA DATASET CONFIGURATION
# =============================================================================

# MARIDA class labels (from official dataset)
MARIDA_CLASSES = {
    1: "Marine Debris",
    2: "Dense Sargassum",
    3: "Sparse Sargassum",
    4: "Natural Organic Material",
    5: "Ship",
    6: "Clouds",
    7: "Marine Water",
    8: "Sediment-Laden Water",
    9: "Foam",
    10: "Turbid Water",
    11: "Shallow Water",
    12: "Waves",
    13: "Cloud Shadows",
    14: "Wakes",
    15: "Mixed Water"
}

# Binary mapping: Which classes count as "debris"?
DEBRIS_CLASSES = [1]  # Only "Marine Debris" class

# =============================================================================
# SPECTRAL DATA CONFIGURATION (Floating-Marine-Debris-Data)
# =============================================================================

SPECTRAL_CLASSES = {
    1: "Water",
    2: "Plastic",
    3: "Driftwood",
    4: "Seaweed",
    5: "Pumice",
    6: "Sea Snot",
    7: "Sea Foam"
}

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Model architecture options
MODEL_ARCHITECTURES = ["unet", "unetpp", "deeplabv3", "fpn"]

# Default model settings
DEFAULT_MODEL = "unetpp"
DEFAULT_ENCODER = "resnet34"
DEFAULT_ENCODER_WEIGHTS = "imagenet"

# Input settings
INPUT_CHANNELS = 12  # 12 Sentinel-2 L2A bands
OUTPUT_CLASSES = 1   # Binary segmentation (debris / no debris)
INPUT_SIZE = 256     # Patch size for training

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Training hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# Data augmentation
AUGMENTATION = True
RANDOM_FLIP = True
RANDOM_ROTATE = True

# Train/Val/Test split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# =============================================================================
# SPECTRAL INDICES THRESHOLDS
# =============================================================================

# Floating Debris Index (FDI) threshold for detection
FDI_THRESHOLD = 0.02

# Normalized Difference Water Index (NDWI) for water masking
NDWI_WATER_THRESHOLD = 0.1

# Normalized Difference Vegetation Index (NDVI) for vegetation
NDVI_VEGETATION_THRESHOLD = 0.3

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================

# RGB band indices for visualization (B04, B03, B02 -> R, G, B)
RGB_BANDS = [3, 2, 1]

# Color map for prediction overlay
DEBRIS_COLOR = [1, 0, 0]  # Red
WATER_COLOR = [0, 0, 0.5]  # Dark Blue

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_paths(is_colab=False):
    """Get appropriate paths based on environment."""
    if is_colab:
        return {
            "data": COLAB_DATA_PATH,
            "marida": COLAB_MARIDA_PATH,
            "spectral": COLAB_SPECTRAL_DATA_PATH
        }
    else:
        return {
            "data": LOCAL_DATA_PATH,
            "marida": LOCAL_MARIDA_PATH,
            "spectral": LOCAL_SPECTRAL_DATA_PATH
        }

def is_running_in_colab():
    """Check if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False
