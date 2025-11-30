"""
Marine Debris Detector - GUI Application

A Tkinter-based graphical interface for:
1. Predicting debris in satellite imagery
2. Training models
3. Visualizing results

Usage:
    python app.py
"""

import os
import sys
import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

# Project setup
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


class MarineDebrisApp:
    """Main application class."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🌊 Marine Debris Detector")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Variables
        self.model_path = tk.StringVar()
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar(value=str(PROJECT_ROOT / "prediction_output"))
        self.threshold = tk.DoubleVar(value=0.5)
        self.tile_size = tk.IntVar(value=128)
        
        # Training variables
        self.train_data_path = tk.StringVar()
        self.train_model_type = tk.StringVar(value="mlp")
        self.train_epochs = tk.IntVar(value=15)
        self.train_batch_size = tk.IntVar(value=256)
        
        # Results storage
        self.current_mask = None
        self.current_probs = None
        
        # Device info
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Status bar (create BEFORE UI so log() works)
        self.status_var = tk.StringVar(value=f"Ready | Device: {self.device.upper()}")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create UI
        self.create_menu()
        self.create_main_layout()
        self.update_model_list()
    
    def create_menu(self):
        """Create menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image...", command=self.browse_input)
        file_menu.add_command(label="Load Model...", command=self.browse_model)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self.show_help)
        help_menu.add_command(label="About", command=self.show_about)
    
    def create_main_layout(self):
        """Create main application layout."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.predict_tab = ttk.Frame(self.notebook)
        self.train_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)
        self.help_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.predict_tab, text="🔮 Predict")
        self.notebook.add(self.train_tab, text="🎓 Train")
        self.notebook.add(self.results_tab, text="📊 Results")
        self.notebook.add(self.help_tab, text="📚 Help")
        
        # Setup each tab
        self.setup_predict_tab()
        self.setup_train_tab()
        self.setup_results_tab()
        self.setup_help_tab()
    
    def setup_predict_tab(self):
        """Setup prediction tab."""
        # Main frame with two columns
        left_frame = ttk.LabelFrame(self.predict_tab, text="Settings", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        right_frame = ttk.LabelFrame(self.predict_tab, text="Preview", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # === Left Panel: Settings ===
        
        # Model Selection
        ttk.Label(left_frame, text="1. Select Model", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        model_frame = ttk.Frame(left_frame)
        model_frame.pack(fill=tk.X, pady=5)
        
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_path, width=40)
        self.model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(model_frame, text="Browse", command=self.browse_model).pack(side=tk.RIGHT, padx=(5, 0))
        
        ttk.Button(left_frame, text="↻ Refresh Models", command=self.update_model_list).pack(anchor=tk.W, pady=5)
        
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Input Selection
        ttk.Label(left_frame, text="2. Select Input", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        input_frame = ttk.Frame(left_frame)
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Entry(input_frame, textvariable=self.input_path, width=40).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(input_frame, text="Browse", command=self.browse_input).pack(side=tk.RIGHT, padx=(5, 0))
        
        ttk.Label(left_frame, text="Supported: .tif, .tiff, .npy, .npz, .SAFE folders", 
                  font=('Helvetica', 8), foreground='gray').pack(anchor=tk.W)
        
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Parameters
        ttk.Label(left_frame, text="3. Parameters", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        param_frame = ttk.Frame(left_frame)
        param_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(param_frame, text="Threshold:").grid(row=0, column=0, sticky=tk.W, pady=2)
        threshold_spin = ttk.Spinbox(param_frame, from_=0.1, to=0.9, increment=0.05, 
                                     textvariable=self.threshold, width=10)
        threshold_spin.grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(param_frame, text="Tile Size:").grid(row=1, column=0, sticky=tk.W, pady=2)
        tile_combo = ttk.Combobox(param_frame, textvariable=self.tile_size, 
                                  values=[64, 128, 256, 512], width=10, state='readonly')
        tile_combo.grid(row=1, column=1, sticky=tk.W, pady=2)
        
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Output
        ttk.Label(left_frame, text="4. Output Directory", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        output_frame = ttk.Frame(left_frame)
        output_frame.pack(fill=tk.X, pady=5)
        
        ttk.Entry(output_frame, textvariable=self.output_path, width=40).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(output_frame, text="Browse", command=self.browse_output).pack(side=tk.RIGHT, padx=(5, 0))
        
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Run Button
        self.predict_btn = ttk.Button(left_frame, text="🚀 Run Detection", 
                                      command=self.run_prediction, style='Accent.TButton')
        self.predict_btn.pack(fill=tk.X, pady=10)
        
        # Progress
        self.predict_progress = ttk.Progressbar(left_frame, mode='indeterminate')
        self.predict_progress.pack(fill=tk.X, pady=5)
        
        # === Right Panel: Preview ===
        self.preview_canvas = tk.Canvas(right_frame, bg='#f0f0f0')
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Placeholder text
        self.preview_canvas.create_text(
            300, 200, 
            text="Results will appear here\nafter running detection",
            font=('Helvetica', 14),
            fill='gray',
            justify=tk.CENTER
        )
    
    def setup_train_tab(self):
        """Setup training tab."""
        # Training type selection
        type_frame = ttk.LabelFrame(self.train_tab, text="Training Type", padding=10)
        type_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.train_type = tk.StringVar(value="classifier")
        ttk.Radiobutton(type_frame, text="Pixel Classification (Fast)", 
                        variable=self.train_type, value="classifier").pack(anchor=tk.W)
        ttk.Radiobutton(type_frame, text="Image Segmentation (Slower)", 
                        variable=self.train_type, value="segmentation").pack(anchor=tk.W)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(self.train_tab, text="Training Settings", padding=10)
        settings_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Data path
        ttk.Label(settings_frame, text="Data Directory:").grid(row=0, column=0, sticky=tk.W, pady=5)
        data_frame = ttk.Frame(settings_frame)
        data_frame.grid(row=0, column=1, sticky=tk.EW, pady=5)
        ttk.Entry(data_frame, textvariable=self.train_data_path, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(data_frame, text="Browse", command=self.browse_train_data).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Model type
        ttk.Label(settings_frame, text="Model:").grid(row=1, column=0, sticky=tk.W, pady=5)
        model_types = ["mlp", "cnn1d", "senet", "resnet", "transformer", "unet", "unet++"]
        ttk.Combobox(settings_frame, textvariable=self.train_model_type, 
                     values=model_types, state='readonly', width=20).grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Epochs
        ttk.Label(settings_frame, text="Epochs:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(settings_frame, from_=5, to=200, textvariable=self.train_epochs, width=10).grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Batch size
        ttk.Label(settings_frame, text="Batch Size:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(settings_frame, textvariable=self.train_batch_size, 
                     values=[32, 64, 128, 256, 512], state='readonly', width=10).grid(row=3, column=1, sticky=tk.W, pady=5)
        
        settings_frame.columnconfigure(1, weight=1)
        
        # Train button
        btn_frame = ttk.Frame(self.train_tab)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(btn_frame, text="🚀 Start Training", command=self.run_training).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="🔬 Compare Models", command=self.run_comparison).pack(side=tk.LEFT, padx=10)
        
        # Training progress
        self.train_progress = ttk.Progressbar(self.train_tab, mode='indeterminate')
        self.train_progress.pack(fill=tk.X, padx=10, pady=5)
        
        # Log output
        log_frame = ttk.LabelFrame(self.train_tab, text="Training Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.train_log = scrolledtext.ScrolledText(log_frame, height=15, font=('Consolas', 9))
        self.train_log.pack(fill=tk.BOTH, expand=True)
    
    def setup_results_tab(self):
        """Setup results tab."""
        # Stats frame
        stats_frame = ttk.LabelFrame(self.results_tab, text="Detection Statistics", padding=10)
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.stats_labels = {}
        stats = [("Debris Pixels:", "debris_pixels"), ("Coverage:", "coverage"), 
                 ("Image Size:", "image_size"), ("Threshold:", "threshold_used")]
        
        for i, (label, key) in enumerate(stats):
            ttk.Label(stats_frame, text=label, font=('Helvetica', 10, 'bold')).grid(row=0, column=i*2, padx=10, pady=5)
            self.stats_labels[key] = ttk.Label(stats_frame, text="-")
            self.stats_labels[key].grid(row=0, column=i*2+1, padx=10, pady=5)
        
        # Visualization frame
        viz_frame = ttk.LabelFrame(self.results_tab, text="Visualization", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create matplotlib figure
        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initially show placeholder
        for ax in self.axes:
            ax.text(0.5, 0.5, 'Run detection to see results', 
                    ha='center', va='center', fontsize=12, color='gray')
            ax.axis('off')
        self.canvas.draw()
        
        # Export buttons
        export_frame = ttk.Frame(self.results_tab)
        export_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(export_frame, text="💾 Save Mask (NPY)", command=self.save_mask_npy).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_frame, text="📷 Save Image (PNG)", command=self.save_mask_png).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_frame, text="🗺️ Save GeoTIFF", command=self.save_mask_geotiff).pack(side=tk.LEFT, padx=5)
    
    def setup_help_tab(self):
        """Setup help tab."""
        help_text = scrolledtext.ScrolledText(self.help_tab, font=('Helvetica', 10), wrap=tk.WORD)
        help_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        help_content = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                         MARINE DEBRIS DETECTOR - HELP                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

📁 SUPPORTED INPUT FORMATS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Sentinel-2 .SAFE Folder
   Download directly from Copernicus (https://browser.dataspace.copernicus.eu/)
   Structure:
   S2A_MSIL2A_*.SAFE/
   └── GRANULE/
       └── L2A_*/
           └── IMG_DATA/
               ├── R10m/  (B02, B03, B04, B08)
               ├── R20m/  (B05, B06, B07, B8A, B11, B12)
               └── R60m/  (B01, B09)

2. GeoTIFF File (.tif, .tiff)
   - 12-band stacked image
   - Band order: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
   - Values: 0-10000 (will be scaled automatically)

3. NumPy Array (.npy, .npz)
   - Shape: (12, height, width) or (height, width, 12)
   - For .npz files, use key 'image' or 'data'


🤖 MODELS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Segmentation Models (for full images):
  • UNet      - Classic encoder-decoder
  • UNet++    - Nested connections (BEST accuracy)
  • DeepLabV3+- Multi-scale features
  • MANet     - Attention-based

Classification Models (for pixel data):
  • MLP       - Fast, accurate (RECOMMENDED)
  • CNN1D     - Good for spectral patterns
  • SENet     - Attention on bands
  • ResNet    - Residual connections
  • Transformer - Self-attention


⚙️ PARAMETERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Threshold (0.1-0.9):
  - Lower = More sensitive (more false positives)
  - Higher = More strict (may miss debris)
  - Default: 0.5

Tile Size:
  - Larger = Faster but needs more memory
  - Smaller = Slower but works on any system
  - Default: 128


💻 COMMAND LINE USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Predict
python predict.py --input image.tif --model checkpoints/model.pt

# Train classifier
python train_classifier.py --data-dir datasets/csv_data --model mlp --epochs 20

# Train segmentation
python train_segmentation.py --data-path /path/to/data --model unet++ --epochs 50

# Compare models
python compare_models.py --epochs 15


📊 OUTPUT CLASSES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Class 0: Water (clean)
Class 1: Marine Debris / Plastic
Class 2: Dense Sargassum
Class 3: Sparse Floating Algae
Class 4: Natural Organic Material
Class 5: Ship
Class 6: Clouds
Class 7: Marine Water


🔧 TROUBLESHOOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"No models found":
  → Train a model first or download a pre-trained checkpoint

"Out of memory":
  → Reduce tile size or batch size

"Model incompatible":
  → Check that model type matches checkpoint (segmentation vs classification)

"Invalid input":
  → Ensure image has 12 Sentinel-2 bands in correct order


📧 SUPPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For issues, please check the README.md or create an issue on GitHub.
        """
        
        help_text.insert(tk.END, help_content)
        help_text.config(state=tk.DISABLED)
    
    # ==================== Helper Methods ====================
    
    def update_model_list(self):
        """Update the list of available models."""
        models = []
        
        # Search directories for models
        search_dirs = [
            PROJECT_ROOT / "checkpoints",
            PROJECT_ROOT / "classifier_output" / "checkpoints",
            PROJECT_ROOT / "segmentation_output" / "checkpoints",
            Path(r"C:\Users\anubh\OneDrive\Desktop\Major\MarineDebrisProject\checkpoints"),
        ]
        
        for search_dir in search_dirs:
            if search_dir.exists():
                for ext in ['*.pt', '*.ckpt', '*.pth']:
                    models.extend(search_dir.rglob(ext))
        
        # Update combobox
        model_paths = [str(m) for m in models]
        self.model_combo['values'] = model_paths
        
        if model_paths:
            self.model_combo.set(model_paths[0])
            self.log(f"Found {len(model_paths)} model(s)")
        else:
            self.log("No models found. Train a model first.")
    
    def browse_model(self):
        """Browse for model file."""
        path = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            filetypes=[
                ("PyTorch Models", "*.pt *.pth *.ckpt"),
                ("All Files", "*.*")
            ]
        )
        if path:
            self.model_path.set(path)
    
    def browse_input(self):
        """Browse for input file or folder."""
        # Ask user for type
        choice = messagebox.askquestion(
            "Input Type",
            "Is your input a folder (.SAFE directory)?\n\nYes = Browse Folder\nNo = Browse File"
        )
        
        if choice == 'yes':
            path = filedialog.askdirectory(title="Select .SAFE Folder")
        else:
            path = filedialog.askopenfilename(
                title="Select Input Image",
                filetypes=[
                    ("GeoTIFF", "*.tif *.tiff"),
                    ("NumPy", "*.npy *.npz"),
                    ("All Files", "*.*")
                ]
            )
        
        if path:
            self.input_path.set(path)
            self.log(f"Input: {Path(path).name}")
    
    def browse_output(self):
        """Browse for output directory."""
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            self.output_path.set(path)
    
    def browse_train_data(self):
        """Browse for training data."""
        path = filedialog.askdirectory(title="Select Training Data Directory")
        if path:
            self.train_data_path.set(path)
    
    def log(self, message):
        """Log message to training log and status."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_var.set(f"{message} | Device: {self.device.upper()}")
        
        if hasattr(self, 'train_log'):
            self.train_log.insert(tk.END, f"[{timestamp}] {message}\n")
            self.train_log.see(tk.END)
    
    # ==================== Prediction ====================
    
    def run_prediction(self):
        """Run debris detection."""
        # Validate inputs
        if not self.model_path.get():
            messagebox.showerror("Error", "Please select a model checkpoint.")
            return
        
        if not self.input_path.get():
            messagebox.showerror("Error", "Please select an input image.")
            return
        
        if not Path(self.model_path.get()).exists():
            messagebox.showerror("Error", f"Model not found: {self.model_path.get()}")
            return
        
        if not Path(self.input_path.get()).exists():
            messagebox.showerror("Error", f"Input not found: {self.input_path.get()}")
            return
        
        # Run in thread
        self.predict_btn.config(state=tk.DISABLED)
        self.predict_progress.start()
        self.log("Starting prediction...")
        
        thread = threading.Thread(target=self._run_prediction_thread)
        thread.start()
    
    def _run_prediction_thread(self):
        """Prediction thread."""
        try:
            from predict import MarineDebrisPredictor
            
            predictor = MarineDebrisPredictor(
                self.model_path.get(),
                threshold=self.threshold.get(),
                device=self.device
            )
            
            self.root.after(0, lambda: self.log("Model loaded, processing image..."))
            
            mask, probs = predictor.predict(
                self.input_path.get(),
                tile_size=self.tile_size.get()
            )
            
            # Store results
            self.current_mask = mask
            self.current_probs = probs
            
            # Update UI on main thread
            self.root.after(0, lambda: self._prediction_complete(mask, probs))
            
        except Exception as e:
            self.root.after(0, lambda: self._prediction_error(str(e)))
    
    def _prediction_complete(self, mask, probs):
        """Handle prediction completion."""
        self.predict_progress.stop()
        self.predict_btn.config(state=tk.NORMAL)
        
        # Update stats
        self.stats_labels['debris_pixels'].config(text=f"{mask.sum():,}")
        self.stats_labels['coverage'].config(text=f"{mask.mean()*100:.2f}%")
        self.stats_labels['image_size'].config(text=f"{mask.shape[0]} x {mask.shape[1]}")
        self.stats_labels['threshold_used'].config(text=f"{self.threshold.get():.2f}")
        
        # Update visualization
        self.axes[0].clear()
        self.axes[0].imshow(mask, cmap='Reds')
        self.axes[0].set_title("Debris Detection Mask")
        self.axes[0].axis('off')
        
        self.axes[1].clear()
        im = self.axes[1].imshow(probs, cmap='hot', vmin=0, vmax=1)
        self.axes[1].set_title("Probability Map")
        self.axes[1].axis('off')
        
        # Add colorbar
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Switch to results tab
        self.notebook.select(self.results_tab)
        
        self.log(f"Detection complete! Found {mask.sum():,} debris pixels")
        messagebox.showinfo("Success", f"Detection complete!\n\nDebris pixels: {mask.sum():,}\nCoverage: {mask.mean()*100:.2f}%")
    
    def _prediction_error(self, error_msg):
        """Handle prediction error."""
        self.predict_progress.stop()
        self.predict_btn.config(state=tk.NORMAL)
        self.log(f"Error: {error_msg}")
        messagebox.showerror("Prediction Error", f"An error occurred:\n\n{error_msg}")
    
    # ==================== Training ====================
    
    def run_training(self):
        """Start model training."""
        if not self.train_data_path.get():
            messagebox.showerror("Error", "Please select training data directory.")
            return
        
        self.train_progress.start()
        self.log("Starting training...")
        
        thread = threading.Thread(target=self._run_training_thread)
        thread.start()
    
    def _run_training_thread(self):
        """Training thread."""
        try:
            import subprocess
            
            if self.train_type.get() == "classifier":
                script = PROJECT_ROOT / "train_classifier.py"
                cmd = [
                    sys.executable, str(script),
                    "--data-dir", self.train_data_path.get(),
                    "--model", self.train_model_type.get(),
                    "--epochs", str(self.train_epochs.get()),
                    "--batch-size", str(self.train_batch_size.get()),
                    "--workers", "0"
                ]
            else:
                script = PROJECT_ROOT / "train_segmentation.py"
                cmd = [
                    sys.executable, str(script),
                    "--data-path", self.train_data_path.get(),
                    "--model", self.train_model_type.get(),
                    "--epochs", str(self.train_epochs.get()),
                    "--batch-size", str(self.train_batch_size.get()),
                    "--workers", "0"
                ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(PROJECT_ROOT)
            )
            
            for line in process.stdout:
                self.root.after(0, lambda l=line: self._log_training(l))
            
            process.wait()
            
            self.root.after(0, lambda: self._training_complete(process.returncode))
            
        except Exception as e:
            self.root.after(0, lambda: self._training_error(str(e)))
    
    def _log_training(self, line):
        """Log training output."""
        self.train_log.insert(tk.END, line)
        self.train_log.see(tk.END)
    
    def _training_complete(self, return_code):
        """Handle training completion."""
        self.train_progress.stop()
        self.update_model_list()
        
        if return_code == 0:
            self.log("Training complete!")
            messagebox.showinfo("Success", "Training completed successfully!")
        else:
            self.log(f"Training failed with code {return_code}")
            messagebox.showerror("Error", "Training failed. Check log for details.")
    
    def _training_error(self, error_msg):
        """Handle training error."""
        self.train_progress.stop()
        self.log(f"Training error: {error_msg}")
        messagebox.showerror("Error", f"Training error:\n\n{error_msg}")
    
    def run_comparison(self):
        """Run model comparison."""
        if not self.train_data_path.get():
            messagebox.showerror("Error", "Please select training data directory.")
            return
        
        self.train_progress.start()
        self.log("Starting model comparison...")
        
        thread = threading.Thread(target=self._run_comparison_thread)
        thread.start()
    
    def _run_comparison_thread(self):
        """Model comparison thread."""
        try:
            import subprocess
            
            script = PROJECT_ROOT / "compare_models.py"
            cmd = [
                sys.executable, str(script),
                "--data-dir", self.train_data_path.get(),
                "--epochs", str(self.train_epochs.get()),
                "--models", "mlp", "cnn1d", "senet", "resnet"
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(PROJECT_ROOT)
            )
            
            for line in process.stdout:
                self.root.after(0, lambda l=line: self._log_training(l))
            
            process.wait()
            
            self.root.after(0, lambda: self._training_complete(process.returncode))
            
        except Exception as e:
            self.root.after(0, lambda: self._training_error(str(e)))
    
    # ==================== Export ====================
    
    def save_mask_npy(self):
        """Save mask as NumPy file."""
        if self.current_mask is None:
            messagebox.showwarning("Warning", "No results to save. Run detection first.")
            return
        
        path = filedialog.asksaveasfilename(
            defaultextension=".npy",
            filetypes=[("NumPy Array", "*.npy")]
        )
        if path:
            np.save(path, self.current_mask)
            self.log(f"Saved: {path}")
            messagebox.showinfo("Success", f"Saved mask to:\n{path}")
    
    def save_mask_png(self):
        """Save visualization as PNG."""
        if self.current_mask is None:
            messagebox.showwarning("Warning", "No results to save. Run detection first.")
            return
        
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")]
        )
        if path:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(self.current_mask, cmap='Reds')
            ax.set_title("Marine Debris Detection")
            ax.axis('off')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            self.log(f"Saved: {path}")
            messagebox.showinfo("Success", f"Saved image to:\n{path}")
    
    def save_mask_geotiff(self):
        """Save mask as GeoTIFF."""
        if self.current_mask is None:
            messagebox.showwarning("Warning", "No results to save. Run detection first.")
            return
        
        try:
            import rasterio
        except ImportError:
            messagebox.showerror("Error", "rasterio not installed. Cannot save GeoTIFF.")
            return
        
        path = filedialog.asksaveasfilename(
            defaultextension=".tif",
            filetypes=[("GeoTIFF", "*.tif")]
        )
        if path:
            # Try to copy georeference from input
            try:
                with rasterio.open(self.input_path.get()) as src:
                    profile = src.profile
                    profile.update(count=1, dtype='uint8')
                    
                    with rasterio.open(path, 'w', **profile) as dst:
                        dst.write(self.current_mask.astype(np.uint8), 1)
            except:
                # Save without georeference
                profile = {
                    'driver': 'GTiff',
                    'height': self.current_mask.shape[0],
                    'width': self.current_mask.shape[1],
                    'count': 1,
                    'dtype': 'uint8'
                }
                with rasterio.open(path, 'w', **profile) as dst:
                    dst.write(self.current_mask.astype(np.uint8), 1)
            
            self.log(f"Saved: {path}")
            messagebox.showinfo("Success", f"Saved GeoTIFF to:\n{path}")
    
    # ==================== Dialogs ====================
    
    def show_help(self):
        """Show help dialog."""
        self.notebook.select(self.help_tab)
    
    def show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About Marine Debris Detector",
            "🌊 Marine Debris Detector\n\n"
            "Version: 1.0.0\n\n"
            "AI-powered detection of floating marine debris\n"
            "from Sentinel-2 satellite imagery.\n\n"
            "Models: UNet, UNet++, MLP, CNN, Transformer\n\n"
            f"Device: {self.device.upper()}\n"
            f"PyTorch: {torch.__version__}"
        )


def main():
    """Main entry point."""
    root = tk.Tk()
    
    # Set theme
    style = ttk.Style()
    style.theme_use('clam')
    
    app = MarineDebrisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
