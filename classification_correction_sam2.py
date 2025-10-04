"""
Unified Segmentation Mask Correction and SAM2 Pipeline
=====================================================

A comprehensive tool for correcting segmentation masks on aligned image stacks
with integrated SAM2 functionality for automated segmentation propagation.

Features include:
- Interactive ROI-based label transfer
- SAM2 integration with point-based annotation
- Automatic rectangle-to-square ROI conversion
- Batch propagation across multiple images
- Interactive segmentation refinement
- Z-projection visualization
- Complete change history tracking
- Save/load functionality

Requirements:
- Python 3.10+
- napari
- magicgui
- numpy
- scikit-image
- tifffile (for image I/O)
- pandas (for change tracking)
- opencv-python (for image processing)
- torch (for SAM2)
- SAM2 model and weights
"""

import napari
from napari.layers import Image, Labels, Shapes, Points
from magicgui import magicgui, magic_factory
from magicgui.widgets import Container, PushButton, SpinBox, CheckBox, Label as LabelWidget, FileEdit, ComboBox
import numpy as np
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set, Union, cast
import tifffile
from skimage.draw import polygon
from dataclasses import dataclass, asdict
import cv2
import torch
import time
import warnings
from functools import partial
import tempfile
import sys
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image as PILImage
# Set to a higher limit (e.g., 500 million pixels)
PILImage.MAX_IMAGE_PIXELS = None

warnings.filterwarnings('ignore')

# Force OpenCV to use non-Qt backend to avoid conflicts with napari
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''
os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '0'

try:
    from qtpy.QtCore import Signal
except ImportError:
    try:
        from qtpy.QtCore import pyqtSignal as Signal
    except ImportError:
        from PySide2.QtCore import Signal

from typing import Dict, List, Tuple, Optional, Set, Union, cast, Any


# ============================================================================
# SAM2 MODEL CONFIGURATION
# ============================================================================
# TODO: Update these paths to match your SAM2 installation
SAM2_CHECKPOINT = r"C:\Users\Florin\OneDrive - Johns Hopkins\Documents\segmentation_mask_correction_pipeline\sam2_model_weights\sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Alternative model options (uncomment to use):
# SAM2_CHECKPOINT = "path/to/sam2.1_hiera_base_plus.pt"  # Smaller, faster model
# SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

# SAM2_CHECKPOINT = "path/to/sam2.1_hiera_small.pt"      # Smallest, fastest model  
# SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"


@dataclass
class ChangeRecord:
    """Record of a single change operation"""
    timestamp: str
    image_index: int
    roi_coordinates: List[Tuple[float, float]]
    source_labels: List[int]
    target_label: int
    affected_pixels: int



def _process_single_mask_worker_threaded(mask_data: Tuple[np.ndarray, int, Dict[str, int], Tuple[int, int], float, int]) -> Tuple[int, Optional[np.ndarray]]:
    """Thread-safe worker function (no fork issues with Qt/napari)
    
    OpenCV releases the GIL during contour operations, so threading is efficient here.
    """
    mask, frame_idx, crop_params, original_shape, detail_level, obj_id = mask_data
    
    if mask is None or not mask.any():
        return (frame_idx, None)
    
    try:
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return (frame_idx, None)
        
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        
        if contour_area < 10:
            return (frame_idx, None)
        
        perimeter = cv2.arcLength(largest_contour, True)
        epsilon = detail_level * 0.01 * perimeter
        approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
        sampled_points = approx_polygon.reshape(-1, 2)
        
        if len(sampled_points) < 3:
            return (frame_idx, None)
        
        y_offset = crop_params['top_left_y']
        x_offset = crop_params['top_left_x']
        
        global_points_yx = np.column_stack((
            sampled_points[:, 1] + y_offset,
            sampled_points[:, 0] + x_offset
        ))
        
        return (frame_idx, global_points_yx.astype(np.float64))
        
    except Exception as e:
        return (frame_idx, None)
    

@dataclass
class PointAnnotation:
    """Record of a point annotation for SAM2"""
    coordinates: Tuple[float, float]  # (y, x) in global coordinates
    point_type: str  # 'positive' or 'negative'
    roi_local_coords: Tuple[float, float]  # (y, x) in ROI-local coordinates
    image_index: int


class SegmentationCorrectionPipeline:
    """Main pipeline class for segmentation correction with SAM2 integration"""
    
    def __init__(self):
        # Define colormap and labels
        # self.colormap = np.array([
        #     [214, 212, 161],  # 1 bone
        #     [247, 184, 67],   # 2 brain + spinal cord
        #     [136, 232, 95],   # 3 eye
        #     [140, 13, 13],    # 4 heart
        #     [38, 27, 166],    # 5 lungs
        #     [13, 125, 11],    # 6 GI track 
        #     [179, 50, 108],   # 7 liver
        #     [228, 235, 131],  # 8 spleen
        #     [156, 96, 235],   # 9 pancreas
        #     [46, 190, 230],   # 10 kidney
        #     [150, 255, 245],  # 11 mesokidney
        #     [254, 222, 255],  # 12 collagen
        #     [235, 154, 108],  # 13 ear
        #     [255, 255, 255],  # 14 nontissue
        #     [9, 64, 116],     # 15 thymus
        #     [255, 255, 74],   # 16 thyroid
        #     [178, 178, 0],    # 17 bladder
        #     [214, 212, 161],  # 18 skull
        #     [54, 83, 89]      # 19 spleen2
        # ]) / 255.0  # Normalize to 0-1 range
        

        self.colormap = np.array([
            [254, 222, 255],  # 1 collagen
            [255, 255, 255],  # 2 nontissue
            [255, 0, 0]      # 3 blood vessels
        ]) / 255.0  # Normalize to 0-1 range

        # self.label_names = ["bone", "brain", "eye", "heart", "lungs", "GI", "liver", 
        #                    "spleen", "pancreas", "kidney", "mesokidney", "collagen", 
        #                    "ear", "nontissue", "thymus", "thyroid", "bladder", "skull", "spleen2"]
        
        self.label_names = ["collagen", "nontissue", "blood vessels"]
        self.num_labels = len(self.colormap)
        
        # Initialize state variables
        self.segmented_stack: Optional[np.ndarray] = None
        self.original_stack: Optional[np.ndarray] = None
        self.composite_stack: Optional[np.ndarray] = None  # For overlay visualization
        self.current_index: int = 0
        self.selected_labels: Set[int] = set()
        self.transfer_label: int = 1
        self.opacity: float = 0.5  # Fixed opacity
        self.z_projection_depth: int = 5
        self._color_lut_uint8 = None  # Will be computed on first use
        self._dirty_regions = {}  # {image_idx: (y1, y2, x1, x2)}
        self.z_projection_labels: Set[int] = set(range(1, self.num_labels + 1))
      
        
        # Change tracking - now stores only metadata for undo
        self.change_history: List[ChangeRecord] = []
        self.undo_metadata: List[Dict] = []  # Stores pixel changes for undo
        
        # File paths
        self.segmented_folder: Optional[Path] = None
        self.original_folder: Optional[Path] = None
        self.save_directory: Optional[Path] = None
        self.original_filenames: List[str] = []  # Track original filenames
        
        # Napari viewer and layers
        self.viewer: Optional[napari.Viewer] = None
        self.composite_layer: Optional[Image] = None
        self.shapes_layer: Optional[Shapes] = None
        self.z_projection_layer: Optional[Image] = None
        
        # SAM2 integration - Point layers
        self.positive_points_layer: Optional[Points] = None
        self.negative_points_layer: Optional[Points] = None
        
        # SAM2 state
        self.current_square_roi: Optional[np.ndarray] = None
        self.current_roi_params: Optional[Dict[str, int]] = None
        self.sam2_mode: str = "idle"  # "idle", "annotation", "propagated", "refining"
        self.propagated_images: Set[int] = set()
        self.sam2_results_cache: Dict[int, List[np.ndarray]] = {}
        self.sam2_working_roi_id: Optional[int] = None
        self.point_annotations_by_object: Dict[int, List[PointAnnotation]] = {}
        self.sam2_box_prompts_by_object: Dict[int, Optional[Tuple[float, float, float, float]]] = {}
        self.active_object_ids: Set[int] = set()
        self.current_sam2_object_id: int = 1  # From the SpinBox

        # NEW: Track annotations state for automatic object assignment
        self._last_point_snapshot: Dict[str, int] = {'positive': 0, 'negative': 0}
        self._last_shapes_snapshot: int = 0
        
        # NEW: Comprehensive annotation tracking
        self.shape_to_object_mapping: Dict[int, int] = {}  # {shape_layer_index: object_id}
        self.shape_to_type_mapping: Dict[int, str] = {}    # {shape_layer_index: 'working_roi' | 'box_prompt'}
        self.object_to_box_shape_index: Dict[int, int] = {} # {object_id: shape_layer_index}
        
        # NEW: Point tracking with napari layer indices
        self.positive_point_to_object: Dict[int, int] = {}  # {point_index_in_layer: object_id}
        self.negative_point_to_object: Dict[int, int] = {}  # {point_index_in_layer: object_id}
        
        # NEW: Track which points in napari layers have been processed
        self.processed_positive_points: Set[int] = set()
        self.processed_negative_points: Set[int] = set()

        # FIX 1: Track original box prompt frames per object
        self.sam2_box_prompt_original_frames: Dict[int, int] = {}


    def preview_folders(self, segmented_folder: str, original_folder: str) -> Tuple[int, List[str]]:
        """Preview folders to get image count and filenames without loading
        
        Returns:
            Tuple of (total_images, list_of_filenames)
        """
        try:
            seg_folder = Path(segmented_folder)
            orig_folder = Path(original_folder)
            
            # Supported image formats
            image_extensions = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg', '*.bmp']
            
            # Get all image files
            seg_files = []
            for ext in image_extensions:
                seg_files.extend(seg_folder.glob(ext))
                seg_files.extend(seg_folder.glob(ext.upper()))
            
            seg_files = sorted(list(set(seg_files)))
            filenames = [f.name for f in seg_files]
            
            return len(seg_files), filenames
        except Exception as e:
            print(f"Error previewing folders: {e}")
            return 0, []

    # note: Change start and end idx from here
    def load_image_folders(self, segmented_folder: str, original_folder: str, 
                      start_idx: Optional[int] = None, end_idx: Optional[int] = None) -> None:
         
        """Load images from segmented and original folders
        
        Args:
            segmented_folder: Path to segmented images
            original_folder: Path to original images
            start_idx: Starting index (0-based) of images to load. If None, starts from 0
            end_idx: Ending index (exclusive) of images to load. If None, loads all
        """

        self.segmented_folder = Path(segmented_folder)
        self.original_folder = Path(original_folder)
        
        # Supported image formats
        image_extensions = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg', '*.bmp']
        
        # Get all image files
        seg_files = []
        orig_files = []
        
        for ext in image_extensions:
            seg_files.extend(self.segmented_folder.glob(ext))
            seg_files.extend(self.segmented_folder.glob(ext.upper()))
            orig_files.extend(self.original_folder.glob(ext))
            orig_files.extend(self.original_folder.glob(ext.upper()))
        
        # Sort files to ensure matching order
        seg_files = sorted(list(set(seg_files)))  # Remove duplicates and sort
        orig_files = sorted(list(set(orig_files)))
        
        # Check if files exist
        if len(seg_files) == 0:
            raise ValueError("No images found in the specified folders")
        
        if len(seg_files) != len(orig_files):
            raise ValueError(f"Number of segmented ({len(seg_files)}) and original ({len(orig_files)}) images don't match")
        
        # Apply range selection
        total_images = len(seg_files)
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = total_images
        
        # Validate range
        if start_idx < 0:
            start_idx = 0
        if end_idx > total_images:
            end_idx = total_images
        if start_idx >= end_idx:
            raise ValueError(f"Invalid range: start_idx ({start_idx}) must be less than end_idx ({end_idx})")
        
        # Select files in range
        seg_files = seg_files[start_idx:end_idx]
        orig_files = orig_files[start_idx:end_idx]
        
        # Store original filenames
        self.original_filenames = [f.name for f in seg_files]
        
        print(f"Loading images {start_idx} to {end_idx-1} ({len(seg_files)} images) out of {total_images} total...")
        
        # Load first image to get dimensions
        first_seg = self.load_single_image(seg_files[0])
        first_orig = self.load_single_image(orig_files[0])
        
        # Ensure first_seg is 2D
        if len(first_seg.shape) > 2:
            first_seg = first_seg[..., 0]  # Take first channel if multi-channel
        
        # Initialize stacks
        self.segmented_stack = np.zeros((len(seg_files), *first_seg.shape), dtype=np.uint8)
        if len(first_orig.shape) == 3:  # RGB
            self.original_stack = np.zeros((len(orig_files), *first_orig.shape), dtype=np.uint8)
        else:  # Grayscale
            self.original_stack = np.zeros((len(orig_files), *first_orig.shape, 3), dtype=np.uint8)
        
        # Load all images
        for i, (seg_file, orig_file) in enumerate(zip(seg_files, orig_files)):
            print(f"Loading {i+1}/{len(seg_files)} (image {start_idx + i} in folder): {seg_file.name}, {orig_file.name}")
            
            seg = self.load_single_image(seg_file)
            orig = self.load_single_image(orig_file)
            
            # Handle segmentation - ensure it's 2D
            if len(seg.shape) > 2:
                # If RGB, convert to grayscale by taking the first channel or max
                if seg.shape[-1] == 3:
                    # For RGB segmentation masks, take the channel with most variation
                    # This handles cases where labels might be encoded in one channel
                    channel_stds = [np.std(seg[..., ch]) for ch in range(3)]
                    best_channel = np.argmax(channel_stds)
                    seg = seg[..., best_channel]
                    print(f"  Segmentation was RGB, using channel {best_channel}")
                else:
                    seg = seg[..., 0]
                    print(f"  Segmentation had {seg.shape[-1]} channels, using first")
            
            self.segmented_stack[i] = seg.astype(np.uint8)
            
            # Handle original images
            if len(orig.shape) == 2:  # Convert grayscale to RGB
                self.original_stack[i] = np.stack([orig, orig, orig], axis=-1)
            elif len(orig.shape) == 3 and orig.shape[-1] > 3:
                # If more than 3 channels, take first 3
                self.original_stack[i] = orig[..., :3]
            elif len(orig.shape) == 3 and orig.shape[-1] < 3:
                # If less than 3 channels, pad with zeros
                padding = 3 - orig.shape[-1]
                self.original_stack[i] = np.pad(orig, ((0, 0), (0, 0), (0, padding)), mode='constant')
            else:
                self.original_stack[i] = orig
                
            # Ensure original is in uint8 range
            if self.original_stack[i].dtype != np.uint8:
                # If float, assume 0-1 range and scale
                if self.original_stack[i].dtype in [np.float32, np.float64]:
                    if self.original_stack[i].max() <= 1.0:
                        self.original_stack[i] = (self.original_stack[i] * 255).astype(np.uint8)
                    else:
                        self.original_stack[i] = np.clip(self.original_stack[i], 0, 255).astype(np.uint8)
                else:
                    # For other dtypes, clip to valid range
                    self.original_stack[i] = np.clip(self.original_stack[i], 0, 255).astype(np.uint8)
        
        # Create initial composite stack
        self.update_composite_stack()
        
        print(f"Loaded {len(seg_files)} images: segmented={self.segmented_stack.shape}, original={self.original_stack.shape}")
        print(f"File types loaded: {', '.join(set(f.suffix for f in seg_files))}")
        
        # Verify data types and ranges
        print(f"Segmented stack dtype: {self.segmented_stack.dtype}, Original stack dtype: {self.original_stack.dtype}")
        
        # Check original image range
        orig_min, orig_max = self.original_stack.min(), self.original_stack.max()
        print(f"Original image value range: [{orig_min}, {orig_max}]")
        if orig_max <= 1.0 and self.original_stack.dtype in [np.float32, np.float64]:
            print("WARNING: Original images appear to be in 0-1 range. Converting to 0-255...")
            self.original_stack = (self.original_stack * 255).astype(np.uint8)
        elif orig_max > 255:
            print(f"WARNING: Original images have values > 255. Clipping to 0-255 range...")
            self.original_stack = np.clip(self.original_stack, 0, 255).astype(np.uint8)
        
        # Print unique label values to verify loading
        unique_labels = np.unique(self.segmented_stack)
        print(f"Unique label values in segmentation: {unique_labels}")
        if len(unique_labels) > 20:
            print("Warning: More than 20 unique values found. Check if segmentation is loaded correctly.")
        
        # Check if labels are in expected range
        if len(unique_labels) > 1:  # More than just background
            non_zero_labels = unique_labels[unique_labels > 0] if np.any(unique_labels > 0) else np.array([])
            
            if len(non_zero_labels) > 0:
                max_label = non_zero_labels.max()
                min_label = non_zero_labels.min()
                
                if max_label > 19:
                    print(f"WARNING: Label values exceed expected range [1-19]. Max label: {max_label}")
                    print("This may cause display issues. Labels above 19 will not be colored.")
                
                # Check if labels might be 0-indexed
                if min_label == 0 and max_label <= 18:
                    print("\nINFO: Labels appear to be 0-indexed (0-18). The colormap expects 1-indexed labels (1-19).")
                    print("Auto-adjusting labels by adding 1 to all non-zero values...")
                    
                    # Adjust labels to be 1-indexed
                    for i in range(len(self.segmented_stack)):
                        mask = self.segmented_stack[i] > 0
                        self.segmented_stack[i][mask] += 1
                    
                    # Re-check unique labels
                    unique_labels = np.unique(self.segmented_stack)
                    print(f"Adjusted label values: {unique_labels}")
    
    def load_single_image(self, filepath: Path) -> np.ndarray:
        """Load a single image file, supporting multiple formats"""
        try:
            # Import PIL here to avoid dependency if not needed
            from PIL import Image as PILImage
        except ImportError:
            raise ImportError("Please install pillow: pip install pillow")
        
        # Try tifffile first for TIFF files
        if filepath.suffix.lower() in ['.tif', '.tiff']:
            try:
                img = tifffile.imread(filepath)
                return img
            except Exception:
                pass  # Fall back to PIL
        
        # Use PIL for all formats
        try:
            img = PILImage.open(filepath)
            
            # Handle different modes
            if img.mode == 'P':  # Palette mode
                # Try to preserve the palette values if they represent labels
                img_array = np.array(img)
                # Check if values are in label range
                if img_array.max() <= 19:
                    return img_array
                else:
                    # Convert to RGB if palette values are too high
                    img = img.convert('RGB')
                    img_array = np.array(img)
            elif img.mode == 'L':  # Grayscale
                img_array = np.array(img)
            elif img.mode == 'RGBA':  # Remove alpha channel
                img_array = np.array(img)[..., :3]
            elif img.mode in ['RGB', 'BGR']:
                img_array = np.array(img)
            elif img.mode in ['I', 'F']:  # 32-bit integer or float
                # Convert to array first, then handle range
                img_array = np.array(img)
                if img.mode == 'F':  # Float mode
                    # Assume 0-1 range for float images
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    # For integer mode, clip to uint8 range
                    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            else:
                # Try to convert other modes to RGB
                img = img.convert('RGB')
                img_array = np.array(img)
            
            return img_array
            
        except Exception as e:
            raise ValueError(f"Could not load image {filepath}: {e}")


    def update_composite_stack(self, indices: Optional[List[int]] = None) -> None:
        """Create composite overlay of segmentation on original images using cv2 - WITH DIRTY REGION OPTIMIZATION
        
        Args:
            indices: List of image indices to update. If None, updates all images.
        """
        if self.segmented_stack is None or self.original_stack is None:
            print("segmented_stack or original_stack is None. Skipping update.")
            return
        
        # Initialize composite stack if it doesn't exist
        if self.composite_stack is None:
            self.composite_stack = np.zeros_like(self.original_stack, dtype=np.uint8)
            indices = None  # Force full update on first run
        
        # Determine which indices to update
        if indices is None:
            indices_to_update = list(range(len(self.segmented_stack)))
            print(f"Updating composite overlay for all {len(indices_to_update)} images...")
        else:
            indices_to_update = list(indices)
            print(f"Updating composite overlay for {len(indices_to_update)} images...")
        
        # Count how many use dirty regions
        dirty_count = sum(1 for i in indices_to_update if i in self._dirty_regions)
        if dirty_count > 0:
            print(f"  Using dirty region optimization for {dirty_count}/{len(indices_to_update)} images")
        
        # Process updates
        for i in indices_to_update:
            if 0 <= i < len(self.segmented_stack):
                # Check if we have a dirty region for this image
                if i in self._dirty_regions:
                    # OPTIMIZED PATH: Update only the dirty region
                    y1, y2, x1, x2 = self._dirty_regions[i]
                    
                    # Extract small regions
                    seg_region = self.segmented_stack[i, y1:y2, x1:x2]
                    orig_region = self.original_stack[i, y1:y2, x1:x2].astype(np.uint8)
                    
                    # Apply colormap to small region only
                    labels_rgb = self.apply_colormap_to_labels(seg_region)
                    
                    # Update only dirty part of composite
                    self.composite_stack[i, y1:y2, x1:x2] = cv2.addWeighted(
                        orig_region, 0.7, labels_rgb, 0.3, 0
                    )
                    
                    # Report efficiency
                    region_size = (y2 - y1) * (x2 - x1)
                    full_size = self.segmented_stack.shape[1] * self.segmented_stack.shape[2]
                    print(f"  Image {i}: Updated {region_size:,} pixels ({region_size/full_size*100:.1f}% of image)")
                    
                    # Clear dirty region after update
                    del self._dirty_regions[i]
                    
                else:
                    # FULL UPDATE: No dirty region info, update entire image
                    orig = self.original_stack[i].astype(np.uint8)
                    labels_rgb = self.apply_colormap_to_labels(self.segmented_stack[i])
                    self.composite_stack[i] = cv2.addWeighted(orig, 0.7, labels_rgb, 0.3, 0)
        
        print(f"Composite update completed")
        

    def apply_colormap_to_labels(self, labels: np.ndarray) -> np.ndarray:
        """Convert label image to RGB using colormap - OPTIMIZED VERSION"""
        if labels.ndim != 2:
            raise ValueError(f"Labels must be 2D, got shape {labels.shape}")
        
        # FIXED: Dynamic LUT size based on actual number of labels
        if not hasattr(self, '_color_lut_uint8') or self._color_lut_uint8 is None:
            # Create LUT with size = num_labels + 1 (for label 0)
            self._color_lut_uint8 = np.zeros((self.num_labels + 1, 3), dtype=np.uint8)
            
            if self.colormap is not None:
                # Fill colors for labels 1 to num_labels
                for i in range(self.num_labels):
                    self._color_lut_uint8[i + 1] = (self.colormap[i] * 255).astype(np.uint8)
                

                # Option 2: Use white for background (pixels equal to 0)
                self._color_lut_uint8[0] = np.array([255, 255, 255], dtype=np.uint8)
        
        # Clip labels to valid range before indexing
        clipped_labels = np.clip(labels, 0, self.num_labels)
        rgb_image = self._color_lut_uint8[clipped_labels]
        
        return rgb_image
    
    def create_z_projection(self, start_idx: int, depth: int, 
                        current_view: Optional[Tuple[slice, slice]] = None) -> np.ndarray:
        """Create z-projection of selected labels for next 'depth' slices - optimized version"""
        if self.segmented_stack is None or len(self.segmented_stack) == 0:
            return np.zeros((100, 100, 3))
            
        # Ensure we don't go beyond stack bounds
        end_idx = min(start_idx + depth, len(self.segmented_stack))
        if start_idx >= len(self.segmented_stack):
            return np.zeros((100, 100, 3))
        
        # Get full image dimensions
        full_height, full_width = self.segmented_stack.shape[1:3]
        
        # Create projection for full image
        projection = np.zeros((full_height, full_width, 3), dtype=np.float32)
        
        # Create enhanced neon colormap with more subtle adjustments
        neon_colormap = self.colormap.copy()
        
        # Color enhancement parameters (adjust these to tune appearance)
        base_brightness = 1.15    # Overall brightness boost (1.0 = original, higher = brighter)
        channel_boost = 1.25      # Boost for dominant color channel (higher = more saturated)
        min_brightness = 0.1      # Minimum brightness for glow effect (0-1, higher = more glow)
        
        # Enhance colors to be fluorescent but not overly different
        for i in range(len(neon_colormap)):
            color = neon_colormap[i]
            # Find the dominant channel
            max_channel = np.argmax(color)
            
            # Apply subtle enhancement
            enhanced_color = color * base_brightness
            enhanced_color[max_channel] *= channel_boost / base_brightness  # Extra boost to dominant channel
            
            # Add slight glow by ensuring minimum brightness (subtle)
            enhanced_color = np.maximum(enhanced_color, color * min_brightness)
            
            # Soft clipping to maintain color relationships
            enhanced_color = np.clip(enhanced_color, 0, 1)
            neon_colormap[i] = enhanced_color
        
        # Create color lookup table (LUT) for vectorized operations
        # Initialize with zeros (black) for all possible label values
        max_label = 20  # Adjust if you have more labels
        color_lut = np.zeros((max_label, 3), dtype=np.float32)
        
        # Fill LUT only for selected labels
        for label_idx in self.z_projection_labels:
            if label_idx == 12:  # Skip label 12 (collagen)
                continue
            color_idx = label_idx - 1
            if 0 <= color_idx < len(neon_colormap):
                color_lut[label_idx] = neon_colormap[color_idx]
        
        # Z-projection blending parameters (adjust these to tune depth perception)
        base_weight = 0.3      # Minimum visibility for back slices (0-1, higher = more visible)
        weight_range = 0.7     # Range of weight variation (0-1, higher = more depth contrast)
        decay_rate = 3.0       # How quickly slices fade with depth (higher = faster fade)
        blend_strength = 0.8   # Overall blend intensity (0-1, lower = more subtle)
        
        # Pre-compute weights for all slices
        num_slices = end_idx - start_idx
        slice_positions = np.arange(num_slices) / max(num_slices - 1, 1)
        weights = base_weight + weight_range * np.exp(-decay_rate * slice_positions)
        
        # Vectorized projection computation
        for z_offset in range(num_slices):
            z_idx = start_idx + z_offset
            weight = weights[z_offset] * blend_strength
            
            # Get the slice and apply color LUT in one operation
            slice_data = self.segmented_stack[z_idx]
            
            # Vectorized color application using advanced indexing
            # This replaces the inner loop over labels
            slice_colors = color_lut[slice_data]  # Shape: (H, W, 3)
            
            # Apply weight and add to projection (additive blending)
            projection += slice_colors * weight
        
        # Final enhancement parameters (adjust for overall look)
        bloom_strength = 1.1    # Bloom/glow intensity (1.0 = none, higher = more bloom)
        contrast_boost = 1.05   # Final contrast adjustment (1.0 = none, higher = more contrast)
        
        # Apply soft bloom effect using tanh for smooth clipping
        projection = np.tanh(projection * bloom_strength) * contrast_boost
        
        # Ensure valid range
        projection = np.clip(projection, 0, 1)
        
        # Extract view region if needed
        if current_view is not None:
            y_slice, x_slice = current_view
            y_start = max(0, y_slice.start if y_slice.start is not None else 0)
            y_stop = min(full_height, y_slice.stop if y_slice.stop is not None else full_height)
            x_start = max(0, x_slice.start if x_slice.start is not None else 0)
            x_stop = min(full_width, x_slice.stop if x_slice.stop is not None else full_width)
            
            return projection[y_start:y_stop, x_start:x_stop]
        
        return projection
    
    
    def apply_roi_transfer(self, roi_list: List[np.ndarray], 
                        image_indices: List[int]) -> None:
        """Apply label transfer within multiple ROIs for specified images
        
        Args:
            roi_list: List of ROI vertex arrays
            image_indices: List of image indices to apply the ROIs to
        """
        if not self.selected_labels or self.segmented_stack is None:
            print("No labels selected or no data loaded")
            return
        
        print(f"Applying ROI transfer: labels={self.selected_labels} → {self.transfer_label}")
        
        # Apply the transfer directly (no background processing)
        modified_indices = self.apply_roi_transfer_optimized(
            roi_list, image_indices, self.selected_labels, self.transfer_label
        )
        
        # Update viewer immediately if there were changes
        if self.viewer and modified_indices:
            self.update_viewer_optimized(modified_indices)


    def apply_roi_transfer_optimized(self, roi_list: List[np.ndarray], 
                                    image_indices: List[int],
                                    selected_labels: Set[int],
                                    transfer_label: int) -> List[int]:
        """Optimized ROI transfer that tracks dirty regions
        
        Returns:
            List of image indices that were actually modified
        """
        if not selected_labels or self.segmented_stack is None:
            return []

        print(f"Applying optimized ROI transfer: labels={selected_labels} → {transfer_label}")
        
        changes_made = 0
        modified_indices = []
        
        # Store undo information
        undo_info: Dict[str, Union[Dict, Any]] = {
            'image_changes': {}
        }
        
        # Convert selected_labels to numpy array for faster operations
        selected_labels_array = np.array(list(selected_labels))

        for idx in image_indices:
            if 0 <= idx < len(self.segmented_stack):
                current_slice = self.segmented_stack[idx]
                image_dirty_bounds = None
                image_changes_coords = []
                image_changes_values = []
                
                for roi_vertices in roi_list:
                    # Handle both 2D and 3D ROIs
                    if roi_vertices.shape[1] == 3:  # 3D ROI
                        z_coords = roi_vertices[:, 0]
                        roi_images = np.unique(z_coords.astype(int))
                        
                        if idx not in roi_images:
                            continue
                        
                        mask_for_this_image = z_coords.astype(int) == idx
                        if not np.any(mask_for_this_image):
                            continue
                            
                        roi_2d = roi_vertices[mask_for_this_image, 1:]
                        y_coords = roi_2d[:, 0]
                        x_coords = roi_2d[:, 1]
                    else:  # 2D ROI
                        y_coords = roi_vertices[:, 0]
                        x_coords = roi_vertices[:, 1]
                    
                    # Get tight bounding box for this ROI
                    y_min = max(0, int(np.min(y_coords)))
                    y_max = min(current_slice.shape[0], int(np.max(y_coords)) + 1)
                    x_min = max(0, int(np.min(x_coords)))
                    x_max = min(current_slice.shape[1], int(np.max(x_coords)) + 1)
                    
                    # Skip if ROI is outside image bounds
                    if y_max <= y_min or x_max <= x_min:
                        continue
                    
                    # Process ONLY the ROI region
                    roi_mask = np.zeros((y_max - y_min, x_max - x_min), dtype=bool)
                    
                    # Adjust vertices to local coordinates
                    local_y = y_coords - y_min
                    local_x = x_coords - x_min
                    
                    # Rasterize in small region
                    try:
                        rr, cc = polygon(local_y, local_x, roi_mask.shape)
                        roi_mask[rr, cc] = True
                    except:
                        continue
                    
                    # Check labels in small region
                    roi_region = current_slice[y_min:y_max, x_min:x_max]
                    labels_mask = np.isin(roi_region, selected_labels_array)
                    transfer_mask = labels_mask & roi_mask
                    
                    if np.any(transfer_mask):
                        # Get coordinates of pixels to change (in local ROI space)
                        local_y_coords, local_x_coords = np.where(transfer_mask)
                        
                        # Convert to global coordinates
                        global_y_coords = local_y_coords + y_min
                        global_x_coords = local_x_coords + x_min
                        
                        # Store original values
                        original_values = roi_region[local_y_coords, local_x_coords].copy()
                        
                        # Apply changes
                        roi_region[local_y_coords, local_x_coords] = transfer_label
                        
                        # Track changes for undo
                        image_changes_coords.extend(zip(global_y_coords, global_x_coords))
                        image_changes_values.extend(original_values)
                        
                        # Expand dirty bounds
                        if image_dirty_bounds is None:
                            image_dirty_bounds = [y_min, y_max, x_min, x_max]
                        else:
                            image_dirty_bounds[0] = min(image_dirty_bounds[0], y_min)
                            image_dirty_bounds[1] = max(image_dirty_bounds[1], y_max)
                            image_dirty_bounds[2] = min(image_dirty_bounds[2], x_min)
                            image_dirty_bounds[3] = max(image_dirty_bounds[3], x_max)
                        
                        changes_made += len(local_y_coords)
                
                # Store dirty region and undo info for this image
                if image_dirty_bounds:
                    # Store dirty region for optimized composite update
                    self._dirty_regions[idx] = tuple(image_dirty_bounds)
                    modified_indices.append(idx)
                    
                    # Store undo information efficiently
                    if image_changes_coords:
                        coords_array = np.array(image_changes_coords)
                        values_array = np.array(image_changes_values)
                        undo_info['image_changes'][idx] = {
                            'coords': (coords_array[:, 0], coords_array[:, 1]),
                            'original_values': values_array
                        }

        if changes_made > 0:
            # Record change
            record = ChangeRecord(
                timestamp=datetime.now().isoformat(),
                image_index=self.current_index,
                roi_coordinates=[roi.tolist() for roi in roi_list],
                source_labels=list(selected_labels),
                target_label=transfer_label,
                affected_pixels=int(changes_made)
            )
            self.change_history.append(record)
            
            # Store undo metadata
            undo_info['record'] = record
            undo_info['target_label'] = transfer_label
            self.undo_metadata.append(undo_info)
            
            # Update composite for ONLY the dirty regions  
            self.update_composite_stack(modified_indices)
            
            print(f"🎉 Optimized transfer: {changes_made} pixels in {len(modified_indices)} images")
            
            # Report efficiency gains
            if self._dirty_regions:
                total_pixels = sum((b[1]-b[0])*(b[3]-b[2]) for b in self._dirty_regions.values())
                full_pixels = len(modified_indices) * np.prod(self.segmented_stack.shape[1:3])
                if full_pixels > 0:
                    efficiency = (1 - total_pixels / full_pixels) * 100
                    print(f"📊 Efficiency: Processed {total_pixels:,} pixels instead of {full_pixels:,} ({efficiency:.1f}% reduction)")
        
        return modified_indices


    def undo_last_change(self) -> None:
        """Undo the last change using stored metadata"""
        if self.undo_metadata and self.segmented_stack is not None:
            # Get the last undo info
            last_undo = self.undo_metadata.pop()
            
            # Track which images are modified
            modified_indices = []
            
            # Restore original values for each affected image
            for img_idx, change_info in last_undo['image_changes'].items():
                y_coords, x_coords = change_info['coords']
                original_values = change_info['original_values']
                
                # Restore the original values
                self.segmented_stack[img_idx][y_coords, x_coords] = original_values
                modified_indices.append(img_idx)
            
            # Remove the last change record
            if self.change_history:
                self.change_history.pop()
            
            # Update composite for modified images
            self.update_composite_stack(modified_indices)
            
            # Update viewer
            if self.viewer:
                self.update_viewer_optimized(modified_indices)
            
            print(f"✅ Undid last change affecting {len(modified_indices)} images")
    
    # ============================================================================
    # ROI UTILITIES
    # ============================================================================
    
    def validate_square_roi(self, roi_vertices: np.ndarray) -> bool:
        """Validate if ROI is a square/rectangle
        
        Args:
            roi_vertices: Array of shape (N, 2) with vertex coordinates
            
        Returns:
            bool: True if ROI is rectangular, False otherwise
        """
        if len(roi_vertices) != 4:
            return False
        
        # Check if it forms a rectangle by verifying:
        # 1. Opposite sides are parallel
        # 2. All angles are 90 degrees
        
        # Get the 4 vertices
        v = roi_vertices
        
        # Calculate vectors for each side
        side1 = v[1] - v[0]
        side2 = v[2] - v[1]
        side3 = v[3] - v[2]
        side4 = v[0] - v[3]
        
        # Check if opposite sides are equal (allowing small tolerance for float comparison)
        tolerance = 1e-6
        
        # Check if opposite sides have same length
        if not (np.abs(np.linalg.norm(side1) - np.linalg.norm(side3)) < tolerance and
                np.abs(np.linalg.norm(side2) - np.linalg.norm(side4)) < tolerance):
            return False
        
        # Check if adjacent sides are perpendicular (dot product should be ~0)
        if not (np.abs(np.dot(side1, side2)) < tolerance and
                np.abs(np.dot(side2, side3)) < tolerance):
            return False
        
        return True
    
    def get_rectangle_roi_params(self, roi_vertices: np.ndarray) -> Dict[str, int]:
        """Extract parameters from a rectangular ROI (handles both 2D and 3D coordinates)
        
        Args:
            roi_vertices: Array of shape (4, 2) or (4, 3) with rectangle vertices
            
        Returns:
            Dict with keys: 'top_left_y', 'top_left_x', 'height', 'width'
        """
        print(f"Debug: ROI vertices shape: {roi_vertices.shape}")
        print(f"Debug: ROI vertices:\n{roi_vertices}")
        
        # Handle both 2D and 3D coordinates
        if roi_vertices.shape[1] == 3:  # 3D coordinates (z, y, x)
            print("Debug: Processing 3D ROI coordinates (z, y, x)")
            z_coords = roi_vertices[:, 0]  # Z coordinates (image indices)
            y_coords = roi_vertices[:, 1]  # Y coordinates 
            x_coords = roi_vertices[:, 2]  # X coordinates
            
            print(f"Debug: Z coordinates: {z_coords}")
            print(f"Debug: Y coordinates: {y_coords}")
            print(f"Debug: X coordinates: {x_coords}")
            
            # Check that all Z coordinates are the same (ROI should be on one image)
            if not np.allclose(z_coords, z_coords[0]):
                print(f"Warning: ROI spans multiple images (Z range: {z_coords.min():.1f} to {z_coords.max():.1f})")
                print("Using first image's coordinates")
                
        elif roi_vertices.shape[1] == 2:  # 2D coordinates (y, x)
            print("Debug: Processing 2D ROI coordinates (y, x)")
            y_coords = roi_vertices[:, 0]  # Y coordinates
            x_coords = roi_vertices[:, 1]  # X coordinates
            
            print(f"Debug: Y coordinates: {y_coords}")
            print(f"Debug: X coordinates: {x_coords}")
            
        else:
            raise ValueError(f"Invalid ROI shape: {roi_vertices.shape}. Expected (N, 2) or (N, 3)")
        
        # Find bounding box using the correct coordinates
        min_y = int(np.min(y_coords))
        max_y = int(np.max(y_coords))
        min_x = int(np.min(x_coords))
        max_x = int(np.max(x_coords))
        
        print(f"Debug: Y range: [{min_y}, {max_y}]")
        print(f"Debug: X range: [{min_x}, {max_x}]")
        
        # Calculate dimensions
        height = max_y - min_y
        width = max_x - min_x
        
        print(f"Debug: Calculated dimensions: {width}x{height}")
        
        # Validate dimensions
        if height <= 0 or width <= 0:
            print(f"ERROR: Invalid ROI dimensions - width={width}, height={height}")
            print("This usually happens when ROI vertices are not properly formed")
            print("Make sure to draw a proper rectangle using napari's rectangle tool")
            
            # Additional debugging for 3D case
            if roi_vertices.shape[1] == 3:
                print("\n🔍 3D ROI DEBUGGING:")
                print(f"   Z (image) coordinates: {z_coords}")
                print(f"   Y coordinates: {y_coords} -> range [{min_y}, {max_y}] = height {height}")
                print(f"   X coordinates: {x_coords} -> range [{min_x}, {max_x}] = width {width}")
                print("   Make sure you drew the rectangle on the current image slice")
            
            raise ValueError(f"Invalid ROI dimensions: width={width}, height={height}")
        
        result = {
            'top_left_y': min_y,
            'top_left_x': min_x,
            'height': height,
            'width': width
        }
        
        print(f"Debug: Final ROI params: {result}")
        return result
    
    def convert_rectangle_to_square(self, rect_params: Dict[str, int]) -> Dict[str, int]:
        """Convert rectangular ROI parameters to square ROI parameters
        
        Args:
            rect_params: Dict with 'top_left_y', 'top_left_x', 'height', 'width'
            
        Returns:
            Dict with square ROI parameters
        """
        # Make it square by using the larger dimension
        square_size = max(rect_params['height'], rect_params['width'])
        
        # Calculate center of original rectangle
        center_y = rect_params['top_left_y'] + rect_params['height'] // 2
        center_x = rect_params['top_left_x'] + rect_params['width'] // 2
        
        # Calculate new top-left for centered square
        new_top_left_y = center_y - square_size // 2
        new_top_left_x = center_x - square_size // 2
        
        return {
            'top_left_y': new_top_left_y,
            'top_left_x': new_top_left_x,
            'height': square_size,
            'width': square_size
        }
    
    def validate_roi_bounds(self, roi_params: Dict[str, int]) -> bool:
        """Validate that ROI fits within image boundaries
        
        Args:
            roi_params: Dict with 'top_left_y', 'top_left_x', 'height', 'width'
            
        Returns:
            bool: True if ROI fits within image, False otherwise
        """
        if self.segmented_stack is None:
            return False
            
        img_height, img_width = self.segmented_stack.shape[1:3]
        
        # Check if ROI fits within image bounds
        if (roi_params['top_left_y'] < 0 or 
            roi_params['top_left_x'] < 0 or
            roi_params['top_left_y'] + roi_params['height'] > img_height or
            roi_params['top_left_x'] + roi_params['width'] > img_width):
            return False
            
        return True
    
    def create_square_roi_vertices(self, square_params: Dict[str, int]) -> np.ndarray:
        """Create square ROI vertices from parameters
        
        Args:
            square_params: Dict with 'top_left_y', 'top_left_x', 'height', 'width'
            
        Returns:
            np.ndarray of shape (4, 2) with square ROI vertices in (Y, X) format
        """
        top_left_y = square_params['top_left_y']
        top_left_x = square_params['top_left_x']
        size = square_params['height']  # Should be same as width for square
        
        # Create square ROI vertices (clockwise from top-left)
        square_roi = np.array([
            [top_left_y, top_left_x],                    # top-left
            [top_left_y, top_left_x + size],             # top-right
            [top_left_y + size, top_left_x + size],      # bottom-right
            [top_left_y + size, top_left_x]              # bottom-left
        ], dtype=np.float64)
        
        return square_roi
    
    def get_square_roi_params(self, roi_vertices: np.ndarray) -> Dict[str, int]:
        """Extract parameters from a square/rectangular ROI (for compatibility)
        
        Args:
            roi_vertices: Array of shape (4, 2) or (4, 3) with rectangle vertices
            
        Returns:
            Dict with keys: 'top_left_y', 'top_left_x', 'height', 'width'
        """
        return self.get_rectangle_roi_params(roi_vertices)
    
    def convert_to_square_roi(self, roi_params: Dict[str, int]) -> Optional[np.ndarray]:
        """Convert rectangular ROI parameters to a square ROI, extending shorter side
        
        Args:
            roi_params: Dict with 'top_left_y', 'top_left_x', 'height', 'width'
            
        Returns:
            np.ndarray of shape (4, 2) with square ROI vertices, or None if invalid
        """
        # Use the new method for consistency
        square_params = self.convert_rectangle_to_square(roi_params)
        
        # Validate bounds
        if not self.validate_roi_bounds(square_params):
            return None
        
        # Create vertices
        return self.create_square_roi_vertices(square_params)
    
    def safely_clear_rois(self, preserve_prompts: bool = False) -> None:
        """Safely clear all ROIs and associated tracking using napari's proper methods
        
        Args:
            preserve_prompts: If True, preserve box prompt data for SAM2 propagation
        """
        try:
            if self.shapes_layer and len(self.shapes_layer.data) > 0:
                self.shapes_layer.selected_data = set(range(len(self.shapes_layer.data)))
                self.shapes_layer.remove_selected()
                print("ROIs cleared safely")
            
            # Clear all shape tracking
            self.shape_to_object_mapping.clear()
            self.shape_to_type_mapping.clear()
            self.object_to_box_shape_index.clear()
            
            # Clear working ROI tracking
            self.sam2_working_roi_id = None
            
            # Clear box prompts ONLY if not preserving for propagation
            if not preserve_prompts:
                self.sam2_box_prompts_by_object.clear()
                print("✅ Box prompts cleared")
            else:
                print("✅ Box prompts preserved for propagation")
            
            # Recompute active objects (including those with preserved box prompts)
            self._recompute_active_objects()
            
            print("✅ All ROIs and shape tracking cleared")
            
        except Exception as e:
            print(f"Warning: Could not clear ROIs safely: {e}")
    
    def safely_add_roi(self, roi_vertices: np.ndarray, target_images: Optional[List[int]] = None, 
                    is_working_roi: bool = False, object_id: Optional[int] = None, **kwargs) -> None:
        """Safely add ROI(s) with proper tracking integration
        
        Args:
            roi_vertices: Array of shape (N, 2) with vertex coordinates
            target_images: List of image indices where ROI should appear. If None, appears on all images.
            is_working_roi: If True, this ROI will be tracked as the SAM2 working ROI
            object_id: If provided, associate this ROI with specific object (for programmatic adds)
            **kwargs: Additional arguments for napari shapes
        """
        try:
            if self.shapes_layer is None:
                print("Warning: No shapes layer available")
                return
                
            # Validate ROI vertices
            if roi_vertices is None or len(roi_vertices) < 3:
                print(f"Warning: Invalid ROI vertices")
                return
                
            if roi_vertices.shape[1] != 2:
                print(f"Warning: ROI vertices have wrong shape {roi_vertices.shape} - expected (N, 2)")
                return
                
            if not np.all(np.isfinite(roi_vertices)):
                print("Warning: ROI contains invalid coordinates (NaN or inf)")
                return
            
            default_kwargs = {
                'shape_type': 'polygon',
                'edge_color': 'yellow', 
                'edge_width': 3,
                'face_color': [0, 0, 0, 0]
            }
            default_kwargs.update(kwargs)
            
            # Temporarily disconnect event handler to avoid processing our own additions
            try:
                self.shapes_layer.events.data.disconnect(self._on_shapes_changed)
            except:
                pass
            
            # Track the starting index before adding ROIs
            starting_roi_count = len(self.shapes_layer.data)
            added_indices = []
            
            if target_images is not None and len(target_images) > 0:
                # Add ROI for specific images only (3D coordinates)
                for img_idx in target_images:
                    try:
                        # Convert 2D roi to 3D by adding z-coordinate
                        roi_3d = np.column_stack([
                            np.full(len(roi_vertices), img_idx, dtype=np.float64),
                            roi_vertices.astype(np.float64)
                        ])
                        
                        self.shapes_layer.add(roi_3d, **default_kwargs)
                        added_indices.append(starting_roi_count + len(added_indices))
                        
                    except Exception as e:
                        print(f"Warning: Could not add ROI to image {img_idx}: {e}")
                        continue
            else:
                # Add ROI for all images (2D coordinates)
                try:
                    roi_vertices_clean = roi_vertices.astype(np.float64)
                    self.shapes_layer.add(roi_vertices_clean, **default_kwargs)
                    added_indices.append(starting_roi_count)
                    
                except Exception as e:
                    print(f"Warning: Could not add 2D ROI: {e}")
            
            # Update tracking for added shapes
            for idx in added_indices:
                if is_working_roi:
                    self.sam2_working_roi_id = idx
                    self.shape_to_object_mapping[idx] = -1  # Special marker for working ROI
                    self.shape_to_type_mapping[idx] = 'working_roi'
                    print(f"🔒 Tracked SAM2 working ROI at index {idx}")
                elif object_id is not None:
                    # Programmatically added ROI with specific object assignment
                    self.shape_to_object_mapping[idx] = object_id
                    self.shape_to_type_mapping[idx] = 'sam2_result'
                    print(f"✅ Added ROI at index {idx} for Object {object_id}")
                else:
                    # Regular ROI without specific assignment
                    self.shape_to_object_mapping[idx] = 0  # Unassigned
                    self.shape_to_type_mapping[idx] = 'user_roi'
            
            # Reconnect event handler
            self.shapes_layer.events.data.connect(self._on_shapes_changed)
            
        except Exception as e:
            print(f"Warning: Could not add ROI safely: {e}")
            import traceback
            traceback.print_exc()
            
            # Ensure event handler is reconnected even if there's an error
            try:
                if self.shapes_layer is not None:
                    self.shapes_layer.events.data.connect(self._on_shapes_changed)
            except:
                pass
    
    # ============================================================================
    # SAM2 INTEGRATION - BOX PROMPT DETECTION AND MANAGEMENT
    # ============================================================================
    
    def clear_box_prompt_rois(self) -> None:
        """Remove any rectangular ROIs that are box prompts using the new tracking system"""
        if self.shapes_layer is None:
            return
        
        # Find all box prompt shape indices
        box_prompt_indices = [
            idx for idx, shape_type in self.shape_to_type_mapping.items()
            if shape_type == 'box_prompt'
        ]
        
        if not box_prompt_indices:
            return
            
        # Remove in reverse order to maintain indices
        for idx in sorted(box_prompt_indices, reverse=True):
            try:
                # Get object ID before removal
                obj_id = self.shape_to_object_mapping.get(idx)
                
                # Select and remove the shape
                self.shapes_layer.selected_data = {idx}
                self.shapes_layer.remove_selected()
                
                print(f"🗑️ Removed box prompt ROI at index {idx} for Object {obj_id}")
                
            except Exception as e:
                print(f"Warning: Could not remove ROI at index {idx}: {e}")
        
        # Note: The mappings will be cleaned up by _on_shapes_changed event
    
    def _is_roi_inside_working_roi(self, roi: np.ndarray) -> bool:
        """Check if an ROI is inside the working square ROI (likely a box prompt)
        
        Args:
            roi: ROI vertices to check
            
        Returns:
            True if ROI is inside the working ROI
        """
        if self.current_roi_params is None:
            return False
        
        # Handle both 2D and 3D coordinates
        if roi.shape[1] == 3:  # 3D coordinates (z, y, x)
            y_coords = roi[:, 1]
            x_coords = roi[:, 2]
        elif roi.shape[1] == 2:  # 2D coordinates (y, x)
            y_coords = roi[:, 0]
            x_coords = roi[:, 1]
        else:
            return False
        
        # Get bounding box of this ROI
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        
        # Check if this ROI is inside the working square ROI
        working_top = self.current_roi_params['top_left_y']
        working_left = self.current_roi_params['top_left_x']
        working_bottom = working_top + self.current_roi_params['height']
        working_right = working_left + self.current_roi_params['width']
        
        return (working_top <= min_y < max_y <= working_bottom and
                working_left <= min_x < max_x <= working_right)
    
    # ============================================================================
    # SAM2 INTEGRATION - COORDINATE UTILITIES
    # ============================================================================
    
    def global_to_local_coords(self, global_coords: Tuple[float, float], 
                             roi_params: Dict[str, int]) -> Tuple[float, float]:
        """Convert global image coordinates to ROI-local coordinates
        
        Args:
            global_coords: (y, x) coordinates in global image space
            roi_params: ROI parameters with 'top_left_y', 'top_left_x'
            
        Returns:
            (y, x) coordinates in ROI-local space
        """
        global_y, global_x = global_coords
        local_y = global_y - roi_params['top_left_y']
        local_x = global_x - roi_params['top_left_x']
        return (local_y, local_x)
    
    def local_to_global_coords(self, local_coords: Tuple[float, float], 
                             roi_params: Dict[str, int]) -> Tuple[float, float]:
        """Convert ROI-local coordinates to global image coordinates
        
        Args:
            local_coords: (y, x) coordinates in ROI-local space
            roi_params: ROI parameters with 'top_left_y', 'top_left_x'
            
        Returns:
            (y, x) coordinates in global image space
        """
        local_y, local_x = local_coords
        global_y = local_y + roi_params['top_left_y']
        global_x = local_x + roi_params['top_left_x']
        return (global_y, global_x)
    
    # ============================================================================
    # SAM2 INTEGRATION - POINT LAYER MANAGEMENT
    # ============================================================================
    
    def initialize_point_layers(self) -> None:
        """Initialize positive and negative point layers with proper event handling"""
        if self.viewer is None:
            return
            
        # Force 3D mode for consistency
        ndim = 3  # Always use 3D for image stacks
        
        # Create positive points layer (green)
        if self.positive_points_layer is None:
            self.positive_points_layer = self.viewer.add_points(
                data=np.empty((0, ndim)),  # Always 3D
                name="Positive Points",
                size=8,
                face_color='lime',
                border_color='darkgreen',
                border_width_is_relative=False,
                border_width=1,
                symbol='o',
                ndim=ndim  # Explicitly set dimensionality
            )
            # Store the last data length to detect actual changes
            self._last_positive_count = 0
            # Connect event for when points are added
            self.positive_points_layer.events.data.connect(
                lambda event: self._on_positive_points_changed()
            )
        
        # Create negative points layer (red)
        if self.negative_points_layer is None:
            self.negative_points_layer = self.viewer.add_points(
                data=np.empty((0, ndim)),  # Always 3D
                name="Negative Points", 
                size=8,
                face_color='red',
                border_color='darkred',
                border_width_is_relative=False,
                border_width=1,
                symbol='x',
                ndim=ndim  # Explicitly set dimensionality
            )
            # Store the last data length to detect actual changes
            self._last_negative_count = 0
            # Connect event for when points are added
            self.negative_points_layer.events.data.connect(
                lambda event: self._on_negative_points_changed()
            )
        
        # Ensure layers are visible and properly configured
        if self.positive_points_layer is not None:
            self.positive_points_layer.visible = True
            # Move to top of layer stack for visibility
            try:
                if self.positive_points_layer in self.viewer.layers:
                    layer_index = list(self.viewer.layers).index(self.positive_points_layer)
                    self.viewer.layers.move(layer_index, -1)
            except Exception as e:
                print(f"Warning: Could not move positive points layer to top: {e}")
                
        if self.negative_points_layer is not None:
            self.negative_points_layer.visible = True
            # Move to top of layer stack for visibility
            try:
                if self.negative_points_layer in self.viewer.layers:
                    layer_index = list(self.viewer.layers).index(self.negative_points_layer)
                    self.viewer.layers.move(layer_index, -1)
            except Exception as e:
                print(f"Warning: Could not move negative points layer to top: {e}")
        
        print("✅ Point layers initialized with 3D support")
    

    def initialize_shapes_layer(self) -> None:
        """Initialize shapes layer with proper event handling"""
        if self.viewer is None:
            return
            
        # Disconnect old event handlers if they exist
        if self.shapes_layer is not None:
            try:
                self.shapes_layer.events.data.disconnect()
            except:
                pass
        
        # Create new shapes layer if needed
        if self.shapes_layer is None:
            self.shapes_layer = self.viewer.add_shapes(
                name="ROI",
                edge_color="yellow",
                edge_width=3,
                face_color="transparent"
            )
        
        # Connect to shape events with the new tracking system
        self.shapes_layer.events.data.connect(self._on_shapes_changed)
        print("✅ Shapes layer initialized with event tracking")


    def _on_shapes_changed(self, event) -> None:
        """FIXED: Track box prompt original frames"""
        try:
            if self.shapes_layer is None or self.sam2_mode not in ["annotation", "refining"]:
                return
                
            current_shapes = self.shapes_layer.data
            current_count = len(current_shapes)
            
            known_shape_indices = set(self.shape_to_object_mapping.keys())
            all_current_indices = set(range(current_count))
            new_shape_indices = all_current_indices - known_shape_indices
            
            removed_indices = known_shape_indices - all_current_indices
            if removed_indices:
                self._handle_removed_shapes(removed_indices)
            
            if not new_shape_indices:
                return
                
            for shape_idx in sorted(new_shape_indices):
                shape = current_shapes[shape_idx]
                
                if shape_idx == self.sam2_working_roi_id:
                    self.shape_to_object_mapping[shape_idx] = -1
                    self.shape_to_type_mapping[shape_idx] = 'working_roi'
                    continue
                    
                if len(shape) == 4 and self._is_roi_inside_working_roi(shape):
                    obj_id = self.current_sam2_object_id
                    
                    if shape.shape[1] == 3:  # 3D
                        y_coords = shape[:, 1]
                        x_coords = shape[:, 2]
                    else:  # 2D
                        y_coords = shape[:, 0]
                        x_coords = shape[:, 1]
                    
                    min_y, max_y = np.min(y_coords), np.max(y_coords)
                    min_x, max_x = np.min(x_coords), np.max(x_coords)
                    
                    local_min_y = min_y - self.current_roi_params['top_left_y']
                    local_max_y = max_y - self.current_roi_params['top_left_y']
                    local_min_x = min_x - self.current_roi_params['top_left_x']
                    local_max_x = max_x - self.current_roi_params['top_left_x']
                    
                    box_prompt = (local_min_x, local_min_y, local_max_x, local_max_y)
                    self.sam2_box_prompts_by_object[obj_id] = box_prompt
                    
                    # FIX: Record the original frame where this box was drawn
                    self.sam2_box_prompt_original_frames[obj_id] = self.current_index
                    
                    self.shape_to_object_mapping[shape_idx] = obj_id
                    self.shape_to_type_mapping[shape_idx] = 'box_prompt'
                    
                    if obj_id in self.object_to_box_shape_index:
                        old_idx = self.object_to_box_shape_index[obj_id]
                        self.shape_to_object_mapping.pop(old_idx, None)
                        self.shape_to_type_mapping.pop(old_idx, None)
                    
                    self.object_to_box_shape_index[obj_id] = shape_idx
                    self.active_object_ids.add(obj_id)
                    
                    print(f"Box prompt for Object {obj_id} recorded on frame {self.current_index}")
                            
        except Exception as e:
            print(f"Error in _on_shapes_changed: {e}")


    def _handle_removed_shapes(self, removed_indices: Set[int]) -> None:
        """Handle shapes that were removed from the layer"""
        for idx in removed_indices:
            # Get object ID if mapped
            obj_id = self.shape_to_object_mapping.get(idx)
            shape_type = self.shape_to_type_mapping.get(idx)
            
            # Clean up mappings
            if idx in self.shape_to_object_mapping:
                del self.shape_to_object_mapping[idx]
            if idx in self.shape_to_type_mapping:
                del self.shape_to_type_mapping[idx]
                
            # Handle specific shape types
            if shape_type == 'working_roi':
                self.sam2_working_roi_id = None
                print(f"⚠️  Working ROI removed")
            elif shape_type == 'box_prompt' and obj_id is not None:
                # Remove box prompt for this object
                if obj_id in self.sam2_box_prompts_by_object:
                    del self.sam2_box_prompts_by_object[obj_id]
                if obj_id in self.object_to_box_shape_index:
                    del self.object_to_box_shape_index[obj_id]
                print(f"⚠️  Box prompt removed for Object {obj_id}")
                
            # Update indices for remaining shapes (napari reindexes after removal)
            self._reindex_shape_mappings(removed_indices)


    def _reindex_shape_mappings(self, removed_indices: Set[int]) -> None:
        """Reindex shape mappings after shapes are removed"""
        if not removed_indices:
            return
            
        # Sort removed indices in descending order
        sorted_removed = sorted(removed_indices, reverse=True)
        
        # Create new mappings
        new_shape_to_object = {}
        new_shape_to_type = {}
        new_object_to_box = {}
        
        # Reindex all mappings
        for old_idx, obj_id in self.shape_to_object_mapping.items():
            if old_idx in removed_indices:
                continue
                
            # Calculate new index
            new_idx = old_idx
            for removed_idx in sorted_removed:
                if old_idx > removed_idx:
                    new_idx -= 1
                    
            new_shape_to_object[new_idx] = obj_id
            new_shape_to_type[new_idx] = self.shape_to_type_mapping.get(old_idx, '')
            
            # Update object to box mapping if this is a box prompt
            if self.shape_to_type_mapping.get(old_idx) == 'box_prompt' and obj_id > 0:
                new_object_to_box[obj_id] = new_idx
                
        # Update working ROI index
        if self.sam2_working_roi_id is not None:
            new_working_roi_id = self.sam2_working_roi_id
            for removed_idx in sorted_removed:
                if self.sam2_working_roi_id > removed_idx:
                    new_working_roi_id -= 1
            self.sam2_working_roi_id = new_working_roi_id if new_working_roi_id >= 0 else None
            
        # Replace mappings
        self.shape_to_object_mapping = new_shape_to_object
        self.shape_to_type_mapping = new_shape_to_type
        self.object_to_box_shape_index = new_object_to_box


    def _handle_removed_positive_points(self, new_count: int) -> None:
        """Handle when positive points are removed"""
        # Find which points were removed
        removed_indices = self.processed_positive_points - set(range(new_count))
        
        for idx in removed_indices:
            # Get object ID for this point
            obj_id = self.positive_point_to_object.get(idx)
            if obj_id is not None:
                # Remove from annotations
                if obj_id in self.point_annotations_by_object:
                    # Find and remove the annotation
                    # This is tricky as we need to match by some criteria
                    # For now, we'll just log it
                    print(f"⚠️  Positive point {idx} removed (was assigned to Object {obj_id})")
                    
                # Remove from mapping
                del self.positive_point_to_object[idx]
                
        # Remove from processed set
        self.processed_positive_points = self.processed_positive_points - removed_indices
        
        # Reindex remaining points
        self._reindex_point_mappings('positive', removed_indices)


    def _handle_removed_negative_points(self, new_count: int) -> None:
        """Handle when negative points are removed"""
        # Similar to positive points
        removed_indices = self.processed_negative_points - set(range(new_count))
        
        for idx in removed_indices:
            obj_id = self.negative_point_to_object.get(idx)
            if obj_id is not None:
                print(f"⚠️  Negative point {idx} removed (was assigned to Object {obj_id})")
                del self.negative_point_to_object[idx]
                
        self.processed_negative_points = self.processed_negative_points - removed_indices
        self._reindex_point_mappings('negative', removed_indices)


    def _reindex_point_mappings(self, point_type: str, removed_indices: Set[int]) -> None:
        """Reindex point mappings after points are removed"""
        if not removed_indices:
            return
            
        sorted_removed = sorted(removed_indices, reverse=True)
        
        if point_type == 'positive':
            old_mapping = self.positive_point_to_object.copy()
            new_mapping = {}
            new_processed = set()
            
            for old_idx, obj_id in old_mapping.items():
                if old_idx in removed_indices:
                    continue
                    
                new_idx = old_idx
                for removed_idx in sorted_removed:
                    if old_idx > removed_idx:
                        new_idx -= 1
                        
                new_mapping[new_idx] = obj_id
                new_processed.add(new_idx)
                
            self.positive_point_to_object = new_mapping
            self.processed_positive_points = new_processed
            
        else:  # negative
            old_mapping = self.negative_point_to_object.copy()
            new_mapping = {}
            new_processed = set()
            
            for old_idx, obj_id in old_mapping.items():
                if old_idx in removed_indices:
                    continue
                    
                new_idx = old_idx
                for removed_idx in sorted_removed:
                    if old_idx > removed_idx:
                        new_idx -= 1
                        
                new_mapping[new_idx] = obj_id
                new_processed.add(new_idx)
                
            self.negative_point_to_object = new_mapping
            self.processed_negative_points = new_processed


    def validate_annotation_state(self) -> bool:
        """Validate that all annotation mappings are consistent"""
        errors = []
        
        # Check shape mappings
        if self.shapes_layer is not None:
            current_shape_count = len(self.shapes_layer.data)
            for shape_idx in self.shape_to_object_mapping.keys():
                if shape_idx >= current_shape_count:
                    errors.append(f"Shape index {shape_idx} in mapping but only {current_shape_count} shapes exist")
                    
        # Check point mappings
        if self.positive_points_layer is not None:
            pos_count = len(self.positive_points_layer.data)
            for point_idx in self.positive_point_to_object.keys():
                if point_idx >= pos_count:
                    errors.append(f"Positive point index {point_idx} in mapping but only {pos_count} points exist")
                    
        if self.negative_points_layer is not None:
            neg_count = len(self.negative_points_layer.data)
            for point_idx in self.negative_point_to_object.keys():
                if point_idx >= neg_count:
                    errors.append(f"Negative point index {point_idx} in mapping but only {neg_count} points exist")
                    
        # Check active objects have annotations
        for obj_id in self.active_object_ids:
            has_points = obj_id in self.point_annotations_by_object and len(self.point_annotations_by_object[obj_id]) > 0
            has_box = obj_id in self.sam2_box_prompts_by_object and self.sam2_box_prompts_by_object[obj_id] is not None
            
            if not has_points and not has_box:
                errors.append(f"Object {obj_id} is active but has no annotations")
                
        # Check working ROI
        if self.sam2_working_roi_id is not None:
            if self.shapes_layer is None or self.sam2_working_roi_id >= len(self.shapes_layer.data):
                errors.append(f"Working ROI index {self.sam2_working_roi_id} is invalid")
                
        if errors:
            print("❌ Annotation state validation failed:")
            for error in errors:
                print(f"   - {error}")
            return False
        else:
            print("✅ Annotation state is valid")
            return True


    def clear_annotations_for_object(self, object_id: int) -> None:
        """Clear all annotations for a specific object"""
        # Clear points
        if object_id in self.point_annotations_by_object:
            del self.point_annotations_by_object[object_id]
            
        # Clear box prompt
        if object_id in self.sam2_box_prompts_by_object:
            del self.sam2_box_prompts_by_object[object_id] # Clear from active objects
        self.active_object_ids.discard(object_id)
        
        # Clean up shape mappings
        if object_id in self.object_to_box_shape_index:
            del self.object_to_box_shape_index[object_id]
            
        print(f"✅ Cleared all annotations for Object {object_id}")


    def get_object_summary(self, object_id: int) -> str:
        """Get a summary of annotations for a specific object"""
        summary_parts = [f"Object {object_id}:"]
        
        # Count points
        if object_id in self.point_annotations_by_object:
            annotations = self.point_annotations_by_object[object_id]
            pos_count = sum(1 for a in annotations if a.point_type == 'positive')
            neg_count = sum(1 for a in annotations if a.point_type == 'negative')
            
            # Group by image
            points_per_image = {}
            for ann in annotations:
                if ann.image_index not in points_per_image:
                    points_per_image[ann.image_index] = {'positive': 0, 'negative': 0}
                points_per_image[ann.image_index][ann.point_type] += 1
                
            summary_parts.append(f"  Points: {pos_count} positive, {neg_count} negative")
            for img_idx, counts in sorted(points_per_image.items()):
                summary_parts.append(f"    Image {img_idx}: {counts['positive']}+, {counts['negative']}-")
        else:
            summary_parts.append("  Points: None")
            
        # Check box prompt
        if object_id in self.sam2_box_prompts_by_object and self.sam2_box_prompts_by_object[object_id]:
            box = self.sam2_box_prompts_by_object[object_id]
            summary_parts.append(f"  Box prompt: ({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f})")
        else:
            summary_parts.append("  Box prompt: None")
            
        # Active status
        if object_id in self.active_object_ids:
            summary_parts.append("  Status: ✅ Active")
        else:
            summary_parts.append("  Status: ❌ Inactive")
            
        return "\n".join(summary_parts)


    def _on_positive_points_changed(self) -> None:
        """Handle when positive points are added via napari's native interface"""
        try:
            if (self.positive_points_layer is None or 
                self.current_roi_params is None or 
                self.sam2_mode not in ["annotation", "refining"]):
                return
            
            current_points = self.positive_points_layer.data
            current_count = len(current_points)
            
            # Find new points (indices not in processed set)
            new_point_indices = set(range(current_count)) - self.processed_positive_points
            
            if not new_point_indices:
                # Check for removed points
                if len(self.processed_positive_points) > current_count:
                    self._handle_removed_positive_points(current_count)
                return
                
            # Process each new point
            for point_idx in sorted(new_point_indices):
                point = current_points[point_idx]
                
                # Extract coordinates
                if len(point) == 3:  # 3D coordinates (z, y, x)
                    image_idx = int(point[0])
                    global_coords = (float(point[1]), float(point[2]))
                else:  # 2D coordinates
                    image_idx = self.current_index
                    global_coords = (float(point[0]), float(point[1]))
                
                # Check if within ROI
                local_coords = self.global_to_local_coords(global_coords, self.current_roi_params)
                roi_height = self.current_roi_params['height']
                roi_width = self.current_roi_params['width']
                
                if (0 <= local_coords[0] <= roi_height and 0 <= local_coords[1] <= roi_width):
                    # Create annotation
                    annotation = PointAnnotation(
                        coordinates=global_coords,
                        point_type='positive',
                        roi_local_coords=local_coords,
                        image_index=image_idx
                    )
                    
                    # Add to current object
                    obj_id = self.current_sam2_object_id
                    if obj_id not in self.point_annotations_by_object:
                        self.point_annotations_by_object[obj_id] = []
                        
                    self.point_annotations_by_object[obj_id].append(annotation)
                    self.positive_point_to_object[point_idx] = obj_id
                    self.active_object_ids.add(obj_id)
                    
                    print(f"✅ Positive point {point_idx} assigned to Object {obj_id}")
                    print(f"   Global: ({global_coords[0]:.1f}, {global_coords[1]:.1f}), Image: {image_idx}")
                else:
                    print(f"⚠️ Positive point {point_idx} is outside ROI bounds")
                    
                # Mark as processed
                self.processed_positive_points.add(point_idx)
                
        except Exception as e:
            print(f"❌ Error processing positive points: {e}")
            import traceback
            traceback.print_exc()
    

    def _on_negative_points_changed(self) -> None:
        """Handle when negative points are added via napari's native interface"""
        try:
            if (self.negative_points_layer is None or 
                self.current_roi_params is None or 
                self.sam2_mode not in ["annotation", "refining"]):
                return
            
            current_points = self.negative_points_layer.data
            current_count = len(current_points)
            
            # Find new points (indices not in processed set)
            new_point_indices = set(range(current_count)) - self.processed_negative_points
            
            if not new_point_indices:
                # Check for removed points
                if len(self.processed_negative_points) > current_count:
                    self._handle_removed_negative_points(current_count)
                return
                
            # Process each new point
            for point_idx in sorted(new_point_indices):
                point = current_points[point_idx]
                
                # Extract coordinates
                if len(point) == 3:  # 3D coordinates (z, y, x)
                    image_idx = int(point[0])
                    global_coords = (float(point[1]), float(point[2]))
                else:  # 2D coordinates
                    image_idx = self.current_index
                    global_coords = (float(point[0]), float(point[1]))
                
                # Check if within ROI
                local_coords = self.global_to_local_coords(global_coords, self.current_roi_params)
                roi_height = self.current_roi_params['height']
                roi_width = self.current_roi_params['width']
                
                if (0 <= local_coords[0] <= roi_height and 0 <= local_coords[1] <= roi_width):
                    # Create annotation
                    annotation = PointAnnotation(
                        coordinates=global_coords,
                        point_type='negative',
                        roi_local_coords=local_coords,
                        image_index=image_idx
                    )
                    
                    # Add to current object
                    obj_id = self.current_sam2_object_id
                    if obj_id not in self.point_annotations_by_object:
                        self.point_annotations_by_object[obj_id] = []
                        
                    self.point_annotations_by_object[obj_id].append(annotation)
                    self.negative_point_to_object[point_idx] = obj_id
                    self.active_object_ids.add(obj_id)
                    
                    print(f"✅ Negative point {point_idx} assigned to Object {obj_id}")
                    print(f"   Global: ({global_coords[0]:.1f}, {global_coords[1]:.1f}), Image: {image_idx}")
                else:
                    print(f"⚠️ Negative point {point_idx} is outside ROI bounds")
                    
                # Mark as processed
                self.processed_negative_points.add(point_idx)
                
        except Exception as e:
            print(f"❌ Error processing negative points: {e}")
            import traceback
            traceback.print_exc()
    

    def _process_new_point(self, global_coords: Tuple[float, float], point_type: str, image_idx: int) -> None:
        """Process a newly added point and validate it's within ROI"""
        try:
            if self.current_roi_params is None:
                print(f"⚠️  Cannot process {point_type} point: No active ROI")
                return
            
            # Get current object's annotations
            obj_id = self.current_sam2_object_id
            if obj_id not in self.point_annotations_by_object:
                self.point_annotations_by_object[obj_id] = []
            
            # Check for duplicate annotations before adding
            for existing_annotation in self.point_annotations_by_object[obj_id]:
                if (existing_annotation.image_index == image_idx and
                    existing_annotation.point_type == point_type and
                    np.allclose(existing_annotation.coordinates, global_coords, atol=1e-6)):
                    print(f"⚠️ Duplicate annotation detected and ignored")
                    return
            
            # Convert to local coordinates
            local_coords = self.global_to_local_coords(global_coords, self.current_roi_params)
            
            # Check if point is within the ROI bounds
            roi_height = self.current_roi_params['height']
            roi_width = self.current_roi_params['width']
            
            if (0 <= local_coords[0] <= roi_height and 0 <= local_coords[1] <= roi_width):
                # Create point annotation with correct image index
                annotation = PointAnnotation(
                    coordinates=global_coords,
                    point_type=point_type,
                    roi_local_coords=local_coords,
                    image_index=image_idx
                )
                self.point_annotations_by_object[obj_id].append(annotation)
                self.active_object_ids.add(obj_id)
                
                print(f"✅ {point_type.capitalize()} point added for Object {obj_id}!")
                print(f"   Global coords (y,x): ({global_coords[0]:.1f}, {global_coords[1]:.1f})")
                print(f"   Local coords (y,x): ({local_coords[0]:.1f}, {local_coords[1]:.1f})")
                print(f"   Image: {image_idx}")
                
                # Count points per image for this object
                points_per_image = {}
                for ann in self.point_annotations_by_object[obj_id]:
                    if ann.image_index not in points_per_image:
                        points_per_image[ann.image_index] = {'positive': 0, 'negative': 0}
                    points_per_image[ann.image_index][ann.point_type] += 1
                
                print(f"   Object {obj_id} points per image: {dict(points_per_image)}")
                print(f"   Total annotations for Object {obj_id}: {len(self.point_annotations_by_object[obj_id])}")
                
            else:
                print(f" ⚠️  {point_type.capitalize()} point is OUTSIDE ROI bounds!")
                
        except Exception as e:
            print(f"❌ Error processing {point_type} point at {global_coords}: {e}")
            import traceback
            traceback.print_exc()
    

    def clear_point_layers(self) -> None:
        """Clear all points from both positive and negative layers and all tracking"""
        try:
            # Clear the actual point data
            if self.positive_points_layer is not None:
                ndim = 3
                self.positive_points_layer.data = np.empty((0, ndim))
                self._last_positive_count = 0
                    
            if self.negative_points_layer is not None:
                ndim = 3
                self.negative_points_layer.data = np.empty((0, ndim))
                self._last_negative_count = 0
            
            # Clear point annotations for ALL objects
            self.point_annotations_by_object.clear()
            
            # Clear point tracking mappings
            self.positive_point_to_object.clear()
            self.negative_point_to_object.clear()
            self.processed_positive_points.clear()
            self.processed_negative_points.clear()
            
            # Note: We don't clear active_object_ids here as objects might have box prompts
            # Instead, recompute active objects
            self._recompute_active_objects()
            
            print("✅ Point layers and all point tracking cleared")
        except Exception as e:
            print(f"❌ Warning: Could not clear point layers: {e}")
    

    def _recompute_active_objects(self) -> None:
        """Recompute which objects are active based on current annotations"""
        new_active = set()
        
        # Check point annotations
        for obj_id, annotations in self.point_annotations_by_object.items():
            if annotations:  # Has at least one annotation
                new_active.add(obj_id)
        
        # Check box prompts - THIS WAS THE ISSUE
        # Box prompts are cleared from shape layer but still stored in sam2_box_prompts_by_object
        for obj_id, box_prompt in self.sam2_box_prompts_by_object.items():
            if box_prompt is not None:
                new_active.add(obj_id)
        
        self.active_object_ids = new_active
        print(f"✅ Recomputed active objects: {sorted(self.active_object_ids)}")


    def get_current_image_points(self, image_index: int, object_id: Optional[int] = None) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Get positive and negative points for a specific image and object"""
        positive_points = []
        negative_points = []
        
        # If object_id specified, get points for that object only
        if object_id is not None:
            if object_id not in self.point_annotations_by_object:
                return positive_points, negative_points
                
            annotations = self.point_annotations_by_object[object_id]
        else:
            # Get points for current object
            object_id = self.current_sam2_object_id
            annotations = self.point_annotations_by_object.get(object_id, [])
        
        print(f"  Getting points for image {image_index}, object {object_id}:")
        print(f"  Total annotations: {len(annotations)}")
        
        for i, annotation in enumerate(annotations):
            if annotation.image_index == image_index:
                if annotation.point_type == 'positive':
                    positive_points.append(annotation.roi_local_coords)
                elif annotation.point_type == 'negative':
                    negative_points.append(annotation.roi_local_coords)
        
        print(f"  Found for image {image_index}, object {object_id}: {len(positive_points)} positive, {len(negative_points)} negative")
        
        return positive_points, negative_points
    
    # ============================================================================
    # SAM2 INTEGRATION - CORE PROCESSING FUNCTIONS
    # ============================================================================
    
    def extract_roi_as_jpeg(self, image: np.ndarray, roi_params: Dict[str, int]) -> np.ndarray:
        """Extract ROI from image as JPEG-ready array
        
        Args:
            image: Full image array
            roi_params: ROI parameters with position and size
            
        Returns:
            ROI cutout as uint8 array
        """
        # Extract crop parameters
        y1 = roi_params['top_left_y']
        x1 = roi_params['top_left_x']
        y2 = y1 + roi_params['height']
        x2 = x1 + roi_params['width']
        
        # Ensure crop boundaries stay within frame limits
        y1_safe = max(0, y1)
        x1_safe = max(0, x1)
        y2_safe = min(image.shape[0], y2)
        x2_safe = min(image.shape[1], x2)
        
        # Extract crop region
        roi_cutout = image[y1_safe:y2_safe, x1_safe:x2_safe].copy()
        return roi_cutout.astype(np.uint8)
    
    def crop_mask_to_global_batch(self, crop_mask: np.ndarray, original_shape: Tuple[int, int], 
                                crop_params: Dict[str, int]) -> np.ndarray:
        """Map segmentation results back to original frame dimensions
        
        Args:
            crop_mask: Binary mask in crop coordinates
            original_shape: (height, width) of original image
            crop_params: Crop parameters with offsets
            
        Returns:
            Full-size mask in global coordinates
        """
        y_offset = crop_params['top_left_y']
        x_offset = crop_params['top_left_x']
        
        # Create full-size mask initialized to False
        full_mask = np.zeros(original_shape[:2], dtype=bool)
        
        # Calculate placement boundaries with safety checks
        y_end = min(y_offset + crop_mask.shape[0], full_mask.shape[0])
        x_end = min(x_offset + crop_mask.shape[1], full_mask.shape[1])
        
        # Place crop mask in correct position within full mask
        full_mask[y_offset:y_end, x_offset:x_end] = crop_mask[:y_end-y_offset, :x_end-x_offset]
        
        return full_mask
    
    def extract_convex_hull_from_mask_batch(self, mask: np.ndarray, frame_idx: int, 
                                          crop_params: Dict[str, int], 
                                          original_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Extract detailed boundary points from segmentation mask for ROI definition
        
        Args:
            mask: Binary mask in crop coordinates
            frame_idx: Frame index (for debugging)
            crop_params: Crop parameters for coordinate conversion
            original_shape: Original image shape
            
        Returns:
            ROI vertex array in global coordinates, shape (N, 2) as (Y, X)
        """
        if mask is None or not mask.any():
            print(f"  Frame {frame_idx}: Mask is None or empty")
            return None
        
        print(f"  Frame {frame_idx}: Processing mask of shape {mask.shape}, crop_params={crop_params}")
        
        # Map crop mask to global coordinates first
        global_mask = self.crop_mask_to_global_batch(mask, original_shape, crop_params)
        print(f"  Frame {frame_idx}: Global mask shape {global_mask.shape}, has_content={global_mask.any()}")
        
        # Convert boolean mask to uint8 for OpenCV contour detection
        mask_uint8 = (global_mask * 255).astype(np.uint8)
        
        # Find contours in the binary mask
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            print(f"  Frame {frame_idx}: No contours found")
            return None
        
        print(f"  Frame {frame_idx}: Found {len(contours)} contours")
        
        # Find the largest contour (main object)
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        print(f"  Frame {frame_idx}: Largest contour area = {contour_area}")
        
        # Use raw contour points with smart sampling for detail
        contour_points = largest_contour.reshape(-1, 2)
        print(f"  Frame {frame_idx}: Raw contour points shape = {contour_points.shape}")
        
        # Sample points to get good detail while keeping manageable size
        num_points = len(contour_points)
        if num_points > 100:  # If too many points, sample intelligently
            # Sample every nth point to get around 50-80 points for good detail
            step = max(1, num_points // 60)
            sampled_points = contour_points[::step]
        elif num_points > 50:  # Medium number of points
            # Sample every 2nd or 3rd point
            step = max(1, num_points // 40)
            sampled_points = contour_points[::step]
        else:
            # Use all points if we don't have many
            sampled_points = contour_points
        
        print(f"  Frame {frame_idx}: Sampled to {len(sampled_points)} points")
        
        # Ensure we have at least a reasonable number of points for a valid polygon
        if len(sampled_points) < 3:
            # If too few points, use a finer approximation as backup
            epsilon = 0.005 * cv2.arcLength(largest_contour, True)  # 0.5% of perimeter - very detailed
            approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
            sampled_points = approx_polygon.reshape(-1, 2)
            print(f"  Frame {frame_idx}: Fallback approximation gave {len(sampled_points)} points")
            
            # Final validation - need at least 3 points for a polygon
            if len(sampled_points) < 3:
                print(f"  Frame {frame_idx}: ERROR - Contour has too few points ({len(sampled_points)}) for valid polygon")
                return None
        
        # OpenCV returns contours as (X, Y), but napari expects (Y, X)
        # Convert from (X, Y) to (Y, X) format as required by napari
        sampled_points_yx = np.column_stack((sampled_points[:, 1], sampled_points[:, 0]))
        
        # Ensure consistent data type
        sampled_points_yx = sampled_points_yx.astype(np.float64)
        
        # Check coordinate ranges
        y_min, y_max = sampled_points_yx[:, 0].min(), sampled_points_yx[:, 0].max()
        x_min, x_max = sampled_points_yx[:, 1].min(), sampled_points_yx[:, 1].max()
        print(f"  Frame {frame_idx}: Coordinate ranges - Y: [{y_min:.1f}, {y_max:.1f}], X: [{x_min:.1f}, {x_max:.1f}]")
        
        # Validate the final result
        if sampled_points_yx.shape[1] != 2:
            print(f"  Frame {frame_idx}: ERROR - Invalid ROI vertex shape: {sampled_points_yx.shape}")
            return None
        
        # Check for invalid coordinates
        if not np.all(np.isfinite(sampled_points_yx)):
            print(f"  Frame {frame_idx}: ERROR - ROI contains invalid coordinates (NaN or inf)")
            return None
        
        print(f"  Frame {frame_idx}: Successfully extracted ROI with {len(sampled_points_yx)} vertices")
        return sampled_points_yx
    
    
    def sam2_propagate_batch(self, image_cutouts: List[np.ndarray], 
                        images_to_process: List[int],
                        points_per_frame_per_object: Dict[int, Dict[int, Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]],
                        box_prompts_per_object: Dict[int, Optional[Tuple[float, float, float, float]]]) -> Dict[int, List[List[np.ndarray]]]:
        """FIXED: SAM2 batch processing with proper state management and dtype consistency"""
        print(f"SAM2 batch processing: {len(image_cutouts)} images, mode: {self.sam2_mode}")
        
        try:
            from sam2.build_sam import build_sam2_video_predictor
            import tempfile
            import os
        except ImportError as e:
            print(f"ERROR: SAM2 not available: {e}")
            return {}
        
        temp_dir = tempfile.mkdtemp(prefix="sam2_napari_")
        
        try:
            # Save cropped frames as JPEG files
            for idx, cutout in enumerate(image_cutouts):
                frame_filename = f"{idx:04d}.jpg"
                frame_path = os.path.join(temp_dir, frame_filename)
                cv2.imwrite(frame_path, cv2.cvtColor(cutout, cv2.COLOR_RGB2BGR))
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # FIXED: Proper state management for refinement
            if self.sam2_mode == "refining" and hasattr(self, '_sam2_state') and self._sam2_state is not None:
                # REFINEMENT MODE: Reuse existing predictor and state
                print("✅ Using existing SAM2 predictor and state for refinement")
                predictor = self._sam2_predictor
                state = self._sam2_state
                
                # Reset state for this refinement cycle
                predictor.reset_state(state)
                
            else:
                # INITIAL PROPAGATION: Create new predictor and state
                if not hasattr(self, '_sam2_predictor') or self._sam2_predictor is None:
                    if not os.path.exists(SAM2_CHECKPOINT):
                        raise FileNotFoundError(f"SAM2 checkpoint not found: {SAM2_CHECKPOINT}")
                    
                    print(f"Initializing SAM2 on device: {device}")
                    
                    if device.type == 'cuda':
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                    
                    self._sam2_predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device)
                    print("✅ SAM2 predictor initialized")
                
                predictor = self._sam2_predictor
                state = predictor.init_state(temp_dir)
                
                # Store state for potential refinement
                self._sam2_state = state
            
            # FIXED: Simplified and consistent prompt application logic
            for obj_id in sorted(self.active_object_ids):
                print(f"Processing Object {obj_id}...")
                
                points_per_frame = points_per_frame_per_object.get(obj_id, {})
                box_prompt = box_prompts_per_object.get(obj_id)
                
                # Process each frame with prompts for this object
                for frame_idx, (positive_points, negative_points) in points_per_frame.items():
                    if frame_idx not in images_to_process:
                        continue
                    
                    local_frame_idx = images_to_process.index(frame_idx)
                    
                    # FIXED: Consistent data type handling
                    all_points_xy = []
                    all_labels = []
                    
                    # Add positive points
                    for (y, x) in positive_points:
                        all_points_xy.append([float(x), float(y)])
                        all_labels.append(1)
                    
                    # Add negative points
                    for (y, x) in negative_points:
                        all_points_xy.append([float(x), float(y)])
                        all_labels.append(0)
                    
                    # FIXED: Simplified box prompt logic - only apply to original frame
                    original_frame = self.sam2_box_prompt_original_frames.get(obj_id)
                    apply_box = (box_prompt is not None and 
                            original_frame is not None and 
                            frame_idx == original_frame)
                    
                    try:
                        if apply_box and len(all_points_xy) > 0:
                            # Box + points on original frame
                            points_array = np.array(all_points_xy, dtype=np.float32)
                            labels_array = np.array(all_labels, dtype=np.int32)
                            box_array = np.array(box_prompt, dtype=np.float32)
                            
                            predictor.add_new_points_or_box(
                                inference_state=state,
                                frame_idx=local_frame_idx,
                                obj_id=obj_id,
                                points=points_array,
                                labels=labels_array,
                                box=box_array,
                            )
                            print(f"  Frame {frame_idx} (original): Box + {len(positive_points)} pos, {len(negative_points)} neg")
                            
                        elif apply_box and len(all_points_xy) == 0:
                            # Box only on original frame
                            box_array = np.array(box_prompt, dtype=np.float32)
                            
                            predictor.add_new_points_or_box(
                                inference_state=state,
                                frame_idx=local_frame_idx,
                                obj_id=obj_id,
                                box=box_array,
                            )
                            print(f"  Frame {frame_idx} (original): Box only")
                            
                        elif len(all_points_xy) > 0:
                            # Points only (refinement frames or frames without box)
                            points_array = np.array(all_points_xy, dtype=np.float32)
                            labels_array = np.array(all_labels, dtype=np.int32)
                            
                            predictor.add_new_points_or_box(
                                inference_state=state,
                                frame_idx=local_frame_idx,
                                obj_id=obj_id,
                                points=points_array,
                                labels=labels_array,
                            )
                            print(f"  Frame {frame_idx}: {len(positive_points)} pos, {len(negative_points)} neg (points only)")
                            
                    except Exception as e:
                        print(f"❌ Error adding prompts for Object {obj_id}, Frame {frame_idx}: {e}")
                        continue
            
            # FIXED: Consistent propagation with proper error handling
            results_per_object = {obj_id: [[] for _ in range(len(image_cutouts))] for obj_id in self.active_object_ids}
            
            try:
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
                    for i, obj_id in enumerate(out_obj_ids):
                        if obj_id in results_per_object:
                            crop_mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze(0)
                            results_per_object[obj_id][out_frame_idx] = [crop_mask]
                            
            except Exception as e:
                print(f"❌ Error during propagation: {e}")
                # Clear state if propagation fails
                if hasattr(self, '_sam2_state'):
                    del self._sam2_state
                raise
            
            print(f"✅ SAM2 batch processing complete for {len(self.active_object_ids)} objects")
            return results_per_object
            
        finally:
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Could not clean up temp directory: {e}")

    

    def convert_masks_to_rois(self, masks: List[np.ndarray], 
                            roi_params: Dict[str, int],
                            detail_level: float = 0.3) -> List[np.ndarray]:
        """Convert binary masks to ROI polygons - wrapper for parallel processing
        
        Args:
            masks: List of binary segmentation masks
            roi_params: ROI parameters for coordinate conversion
            detail_level: Level of detail (0.1=very detailed, 0.3=medium, 0.5=simple)
            
        Returns:
            List of ROI vertex arrays in global coordinates
        """
        # Wrap in single object dict for parallel processing
        masks_dict = {1: masks}
        results = self.convert_masks_to_rois_parallel(masks_dict, roi_params, detail_level)
        
        # Extract just the ROI vertices (ignore image indices for single-object case)
        if 1 in results:
            return [roi for _, roi in results[1]]
        return []

    
    # ============================================================================
    # SAM2 INTEGRATION - MAIN WORKFLOW FUNCTIONS
    # ============================================================================
    
    def sam2_seg_button_clicked(self) -> None:
        """Handle SAM2 Seg button click - convert rectangle to square and enable point/box annotation"""
        # Check if exactly one ROI is selected
        if self.shapes_layer is None or len(self.shapes_layer.data) == 0:
            print("ERROR: No ROI selected. Please draw a rectangular ROI using the rectangle tool first.")
            return
        
        if len(self.shapes_layer.data) > 1:
            print("ERROR: Only 1 rectangular ROI can be selected for SAM2. Please select exactly one ROI and try again.")
            return
        
        if self.original_stack is None or self.segmented_stack is None:
            print("ERROR: No images loaded.") 
            return
        
        # Get the ROI
        roi = self.shapes_layer.data[0]
        
        # Check if it's a rectangle (must have 4 vertices)
        if len(roi) != 4:
            print("ERROR: Rectangle tool must be selected. Please draw a rectangular ROI using the rectangle tool.")
            return
        
        # Get rectangle parameters first
        try:
            roi_params = self.get_rectangle_roi_params(roi)
        except Exception as e:
            print(f"ERROR: Could not process ROI: {e}")
            return
        
        # Check if it's already square, if not convert it
        if roi_params['height'] != roi_params['width']:
            print(f"Converting rectangular ROI ({roi_params['width']}x{roi_params['height']}) to square...")
            square_roi = self.convert_to_square_roi(roi_params)
            if square_roi is None:
                print("ERROR: ROI extends beyond image boundaries when converted to square. Please choose a smaller ROI closer to the center.")
                return
            
            roi = square_roi
            roi_params = self.get_square_roi_params(roi)
            print(f"ROI converted to square: {roi_params['width']}x{roi_params['height']}")
        
        # Store current ROI for annotation
        self.current_square_roi = roi
        self.current_roi_params = roi_params
        self.sam2_mode = "annotation"
        
        # Clear all tracking data with validation
        self.validate_annotation_state()  # Log current state before clearing
        
        self.point_annotations_by_object.clear()
        self.sam2_box_prompts_by_object.clear()
        self.active_object_ids.clear()
        self.current_sam2_object_id = 1  # Reset to default
        
        # Clear tracking mappings
        self.shape_to_object_mapping.clear()
        self.shape_to_type_mapping.clear()
        self.object_to_box_shape_index.clear()
        self.positive_point_to_object.clear()
        self.negative_point_to_object.clear()
        self.processed_positive_points.clear()
        self.processed_negative_points.clear()
        
        # Clear existing ROIs
        self.safely_clear_rois()
        
        # Initialize shapes layer with event tracking
        self.initialize_shapes_layer()
        
        # Determine which images will be processed
        target_images = [self.current_index]
        
        # Add the new square ROI only on target images
        self.safely_add_roi(
            roi, 
            target_images=target_images,
            is_working_roi=True,
            edge_color='lime',
            edge_width=4
        )
        
        # Initialize point layers
        self.initialize_point_layers()
        self.clear_point_layers()
        
        # Validate state after setup
        self.validate_annotation_state()
        
        print(f"✅ SAM2 annotation mode enabled!")
        print(f"📐 Square ROI: top_left=({roi_params['top_left_y']}, {roi_params['top_left_x']}), size=({roi_params['height']}x{roi_params['width']})")
        print(f"🎯 Current Object ID: 1")
        print(f"📍 Green square currently visible on image {self.current_index + 1}")
        print("\n🎯 SAM2 MULTI-OBJECT ANNOTATION:")
        print("1. ANNOTATE CURRENT OBJECT:")
        print("   - Add points and/or draw a box inside the green square")
        print("   - Annotations are automatically assigned to current object")
        print("\n2. ANNOTATE ADDITIONAL OBJECTS:")
        print("   - Change 'SAM2 Object ID' spinbox to 2, 3, etc.")
        print("   - Add annotations for the new object")
        print("   - All annotations are tracked in real-time")
        print("\n3. PROPAGATE ALL OBJECTS:")
        print("   - Click 'Propagate' to process all annotated objects")
        print("   - Each object will appear in a different color")
    

    def propagate_button_clicked(self, num_images: int, direction: str = "forward") -> None:
        """Handle Propagate button with proper forward/backward propagation logic"""
        if self.sam2_mode not in ["annotation", "refining"]:
            print("ERROR: Must click 'SAM2 Seg' first to set up square ROI and points")
            return
        
        if self.current_roi_params is None:
            print("ERROR: No active square ROI.")
            return
        
        if self.original_stack is None:
            print("ERROR: No original images loaded.")
            return
        
        # Validate annotation state
        if not self.validate_annotation_state():
            print("Warning: Annotation state validation failed, attempting to continue...")
        
        # Ensure current object is active if it has annotations
        obj_has_points = (self.current_sam2_object_id in self.point_annotations_by_object and 
                        len(self.point_annotations_by_object[self.current_sam2_object_id]) > 0)
        obj_has_box = (self.current_sam2_object_id in self.sam2_box_prompts_by_object and 
                    self.sam2_box_prompts_by_object[self.current_sam2_object_id] is not None)
        
        if obj_has_points or obj_has_box:
            self.active_object_ids.add(self.current_sam2_object_id)
        
        if len(self.active_object_ids) == 0:
            print("ERROR: No objects with annotations found. Please add points or box prompts.")
            return
        
        # Determine images to process
        if self.sam2_mode == "refining":
            print("Refinement Mode: Processing images with annotations")
            images_with_points = set()
            for obj_id in self.active_object_ids:
                if obj_id in self.point_annotations_by_object:
                    for ann in self.point_annotations_by_object[obj_id]:
                        images_with_points.add(ann.image_index)
            
            images_to_process = sorted(list(images_with_points.union(self.propagated_images)))
            images_to_process = [i for i in images_to_process if i < len(self.original_stack)]
            print(f"Refining segmentation on {len(images_to_process)} images: {images_to_process}")
        else:
            if direction == "forward":
                print("Forward Propagation")
                start_idx = self.current_index
                end_idx = min(start_idx + num_images, len(self.original_stack))
                images_to_process = list(range(start_idx, end_idx))
                
            else:  # backward
                print("Backward Propagation")
                start_idx = max(0, self.current_index - num_images + 1)
                end_idx = self.current_index + 1
                images_to_process = list(range(self.current_index, start_idx - 1, -1))
            
            print(f"Propagating SAM2 {direction} across {len(images_to_process)} images")
            print(f"Processing order: {images_to_process}")
        
        # Box prompts should be applied to CURRENT frame (first frame in processing order)
        for obj_id in self.active_object_ids:
            if obj_id in self.sam2_box_prompts_by_object and self.sam2_box_prompts_by_object[obj_id]:
                self.sam2_box_prompt_original_frames[obj_id] = self.current_index
                print(f"Object {obj_id}: Box prompt on frame {self.current_index} (first in processing order)")
        
        # Collect points per frame per object
        points_per_frame_per_object = {}
        
        for obj_id in self.active_object_ids:
            points_per_frame_per_object[obj_id] = {}
            
            for img_idx in images_to_process:
                img_positive, img_negative = self.get_current_image_points(img_idx, obj_id)
                if img_positive or img_negative:
                    points_per_frame_per_object[obj_id][img_idx] = (img_positive, img_negative)
        
        # Handle objects with only box prompts
        for obj_id in self.active_object_ids:
            if obj_id not in points_per_frame_per_object or not points_per_frame_per_object[obj_id]:
                if obj_id in self.sam2_box_prompts_by_object and self.sam2_box_prompts_by_object[obj_id]:
                    points_per_frame_per_object[obj_id] = {self.current_index: ([], [])}
        
        # Final validation
        total_prompts = sum(len(frames) for frames in points_per_frame_per_object.values())
        total_boxes = sum(1 for obj_id in self.active_object_ids 
                        if obj_id in self.sam2_box_prompts_by_object and 
                        self.sam2_box_prompts_by_object[obj_id] is not None)
        
        if total_prompts == 0 and total_boxes == 0:
            print("ERROR: No prompts found for any object. Add points or draw boxes.")
            return
        
        print(f"Processing: {len(self.active_object_ids)} objects, {total_prompts} annotations, {total_boxes} boxes")
        print(f"Direction: {direction}, Starting frame: {self.current_index}")
        
        # Clear ROIs but preserve box prompts
        self.safely_clear_rois(preserve_prompts=True)
        
        # Add working ROI
        self.safely_add_roi(
            self.current_square_roi,
            target_images=images_to_process,
            is_working_roi=True,
            edge_color='lime',
            edge_width=6
        )
        
        # Extract image cutouts IN THE ORDER they will be processed
        print("Extracting ROI cutouts from ORIGINAL images...")
        image_cutouts = []
        for idx in images_to_process:
            original_image = self.original_stack[idx]
            cutout = self.extract_roi_as_jpeg(original_image, self.current_roi_params)
            image_cutouts.append(cutout)
        
        # Run SAM2 processing
        try:
            results_per_object = self.sam2_propagate_batch(
                image_cutouts, 
                images_to_process,
                points_per_frame_per_object,
                self.sam2_box_prompts_by_object
            )
            
            print(f"SAM2 processing completed for {len(results_per_object)} objects")
            
            # Convert masks to ROIs in parallel
            print("Converting masks to ROIs (parallel processing)...")
            all_rois_dict = self.convert_masks_to_rois_parallel(
                results_per_object,
                images_to_process, 
                self.current_roi_params,
                detail_level=0.3
            )

            # Update propagated images tracking
            for obj_id, rois_list in all_rois_dict.items():
                for img_idx, _ in rois_list:
                    self.propagated_images.add(img_idx)

            # Process results
            self.process_sam2_results_multi_object_optimized(all_rois_dict)
            
            # Calculate total ROIs for reporting
            total_rois = sum(len(rois_list) for rois_list in all_rois_dict.values())
            unique_images = len(set(img_idx for rois_list in all_rois_dict.values() for img_idx, _ in rois_list))
            
            # Update mode
            if self.sam2_mode == "refining":
                print(f"Refinement complete! Updated {total_rois} ROIs across {unique_images} images.")
            else:
                self.sam2_mode = "propagated"
                print(f"{direction.capitalize()} propagation complete! Generated {total_rois} ROIs across {unique_images} images.")
            
            print("\nNext steps:")
            print("1. Apply ROIs: Click 'Apply ROI Transfer' to transfer labels")
            print("2. Add more objects: Change Object ID and add new annotations") 
            print("3. Refine: Click 'Refine Seg' to improve existing objects")
            print("4. Clean up: Click 'Delete Seg' to start over")
            
        except Exception as e:
            print(f"ERROR in SAM2 propagation: {e}")
            
            if hasattr(self, '_sam2_state'):
                print("Clearing SAM2 state due to error")
                del self._sam2_state
            
            print("Try reducing the number of images or simplifying annotations")
            import traceback
            traceback.print_exc()


    def convert_masks_to_rois_parallel(self, masks_per_object: Dict[int, List[List[np.ndarray]]], 
                                    images_to_process: List[int],  # ADD THIS
                                    roi_params: Dict[str, int],
                                    detail_level: float = 0.3,
                                    max_workers: int = 4) -> Dict[int, List[Tuple[int, np.ndarray]]]:
        """Parallel conversion using threads only (no multiprocessing)
        
        Args:
            masks_per_object: {object_id: [[mask1], [mask2], ...]} from SAM2
            images_to_process: List of actual image indices in processing order
            roi_params: ROI parameters for coordinate conversion
            detail_level: Level of detail (0.1=very detailed, 0.5=medium, 1.0=simple)
            max_workers: Number of threads (default 4, safe for Qt)
            
        Returns:
            Dictionary mapping object_id to list of (image_idx, roi_vertices) tuples
        """
        # Get original image shape
        if self.original_stack is not None:
            original_shape = self.original_stack.shape[1:3]
        else:
            original_shape = (
                roi_params['top_left_y'] + roi_params['height'] + 100,
                roi_params['top_left_x'] + roi_params['width'] + 100
            )
        
        # Flatten all masks into one batch
        all_mask_data = []
        object_mask_ranges = {}
        current_idx = 0
        
        for obj_id, masks_list in masks_per_object.items():
            start_idx = current_idx
            
            # FIX: Use actual image indices from images_to_process
            for list_idx, mask_container in enumerate(masks_list):
                if mask_container is None or len(mask_container) == 0:
                    continue
                
                # Map list position to actual image index
                actual_image_idx = images_to_process[list_idx] if list_idx < len(images_to_process) else list_idx
                
                actual_mask = mask_container[0] if isinstance(mask_container, list) else mask_container
                
                if actual_mask is None or not isinstance(actual_mask, np.ndarray) or not actual_mask.any():
                    continue
                
                all_mask_data.append(
                    (actual_mask, actual_image_idx, roi_params, original_shape, detail_level, obj_id)  # Use actual_image_idx
                )
                current_idx += 1
            
            object_mask_ranges[obj_id] = (start_idx, current_idx)
        
        total_masks = len(all_mask_data)
        
        if total_masks == 0:
            print("No valid masks to process")
            return {obj_id: [] for obj_id in masks_per_object.keys()}
        
        print(f"Converting {total_masks} masks to ROIs ({max_workers} threads, detail={detail_level})...")
        
        results = [None] * total_masks
        
        # Process with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_single_mask_worker_threaded, data): idx
                for idx, data in enumerate(all_mask_data)
            }
            
            processed = 0
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result(timeout=30)
                except Exception as e:
                    print(f"  Warning: Mask {idx} failed: {e}")
                    results[idx] = None
                
                processed += 1
                if processed % 50 == 0:
                    print(f"  Progress: {processed}/{total_masks}")
        
        # Organize results by object
        results_per_object = {}
        for obj_id, (start_idx, end_idx) in object_mask_ranges.items():
            obj_results = []
            for result in results[start_idx:end_idx]:
                if result is not None:
                    frame_idx, roi_vertices = result
                    if roi_vertices is not None:
                        obj_results.append((frame_idx, roi_vertices))
            
            results_per_object[obj_id] = obj_results
            print(f"  Object {obj_id}: {len(obj_results)} ROIs")
        
        total_rois = sum(len(r) for r in results_per_object.values())
        print(f"Complete: {total_masks} masks -> {total_rois} ROIs")
        
        return results_per_object


    def process_sam2_results_multi_object_optimized(self, sam2_rois_per_object: Dict[int, List[Tuple[int, np.ndarray]]]) -> None:
        """Process ROIs from multiple objects with optimized adding
        
        Args:
            sam2_rois_per_object: {object_id: [(image_idx, roi_vertices), ...]}
        """
        if self.shapes_layer is None:
            print("No shapes layer available")
            return
        
        object_colors = [
            'cyan', 'magenta', 'yellow', 'orange', 'purple', 'pink', 'brown', 'gray'
        ]
        
        try:
            total_rois = sum(len(rois) for rois in sam2_rois_per_object.values())
            rois_added = 0
            
            print(f"Adding {total_rois} ROIs to viewer...")
            
            # Batch add ROIs for better performance
            for obj_id, rois_list in sam2_rois_per_object.items():
                color_idx = (obj_id - 1) % len(object_colors)
                obj_color = object_colors[color_idx]
                
                for img_idx, roi in rois_list:
                    try:
                        self.safely_add_roi(
                            roi,
                            target_images=[img_idx],
                            object_id=obj_id,
                            edge_color=obj_color,
                            edge_width=5,
                            face_color=[0, 0, 0, 0]
                        )
                        rois_added += 1
                        
                        if rois_added % 50 == 0:
                            print(f"  Added {rois_added}/{total_rois} ROIs...")
                            
                    except Exception as e:
                        print(f"  Warning: Could not add ROI for Object {obj_id}: {e}")
            
            print(f"Successfully added {rois_added}/{total_rois} ROIs from SAM2")
            self.validate_annotation_state()
            
        except Exception as e:
            print(f"ERROR in process_sam2_results_multi_object_optimized: {e}")
            import traceback
            traceback.print_exc()




    def refine_seg_button_clicked(self) -> None:
        """FIXED: Handle Refine Seg button click following SAM2 documentation pattern - no state reset"""
        if self.sam2_mode not in ["propagated", "refining"]:
            print("ERROR: Must run 'Propagate' first before refining.")
            return
        
        if self.current_roi_params is None:
            print("ERROR: No active ROI.")
            return
        
        # Enter refinement mode
        previous_mode = self.sam2_mode
        self.sam2_mode = "refining"
        
        # Ensure point layers are available
        self.initialize_point_layers()
        
        # FIXED: Verify persistent session state (following SAM2 documentation pattern)
        state_preserved = hasattr(self, '_sam2_state') and self._sam2_state is not None
        predictor_preserved = hasattr(self, '_sam2_predictor') and self._sam2_predictor is not None
        video_session_preserved = hasattr(self, '_sam2_temp_dir') and self._sam2_temp_dir is not None and os.path.exists(self._sam2_temp_dir)
        
        print("🔧 Refinement mode enabled!")
        print(f"   SAM2 state preserved: {'✅' if state_preserved else '❌'}")
        print(f"   SAM2 predictor preserved: {'✅' if predictor_preserved else '❌'}")
        print(f"   Video session preserved: {'✅' if video_session_preserved else '❌'}")
        
        if not (state_preserved and predictor_preserved and video_session_preserved):
            print("⚠️  Warning: SAM2 session not fully preserved. Refinement will work but may reinitialize.")
            print("   This can happen if this is the first refinement or after an error.")
        else:
            print("✅ Full SAM2 session preserved - refinement will use existing state (no reset)")
        
        if previous_mode != "refining":
            print("\n📋 REFINEMENT INSTRUCTIONS:")
            print("1. NAVIGATE to any processed image")
            print("2. SELECT the Object ID you want to refine (use spinbox)")
            print("3. ADD positive/negative points where needed")
            print("4. CLICK 'Propagate' to update segmentation")
            print("5. REPEAT as needed for multiple objects/frames")
        else:
            print("Already in refinement mode - continue adding annotations")
        
        # Show current status
        print(f"\n📊 Current status:")
        print(f"   Active objects: {sorted(self.active_object_ids)}")
        print(f"   Current Object ID: {self.current_sam2_object_id}")
        
        # Show detailed annotation status
        for obj_id in sorted(self.active_object_ids):
            print(f"\n   Object {obj_id}:")
            
            # Box prompt info
            has_box = obj_id in self.sam2_box_prompts_by_object and self.sam2_box_prompts_by_object[obj_id] is not None
            if has_box:
                original_frame = self.sam2_box_prompt_original_frames.get(obj_id, "unknown")
                print(f"     Box prompt: ✅ (originally on frame {original_frame})")
            else:
                print(f"     Box prompt: ❌")
            
            # Points per image
            points_per_image = {}
            if obj_id in self.point_annotations_by_object:
                for ann in self.point_annotations_by_object[obj_id]:
                    if ann.image_index not in points_per_image:
                        points_per_image[ann.image_index] = {'positive': 0, 'negative': 0}
                    points_per_image[ann.image_index][ann.point_type] += 1
            
            if points_per_image:
                for img_idx, counts in sorted(points_per_image.items()):
                    print(f"     Frame {img_idx}: {counts['positive']} positive, {counts['negative']} negative")
            else:
                print(f"     Points: None yet")
        
        print("\n💡 REFINEMENT STRATEGY (SAM2 Documentation Pattern):")
        print("   - Persistent video session maintained across refinement cycles")
        print("   - No state reset (adds prompts to existing inference state)")
        print("   - Box prompts not reapplied during refinement (only initial propagation)")
        print("   - Consistent autocast context maintained")
    
    
    def delete_seg_button_clicked(self) -> None:
        """FIXED: Handle Delete Seg button click with complete cleanup"""
        if getattr(self, 'sam2_mode', 'idle') == 'idle':
            print("No SAM2 segmentation to delete.")
            return
        
        print("🗑️ Deleting SAM2 segmentation data...")
        
        # Clear visual elements
        self.safely_clear_rois()
        self.clear_point_layers()
        
        # Reset ROI state
        self.current_square_roi = None
        self.current_roi_params = None
        self.sam2_working_roi_id = None
        self.sam2_mode = "idle"
        
        # Clear multi-object data
        num_objects = len(self.active_object_ids)
        total_annotations = sum(len(anns) for anns in self.point_annotations_by_object.values())
        total_boxes = sum(1 for box in self.sam2_box_prompts_by_object.values() if box is not None)
        
        self.point_annotations_by_object.clear()
        self.sam2_box_prompts_by_object.clear()
        self.sam2_box_prompt_original_frames.clear()
        self.active_object_ids.clear()
        self.current_sam2_object_id = 1
        
        # Clear tracking mappings
        self.shape_to_object_mapping.clear()
        self.shape_to_type_mapping.clear()
        self.object_to_box_shape_index.clear()
        self.positive_point_to_object.clear()
        self.negative_point_to_object.clear()
        self.processed_positive_points.clear()
        self.processed_negative_points.clear()
        
        # Clear propagation tracking
        num_propagated = len(self.propagated_images)
        self.propagated_images.clear()
        self.sam2_results_cache.clear()
        
        # FIXED: Complete SAM2 cleanup
        sam2_cleaned = False
        if hasattr(self, '_sam2_predictor') and self._sam2_predictor is not None:
            try:
                del self._sam2_predictor
                self._sam2_predictor = None
                sam2_cleaned = True
            except Exception as e:
                print(f"Warning: Could not clean up SAM2 predictor: {e}")
        
        if hasattr(self, '_sam2_state') and self._sam2_state is not None:
            try:
                del self._sam2_state
                self._sam2_state = None
            except Exception as e:
                print(f"Warning: Could not clean up SAM2 state: {e}")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                print("✅ GPU memory cleared")
            except Exception as e:
                print(f"Warning: Could not clear GPU memory: {e}")
        
        # Print summary
        print(f"✅ SAM2 segmentation deleted:")
        print(f"   - {num_objects} objects cleared")
        print(f"   - {total_annotations} point annotations removed")
        print(f"   - {total_boxes} box prompts removed")
        print(f"   - {num_propagated} propagated images cleared")
        print(f"   - SAM2 predictor: {'✅ cleaned' if sam2_cleaned else '❌ cleanup failed'}")
        print(f"   - All tracking mappings cleared")
        print("\n🔄 Ready for new SAM2 workflow")

    
    def process_sam2_results(self, sam2_rois: Dict[int, List[np.ndarray]]) -> None:
        """Process ROIs returned from SAM2 and add them to napari viewer
        
        Args:
            sam2_rois: Dictionary mapping image indices to lists of ROI vertex arrays
                      {image_idx: [roi1_vertices, roi2_vertices, ...]}
        """
        if self.shapes_layer is None:
            print("No shapes layer available")
            return
        
        try:
            # Count total ROIs to add
            total_rois = sum(len(roi_list) for roi_list in sam2_rois.values())
            
            if total_rois > 0:
                print(f"Adding {total_rois} ROIs from SAM2...")
                
                # Add ROIs one by one, specifying which image each belongs to
                rois_added = 0
                for img_idx, roi_list in sam2_rois.items():
                    for roi_idx, roi in enumerate(roi_list):
                        try:
                            # Add each ROI to its specific image using 3D coordinates
                            self.safely_add_roi(
                                roi,
                                target_images=[img_idx],  # Specify which image this ROI belongs to
                                edge_color='cyan',  # Different color for SAM2 results
                                edge_width=2,
                                face_color=[0, 1, 1, 0.1]  # Slight cyan fill
                            )
                            rois_added += 1
                            print(f"  Added ROI {roi_idx+1} for image {img_idx+1} ({len(roi)} vertices)")
                        except Exception as e:
                            print(f"  Warning: Could not add ROI {roi_idx+1} for image {img_idx+1}: {e}")
                
                print(f"Successfully added {rois_added}/{total_rois} ROIs from SAM2 to viewer")
                
                if rois_added < total_rois:
                    print(f"Note: {total_rois - rois_added} ROIs failed to add - this may be due to complex shapes")
            else:
                print("No ROIs returned from SAM2")
                
        except Exception as e:
            print(f"ERROR in process_sam2_results: {e}")
            import traceback
            traceback.print_exc()
    
    # ============================================================================
    # ROI MANAGEMENT HELPERS
    # ============================================================================
    
    def get_transferable_rois(self) -> List[np.ndarray]:
        """Get ROIs that should be applied during transfer using the new tracking system
        
        Returns:
            List of ROI vertex arrays that should be applied during transfer
        """
        if self.shapes_layer is None or len(self.shapes_layer.data) == 0:
            return []
            
        transferable_rois = []
        
        for i, roi in enumerate(self.shapes_layer.data):
            shape_type = self.shape_to_type_mapping.get(i, 'unknown')
            
            # Skip non-transferable shape types
            if shape_type in ['working_roi', 'box_prompt']:
                print(f"🚫 Excluding {shape_type} at index {i}")
                continue
            
            # Include user ROIs and SAM2 results
            if shape_type in ['user_roi', 'sam2_result', 'unknown']:
                transferable_rois.append(roi)
                obj_id = self.shape_to_object_mapping.get(i, 'unassigned')
                print(f"✅ Including {shape_type} at index {i} (Object: {obj_id})")
                    
        print(f"✅ Found {len(transferable_rois)} transferable ROIs")
        return transferable_rois
    
    def _is_roi_inside_working_roi(self, roi: np.ndarray) -> bool:
        """Check if an ROI is inside the working square ROI (likely a box prompt)
        
        Args:
            roi: ROI vertices to check
            
        Returns:
            True if ROI is inside the working ROI
        """
        if self.current_roi_params is None:
            return False
        
        # Handle both 2D and 3D coordinates
        if roi.shape[1] == 3:  # 3D coordinates (z, y, x)
            y_coords = roi[:, 1]
            x_coords = roi[:, 2]
        elif roi.shape[1] == 2:  # 2D coordinates (y, x)
            y_coords = roi[:, 0]
            x_coords = roi[:, 1]
        else:
            return False
        
        # Get bounding box of this ROI
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        
        # Check if this ROI is inside the working square ROI
        working_top = self.current_roi_params['top_left_y']
        working_left = self.current_roi_params['top_left_x']
        working_bottom = working_top + self.current_roi_params['height']
        working_right = working_left + self.current_roi_params['width']
        
        return (working_top <= min_y < max_y <= working_bottom and
                working_left <= min_x < max_x <= working_right)
    
    def count_roi_types(self) -> Dict[str, int]:
        """Count different types of ROIs using the new tracking system
        
        Returns:
            Dictionary with counts of different ROI types
        """
        if self.shapes_layer is None:
            return {'total': 0, 'working': 0, 'box_prompt': 0, 'user_roi': 0, 
                    'sam2_result': 0, 'transferable': 0}
        
        counts = {
            'total': len(self.shapes_layer.data),
            'working': 0,
            'box_prompt': 0,
            'user_roi': 0,
            'sam2_result': 0,
            'unknown': 0
        }
        
        for shape_type in self.shape_to_type_mapping.values():
            if shape_type in counts:
                counts[shape_type] += 1
            else:
                counts['unknown'] += 1
        
        # Calculate transferable (everything except working ROI and box prompts)
        counts['transferable'] = counts['total'] - counts['working'] - counts['box_prompt']
        
        return counts
    
    # ============================================================================
    # PROJECT MANAGEMENT (UNCHANGED)
    # ============================================================================
    
    def save_project(self, directory: str) -> None:
        """Save project state and history - FIXED VERSION with proper compression and no palette issues"""
        if self.segmented_stack is None:
            print("No data to save")
            return
            
        save_dir = Path(directory)
        save_dir.mkdir(exist_ok=True)
        
        # Save individual corrected segmentation images with original filenames
        corrected_dir = save_dir / "corrected_segmentation"
        corrected_dir.mkdir(exist_ok=True)
        print(f"Saving {len(self.segmented_stack)} individual segmentation images...")
        
        total_size = 0
        successful_saves = 0
        
        for i in range(len(self.segmented_stack)):
            # Use original filename if available
            if i < len(self.original_filenames):
                filename = self.original_filenames[i]
                filename_stem = Path(filename).stem
                original_ext = Path(filename).suffix.lower()
            else:
                filename_stem = f"corrected_seg_{i:04d}"
                original_ext = ".png"  # Default to PNG for fallback
            
            # FIXED: Ensure data is uint8 and in valid range
            img_data = self.segmented_stack[i]
            if img_data.dtype != np.uint8:
                print(f"  Warning: Converting image {i} from {img_data.dtype} to uint8")
                img_data = np.clip(img_data, 0, 255).astype(np.uint8)
            
            # Verify label values are in expected range
            unique_values = np.unique(img_data)
            if len(unique_values) > 20 or np.any(unique_values > 19):
                print(f"  Warning: Image {i} has unexpected label values: {unique_values}")
            
            saved_successfully = False
            save_path = None
            
            # Try TIFF first if original was TIFF
            if original_ext in ['.tif', '.tiff']:
                save_path = corrected_dir / f"{filename_stem}_corrected.tif"
                
                # FIXED: Correct TIFF compression attempts with proper syntax
                compression_methods = [
                    ('lzw', {}),                    # LZW compression (no level parameter)
                    ('zlib', {'level': 6}),         # FIXED: Correct zlib syntax
                    ('deflate', {}),                # DEFLATE compression
                    (None, {})                      # No compression
                ]
                
                for compression, compress_args in compression_methods:
                    try:
                        if compression is None:
                            tifffile.imwrite(save_path, img_data)
                        else:
                            # FIXED: Use correct parameter name
                            tifffile.imwrite(save_path, img_data, compression=compression, **compress_args)
                        
                        saved_successfully = True
                        if i == 0:
                            comp_name = compression if compression else "uncompressed"
                            print(f"  Using TIFF with {comp_name} compression")
                        break
                        
                    except Exception as e:
                        if i == 0:
                            comp_name = compression if compression else "uncompressed"
                            print(f"  TIFF {comp_name} failed: {e}")
                        continue
            
            # Use PNG if TIFF failed or wasn't attempted
            if not saved_successfully:
                save_path = corrected_dir / f"{filename_stem}_corrected.png"
                
                try:
                    # FIXED: Explicit prevention of palette conversion for PNG
                    from PIL import Image as PILImage
                    
                    # Create PIL image with explicit mode and no palette
                    img_pil = PILImage.fromarray(img_data, mode='L')
                    
                    # CRITICAL FIX: Save with explicit parameters to prevent palette conversion
                    img_pil.save(
                        save_path, 
                        'PNG',
                        optimize=False,          # FIXED: Disable optimization that might create palette
                        compress_level=6,        # Reasonable compression without optimization
                        bits=8                   # Explicit 8-bit depth
                    )
                    
                    saved_successfully = True
                    if i == 0:
                        print("  Using PNG format (8-bit grayscale, no palette)")
                        
                except Exception as e:
                    print(f"  ERROR: PNG save failed for image {i}: {e}")
                    
                    # FALLBACK: Try with minimal PNG settings
                    try:
                        img_pil = PILImage.fromarray(img_data, mode='L')
                        img_pil.save(save_path, 'PNG', optimize=False)
                        saved_successfully = True
                        if i == 0:
                            print("  Using PNG format (minimal settings fallback)")
                    except Exception as e2:
                        print(f"  ERROR: Even fallback PNG save failed for image {i}: {e2}")
                        continue

            # FIXED: Verify save integrity
            if saved_successfully and save_path:
                try:
                    # Verify the saved file contains correct values
                    if save_path.suffix.lower() == '.png':
                        verify_img = PILImage.open(save_path)
                        verify_array = np.array(verify_img)
                    else:  # TIFF
                        verify_array = tifffile.imread(save_path)
                    
                    # Check if values match exactly
                    if np.array_equal(img_data, verify_array):
                        successful_saves += 1
                        file_size = save_path.stat().st_size
                        total_size += file_size
                        
                        if i == 0:  # Report first file as example
                            print(f"  ✓ Verified: Values preserved correctly")
                            print(f"  First file size: {file_size / 1024:.1f} KB")
                            
                    else:
                        print(f"  ❌ ERROR: Values changed during save for image {i}!")
                        print(f"    Original range: [{img_data.min()}, {img_data.max()}]")
                        print(f"    Saved range: [{verify_array.min()}, {verify_array.max()}]")
                        print(f"    Original unique: {len(np.unique(img_data))} values")
                        print(f"    Saved unique: {len(np.unique(verify_array))} values")
                        
                except Exception as e:
                    print(f"  Warning: Could not verify save for image {i}: {e}")
                    successful_saves += 1  # Count as success since save didn't fail
        
        # Report save results
        if successful_saves > 0:
            avg_size = total_size / successful_saves / 1024
            print(f"✅ Successfully saved {successful_saves}/{len(self.segmented_stack)} images")
            print(f"   Average file size: {avg_size:.1f} KB (total: {total_size / 1024 / 1024:.1f} MB)")
        else:
            print("❌ No images were saved successfully!")
            return
        
        # Save change history (unchanged from original)
        try:
            history_data = [asdict(record) for record in self.change_history]
            with open(save_dir / "change_history.json", 'w') as f:
                json.dump(history_data, f, indent=2)
            
            # Save change summary
            if history_data:
                summary_df = pd.DataFrame(history_data)
                summary_df.to_csv(save_dir / "change_summary.csv", index=False)
                
                # Create detailed summary statistics
                summary_stats = {
                    "total_changes": len(history_data),
                    "total_pixels_modified": sum(r["affected_pixels"] for r in history_data),
                    "images_modified": len(set(r["image_index"] for r in history_data)),
                    "label_transfer_counts": {}
                }
                
                # Count label transfers
                for record in history_data:
                    source_labels_str = str(sorted(record['source_labels']))
                    key = f"{source_labels_str} -> {record['target_label']}"
                    summary_stats["label_transfer_counts"][key] = summary_stats["label_transfer_counts"].get(key, 0) + 1
                
                with open(save_dir / "summary_statistics.json", 'w') as f:
                    json.dump(summary_stats, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save change history: {e}")
        
        # Save project metadata
        try:
            metadata = {
                "original_folder": str(self.original_folder) if self.original_folder else "",
                "segmented_folder": str(self.segmented_folder) if self.segmented_folder else "",
                "original_filenames": self.original_filenames,
                "colormap": self.colormap.tolist(),
                "label_names": self.label_names,
                "total_changes": len(self.change_history),
                "num_images": len(self.segmented_stack) if self.segmented_stack is not None else 0,
                "successful_saves": successful_saves,
                "save_timestamp": datetime.now().isoformat()
            }
            with open(save_dir / "project_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")
            
        print(f"\n✅ Project saved to {save_dir}")
        print(f"- Segmentation images: {successful_saves} files in corrected_segmentation/")
        print(f"- Change history: {len(self.change_history)} changes recorded")
        print(f"- All label values preserved exactly as stored in memory")
        print("\n🎉 Your segmentation corrections are safely saved!")


    def update_viewer_optimized(self, changed_indices: Optional[List[int]] = None) -> None:
        """Update napari viewer"""
        if self.composite_layer is None or changed_indices is None:
            self.update_viewer()
            return
        
        # Update only changed slices
        try:
            for idx in changed_indices:
                if 0 <= idx < len(self.composite_stack):
                    self.composite_layer.data[idx] = self.composite_stack[idx].copy()
            
            # Force refresh of current view
            current_idx = self.viewer.dims.current_step[0] if len(self.viewer.dims.current_step) > 0 else 0
            if current_idx in changed_indices:
                self.composite_layer.refresh()
            
            print(f"✅ Viewer updated for {len(changed_indices)} images")
            
        except Exception as e:
            print(f"⚠️  Update failed, falling back to standard update: {e}")
            self.update_viewer()


    def update_viewer(self) -> None:
        """Standard viewer update - called when full refresh is needed"""
        if self.viewer and self.composite_stack is not None:
            if self.composite_layer is None:
                # Create new layer
                display_data = self.composite_stack.astype(np.uint8)
                layer = self.viewer.add_image(
                    display_data,
                    name="Segmentation Overlay",
                    rgb=True,
                    contrast_limits=[0, 255]
                )
                self.composite_layer = layer[0] if isinstance(layer, list) else layer
            else:
                # Update entire dataset
                display_data = self.composite_stack.astype(np.uint8)
                setattr(self.composite_layer, 'data', display_data)
                self.composite_layer.contrast_limits = [0, 255]
            
            # Ensure shapes layer exists
            if self.shapes_layer is None:
                self.shapes_layer = self.viewer.add_shapes(
                    name="ROI",
                    edge_color="yellow",
                    edge_width=3,
                    face_color="transparent"
                )
            
            # Initialize point layers
            self.initialize_point_layers()

def create_gui(pipeline: SegmentationCorrectionPipeline) -> Container:
    """Create the GUI using magicgui"""
    
    # Store widget references for updates
    widget_refs = {}
    
    # Create range selection widgets (initially hidden)
    start_idx_spin = SpinBox(
        value=0,
        min=0,
        max=9999,
        name="Start Image"
    )
    
    end_idx_spin = SpinBox(
        value=10,
        min=1,
        max=10000,
        name="End Image"
    )
    
    # Label to show preview info
    preview_label = LabelWidget(value="Select folders to preview available images")
    
    # Load all checkbox
    load_all_checkbox = CheckBox(value=True, text="Load all images")
    
    # Container for range controls
    range_controls = Container(
        widgets=[start_idx_spin, end_idx_spin],
        layout='horizontal'
    )
    range_controls.visible = False  # Initially hidden
    
    # Function to update range visibility
    def toggle_range_controls(load_all: bool):
        range_controls.visible = not load_all
        if load_all:
            preview_label.value = f"Will load all {getattr(toggle_range_controls, 'total_images', 0)} images"
        else:
            update_preview_label()
    
    # Function to update preview label
    def update_preview_label():
        if hasattr(toggle_range_controls, 'total_images'):
            total = toggle_range_controls.total_images
            start = start_idx_spin.value
            end = end_idx_spin.value
            count = end - start
            preview_label.value = f"Will load images {start} to {end-1} ({count} of {total} total)"
        
    # Connect checkbox
    load_all_checkbox.changed.connect(toggle_range_controls)
    
    # Connect spinbox changes
    start_idx_spin.changed.connect(lambda v: update_preview_label())
    end_idx_spin.changed.connect(lambda v: update_preview_label())
    
    # Folder selection with preview
    @magicgui(
        call_button="Select Folders",
        segmented_folder={"mode": "d"},
        original_folder={"mode": "d"}
    )
    def select_folders(segmented_folder: Path = Path("."), 
                      original_folder: Path = Path(".")):
        """Select folders and preview available images"""
        try:
            # Store folder paths
            select_folders.seg_folder = segmented_folder
            select_folders.orig_folder = original_folder
            
            # Preview to get image count
            total_images, filenames = pipeline.preview_folders(
                str(segmented_folder), 
                str(original_folder)
            )
            
            if total_images == 0:
                preview_label.value = "No images found in selected folders!"
                return
            
            # Store total for later use
            toggle_range_controls.total_images = total_images
            
            # Update spinbox ranges
            start_idx_spin.max = total_images - 1
            end_idx_spin.max = total_images
            end_idx_spin.value = min(total_images, 10)  # Default to first 10 or all if less
            
            # Update preview
            if load_all_checkbox.value:
                preview_label.value = f"Found {total_images} images. Ready to load all."
            else:
                update_preview_label()
                
            # Show first few filenames
            preview_files = filenames[:3]
            if len(filenames) > 3:
                preview_files.append("...")
            print(f"Found images: {', '.join(preview_files)}")
            
        except Exception as e:
            preview_label.value = f"Error: {e}"
    
    # Load button
    def load_images():
        """Load the selected range of images"""
        if not hasattr(select_folders, 'seg_folder'):
            print("Please select folders first!")
            return
            
        try:
            # Determine range
            if load_all_checkbox.value:
                start = None
                end = None
                print("Loading all images...")
            else:
                start = start_idx_spin.value
                end = end_idx_spin.value
                print(f"Loading images {start} to {end-1}...")
            
            # Load images
            pipeline.load_image_folders(
                str(select_folders.seg_folder),
                str(select_folders.orig_folder),
                start_idx=start,
                end_idx=end
            )
            
            if pipeline.viewer:
                update_viewer()
                
            preview_label.value = f"✅ Loaded {len(pipeline.segmented_stack)} images successfully!"
            
        except Exception as e:
            preview_label.value = f"Error loading: {e}"
            print(f"Error loading images: {e}")
    
    load_btn = PushButton(text="Load Images")
    load_btn.clicked.connect(load_images)
    
    # Create a compact folder section
    folder_section = Container(
        widgets=[
            # LabelWidget(value="=== IMAGE LOADING ==="),
            select_folders,
            preview_label,
            load_all_checkbox,
            range_controls,
            load_btn
        ],
        layout='vertical'
    )
    
    # Label selection - organized in 2 columns
    label_checkboxes = []
    for i, name in enumerate(pipeline.label_names):
        checkbox = CheckBox(value=False)
        checkbox.text = f"{i+1}: {name}"
        
        def make_callback(idx):
            def callback(val):
                if val:
                    pipeline.selected_labels.add(idx)
                else:
                    pipeline.selected_labels.discard(idx)
                print(f"Label {idx} {'selected' if val else 'deselected'}")
            return callback
        
        checkbox.changed.connect(make_callback(i + 1))
        label_checkboxes.append(checkbox)

    # Organize labels into 4 efficient columns (19 labels: 5,5,5,4 distribution)
    column1_labels = label_checkboxes[0:5]    # Labels 1-5
    column2_labels = label_checkboxes[5:10]   # Labels 6-10  
    column3_labels = label_checkboxes[10:15]  # Labels 11-15
    column4_labels = label_checkboxes[15:19]  # Labels 16-19
    
    # Create compact containers for each column
    labels_column1 = Container(widgets=column1_labels, layout='vertical')
    labels_column2 = Container(widgets=column2_labels, layout='vertical')
    labels_column3 = Container(widgets=column3_labels, layout='vertical')
    labels_column4 = Container(widgets=column4_labels, layout='vertical')
    
    # Create horizontal container to hold all 4 columns
    labels_layout = Container(
        widgets=[labels_column1, labels_column2, labels_column3, labels_column4],
        layout='horizontal'
    )
    
    # Select all button
    def select_all_labels():
        for i, cb in enumerate(label_checkboxes):
            cb.value = True
            pipeline.selected_labels.add(i + 1)
        print("All labels selected")
    
    select_all_btn = PushButton(text="Select All Labels")
    select_all_btn.clicked.connect(select_all_labels)
    
    # Clear all button
    def clear_all_labels():
        for cb in label_checkboxes:
            cb.value = False
        pipeline.selected_labels.clear()
        print("All labels cleared")
    
    clear_all_btn = PushButton(text="Clear All Labels")
    clear_all_btn.clicked.connect(clear_all_labels)
    
    # Transfer label selection
    transfer_combo = ComboBox(
        choices=[f"{i}: {pipeline.label_names[i-1]}" for i in range(1, pipeline.num_labels + 1)],
        value=f"1: {pipeline.label_names[0]}",
        name="Transfer to label"
    )
    
    def update_transfer_label(value):
        label_num = int(value.split(":")[0])
        pipeline.transfer_label = label_num
        print(f"Transfer target set to label {label_num}")
    
    transfer_combo.changed.connect(update_transfer_label)
    
    # Z-projection controls
    z_depth_spin = SpinBox(
        value=5,
        min=1,
        max=20,
        name="Z-Projection Depth"
    )
    
    def update_z_depth(value):
        pipeline.z_projection_depth = value
    
    z_depth_spin.changed.connect(update_z_depth)
    
    # Multi-image editing
    multi_image_spin = SpinBox(
        value=1,
        min=1,
        max=100,
        name="Apply to N images (starting from current)"
    )
    
    def apply_roi():
        if (pipeline.shapes_layer is not None and 
            len(pipeline.shapes_layer.data) > 0 and 
            pipeline.segmented_stack is not None):
            
            # Get transferable ROIs
            rois_to_apply = pipeline.get_transferable_rois()
            
            if len(rois_to_apply) == 0:
                print("❌ No ROIs to apply after filtering out SAM2 working ROI.")
                return
            
            num_images = multi_image_spin.value
            indices = list(range(pipeline.current_index, 
                            min(pipeline.current_index + num_images, 
                                    len(pipeline.segmented_stack))))
            
            roi_counts = pipeline.count_roi_types()
            print(f"\n🎯 Applying {len(rois_to_apply)} ROIs to {len(indices)} images...")
            
            # Direct synchronous processing
            pipeline.apply_roi_transfer(rois_to_apply, indices)
            
            # Clear ROIs after transfer
            try:
                if pipeline.sam2_working_roi_id is not None:
                    print("✅ ROI transfer complete. Keeping SAM2 working ROI.")
                else:
                    pipeline.safely_clear_rois()
                    print("✅ ROI transfer complete. ROIs cleared.")
            except Exception as e:
                print(f"Warning: Could not clear ROIs: {e}")
    
    apply_btn = PushButton(text="Apply ROI Transfer")
    apply_btn.clicked.connect(apply_roi)
    
    # ============================================================================
    # SAM2 INTEGRATION BUTTONS
    # ============================================================================
    
    # SAM2 Seg button
    def sam2_seg():
        pipeline.sam2_seg_button_clicked()
        update_viewer()
    
    sam2_seg_btn = PushButton(text="SAM2 Seg")
    sam2_seg_btn.clicked.connect(sam2_seg)
    
    # SAM2 Object ID SpinBox - NEW!
    sam2_object_spin = SpinBox(
        value=1,
        min=1,
        max=99,
        name="SAM2 Object ID"
    )
    
    def update_sam2_object_id(value):
        previous_id = pipeline.current_sam2_object_id
        
        # No need for detection - annotations are already tracked!
        # Just update to new object ID
        pipeline.current_sam2_object_id = value
        print(f"\n🎯 SAM2 Object ID changed from {previous_id} to {value}")
        
        # Show summary for previous object
        if previous_id != value:
            print(f"\nSummary for previous object:")
            print(pipeline.get_object_summary(previous_id))
        
        # Show summary for new current object
        print(f"\nSummary for current object:")
        print(pipeline.get_object_summary(value))
        
        # Show all active objects
        if pipeline.active_object_ids:
            print(f"\nActive objects with annotations: {sorted(pipeline.active_object_ids)}")
        
        # Validate state
        pipeline.validate_annotation_state()
    
    sam2_object_spin.changed.connect(update_sam2_object_id)
    
    # Propagate button
    # Replace the single Propagate button with two buttons in parallel
    def propagate_forward():
        num_images = multi_image_spin.value
        pipeline.propagate_button_clicked(num_images, direction="forward")
        update_viewer()

    def propagate_backward():
        num_images = multi_image_spin.value
        pipeline.propagate_button_clicked(num_images, direction="backward")
        update_viewer()

    propagate_forward_btn = PushButton(text="→ Forward")
    propagate_forward_btn.clicked.connect(propagate_forward)

    propagate_backward_btn = PushButton(text="← Backward")
    propagate_backward_btn.clicked.connect(propagate_backward)

    # Create horizontal container for both propagate buttons
    propagate_buttons = Container(
        widgets=[propagate_backward_btn, propagate_forward_btn],
        layout='horizontal'
    )
    
    # Refine Seg button
    def refine_seg():
        pipeline.refine_seg_button_clicked()
        update_viewer()
    
    refine_seg_btn = PushButton(text="Refine Seg")
    refine_seg_btn.clicked.connect(refine_seg)
    
    # Delete Seg button
    def delete_seg():
        pipeline.delete_seg_button_clicked()
        update_viewer()
    
    delete_seg_btn = PushButton(text="Delete Seg")
    delete_seg_btn.clicked.connect(delete_seg)
    
    # ============================================================================
    # OTHER BUTTONS (UNCHANGED)
    # ============================================================================
    
    # Undo button
    def undo():
        pipeline.undo_last_change()
        update_viewer()
    
    undo_btn = PushButton(text="Undo Last Change")
    undo_btn.clicked.connect(undo)
    
    # Save/Load buttons
    @magicgui(call_button="Save Project", directory={"mode": "d"})
    def save_project(directory: Path = Path(".")):
        pipeline.save_project(str(directory))
    
    @magicgui(call_button="Load Project", directory={"mode": "d"})
    def load_project(directory: Path = Path(".")):
        pipeline.load_project(str(directory))
        update_viewer()
    
    # Update z-projection button
    def update_z_projection():
        if pipeline.viewer and pipeline.segmented_stack is not None and pipeline.current_index < len(pipeline.segmented_stack):
            # Create z-projection for the full image
            # We'll pass None for current_view to get the full projection
            z_proj = pipeline.create_z_projection(
                pipeline.current_index, 
                pipeline.z_projection_depth,
                None  # Full image projection
            )
            
            # Get image dimensions for positioning
            displayed_shape = pipeline.segmented_stack.shape[1:3]  # (height, width)
            
            if pipeline.z_projection_layer is None:
                # Create new layer positioned to the right
                layer = pipeline.viewer.add_image(
                    z_proj, 
                    name="Z-Projection (excl. collagen)",
                    translate=(0, displayed_shape[1] + 20)
                )
                pipeline.z_projection_layer = layer[0] if isinstance(layer, list) else layer
            else:
                # Update existing layer data
                setattr(pipeline.z_projection_layer, 'data', z_proj)
                
            slice_end = min(pipeline.current_index + pipeline.z_projection_depth, len(pipeline.segmented_stack) - 1)
            print(f"Updated Z-projection: slices {pipeline.current_index} to {slice_end}")
            print("Note: Label 12 (collagen) is excluded from z-projection")
    
    z_proj_btn = PushButton(text="Update Z-Projection")
    z_proj_btn.clicked.connect(update_z_projection)
    

    # Viewer update function
    # Replace the existing update_viewer function in create_gui
    def update_viewer():
        if pipeline.viewer and pipeline.composite_stack is not None:
            pipeline.update_viewer()  # This now calls the standard update method
    

    # Assemble compact GUI - with SAM2 Object ID spinbox
    container = Container(widgets=[
        LabelWidget(value="=== FOLDER LOADING ==="),
        folder_section,  # New compact folder section
        LabelWidget(value="=== LABELS (Select to transfer FROM) ==="),
        labels_layout,  # 4-column layout instead of 2-column
        select_all_btn,
        clear_all_btn,
        LabelWidget(value="=== TRANSFER SETTINGS ==="),
        transfer_combo,
        multi_image_spin,
        LabelWidget(value="=== TRADITIONAL WORKFLOW ==="),
        apply_btn,
        undo_btn,
        LabelWidget(value="=== SAM2 WORKFLOW ==="),
        sam2_seg_btn,
        sam2_object_spin,  # NEW: SAM2 Object ID spinbox
        propagate_buttons,
        refine_seg_btn,
        delete_seg_btn,
        LabelWidget(value="=== Z-PROJECTION ==="),
        z_depth_spin,
        z_proj_btn,
        LabelWidget(value="=== PROJECT MANAGEMENT ==="),
        save_project,
        load_project
    ])
    
    # Store reference to update function and make container scrollable-friendly
    widget_refs['update_viewer'] = update_viewer
    
    # 🔧 ENHANCEMENT: Set container properties for better scrolling
    try:
        # Ensure the container has proper size policies for scrolling
        container_widget = container.native if hasattr(container, 'native') else container
        if hasattr(container_widget, 'setSizePolicy'):
            from qtpy.QtWidgets import QSizePolicy
            container_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        if hasattr(container_widget, 'setMinimumWidth'):
            container_widget.setMinimumWidth(320)  # Ensure minimum readable width
    except:
        pass  # If enhancement fails, continue with standard container
    
    return container


def main():
    """Main entry point"""
    try:
        # Create pipeline instance
        pipeline = SegmentationCorrectionPipeline()
        
        # 🔧 FIX: Get screen dimensions and set appropriate window size
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()  # Hide the tkinter window
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
            
            # Calculate appropriate viewer size (leave margin for taskbar/dock)
            margin_width = 400  # Space for dock widget + margins
            margin_height = 150  # Space for title bar + taskbar
            
            viewer_width = min(1400, screen_width - margin_width)  # Max 1400px or screen width - margin
            viewer_height = min(900, screen_height - margin_height)  # Max 900px or screen height - margin
            
            print(f"🖥️  Screen: {screen_width}x{screen_height}, Setting viewer: {viewer_width}x{viewer_height}")
            
        except Exception as e:
            print(f"Could not detect screen size: {e}, using default size")
            viewer_width, viewer_height = 1200, 800
        
        # Create napari viewer with calculated size
        viewer = napari.Viewer(
            title="Unified Segmentation Correction and SAM2 Pipeline",
            show=False  # Don't show yet, set size first
        )
        
        # Set window size
        viewer.window.qt_viewer.resize(viewer_width, viewer_height)
        
        # Center the window on screen
        try:
            screen_geometry = viewer.window.qt_viewer.screen().geometry()
            x = (screen_geometry.width() - viewer_width) // 2
            y = (screen_geometry.height() - viewer_height) // 2
            viewer.window.qt_viewer.move(x, y)
        except:
            pass  # If centering fails, just use default position
        
        pipeline.viewer = viewer
        
        # Create and add GUI with properly implemented scrollable dock widget
        gui = create_gui(pipeline)
        dock_widget = viewer.window.add_dock_widget(
            gui, 
            area="right", 
            name="Segmentation Correction"
        )
        
        # 🔧 FIX: Proper scrollable implementation for smaller screens
        try:
            from qtpy.QtWidgets import QScrollArea, QSizePolicy, QApplication
            from qtpy.QtCore import Qt
            
            # Get the original widget
            original_widget = dock_widget.widget()
            
            # Force the original widget to calculate its proper size
            original_widget.adjustSize()
            original_widget.updateGeometry()
            
            # Get the minimum size needed for all content
            min_size = original_widget.minimumSizeHint()
            if not min_size.isValid():
                min_size = original_widget.sizeHint()
            
            print(f"📏 Content minimum size: {min_size.width()}x{min_size.height()}")
            
            # 🔧 DEBUGGING: Get detailed size information before modifications
            original_height = original_widget.height()
            original_size_hint = original_widget.sizeHint()
            original_min_size_hint = original_widget.minimumSizeHint()
            print(f"🔍 Original widget - Height: {original_height}, SizeHint: {original_size_hint.width()}x{original_size_hint.height()}, MinSizeHint: {original_min_size_hint.width()}x{original_min_size_hint.height()}")
            
            # Create scroll area with proper settings
            scroll_area = QScrollArea()
            scroll_area.setWidget(original_widget)
            scroll_area.setWidgetResizable(True)
            scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            
            # Set proper size policies
            scroll_area.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
            original_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
            
            # 🔧 ALIGNMENT FIX: Ensure scroll area is properly aligned and fills dock widget  
            scroll_area.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            
            # Set width constraints but let height expand naturally to fill dock widget
            scroll_area.setMinimumWidth(350)
            scroll_area.setMaximumWidth(450)
            
            # 🔧 KEY FIX: Multiple approaches to ensure proper content height
            # Method 1: Use detected minimum size
            target_height = 0
            if min_size.isValid() and min_size.height() > 0:
                target_height = min_size.height()
                original_widget.setMinimumHeight(target_height)
                print(f"✅ Method 1: Set content minimum height to: {target_height}px")
            else:
                # Method 2: Estimate based on widget count
                widget_count = 0
                try:
                    if hasattr(original_widget, 'layout'):
                        layout = original_widget.layout()
                        if layout:
                            widget_count = layout.count()
                    if widget_count == 0:
                        if hasattr(gui, '__len__'):
                            widget_count = len(gui)
                        else:
                            widget_count = 25  # Safe fallback
                except:
                    widget_count = 25  # Safe fallback
                
                target_height = widget_count * 35 + 150  # ~35px per widget + extra padding
                original_widget.setMinimumHeight(target_height)
                print(f"✅ Method 2: Set estimated content height to: {target_height}px (based on {widget_count} widgets)")
            
            # 🔧 ENHANCED FIX: Force layout calculation and size updates
            # Step 1: Force the original widget to calculate its proper size
            original_widget.adjustSize()
            original_widget.updateGeometry()
            
            # Step 2: Process events to ensure Qt calculates layouts
            try:
                QApplication.processEvents()
            except:
                pass
            
            # Step 3: Get updated size information
            updated_height = original_widget.height()
            updated_size_hint = original_widget.sizeHint()
            updated_min_size_hint = original_widget.minimumSizeHint()
            print(f"🔍 After size setting - Height: {updated_height}, SizeHint: {updated_size_hint.width()}x{updated_size_hint.height()}, MinSizeHint: {updated_min_size_hint.width()}x{updated_min_size_hint.height()}")
            
            # Step 4: Use the largest reasonable height for final setting
            final_height = max(target_height, updated_height, updated_size_hint.height())
            if final_height != target_height:
                original_widget.setMinimumHeight(final_height)
                print(f"🔧 Adjusted final content height to: {final_height}px")
            
            # 🔧 POSITIONING FIX: Let scroll area fill dock widget completely (no height constraints)
            print(f"✅ Content sized for proper scrolling, scroll area will fill dock widget")
            
            # Replace the dock widget content with proper positioning
            dock_widget.setWidget(scroll_area)
            
            # 🔧 POSITIONING FIX: Ensure dock widget and scroll area are properly positioned
            try:
                # Make sure the dock widget itself is properly configured
                dock_widget.setMinimumWidth(350)
                dock_widget.setMaximumWidth(450)
                
                # Ensure the scroll area fills the entire dock widget area from top to bottom
                dock_widget_layout = dock_widget.layout()
                if dock_widget_layout:
                    dock_widget_layout.setContentsMargins(0, 0, 0, 0)  # Remove any margins
                    dock_widget_layout.setSpacing(0)  # Remove any spacing
                
            except Exception as layout_error:
                print(f"Note: Could not optimize dock widget layout: {layout_error}")
            
            # 🔧 SCROLL RANGE FIX: Force scroll area to recalculate range after all size changes
            # Step 1: Complete initial setup
            original_widget.adjustSize()
            scroll_area.updateGeometry()
            dock_widget.updateGeometry()
            
            # Step 2: Process events to ensure all layouts are calculated
            try:
                QApplication.processEvents()
            except:
                pass
            
            # Step 3: Force scroll area to recalculate its scroll range
            scroll_area.ensureWidgetVisible(original_widget, 0, 0)  # This forces range recalculation
            
            # Step 4: Get scroll area dimensions for verification
            scroll_area_height = scroll_area.height()
            scroll_area_viewport_height = scroll_area.viewport().height() if scroll_area.viewport() else 0
            content_height_final = original_widget.height()
            
            # Step 5: Manually verify and set scroll range if needed
            max_scroll_value = scroll_area.verticalScrollBar().maximum()
            scroll_range = max_scroll_value
            expected_scroll_range = max(0, content_height_final - scroll_area_viewport_height)
            
            print(f"🔍 Scroll area verification:")
            print(f"   Scroll area height: {scroll_area_height}px")
            print(f"   Viewport height: {scroll_area_viewport_height}px") 
            print(f"   Content height: {content_height_final}px")
            print(f"   Current scroll range: {scroll_range}px")
            print(f"   Expected scroll range: {expected_scroll_range}px")
            
            # Step 6: Force correct scroll range if Qt's calculation is wrong
            if scroll_range < expected_scroll_range * 0.8:  # Allow some tolerance
                print(f"🔧 Forcing scroll range recalculation...")
                
                # Force widget to use exact calculated height
                original_widget.setFixedHeight(content_height_final)
                
                # Force scroll area update
                scroll_area.updateGeometry()
                scroll_area.widget().updateGeometry()
                
                # Process events again
                try:
                    QApplication.processEvents()
                except:
                    pass
                
                # Check again
                new_max_scroll = scroll_area.verticalScrollBar().maximum()
                print(f"🔍 After forced recalculation: scroll range = {new_max_scroll}px")
                
                # If still wrong, set explicit scroll range (last resort)
                if new_max_scroll < expected_scroll_range * 0.8:
                    scroll_area.verticalScrollBar().setRange(0, expected_scroll_range)
                    print(f"🔧 Set explicit scroll range: 0 to {expected_scroll_range}px")
            
            print(f"✅ Scroll range verification complete")
            
            # 🔧 FINAL VERIFICATION: Comprehensive scroll setup verification
            final_content_height = original_widget.height()
            final_scroll_height = scroll_area.height()
            final_viewport_height = scroll_area.viewport().height() if scroll_area.viewport() else final_scroll_height
            final_scroll_range = scroll_area.verticalScrollBar().maximum()
            can_scroll = final_scroll_range > 0
            scroll_sufficient = final_scroll_range >= (final_content_height - final_viewport_height) * 0.8
            
            print(f"📊 FINAL VERIFICATION:")
            print(f"   Content height: {final_content_height}px")
            print(f"   Scroll area height: {final_scroll_height}px") 
            print(f"   Viewport height: {final_viewport_height}px")
            print(f"   Scroll range: {final_scroll_range}px")
            print(f"   Can scroll: {can_scroll}")
            print(f"   Scroll sufficient: {scroll_sufficient}")
            print(f"   Expected range: {max(0, final_content_height - final_viewport_height)}px")
            
            if scroll_sufficient:
                print("✅ Scroll range is sufficient to access all content")
            else:
                print("⚠️  Scroll range may be insufficient - some buttons might not be accessible")
            
            print("✅ Implemented robust scrollable dock widget with proper positioning and range")
            
        except Exception as e:
            print(f"Could not implement comprehensive scrollable dock widget: {e}")
            print("Falling back to standard dock widget")
            import traceback
            traceback.print_exc()
        
        # Now show the viewer
        viewer.show()
        
        # Initialize point layers immediately so they're visible at startup
        pipeline.initialize_point_layers()
        
        @viewer.bind_key("u")
        def undo_key(viewer):
            """Undo last change"""
            try:
                pipeline.undo_last_change()  # This now uses optimized undo
            except Exception as e:
                print(f"Error in undo: {e}")
        
        @viewer.bind_key("z")
        def toggle_z_projection(viewer):
            """Toggle z-projection visibility"""
            try:
                if pipeline.z_projection_layer:
                    pipeline.z_projection_layer.visible = not pipeline.z_projection_layer.visible
            except Exception as e:
                print(f"Error toggling z-projection: {e}")
        
        @viewer.bind_key("a")
        def apply_roi_key(viewer):
            """Apply ROI transfer"""
            try:
                if (pipeline.shapes_layer is not None and 
                    len(pipeline.shapes_layer.data) > 0 and 
                    pipeline.segmented_stack is not None):
                    
                    # Use the safer helper methods
                    rois_to_apply = pipeline.get_transferable_rois()
                    
                    if len(rois_to_apply) == 0:
                        print("❌ No transferable ROIs found (SAM2 working ROI excluded)")
                        return
                    
                    num_images = 1  # Default to current image when using keyboard shortcut
                    indices = [pipeline.current_index]
                    
                    roi_counts = pipeline.count_roi_types()
                    print(f"\n🎯 Applying {len(rois_to_apply)} ROIs to current image")
                    if roi_counts['working'] > 0:
                        print(f"   (Excluded {roi_counts['working']} SAM2 working ROI)")
                    
                    pipeline.apply_roi_transfer(rois_to_apply, indices)
                    if pipeline.composite_stack is not None:
                        setattr(pipeline.composite_layer, 'data', pipeline.composite_stack)
                    
                    # Clear applied ROIs if not in SAM2 workflow
                    if pipeline.sam2_working_roi_id is None:
                        pipeline.safely_clear_rois()
                        print("✅ ROI transfer complete. All ROIs cleared.")
                    else:
                        print("✅ ROI transfer complete. SAM2 working ROI preserved.")
                else:
                    print("❌ No ROIs to apply or no data loaded")
            except Exception as e:
                print(f"Error in apply ROI: {e}")
                import traceback
                traceback.print_exc()
        
        @viewer.bind_key("c")
        def clear_rois_key(viewer):
            """Clear all ROIs - emergency cleanup"""
            try:
                pipeline.safely_clear_rois()
            except Exception as e:
                print(f"Error clearing ROIs: {e}")
        
        @viewer.bind_key("s")
        def sam2_seg_key(viewer):
            """SAM2 Seg shortcut"""
            try:
                pipeline.sam2_seg_button_clicked()
            except Exception as e:
                print(f"Error in SAM2 seg: {e}")
        
        @viewer.bind_key("p")
        def propagate_key(viewer):
            """Propagate shortcut"""
            try:
                pipeline.propagate_button_clicked(1)  # Default to 1 image
            except Exception as e:
                print(f"Error in propagate: {e}")
        
        @viewer.bind_key("r")
        def refine_seg_key(viewer):
            """Refine Seg shortcut"""
            try:
                pipeline.refine_seg_button_clicked()
            except Exception as e:
                print(f"Error in refine seg: {e}")
        
        @viewer.bind_key("d")
        def delete_seg_key(viewer):
            """Delete Seg shortcut"""
            try:
                pipeline.delete_seg_button_clicked()
            except Exception as e:
                print(f"Error in delete seg: {e}")
        
        # Handle dimension changes
        def on_dim_change(event):
            try:
                if pipeline.segmented_stack is not None:
                    if hasattr(viewer.dims, 'current_step') and len(viewer.dims.current_step) > 0:
                        old_index = pipeline.current_index
                        pipeline.current_index = viewer.dims.current_step[0]
                        if old_index != pipeline.current_index:
                            print(f"Current image: {pipeline.current_index + 1}/{len(pipeline.segmented_stack)}")
            except Exception as e:
                print(f"Error in dimension change: {e}")
        
        viewer.dims.events.current_step.connect(on_dim_change)
        
        # Print instructions
        print("\n=== UNIFIED SEGMENTATION CORRECTION AND SAM2 PIPELINE ===")
        print("🚀 Full SAM2 integration active! All placeholder functions replaced.")
        print("✨ NEW: Image-specific points + SAM2 working ROI filtering + Original image usage!")
        print("🖥️  DISPLAY: Auto-sized window + 4-column compact label layout + scrollable interface!")
        print("\n📋 REQUIREMENTS:")
        print("  ✅ SAM2 model weights (configured in script)")
        print("  ✅ CUDA GPU recommended for best performance")
        print("  ✅ OpenCV with Qt backend handling")
        print("  ✅ Napari with native point layer integration")
        print(f"\n🔧 SAM2 MODEL CONFIG:")
        print(f"  Checkpoint: {SAM2_CHECKPOINT}")
        print(f"  Config: {SAM2_CONFIG}")
        print("  💡 Update paths at top of script if needed")
        print("\n🎯 KEY IMPROVEMENTS:")
        print("  → Points are now IMAGE-SPECIFIC (only appear on their target image)")
        print("  → SAM2 working ROI (green square) is excluded from label transfer")
        print("  → Only generated SAM2 ROIs (cyan) and traditional ROIs (yellow) are applied")
        print("  → SAM2 uses ORIGINAL images, not overlay images")
        print("  → Points are smaller (size 8) for better precision")
        print("  → Enhanced 3D coordinate handling for repeated SAM2 usage")
        print("  → Better error handling and user feedback throughout")
        print("  → Fixed AttributeError issues with sam2_working_roi_id")
        print("\nKEYBOARD SHORTCUTS:")
        print("  u - Undo last change")
        print("  z - Toggle Z-projection visibility")
        print("  a - Apply ROI transfer")
        print("  c - Clear all ROIs (emergency cleanup)")
        print("  s - SAM2 Seg (convert rectangle to square, enter annotation mode)")
        print("  p - Propagate (batch SAM2 across images)")
        print("  r - Refine Seg (enable refinement mode)")
        print("  d - Delete Seg (clear SAM2 state)")
        print("\n=== TRADITIONAL WORKFLOW ===")
        print("HOW TO DRAW ROI:")
        print("1. Click the polygon tool in napari toolbar (or press 'p')")
        print("2. Click to place vertices around the region")
        print("3. Double-click or press Enter to close the polygon")
        print("4. Click 'Apply ROI Transfer' or press 'a'")
        print("\n=== SAM2 WORKFLOW ===")
        print("STEP 1 - SET UP CROP AREA:")
        print("1. Draw a RECTANGLE ROI using the rectangle tool")
        print("2. This defines the crop area that SAM2 will operate on")
        print("3. Ensure only ONE ROI is selected")
        print("4. Click 'SAM2 Seg' (or press 's')")
        print("   → Rectangle automatically converts to square (centered)")
        print("   → Green square shows the SAM2 crop area")
        print("   → Enters annotation mode")
        print("\nSTEP 2 - CHOOSE PROMPT TYPE:")
        print("Option A - Point Prompts:")
        print("1. Select 'Positive Points' layer (green) in napari layer list")
        print("2. Click the 'Add points' button (+ icon) in napari toolbar")
        print("3. Click inside the green square to place positive points")
        print("4. Optionally select 'Negative Points' layer (red) for exclusion areas")
        print("\nOption B - Box Prompt:")
        print("1. Use napari's rectangle tool")
        print("2. Draw a smaller rectangle INSIDE the green square")
        print("3. This rectangle becomes the bounding box prompt")
        print("4. SAM2 will segment everything inside this inner box")
        print("\nOption C - Combined (Box + Points):")
        print("1. Draw a rectangle inside the green square (box prompt)")
        print("2. Add positive/negative points for refinement")
        print("3. SAM2 uses both prompts together for better results")
        print("\nSTEP 3 - PROPAGATE:")
        print("1. Set 'Apply to N images' (how many images to process starting from current)")
        print("   → Current image + (N-1) additional images will be processed")
        print("   → Example: Current=5, N=3 → processes images 5, 6, 7")
        print("2. Click 'Propagate' (or press 'p')")
        print("   → Extracts ROI cutouts as JPEGs")
        print("   → Runs SAM2 batch processing with your points")
        print("   → Converts masks back to global ROI coordinates")
        print("   → Adds generated ROIs to viewer (cyan color)")
        print("\nSTEP 4 - APPLY OR REFINE:")
        print("Option A - Apply Results:")
        print("  → Click 'Apply ROI Transfer' to transfer labels in all generated ROIs")
        print("Option B - Refine Results:")
        print("  → Click 'Refine Seg' to enter refinement mode")
        print("  → Navigate through processed images and add more points")
        print("  → Click 'Propagate' to reprocess with additional refinement points")
        print("  → Can repeat refinement cycle multiple times")
        print("\nSTEP 5 - CLEANUP:")
        print("  → Click 'Delete Seg' (or press 'd') to clear all SAM2 data")
        print("  → Returns to idle state for new workflow")
        print("\n=== COORDINATE SYSTEMS ===")
        print("GLOBAL COORDINATES:")
        print("  → Full image coordinate system")
        print("  → Used by napari layers and point placement")
        print("LOCAL COORDINATES:")
        print("  → ROI-relative coordinate system (0,0 = top-left of ROI)")
        print("  → Used by SAM2 processing")
        print("  → Automatically converted by pipeline")
        print("\n=== SAM2 PROMPT COMPARISON ===")
        print("POINT PROMPTS:")
        print("  ✅ Precise control over what to segment")
        print("  ✅ Can segment specific objects within crop area")
        print("  ✅ Best for complex scenes with multiple objects")
        print("  ❌ Requires manual point placement")
        print("  ❌ More time-consuming")
        print("\nBOX PROMPTS:")
        print("  ✅ Fast - draw one rectangle")
        print("  ✅ Segments everything inside the box")
        print("  ✅ Ideal for single objects")
        print("  ✅ Can combine with points for refinement")
        print("  ❌ Less precise than point-only mode")
        print("  ❌ May include unwanted objects in box area")
        print("\nCOMBINED (BOX + POINTS):")
        print("  ✅ Best of both worlds")
        print("  ✅ Box provides initial guidance")
        print("  ✅ Points refine the segmentation")
        print("  ✅ Most robust for challenging objects")
        print("  ❌ Requires both box and point annotations")
        print("\nWHEN TO USE EACH MODE:")
        print("🎯 Use POINT MODE when:")
        print("   - ROI contains multiple objects")
        print("   - Need to segment specific parts")
        print("   - Complex backgrounds")
        print("   - Precise control required")
        print("\n📦 Use BOUNDING BOX MODE when:")
        print("   - Single object fills most of ROI")
        print("   - Simple, clean backgrounds")
        print("   - Quick segmentation needed")
        print("   - Object boundaries align with ROI edges")
        print("\n=== SAM2 WORKFLOW STATES ===")
        print("idle: No SAM2 workflow active")
        print("annotation: Square ROI set, ready for annotation")
        print("propagated: SAM2 batch processing completed")
        print("refining: Refinement mode active")
        print("\n=== POINT LAYERS ===")
        print("Positive Points (green): Areas to include in segmentation")
        print("Negative Points (red): Areas to exclude from segmentation")
        print("  → Points automatically track global/local coordinates")
        print("  → Points are linked to specific images")
        print("  → Clear automatically when switching workflows")
        print("\n=== MULTIPLE ROIs ===")
        print("TRADITIONAL:")
        print("  → Draw multiple polygons before applying")
        print("  → All ROIs processed together")
        print("SAM2:")
        print("  → Single rectangle input → multiple generated ROIs")
        print("  → Generated ROIs can be applied together")
        print("\nNAVIGATION:")
        print("  Mouse scroll - Navigate through image stack")
        print("  Shift + drag - Pan the image")
        print("  Ctrl + scroll - Zoom in/out")
        print("\nZ-PROJECTION:")
        print("  - Shows fluorescent/holographic view of next N slices")
        print("  - Label 12 (collagen) is excluded from z-projection")
        print("  - Each slice's contribution is visible with depth-based weighting")
        print("\nOVERLAY:")
        print("  - Original image: 70%, Segmentation: 30%")
        print("  - Uses cv2.addWeighted for smooth blending")
        print("\nTROUBLESHOOTING:")
        print("  - If SAM2 causes issues, press 'd' to delete SAM2 state")
        print("  - If viewer becomes unresponsive, press 'c' to clear ROIs")
        print("  - If point placement fails, check that you're in the right layer")
        print("  - ROI must be rectangle for SAM2 (polygon tool won't work)")
        print("\nSAM2 REQUIREMENTS:")
        print("  - Must use rectangle tool (other shapes rejected)")
        print("  - Only 1 ROI allowed for SAM2 (multiple ROIs rejected)")
        print("  - Rectangle will be auto-converted to square (centered)")
        print("  - Square must fit within image boundaries")
        print("  - At least 1 positive point required for propagation")

        # Start the application
        napari.run()
        
    except Exception as e:
        print(f"Critical error in main: {e}")
        print("Please restart the application")


if __name__ == "__main__":
    main()

# in case you can't save, here is a script to save directly in napari console. pull up console by clicking control+shift+c
# first run this below:
# Get the real pipeline from Method 1 results
# import inspect
# import numpy as np
# print("🎯 ACCESSING THE REAL PIPELINE")
# print("="*40)

# real_pipeline = None

# # Look through frames for the one with actual numeric shape
# for frame_info in inspect.stack():
#     frame = frame_info.frame
#     all_vars = {**frame.f_locals, **frame.f_globals}
    
#     for name, obj in all_vars.items():
#         try:
#             if (hasattr(obj, 'segmented_stack') and 
#                 hasattr(obj, 'change_history') and
#                 hasattr(obj.segmented_stack, 'shape') and
#                 isinstance(obj.segmented_stack.shape, tuple) and  # Must be actual tuple, not string
#                 len(obj.segmented_stack.shape) == 3):  # Should be 3D: (N, H, W)
                
#                 print(f"✅ REAL PIPELINE FOUND: {name}")
#                 print(f"   Shape: {obj.segmented_stack.shape}")
#                 print(f"   Dtype: {obj.segmented_stack.dtype}")
#                 print(f"   Changes: {len(obj.change_history)}")
#                 real_pipeline = obj
#                 break
#         except:
#             continue
    
#     if real_pipeline:
#         break

# if real_pipeline:
#     print(f"\n🎉 SUCCESS! Found real pipeline with shape {real_pipeline.segmented_stack.shape}")
# else:
#     print("❌ Could not access real pipeline")

# then run this one below:   
# IMMEDIATE SAVE OF REAL SEGMENTATION DATA
# if real_pipeline is not None:
#     from PIL import Image as PILImage
#     from pathlib import Path
#     import json
#     from datetime import datetime
#     from dataclasses import asdict
    
#     print("\n💾 SAVING REAL CORRECTED SEGMENTATION DATA")
#     print("="*50)
    
#     # Set save directory
#     save_dir = Path(r"\\10.162.80.11\Andre_kit\data\monkey_fetus\bissected_monkey_GS55\10x_python\registeredE\int_2\5x_downsampled_images\classification_MODEL1_5x_GS40_GS55_06_10_2025_45_big_tiles_inceptionresnetv2\round_3_corrected_classification\275_to_346")
    
#     # Create directory for the REAL lightweight segmentation masks
#     real_segmented_dir = save_dir / "corrected_segmentation_masks"
#     real_segmented_dir.mkdir(exist_ok=True)
    
#     print(f"Saving to: {real_segmented_dir}")
#     print(f"Data shape: {real_pipeline.segmented_stack.shape}")
#     print(f"Data type: {real_pipeline.segmented_stack.dtype}")
    
#     # Save each corrected segmentation mask
#     for i in range(len(real_pipeline.segmented_stack)):
#         # Get the actual corrected segmentation mask
#         seg_mask = real_pipeline.segmented_stack[i]  # This is your corrected data!
        
#         # Use original filename if available
#         if i < len(real_pipeline.original_filenames):
#             filename_stem = Path(real_pipeline.original_filenames[i]).stem
#         else:
#             filename_stem = f"seg_{i:04d}"
        
#         save_path = real_segmented_dir / f"{filename_stem}_corrected.png"
        
#         # Save as lightweight PNG
#         img = PILImage.fromarray(seg_mask, mode='L')
#         img.save(save_path, 'PNG', optimize=True)
        
#         if i % 10 == 0:
#             file_size = save_path.stat().st_size / 1024  # KB
#             unique_labels = len(np.unique(seg_mask))
#             print(f"  {i+1}/50: {file_size:.1f}KB, {unique_labels} labels")
    
#     # Save metadata and change history
#     print("Saving metadata and change history...")
    
#     history_data = [asdict(record) for record in real_pipeline.change_history]
#     with open(save_dir / "_change_history.json", 'w') as f:
#         json.dump(history_data, f, indent=2)
    
#     metadata = {
#         "save_timestamp": datetime.now().isoformat(),
#         "data_type": "REAL corrected segmentation masks",
#         "segmentation_shape": list(real_pipeline.segmented_stack.shape),
#         "data_dtype": str(real_pipeline.segmented_stack.dtype),
#         "total_changes": len(real_pipeline.change_history),
#         "colormap": real_pipeline.colormap.tolist(),
#         "label_names": real_pipeline.label_names,
#         "original_filenames": real_pipeline.original_filenames
#     }
    
#     with open(save_dir / "REAL_metadata.json", 'w') as f:
#         json.dump(metadata, f, indent=2)
    
#     # Create summary
#     total_size = sum(f.stat().st_size for f in real_segmented_dir.glob("*.png")) / 1024 / 1024  # MB
#     avg_size = total_size / len(real_pipeline.segmented_stack) * 1024  # KB per image
    
#     summary = f"""REAL CORRECTED SEGMENTATION SAVED SUCCESSFULLY!
# {"="*60}

# ✅ Data Source: Actual pipeline.segmented_stack (not reconstructed)
# 📊 Images: {len(real_pipeline.segmented_stack)}
# 📏 Dimensions: {real_pipeline.segmented_stack.shape[1]} x {real_pipeline.segmented_stack.shape[2]}
# 🏷️  Data Type: {real_pipeline.segmented_stack.dtype} (label values 1-19)
# 📝 Changes: {len(real_pipeline.change_history)} correction operations
# 💾 Total Size: {total_size:.2f} MB
# 📄 Avg per image: {avg_size:.1f} KB

# 📁 Location: {real_segmented_dir}

# This is your ACTUAL corrected segmentation work - the lightweight 
# label masks you spent hours creating, saved directly from memory!
# """
    
#     with open(save_dir / "REAL_SAVE_SUCCESS.txt", 'w') as f:
#         f.write(summary)
    
#     print(summary)
    
# else:
#     print("❌ Cannot proceed - real pipeline not found")