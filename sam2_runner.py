# Force OpenCV to use non-Qt backend to avoid conflicts with napari
import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''  # Disable Qt for OpenCV
os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '0'  # Disable problematic backends

import glob
import re
import numpy as np
import torch
import cv2
import tempfile
import shutil

# Explicitly set OpenCV to use a compatible backend
cv2.namedWindow("test")
cv2.destroyWindow("test")  # Quick test to force backend initialization

from sam2.build_sam import build_sam2_video_predictor
from typing import List, Dict

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION AND PATHS
# ═══════════════════════════════════════════════════════════════════════════════
"""
GLOBAL PURPOSE: Define all file paths and directories needed for the SAM2 segmentation pipeline
TECHNICAL PURPOSE: Centralize path configuration to avoid hardcoding throughout the application
"""

checkpoint = r"C:\Users\Florin\OneDrive - Johns Hopkins\Documents\segmentation_mask_correction_pipeline\sam2_model_weights\sam2.1_hiera_large.pt"
model_cfg  = "configs/sam2.1/sam2.1_hiera_l.yaml"

# ═══════════════════════════════════════════════════════════════════════════════
# QT CONFLICT DETECTION AND HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

def detect_qt_conflicts():
    """
    GLOBAL PURPOSE: Detect potential Qt conflicts with napari and other GUI applications
    TECHNICAL PURPOSE: Checks for existing Qt applications and provides warnings/solutions
    """
    import sys
    
    # Check if QApplication already exists (indicating napari or other Qt app is running)
    try:
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            print("WARNING: Qt application detected (likely napari)")
            print("This may cause conflicts with OpenCV windows")
            print("Recommended: Close napari before running SAM2 pipeline")
            return True
    except ImportError:
        try:
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            if app is not None:
                print("WARNING: Qt application detected (likely napari)")
                print("This may cause conflicts with OpenCV windows") 
                print("Recommended: Close napari before running SAM2 pipeline")
                return True
        except ImportError:
            pass
    
    return False

def setup_opencv_safely():
    """
    GLOBAL PURPOSE: Configure OpenCV to avoid Qt conflicts
    TECHNICAL PURPOSE: Sets environment variables and tests OpenCV window creation
    """
    # Test OpenCV window creation safely
    try:
        # Try to create a small test window to check if OpenCV GUI works
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.namedWindow("opencv_test", cv2.WINDOW_NORMAL)
        cv2.imshow("opencv_test", test_img)
        cv2.waitKey(1)
        cv2.destroyWindow("opencv_test")
        cv2.waitKey(1)  # Ensure cleanup
        print("OpenCV GUI test successful")
        return True
    except Exception as e:
        print(f"OpenCV GUI test failed: {e}")
        print("Continuing without OpenCV GUI support...")
        return False

def try_block(step_name, fn, *args, **kwargs):
    """
    GLOBAL PURPOSE: Provide standardized error handling and logging for critical operations
    TECHNICAL PURPOSE: Wraps function calls with try-catch blocks, providing consistent
                      error reporting and execution status logging to help debug issues
                      in the complex SAM2 pipeline. Enhanced with Qt conflict detection.
    """
    try:
        print(f"[*] {step_name}...")
        res = fn(*args, **kwargs)
        print(f"[+] {step_name} done")
        return res
    except Exception as e:
        error_msg = str(e).lower()
        if 'qt' in error_msg or 'qapplication' in error_msg or 'qwidget' in error_msg:
            print(f"[!] Qt conflict detected in {step_name}: {e}")
            print("    This is likely due to napari/OpenCV Qt backend conflict")
            print("    Try closing napari before running SAM2 pipeline")
        else:
            print(f"[!] Error in {step_name}: {e}")
        raise

def normalize_filenames(directory, padding=4):
    """
    GLOBAL PURPOSE: Ensure consistent filename formatting for frame sequence processing
    TECHNICAL PURPOSE: Converts numeric filenames to zero-padded format (e.g., "1.jpg" -> "0001.jpg")
                      This is critical for SAM2 which expects lexicographically sorted filenames
                      to maintain temporal frame order during video processing
    """
    for p in glob.glob(os.path.join(directory, '*.jpg')):
        # Extract numeric part from filename using regex
        m = re.search(r"(\d+)$", os.path.splitext(os.path.basename(p))[0])
        if m:
            # Create zero-padded filename
            new = os.path.join(directory, f"{m.group(1).zfill(padding)}.jpg")
            # Only rename if new name doesn't exist to avoid conflicts
            if p != new and not os.path.exists(new):
                os.rename(p, new)

# ═══════════════════════════════════════════════════════════════════════════════
# SAM2 MODEL INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
"""
GLOBAL PURPOSE: Initialize the SAM2 model with optimal hardware configuration
TECHNICAL PURPOSE: Detect available compute devices (CUDA/CPU), configure tensor operations
                  for performance, and build the SAM2 video predictor instance
"""

# Detect and configure optimal compute device
device = try_block("device selection", lambda: torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
print(f"Using device: {device}")

# Enable Tensor Float-32 (TF32) operations on modern NVIDIA GPUs for performance
if device.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Build the SAM2 video predictor with specified model configuration and weights
predictor = try_block("build predictor", build_sam2_video_predictor, model_cfg, checkpoint, device)

# ═══════════════════════════════════════════════════════════════════════════════
# COORDINATE SYSTEM MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════
"""
GLOBAL PURPOSE: Handle coordinate transformations between full frame and crop coordinate systems
TECHNICAL PURPOSE: Provide seamless coordinate conversion to allow user interaction in global
                  coordinates while SAM2 processes in crop coordinates
"""

def crop_to_global_coords_batch(point, crop_params):
    """
    GLOBAL PURPOSE: Convert SAM2 processing coordinates back to full frame coordinates
    TECHNICAL PURPOSE: Translates crop-relative coordinates to global frame coordinates
                      by adding crop offset, enabling visualization in original frame context
    """
    x_offset = crop_params['top_left_x']
    y_offset = crop_params['top_left_y']
    return [point[0] + y_offset, point[1] + x_offset]  # Note: keeping (Y, X) format as specified

def crop_mask_to_global_batch(crop_mask, original_shape, crop_params):
    """
    GLOBAL PURPOSE: Map segmentation results back to original frame dimensions
    TECHNICAL PURPOSE: Creates full-size mask array and places crop mask in correct
                      position, enabling proper overlay visualization and final output
                      in original frame coordinates
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

# ═══════════════════════════════════════════════════════════════════════════════
# ROI CONVEX HULL EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_convex_hull_from_mask_batch(mask, frame_idx, crop_params, original_shape):
    """
    GLOBAL PURPOSE: Extract detailed boundary points from segmentation mask for ROI definition
    TECHNICAL PURPOSE: Uses OpenCV contour detection to find a detailed boundary that closely
                      follows the actual mask shape with sufficient points for accurate representation
    """
    if mask is None or not mask.any():
        return None
    
    # Map crop mask to global coordinates first
    global_mask = crop_mask_to_global_batch(mask, original_shape, crop_params)
    
    # Convert boolean mask to uint8 for OpenCV contour detection
    mask_uint8 = (global_mask * 255).astype(np.uint8)
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return None
    
    # Find the largest contour (main object)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Use raw contour points with smart sampling for detail
    contour_points = largest_contour.reshape(-1, 2)
    
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
    
    # Ensure we have at least a reasonable number of points
    if len(sampled_points) < 8:
        # If still too few points, use a finer approximation as backup
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)  # 0.5% of perimeter - very detailed
        approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
        sampled_points = approx_polygon.reshape(-1, 2)
    
    # Convert from (X, Y) to (Y, X) format as required by the specification
    sampled_points_yx = np.column_stack((sampled_points[:, 1], sampled_points[:, 0]))
    
    return sampled_points_yx

# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE REFINEMENT SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# ROI CONVEX HULL EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_convex_hull_from_mask_batch(mask, frame_idx, crop_params, original_shape):
    """
    GLOBAL PURPOSE: Extract detailed boundary points from segmentation mask for ROI definition
    TECHNICAL PURPOSE: Uses OpenCV contour detection to find a detailed boundary that closely
                      follows the actual mask shape with sufficient points for accurate representation
    """
    if mask is None or not mask.any():
        return None
    
    # Map crop mask to global coordinates first
    global_mask = crop_mask_to_global_batch(mask, original_shape, crop_params)
    
    # Convert boolean mask to uint8 for OpenCV contour detection
    mask_uint8 = (global_mask * 255).astype(np.uint8)
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return None
    
    # Find the largest contour (main object)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Use raw contour points with smart sampling for detail
    contour_points = largest_contour.reshape(-1, 2)
    
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
    
    # Ensure we have at least a reasonable number of points
    if len(sampled_points) < 8:
        # If still too few points, use a finer approximation as backup
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)  # 0.5% of perimeter - very detailed
        approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
        sampled_points = approx_polygon.reshape(-1, 2)
    
    # Convert from (X, Y) to (Y, X) format as required by the specification
    sampled_points_yx = np.column_stack((sampled_points[:, 1], sampled_points[:, 0]))
    
    return sampled_points_yx

def annotate_initial_frame_batch(cropped_frames, image_indices):
    """
    GLOBAL PURPOSE: Collect user annotations on the first cropped frame to initialize object tracking
    TECHNICAL PURPOSE: Implements interactive point collection with visual feedback,
                      real-time preview functionality, and storage of annotations in crop coordinate system for SAM2 processing
    """
    frame_idx = 0
    points = []  # Store annotations in crop coordinates
    labels = []  # 1 for positive, 0 for negative clicks
    
    # Create initial display of cropped frame
    crop_frame = cropped_frames[frame_idx]
    display_frame = cv2.cvtColor(crop_frame, cv2.COLOR_RGB2BGR).copy()
    
    win = "Initial Annotation - Cropped Frame 0"
    
    # Try to create window with error handling for Qt conflicts
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 800, 600)
    except Exception as e:
        print(f"Warning: OpenCV window creation issue (Qt conflict): {e}")
        print("Continuing with default window settings...")
    
    actual_image_idx = image_indices[frame_idx]
    print(f"\n=== Initial Object Annotation (Cropped Image {actual_image_idx}) ===")
    print("You are viewing the cropped region only")
    print("Controls:")
    print("  Left click: Add positive point")
    print("  Right click: Add negative point") 
    print("  'r': Reset all points")
    print("  'p': Preview segmentation")
    print("  'q': Finish annotation")
    
    def click_callback(event, x, y, flags, param):
        """
        TECHNICAL PURPOSE: Process mouse clicks and update visual feedback
        """
        nonlocal points, labels, display_frame
        
        # Points are already in crop coordinates since we're displaying cropped regions
        crop_point = [x, y]
        
        if event == cv2.EVENT_LBUTTONDOWN:  # Positive annotation
            points.append(crop_point)
            labels.append(1)
            # Draw green circle for positive point
            cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(display_frame, (x, y), 7, (255, 255, 255), 2)  # White border
            print(f"  Added positive point at crop({x}, {y})")
            
        elif event == cv2.EVENT_RBUTTONDOWN:  # Negative annotation
            points.append(crop_point)
            labels.append(0)
            # Draw red circle for negative point
            cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
            cv2.circle(display_frame, (x, y), 7, (255, 255, 255), 2)  # White border
            print(f"  Added negative point at crop({x}, {y})")
        
        try:
            cv2.imshow(win, display_frame)
        except Exception as e:
            print(f"Warning: Display update issue: {e}")
    
    # Register mouse callback and show initial display
    try:
        cv2.setMouseCallback(win, click_callback)
        cv2.imshow(win, display_frame)
    except Exception as e:
        print(f"Warning: OpenCV display issue: {e}")
        return False, [], []
    
    # Main annotation interaction loop
    done = False
    while not done:
        try:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Finish annotation
                if len(points) > 0:
                    done = True
                else:
                    print("  Please add at least one positive point")
                    
            elif key == ord('r'):  # Reset all points
                points = []
                labels = []
                display_frame = cv2.cvtColor(crop_frame, cv2.COLOR_RGB2BGR).copy()
                cv2.imshow(win, display_frame)
                print("  Points reset")
                
            elif key == ord('p'):  # Preview segmentation
                if len(points) > 0:
                    print("  Preview functionality available after initial propagation")
                    
        except KeyboardInterrupt:
            print("\nUser interrupted annotation")
            break
        except Exception as e:
            print(f"Warning: Key handling issue: {e}")
            break
    
    try:
        cv2.destroyWindow(win)
    except Exception as e:
        print(f"Warning: Window cleanup issue: {e}")
    
    # Return annotation results
    if len(points) > 0:
        print(f"Initial annotation complete: {len(points)} points on image {actual_image_idx} (crop coordinates)")
        return True, points, labels
    #return False, [], []
    """
    GLOBAL PURPOSE: Provide frame-by-frame review and annotation refinement interface for batch processing
    TECHNICAL PURPOSE: Implements trackbar-based frame navigation, real-time visual feedback,
                      coordinate validation, temporary annotation storage, and batch refinement
                      application with immediate re-propagation capability - adapted for cropped regions only
    """
    if len(current_masks) == 0:
        print("No masks to refine!")
        return False, {}
    
    print("\n=== Interactive Refinement (Cropped Regions Only) ===")
    print("Browse through cropped frames and add refinement points where needed")
    print("You are viewing only the cropped regions, not the full images")
    
    # Setup frame navigation
    frame_indices = sorted(current_masks.keys())
    current_frame_idx = 0
    max_frames = len(frame_indices)
    
    # Create main window with frame navigation trackbar
    win = "Frame Browser - Crop Refinement"
    
    # Enhanced error handling for Qt conflicts
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 800, 600)
        cv2.createTrackbar('Frame', win, 0, max_frames - 1, lambda x: None)
    except Exception as e:
        print(f"Warning: OpenCV window/trackbar creation issue (Qt conflict): {e}")
        print("Attempting alternative window creation...")
        try:
            cv2.namedWindow(win)
            cv2.createTrackbar('Frame', win, 0, max_frames - 1, lambda x: None)
        except Exception as e2:
            print(f"Error: Could not create refinement interface: {e2}")
            print("Skipping refinement due to GUI conflicts")
            return False, current_masks
    
    # Refinement annotation state
    temp_points = []  # Temporary annotations in crop coordinates
    temp_labels = []
    frame_prompts = {}  # Store refinement annotations
    refinement_made = False
    obj_id = 2  # SAM2 object identifier
    
    def create_crop_overlay(crop_frame, crop_mask, alpha=0.3):
        """
        TECHNICAL PURPOSE: Create overlay visualization for cropped region only
        """
        overlay = crop_frame.copy()
        
        # Apply red color to mask regions
        if crop_mask is not None and crop_mask.any():
            overlay[crop_mask] = [255, 0, 0]  # Red mask overlay
        
        # Blend original frame with mask overlay
        result = cv2.addWeighted(crop_frame, 1-alpha, overlay, alpha, 0)
        return result
    
    def update_display():
        """
        TECHNICAL PURPOSE: Refresh display with current cropped frame, mask overlay, and annotations
        """
        nonlocal current_frame_idx
        actual_frame_idx = frame_indices[current_frame_idx]
        
        # Get current cropped frame and mask
        crop_frame = cropped_frames[actual_frame_idx]
        crop_mask = current_masks.get(actual_frame_idx, None)
        
        # Create base overlay for cropped region
        overlay = create_crop_overlay(crop_frame, crop_mask)
        display_frame = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        
        # Draw existing refinement annotations (already in crop coordinates)
        if actual_frame_idx in frame_prompts:
            existing_points = frame_prompts[actual_frame_idx]['points']
            existing_labels = frame_prompts[actual_frame_idx]['labels']
            for crop_point, label in zip(existing_points, existing_labels):
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                cv2.circle(display_frame, tuple(map(int, crop_point)), 5, color, -1)
                cv2.circle(display_frame, tuple(map(int, crop_point)), 7, (255, 255, 255), 2)
        
        # Draw temporary annotations (already in crop coordinates)
        for crop_point, label in zip(temp_points, temp_labels):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(display_frame, tuple(map(int, crop_point)), 3, color, -1)
            cv2.circle(display_frame, tuple(map(int, crop_point)), 5, (255, 255, 0), 1)  # Yellow border for temp
        
        # Add informational overlay text
        actual_image_idx = image_indices[actual_frame_idx]
        text = f"Image {actual_image_idx} (Frame {actual_frame_idx}/{len(cropped_frames)-1}) | Points: {len(temp_points)}"
        cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        try:
            cv2.imshow(win, display_frame)
        except Exception as e:
            print(f"Warning: Display update issue (Qt conflict): {e}")
    
    def click_callback(event, x, y, flags, param):
        """
        TECHNICAL PURPOSE: Process mouse clicks for refinement annotation in crop coordinates
        """
        nonlocal temp_points, temp_labels, refinement_made
        
        # Points are already in crop coordinates since we're displaying cropped regions
        crop_point = [x, y]
        
        if event == cv2.EVENT_LBUTTONDOWN:  # Positive refinement
            temp_points.append(crop_point)
            temp_labels.append(1)
            actual_frame_idx = frame_indices[current_frame_idx]
            actual_image_idx = image_indices[actual_frame_idx]
            print(f"  Added positive refinement at crop({x}, {y}) on image {actual_image_idx}")
            update_display()
            
        elif event == cv2.EVENT_RBUTTONDOWN:  # Negative refinement
            temp_points.append(crop_point)
            temp_labels.append(0)
            actual_frame_idx = frame_indices[current_frame_idx]
            actual_image_idx = image_indices[actual_frame_idx]
            print(f"  Added negative refinement at crop({x}, {y}) on image {actual_image_idx}")
            update_display()
    
    # Register mouse callback and display controls
    try:
        cv2.setMouseCallback(win, click_callback)
    except Exception as e:
        print(f"Warning: Mouse callback setup issue: {e}")
        print("Mouse interaction may not work properly")
    
    print("Controls:")
    print("  Trackbar: Navigate frames")
    print("  Left click: Add positive refinement point")
    print("  Right click: Add negative refinement point")
    print("  'a': Apply points to current frame")
    print("  'r': Reset points on current frame")
    print("  'c': Clear all points on current frame")
    print("  'p': Re-propagate with all refinements")
    print("  'q': Finish refinement and return results")
    
    # Force initial display
    update_display()
    
    # Main refinement interaction loop
    done = False
    while not done:
        try:
            key = cv2.waitKey(1) & 0xFF
            
            # Check for frame navigation via trackbar
            try:
                new_pos = cv2.getTrackbarPos('Frame', win)
                if new_pos != current_frame_idx:
                    current_frame_idx = new_pos
                    temp_points = []  # Clear temporary annotations when changing frames
                    temp_labels = []
                    update_display()
            except Exception as e:
                # Trackbar may not work due to Qt conflicts, continue without frame navigation
                pass
            
            if key == ord('q'):  # Finish refinement
                done = True
                
            elif key == ord('a'):  # Apply current temporary annotations
                if len(temp_points) > 0:
                    actual_frame_idx = frame_indices[current_frame_idx]
                    
                    # Merge with existing annotations or create new entry
                    if actual_frame_idx in frame_prompts:
                        frame_prompts[actual_frame_idx]['points'].extend(temp_points)
                        frame_prompts[actual_frame_idx]['labels'].extend(temp_labels)
                    else:
                        frame_prompts[actual_frame_idx] = {
                            'points': temp_points.copy(),
                            'labels': temp_labels.copy()
                        }
                    
                    actual_image_idx = image_indices[actual_frame_idx]
                    print(f"  Applied {len(temp_points)} points to image {actual_image_idx}")
                    temp_points = []
                    temp_labels = []
                    refinement_made = True
                    update_display()
                
            elif key == ord('r'):  # Reset temporary annotations
                temp_points = []
                temp_labels = []
                print("  Reset temporary points")
                update_display()
                
            elif key == ord('c'):  # Clear all annotations on current frame
                actual_frame_idx = frame_indices[current_frame_idx]
                if actual_frame_idx in frame_prompts and actual_frame_idx != 0:  # Protect initial frame
                    del frame_prompts[actual_frame_idx]
                    actual_image_idx = image_indices[actual_frame_idx]
                    print(f"  Cleared all points on image {actual_image_idx}")
                    update_display()
                
            elif key == ord('p'):  # Re-propagate with current refinements
                if refinement_made:
                    print("  Re-propagating with refinements...")
                    success, updated_masks = run_refined_propagation_batch(frame_prompts, state, predictor, obj_id)
                    if success:
                        current_masks.update(updated_masks)
                        print("  Re-propagation complete! Updated masks.")
                        refinement_made = False
                        update_display()
                else:
                    print("  No refinements to apply")
        
        except KeyboardInterrupt:
            print("\nUser interrupted refinement")
            break
        except Exception as e:
            print(f"Warning: Refinement loop issue (likely Qt conflict): {e}")
            # Continue the loop but with reduced functionality
            continue
    
    try:
        cv2.destroyWindow(win)
    except Exception as e:
        print(f"Warning: Window cleanup issue: {e}")
    
    return refinement_made, frame_prompts, current_masks

def run_refined_propagation_batch(frame_prompts, state, predictor, obj_id):
    """
    GLOBAL PURPOSE: Re-execute tracking with accumulated user refinements from multiple frames
    TECHNICAL PURPOSE: Resets SAM2 state, applies all collected annotations across frames,
                      runs propagation with enhanced temporal context, and updates all
                      segmentation results with improved accuracy
    """
    try:
        print("\n=== Running Refined Propagation ===")
        print(f"Frames with prompts: {sorted(frame_prompts.keys())}")
        
        # Reset SAM2 state for clean re-propagation
        predictor.reset_state(state)
        
        # Apply annotations from all annotated frames
        for frame_idx in sorted(frame_prompts.keys()):
            points = np.array(frame_prompts[frame_idx]['points'], dtype=np.float32)
            labels = np.array(frame_prompts[frame_idx]['labels'], dtype=np.int32)
            
            print(f"  Adding {len(points)} points to frame {frame_idx}")
            
            # Add frame-specific annotations to SAM2
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
            )
        
        print(f"Refined propagation setup complete.")
        return True
        
    except Exception as e:
        print(f"Error in refined propagation: {e}")
        return False

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def process_images_with_sam2(images: List[np.ndarray], 
                            image_indices: List[int], 
                            crop_params: Dict[str, int]) -> Dict[int, List[np.ndarray]]:
    """
    Process images with SAM2 to generate segmentation ROIs.
    
    Args:
        images: List of original images (RGB, shape: (H, W, 3), dtype: uint8)
                These are the original unmodified images, not the segmentation masks
        image_indices: List of image indices corresponding to the images
                      e.g., [5, 6, 7] if processing images 5, 6, and 7
        crop_params: Dictionary with cropping parameters:
                    {
                        'top_left_y': int,  # Y coordinate of top-left corner
                        'top_left_x': int,  # X coordinate of top-left corner
                        'height': int,      # Height of the crop region
                        'width': int        # Width of the crop region
                    }
    
    Returns:
        Dictionary mapping image indices to lists of ROI vertex arrays:
        {
            image_idx: [roi1_vertices, roi2_vertices, ...],
            ...
        }
        
        Each roi_vertices should be a numpy array of shape (N, 2) where:
        - N is the number of vertices in the polygon
        - First column is Y coordinates
        - Second column is X coordinates
        - Coordinates should be in the original image space (not cropped space)
    """
    
    print(f"=== SAM2 Batch Processing ===")
    print(f"Processing {len(images)} images with indices: {image_indices}")
    print(f"Crop parameters: {crop_params}")
    
    # Detect potential Qt conflicts before starting
    qt_detected = detect_qt_conflicts()
    opencv_gui_ok = setup_opencv_safely()
    
    if qt_detected and not opencv_gui_ok:
        print("\nERROR: Qt conflict detected and OpenCV GUI failed")
        print("Please close napari and restart the SAM2 pipeline")
        return {}
    
    # Initialize variables for cleanup
    cropped_frames = []  # For SAM2 processing only
    current_masks = {}
    results = {}
    original_shapes = []
    
    # Create temporary directory for cropped frames
    temp_dir = tempfile.mkdtemp(prefix="sam2_batch_")
    
    try:
        # Extract crop parameters
        y1 = crop_params['top_left_y']
        x1 = crop_params['top_left_x']
        y2 = y1 + crop_params['height']
        x2 = x1 + crop_params['width']
        
        # Crop all images and save them with numeric names for SAM2 processing
        for idx, image in enumerate(images):
            # Store original shape for coordinate conversion
            original_shapes.append(image.shape)
            
            # Ensure crop boundaries stay within frame limits
            y1_safe = max(0, y1)
            x1_safe = max(0, x1)
            y2_safe = min(image.shape[0], y2)
            x2_safe = min(image.shape[1], x2)
            
            # Extract crop region
            cropped = image[y1_safe:y2_safe, x1_safe:x2_safe].copy()
            cropped_frames.append(cropped)
            
            # Save cropped frame with numeric-only name for SAM2
            frame_filename = f"{idx:04d}.jpg"
            frame_path = os.path.join(temp_dir, frame_filename)
            cv2.imwrite(frame_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
        
        print(f"Created {len(cropped_frames)} cropped frames for SAM2 processing")
        print(f"Cropped frame size: {cropped_frames[0].shape[:2]}")
        
        # Apply filename normalization to ensure proper frame ordering
        try_block("normalize filenames", normalize_filenames, temp_dir)
        
        # Initialize SAM2 with cropped frame directory
        state = try_block("init state", predictor.init_state, temp_dir)
        predictor.reset_state(state)
        
        # Single object tracking configuration
        obj_id = 2  # SAM2 object identifier
        
        # Phase 1: Initial Object Annotation (user input required)
        print(f"\n=== Phase 1: Initial Annotation ===")
        annotation_success, initial_points, initial_labels = annotate_initial_frame_batch(cropped_frames, image_indices)
        
        if not annotation_success:
            print("Initial annotation failed or cancelled.")
            return {}
        
        # Phase 2: Initial Propagation with user annotations
        print(f"\n=== Phase 2: Initial Propagation ===")
        frame_idx = 0  # Always use first frame for initial annotation
        
        # Convert user annotations to numpy arrays for SAM2
        points_array = np.array(initial_points, dtype=np.float32)
        labels_array = np.array(initial_labels, dtype=np.int32)
        
        print(f"Applying {len(initial_points)} user annotations for initial propagation...")
        
        # Add initial annotations to SAM2
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points_array,
            labels=labels_array,
        )
        
        # Store masks and extract ROIs
        current_masks = {}
        results = {}
        
        # Propagate tracking across entire video sequence with timeout protection
        import threading
        import time
        
        propagation_results = []
        propagation_error = []
        
        def run_propagation():
            try:
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
                    if len(out_mask_logits) > 0:
                        # Extract binary mask from SAM2 output
                        crop_mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze(0)
                        current_masks[out_frame_idx] = crop_mask
                        
                        # Extract ROI coordinates in global coordinate system
                        actual_image_idx = image_indices[out_frame_idx]
                        original_shape = original_shapes[out_frame_idx]
                        
                        roi_vertices = extract_convex_hull_from_mask_batch(
                            crop_mask, out_frame_idx, crop_params, original_shape
                        )
                        
                        if roi_vertices is not None:
                            results[actual_image_idx] = [roi_vertices]
                            print(f"Generated ROI for image {actual_image_idx}: {len(roi_vertices)} vertices")
                        else:
                            results[actual_image_idx] = []
                            print(f"No valid ROI generated for image {actual_image_idx}")
                
                propagation_results.append("SUCCESS")
            
            except Exception as e:
                propagation_error.append(str(e))
        
        # Run propagation in separate thread with timeout
        propagation_thread = threading.Thread(target=run_propagation)
        propagation_thread.daemon = True
        propagation_thread.start()
        
        # Wait for completion or timeout (60 seconds)
        propagation_thread.join(timeout=60)
        
        if propagation_thread.is_alive():
            print("ERROR: SAM2 propagation timed out after 60 seconds!")
            print("This is likely due to Qt conflicts with napari")
            print("Solutions:")
            print("1. Close napari completely before running SAM2")
            print("2. Restart Python kernel")
            print("3. Try running in a separate Python process")
            return {}
        
        if propagation_error:
            error_msg = propagation_error[0]
            if 'qt' in error_msg.lower():
                print(f"Qt conflict during propagation: {error_msg}")
                print("Close napari and try again")
            raise Exception(error_msg)
        print(f"SAM2 propagation complete. Processed {len(current_masks)} frames.")
        print(f"Generated ROI data for {len([k for k, v in results.items() if v])} images.")
        
        # Interactive refinement phase
        print(f"\n=== Phase 3: Review and Refinement ===")
        print(f"Initial segmentation complete!")
        print(f"Processed {len(current_masks)} frames with crop region: {crop_params}")
        
        # User decision point for refinement
        refine = input("\nWould you like to refine the segmentation? (y/n): ").lower().strip()
        
        if refine == 'y':
            # Interactive refinement cycle
            print(f"Starting interactive refinement for cropped regions...")
            refinements_made, updated_masks = interactive_refinement_batch(
                cropped_frames, current_masks, image_indices, state, predictor
            )
            
            if refinements_made:
                # Update current_masks with refined results
                current_masks = updated_masks
                
                # Regenerate ROI data with refined masks
                results = {}
                for out_frame_idx, crop_mask in current_masks.items():
                    actual_image_idx = image_indices[out_frame_idx]
                    original_shape = original_shapes[out_frame_idx]
                    
                    roi_vertices = extract_convex_hull_from_mask_batch(
                        crop_mask, out_frame_idx, crop_params, original_shape
                    )
                    
                    if roi_vertices is not None:
                        results[actual_image_idx] = [roi_vertices]
                        print(f"Refined ROI for image {actual_image_idx}: {len(roi_vertices)} vertices")
                    else:
                        results[actual_image_idx] = []
                        print(f"No valid refined ROI for image {actual_image_idx}")
                
                print(f"\nFinal refined segmentation complete!")
                print(f"Updated ROI data for {len([k for k, v in results.items() if v])} images.")
            else:
                print("No additional refinements were made.")
        
        # Workflow completion summary
        print("\nSegmentation workflow complete!")
        print(f"Final ROI data covers crop region {crop_params} mapped to global coordinates")
        
        return results
        
    except Exception as e:
        print(f"Error in SAM2 processing: {e}")
        raise
    finally:
        # Comprehensive cleanup to prevent Qt conflicts and memory leaks
        
        # Close any remaining OpenCV windows first
        try:
            cv2.destroyAllWindows()
            # Force Qt event processing to complete
            cv2.waitKey(1)
        except Exception as e:
            print(f"Warning: OpenCV window cleanup issue: {e}")
        
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory {temp_dir}: {e}")
        
        # Clear variables from memory
        try:
            if 'cropped_frames' in locals():
                cropped_frames.clear()
            if 'current_masks' in locals():
                current_masks.clear()
            print("Ejected cropped frames and masks from memory")
        except Exception as e:
            print(f"Warning: Memory cleanup issue: {e}")
        
        # Final Qt event loop cleanup
        try:
            cv2.waitKey(1)  # Final event processing
        except:
            pass

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def convert_mask_to_polygon(mask: np.ndarray, offset_y: int = 0, offset_x: int = 0) -> np.ndarray:
    """
    Helper function to convert a binary mask to polygon vertices.
    
    Args:
        mask: Binary mask (2D array)
        offset_y: Y offset to add to coordinates (for converting from crop to full image space)
        offset_x: X offset to add to coordinates
        
    Returns:
        Array of vertices shape (N, 2) in (y, x) format
    """
    if mask is None or not mask.any():
        return None
    
    # Convert boolean mask to uint8 for OpenCV contour detection
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find the largest contour (main object)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Convert contour points to the required format
    contour_points = largest_contour.reshape(-1, 2)
    
    # Convert from (X, Y) to (Y, X) format and add offsets
    vertices = np.column_stack((
        contour_points[:, 1] + offset_y,  # Y coordinates with offset
        contour_points[:, 0] + offset_x   # X coordinates with offset
    ))
    
    return vertices