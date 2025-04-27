import json
import os
import cv2
import torch
import torchvision.transforms as T
import numpy as np
import supervision as sv
# from sam2.build_sam import build_sam2_video_predictor, build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
# from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
# from grounding_dino.groundingdino.groundingdino.util.inference import load_model, predict, load_image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import gc
import shutil

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.version.cuda)

ds_folder = 'output'




def normalize_box(box, image_width, image_height):
    """
    Normalize YOLO bbox to [0,1] range.
    Input box = [x1, y1, x2, y2] in pixels
    Returns: x_center, y_center, width, height in normalized [0,1]
    """
    x1, y1, x2, y2 = box
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    cx = ((x1 + x2) / 2) / image_width
    cy = ((y1 + y2) / 2) / image_height
    w = (x2 - x1) / image_width
    h = (y2 - y1) / image_height
    return cx, cy, w, h



import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from skimage import measure
from tqdm import tqdm

import numpy as np
from PIL import Image

import numpy as np
from PIL import Image
import os

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

# Constants
SRC_ROOT = Path("yolo_dataset")
DST_ROOT = Path("yolo_dataset_filtered")
MAX_POLY_POINTS = 10
TOP_K = 50  # Number of best samples to keep




import cv2
import numpy as np


import cv2

import numpy as np

import matplotlib.pyplot as plt

# def mask_to_yolo_poly(mask, image_size, min_area=256, max_points=20, image=None, mask_idx=None, vis_dir=None):
#     """
#     Converts a binary mask to YOLO polygon format and optionally visualize it.

#     Args:
#         mask (np.array): Binary mask (H, W).
#         image_size (tuple): (W, H) of the image.
#         min_area (int): Minimum area to keep a polygon.
#         max_points (int): Maximum number of points per polygon.
#         visualize_every_n (int): Visualize every Nth mask (0 disables).
#         image (np.array): Original image to plot on (H, W, 3).
#         mask_idx (int): Current mask index (used with visualize_every_n).

#     Returns:
#         List[List[float]]: List of polygons with normalized points.
#     """
#     w, h = image_size

#     # Step 1: Smooth mask
#     kernel = np.ones((5, 5), np.uint8)
#     mask_smooth = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
#     mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_OPEN, kernel)

#     # Step 2: Find contours
#     contours, _ = cv2.findContours(mask_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     yolo_polygons = []

#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area < min_area:
#             continue  # Skip tiny objects

#         # Simplify contour
#         epsilon = 0.01 * cv2.arcLength(cnt, True)  # 1% of contour length
#         approx = cv2.approxPolyDP(cnt, epsilon, True)

#         # Limit number of points
#         if len(approx) > max_points:
#             idx = np.linspace(0, len(approx) - 1, max_points, dtype=int)
#             approx = approx[idx]

#         # Normalize points
#         poly = []
#         for pt in approx:
#             x, y = pt[0]
#             poly.append(x / w)
#             poly.append(y / h)

#         if len(poly) >= 6:  # At least 3 points
#             yolo_polygons.append(poly)

#     # Save visualization
#     vis_dir = os.path.join(out_root, f"vis/{subset}/{img_id}") if vis else None
#     os.makedirs(vis_dir, exist_ok=True) if vis_dir else None
    
#     if image is not None:

#         vis_image = np.array(image)

#         # Add green mask overlay
#         mask_overlay = np.zeros_like(vis_image)
#         mask_overlay[mask_smooth > 0] = (0, 255, 0)
#         vis_image = cv2.addWeighted(vis_image, 0.7, mask_overlay, 0.3, 0)

#         # Draw blue polygon edges
#         for poly in yolo_polygons:
#             points = np.array(poly).reshape(-1, 2) * np.array([w, h])
#             points = points.astype(int)
#             cv2.polylines(vis_image, [points], isClosed=True, color=(255, 0, 0), thickness=2)

#         vis_path = os.path.join(vis_dir, f"mask_{mask_idx:05d}.jpg")
#         cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

#     return yolo_polygons


import numpy as np
import cv2
import os
from PIL import Image

def mask_to_yolo_poly(mask, image_size, n_points=12, min_area=256, use_convex_hull=True, image=None, mask_idx=None, vis_dir=None):
    """
    Converts a binary mask into a YOLO polygon format by connecting N outer points.
    
    Args:
        mask (np.array): Binary mask (H, W).
        image_size (tuple): (W, H) of the image.
        n_points (int): Number of points to sample along the outer contour.
        min_area (int): Minimum area threshold for a valid object.
        use_convex_hull (bool): Whether to use convex hull or raw contour.
        image (np.array): Optional image for visualization.
        mask_idx (int): Mask index for visualization saving.
        vis_dir (str): Directory to save visualizations if needed.

    Returns:
        List[List[float]]: List of polygons with normalized points.
    """

    h, w = mask.shape

    # 1. Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # 2. Take the largest contour
    contour = max(contours, key=cv2.contourArea)

    # 3. Check area
    if cv2.contourArea(contour) < min_area:
        return []

    # 4. Optionally use convex hull
    if use_convex_hull:
        contour = cv2.convexHull(contour)

    # 5. Uniformly sample N points
    indices = np.linspace(0, len(contour) - 1, n_points, dtype=int)
    selected_points = contour[indices][:, 0, :]  # shape: (n_points, 2)

    # 6. Normalize points
    poly = []
    for x, y in selected_points:
        poly.append(x / w)
        poly.append(y / h)

    polygons = [poly]

    # 7. Visualization
    if image is not None and vis_dir is not None and mask_idx is not None:
        os.makedirs(vis_dir, exist_ok=True)

        vis_image = np.array(image)
        mask_overlay = np.zeros_like(vis_image)
        mask_overlay[mask > 0] = (0, 255, 0)  # Green mask
        vis_image = cv2.addWeighted(vis_image, 0.7, mask_overlay, 0.3, 0)

        points = (np.array(poly).reshape(-1, 2) * np.array([w, h])).astype(int)
        cv2.polylines(vis_image, [points], isClosed=True, color=(255, 0, 0), thickness=2)

        vis_path = os.path.join(vis_dir, f"mask_{mask_idx:05d}.jpg")
        cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

    return polygons




def get_bbox_fast(mask_path):
    """Get bounding box from binary mask (fast)."""
    with Image.open(mask_path) as im:
        mask = np.array(im) > 0
    if mask.any():
        y_indices, x_indices = np.where(mask)
        return [int(np.min(x_indices)), int(np.min(y_indices)),
                int(np.max(x_indices)), int(np.max(y_indices))]
    else:
        return None

def bbox_iou(a, b):
    """Compute intersection area of two bounding boxes."""
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    return interW * interH

def total_bbox_overlap(mask_paths):
    """Sum all pairwise bbox intersections (fast heuristic)."""
    bboxes = [get_bbox_fast(p) for p in mask_paths]
    bboxes = [b for b in bboxes if b is not None]
    total_overlap = 0
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            total_overlap += bbox_iou(bboxes[i], bboxes[j])
    return total_overlap

def load_samples(meta_path):
    with open(meta_path) as f:
        return json.load(f)



# def rank_samples_by_overlap(samples):
#     ranked = []
#     for sample in tqdm(samples, desc="Ranking by mask overlap"):
#         masks = sample.get("mask_paths", [])
#         if masks is None:
#             continue
#         if len(masks) < 2:
#             continue

#         overlap = total_bbox_overlap(masks)
#         print(f"Sample {sample['idx']}: {len(masks)} masks, overlap {overlap}")
        
#         ranked.append((overlap, sample))
#     # ranked.sort(key=lambda x: (len(x[1]["mask_paths"]), x[0]))  # Prefer fewer masks + less overlap
#     # ranked.sort(key=lambda x: len(x[1]["mask_paths"]))  # Prefer fewer masks + less overlap
#     ranked.sort(key=lambda x: x[0])  # Prefer less overlap
    
#     return [s for _, s in ranked]





ds = []
for idx in range(1,60):
    file_path = os.path.join(ds_folder, f'processed_batch_{idx}.json')
    with open(file_path, 'r') as f:
        batch = json.load(f)  
    ds.extend(batch)

# print(f"{len(ds)} samples loaded")
# ds = ds[:1000]
# print(f"{len(ds)} samples pre-selected")



# top_samples = rank_samples_by_overlap(ds)
# # Export to filtered dataset or JSON
# with open("output/ranked_ds.json", "w") as f:
#     json.dump(top_samples, f)


# top_samples = top_samples[:100]
# print(f" {len(top_samples)} low-overlap samples.")

# for i in range(10):
#     with open(f'yolo_dataset/labels/val/{i}.txt') as f:
#         lines = f.readlines()
#     print(f"{len(lines)} instances in this image")



# def make_masks_exclusive(masks, return_void_count=False):
#     """
#     Accepts a list of binary masks (H, W) and returns a list of mutually exclusive masks.
#     Optionally returns how many masks were empty or became empty (void).
#     """
#     if not masks:
#         return ([], 0) if return_void_count else []

#     exclusive_masks = []
#     current_union = np.zeros_like(masks[0], dtype=bool)
#     void_count = 0

#     for mask in masks:
#         mask = mask.astype(bool)
#         clean_mask = np.logical_and(mask, ~current_union)

#         if np.count_nonzero(clean_mask) == 0:
#             void_count += 1
#             continue  # skip void mask

#         exclusive_masks.append(clean_mask)
#         current_union = np.logical_or(current_union, clean_mask)

#     return (exclusive_masks, void_count) if return_void_count else exclusive_masks


# def mask_to_smart_polygon(mask, image_size, max_points=10):
#     """
#     Converts a mask into a simplified, denoised polygon that retains core structure.
#     """
#     w, h = image_size
#     mask = mask.astype(np.uint8)
    
#     # Remove small noise
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

#     # Find contours
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     polygons = []
    
#     for cnt in contours:
#         if len(cnt) < 3:
#             continue
#         # Simplify polygon with increasing epsilon until points are within limit
#         epsilon = 0.005 * cv2.arcLength(cnt, True)
#         approx = cv2.approxPolyDP(cnt, epsilon, True)

#         while len(approx) > max_points:
#             epsilon *= 1.5
#             approx = cv2.approxPolyDP(cnt, epsilon, True)

#         poly = [coord / dim for point in approx[:, 0] for coord, dim in zip(point, (w, h))]
#         if len(poly) >= 6:
#             polygons.append(poly)

#     return polygons


import os
import shutil
from PIL import Image

MIN_PIXELS = 128  # minimum number of non-zero pixels to consider valid
MAX_MASKS = 200



def export_yolov8_seg_format(sample, out_root="yolo_dataset", subset="train", vis=None):
    img_path = sample["image_path"].replace('/SKU110K_fixed/', '/SKU110K_fixed/images/')
    img_id = str(sample["idx"])
    mask_paths = sample.get("mask_paths", [])

    if not mask_paths:
        print(f"Skipping {img_id}: No mask paths")
        return

    # Load image and masks
    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    masks = [np.array(Image.open(p).convert("L")) > 0 for p in mask_paths]

    # Output paths
    img_out = os.path.join(out_root, f"images/{subset}/{img_id}.jpg")
    lbl_out = os.path.join(out_root, f"labels/{subset}/{img_id}.txt")
    mask_dir = os.path.join(out_root, f"masks/{subset}/{img_id}")

    os.makedirs(os.path.dirname(img_out), exist_ok=True)
    os.makedirs(os.path.dirname(lbl_out), exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    shutil.copy(img_path, img_out)

    # Save masks and label file
    with open(lbl_out, "w") as f:
        for i, mask in enumerate(masks):
            mask_save_path = os.path.join(mask_dir, f"{i}.png")
            Image.fromarray((mask.astype(np.uint8)) * 255).save(mask_save_path)

            if vis:
                vis_dir = os.path.join(out_root, f"vis/{subset}/{img_id}")
                polys = mask_to_yolo_poly(mask, (w, h), n_points=12, image=image, mask_idx=i, vis_dir=vis_dir)

            else:
                polys = mask_to_yolo_poly(mask, (w, h), n_points=12)


            if polys:
                
                # # Choosing the largest-area poly...       
                # largest_poly = max(polys, key=lambda p: cv2.contourArea((np.array(p).reshape(-1, 2) * np.array([w, h])).astype(np.int32)))
                # if len(largest_poly) >= 6:
                #     f.write("0 " + " ".join(f"{x:.6f}" for x in largest_poly) + "\n")

                # OR merging them into one:
                # Convert all normalized polygons back to pixel coordinates
                pixel_polys = [ (np.array(p).reshape(-1, 2) * np.array([w, h])).astype(np.int32) for p in polys ]
                
                # Combine all contours into one
                all_points = np.vstack(pixel_polys)
                
                # Optional: Convex Hull to get a clean outer boundary
                merged_contour = cv2.convexHull(all_points)
                
                # Normalize back
                merged_poly = merged_contour.reshape(-1, 2) / np.array([w, h])
                merged_poly = merged_poly.flatten().tolist()
                
                # Then write merged_poly instead of largest_poly:
                if len(merged_poly) >= 6:
                    f.write("0 " + " ".join(f"{x:.6f}" for x in merged_poly) + "\n")
            
            




# with open('output/ranked_ds.json', 'r') as f:
#     top_samples = json.load(f)  

top_samples = ds
ds_len = len(top_samples)
train_len = int(0.8*ds_len)
val_len = ds_len - train_len

top_samples = top_samples[:ds_len]
print(f"{len(top_samples)} top_samples selected")

# for i, sample in enumerate(top_samples):
#     overlap_pixels = check_mask_overlap(sample['mask_paths'])
#     is_void = [is_void_mask(x) for x in sample['mask_paths']]

#     voids = [x for x in is_void if x == True]
#     print(f"Overlap: {overlap_pixels/len(sample['mask_paths']):1f} per mask, void: {len(voids)}/{len(is_void)}")
#     if i == 5:
#         break



# def check_mask_overlap(mask_paths):
#     combined = np.zeros(cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE).shape, dtype=np.uint16)
#     for mask_path in mask_paths:
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         mask = (mask > 127).astype(np.uint8)
#         combined += mask

#     # Overlap occurs where sum > 1
#     overlap_pixels = np.sum(combined > 1)
#     return overlap_pixels
    
# def is_void_mask(mask_path):
#     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     return np.sum(mask > 0) == 0

    






for i, sample in enumerate(top_samples[:train_len]):
    vis = True if i%20 == 0 else None
    export_yolov8_seg_format(sample, out_root="yolo_dataset_v3", subset="train", vis=vis)
    print(f"{i+1}/{len(top_samples[:train_len])} preprocessed for Train split")


for i, sample in enumerate(top_samples[train_len:]):
    vis = True if i%20 == 0 else None
    export_yolov8_seg_format(sample, out_root="yolo_dataset_v3", subset="val", vis=vis)
    print(f"{i+1}/{len(top_samples[train_len:])}  preprocessed for Val split")

print("YOLOv8-seg export complete.")


import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# def analyze_masks(mask_dir):
#     mask_paths = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')]
#     if not mask_paths:
#         return 0, 0, 0

#     masks = [np.array(Image.open(p).convert("L")) > 0 for p in mask_paths]
#     stack = np.stack(masks)

#     total_masks = len(masks)
#     total_pixels = np.sum(stack)
#     overlap_pixels = np.sum(np.sum(stack, axis=0) > 1)

#     return total_masks, total_pixels, overlap_pixels

# def walk_dataset(dataset_root="yolo_dataset", subset="train"):
#     mask_root = os.path.join(dataset_root, "masks", subset)
#     stats = []

#     if not os.path.isdir(mask_root):
#         raise FileNotFoundError(f"No mask directory found: {mask_root}")

#     sample_ids = os.listdir(mask_root)
#     for sample_id in tqdm(sample_ids, desc="Analyzing masks"):
#         sample_dir = os.path.join(mask_root, sample_id)
#         if not os.path.isdir(sample_dir):
#             continue

#         num_masks, nonzero_pixels, overlap = analyze_masks(sample_dir)
#         stats.append({
#             "id": sample_id,
#             "num_masks": num_masks,
#             "nonzero_pixels": int(nonzero_pixels),
#             "overlap_pixels": int(overlap)
#         })

#     return stats



# import pandas as pd
# subset = "val"
# dataset_stats = walk_dataset("/SAM/Grounded-SAM-2/yolo_dataset_cleaned", subset=subset)
# df = pd.DataFrame(dataset_stats)
# df.to_csv(f"yolo_dataset_mask_stats_{subset}.csv", index=False)
# print(df.head())





from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")
model.train(
    data="yolo_dataset_v3/data.yaml",
    epochs=300,
    imgsz=512,
    batch=8,
    single_cls=True,
    device=0,
    workers=1,
    plots=False,
    verbose=True,
    augment=False,
    name="experiment_yolo_seg"
)
