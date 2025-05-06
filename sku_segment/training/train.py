import json
import os
import cv2
import torchvision.transforms as T
import numpy as np
import supervision as sv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import shutil
from PIL import Image, ImageFile
from skimage import measure
from tqdm import tqdm
import json
import yaml


ImageFile.LOAD_TRUNCATED_IMAGES = True


# Constants
SRC_ROOT = Path("yolo_dataset")
DST_ROOT = Path("yolo_dataset_filtered")
MAX_POLY_POINTS = 10
TOP_K = 50  # Number of best samples to keep




class TrainYOLO:
    def __init__(self, dataset_dir, input_dir):
        self.dataset_dir = dataset_dir
        self.input_dir = input_dir

    def normalize_box(self, box, image_width, image_height):
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


    def mask_to_yolo_poly(self, mask, image_size, n_points=12, min_area=256, use_convex_hull=True, image=None, mask_idx=None, vis_dir=None):
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


    def get_bbox_fast(self, mask_path):
        """Get bounding box from binary mask (fast)."""
        with Image.open(mask_path) as im:
            mask = np.array(im) > 0
        if mask.any():
            y_indices, x_indices = np.where(mask)
            return [int(np.min(x_indices)), int(np.min(y_indices)),
                    int(np.max(x_indices)), int(np.max(y_indices))]
        else:
            return None

    def bbox_iou(self, a, b):
        """Compute intersection area of two bounding boxes."""
        xA = max(a[0], b[0])
        yA = max(a[1], b[1])
        xB = min(a[2], b[2])
        yB = min(a[3], b[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        return interW * interH

    def total_bbox_overlap(self, mask_paths):
        """Sum all pairwise bbox intersections (fast heuristic)."""
        bboxes = [self.get_bbox_fast(p) for p in mask_paths]
        bboxes = [b for b in bboxes if b is not None]
        total_overlap = 0
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                total_overlap += self.bbox_iou(bboxes[i], bboxes[j])
        return total_overlap

    def load_batched_samples(self, max_batches=10):
        preprocessed_ds = []
        i = 0
        for subdir, dirs, files in os.walk(self.input_dir):
            for file in files:
                if i>max_batches:
                    break  

                file_path = subdir + os.sep + file
                
                if file_path.endswith('.json'):
                    with open(file_path, 'r') as f:
                        batch = json.load(f)  
                    preprocessed_ds.extend(batch)
                    i+=1

            if i>max_batches:
                break

        return preprocessed_ds


    def rank_samples_by_overlap(self, samples, method='overlap'):
        ranked = []
        for sample in tqdm(samples, desc="Ranking by mask overlap"):
            masks = sample.get("mask_paths", [])
            if masks is None:
                continue
            if len(masks) < 2:
                continue

            overlap = self.total_bbox_overlap(masks)
            print(f"Sample {sample['idx']}: {len(masks)} masks, overlap {overlap}")
            
            ranked.append((overlap, sample))

        # Ranking logic
        if method == 'overlap':
            ranked.sort(key=lambda x: x[0])  # Prefer less overlap
        elif method == 'masks':
            ranked.sort(key=lambda x: len(x[1]["mask_paths"]))  # Prefer fewer masks
        else:
            ranked.sort(key=lambda x: (len(x[1]["mask_paths"]), x[0]))  # Prefer fewer masks + less overlap


        return [s for _, s in ranked]

    def make_masks_exclusive(self, masks, return_void_count=False):
        """
        Accepts a list of binary masks (H, W) and returns a list of mutually exclusive masks.
        Optionally returns how many masks were empty or became empty (void).
        """
        if not masks:
            return ([], 0) if return_void_count else []

        exclusive_masks = []
        current_union = np.zeros_like(masks[0], dtype=bool)
        void_count = 0

        for mask in masks:
            mask = mask.astype(bool)
            clean_mask = np.logical_and(mask, ~current_union)

            if np.count_nonzero(clean_mask) == 0:
                void_count += 1
                continue  # skip void mask

            exclusive_masks.append(clean_mask)
            current_union = np.logical_or(current_union, clean_mask)

        return (exclusive_masks, void_count) if return_void_count else exclusive_masks

    def export_yolov8_seg_format(self, sample, out_root="yolo_dataset", subset="train", vis=None):
        img_path = sample["image_path"].replace('/SKU110K_fixed/', 'images/')
        img_path = os.path.join(self.dataset_dir, img_path)
        img_id = str(sample["idx"])
        mask_paths = sample.get("mask_paths", [])

        if not mask_paths:
            print(f"Skipping {img_id}: No mask paths")
            return

        # Load image and masks
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        masks = [np.array(Image.open(os.path.join(self.input_dir,p)).convert("L")) > 0 for p in mask_paths]

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
                    polys = self.mask_to_yolo_poly(mask, (w, h), n_points=12, image=image, mask_idx=i, vis_dir=vis_dir)

                else:
                    polys = self.mask_to_yolo_poly(mask, (w, h), n_points=12)


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

    def check_mask_overlap(self, mask_paths):
        combined = np.zeros(cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE).shape, dtype=np.uint16)
        for mask_path in mask_paths:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8)
            combined += mask

        # Overlap occurs where sum > 1
        overlap_pixels = np.sum(combined > 1)
        return overlap_pixels
        
    def is_void_mask(self, mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return np.sum(mask > 0) == 0


    def prepare_dataset(self,ds,train_len,output_dir):

        for i, sample in enumerate(ds[:train_len]):
            vis = True if i%20 == 0 else None
            self.export_yolov8_seg_format(sample, out_root=output_dir, subset="train", vis=vis)
            print(f"{i+1}/{len(ds[:train_len])} preprocessed for Train split")

        for i, sample in enumerate(ds[train_len:]):
            vis = True if i%20 == 0 else None
            self.export_yolov8_seg_format(sample, out_root=output_dir, subset="val", vis=vis)
            print(f"{i+1}/{len(ds[train_len:])}  preprocessed for Val split")

        yaml_data = {
        "path": output_dir,
        "train": "images/train",
        "val": "images/val",
        "names": {0: "product"}
        }

        yaml_path = os.path.join(output_dir,'data.yml')
        with open(yaml_path, 'w') as yaml_file:
            yaml.dump(yaml_data, yaml_file, default_flow_style=False)

        print(f"YOLOv8-seg export complete. YAML saved to {yaml_path}")

        return yaml_path



    def analyze_masks(self, mask_dir):
        mask_paths = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')]
        if not mask_paths:
            return 0, 0, 0

        masks = [np.array(Image.open(p).convert("L")) > 0 for p in mask_paths]
        stack = np.stack(masks)

        total_masks = len(masks)
        total_pixels = np.sum(stack)
        overlap_pixels = np.sum(np.sum(stack, axis=0) > 1)

        return total_masks, total_pixels, overlap_pixels

    def walk_dataset(self, dataset_root="yolo_dataset", subset="train"):
        mask_root = os.path.join(dataset_root, "masks", subset)
        stats = []

        if not os.path.isdir(mask_root):
            raise FileNotFoundError(f"No mask directory found: {mask_root}")

        sample_ids = os.listdir(mask_root)
        for sample_id in tqdm(sample_ids, desc="Analyzing masks"):
            sample_dir = os.path.join(mask_root, sample_id)
            if not os.path.isdir(sample_dir):
                continue

            num_masks, nonzero_pixels, overlap = self.analyze_masks(sample_dir)
            stats.append({
                "id": sample_id,
                "num_masks": num_masks,
                "nonzero_pixels": int(nonzero_pixels),
                "overlap_pixels": int(overlap)
            })

        return stats
