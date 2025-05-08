import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, ROOT_DIR)
BASE_DIR = os.path.dirname(ROOT_DIR)
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

import json

import cv2
import torch

import torchvision.transforms as T
import numpy as np
import supervision as sv
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import torchvision

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import gc
# from torchvision.ops import box_iou

from shapely.geometry import box as shapely_box
from shapely.geometry import Polygon

# from grounding_dino.groundingdino.util.inference import load_model, predict, load_image
from sku_segment.grounding_dino.groundingdino.util.inference import load_model, predict, load_image
# from sku_segment.grounding_dino.groundingdino import _C

from sam2.utils.track_utils import sample_points_from_masks
from sam2.utils.video_utils import create_video_from_images
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor



MIN_PIXELS = 128 # Minimum pixels in mask to be considered "good"
IOU_THRESHOLD = 0.2 # Minimum pixels in mask to be considered "good"

GOOD_MASK_RATIO = 0.2 # exclude samples having less than GOOD_MASK_RATIO*len(masks) "good masks"



class MaskGenerator:
    def __init__(self, dataset_name, device='cpu'):
        self.dataset_dir = os.path.join(BASE_DIR, 'datasets', dataset_name)
        self.output_dir = OUTPUT_DIR
        self.device = device

        # Transformation for GroundingDINO
        self.dino_transform = T.Compose([
            T.Resize(800),  # DINO expects 800px short side
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        

    def set_cuda(self):
        if 'cuda' in self.device:
            # Check CUDA 
            print(f"Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})\n")

            # use bfloat16 for the entire notebook
            torch.autocast(device_type=self.device, dtype=torch.bfloat16).__enter__()

            if torch.cuda.get_device_properties(0).major >= 8:
                # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True    
        else:
            print('CUDA is not available or has not been selected')    


    def set_sam2(self, sam2_checkpoint_name='sam2.1_hiera_large.pt', sam2_config_name='sam2.1_hiera_l.yaml'):
        # init sam image predictor and video predictor model

        sam2_config_dir = os.path.join(ROOT_DIR, 'configs')
        checkpoint_dir = os.path.join(BASE_DIR, 'checkpoints')


        sam2_checkpoint = os.path.join(checkpoint_dir,sam2_checkpoint_name)
        sam2_model_cfg = os.path.join(sam2_config_dir,sam2_config_name)

        # self.video_predictor = build_sam2_video_predictor(sam2_config_name, sam2_checkpoint, device=self.device)
        # print(f"Video predictor loaded from config {sam2_model_cfg}, checkpoint {sam2_checkpoint}")

        self.sam2_image_model = build_sam2(sam2_config_name, sam2_checkpoint, device=self.device)
        print(f"SAM2 image model loaded from {sam2_checkpoint}")

        self.image_predictor = SAM2ImagePredictor(self.sam2_image_model, device=self.device)
        print(f"image_predictor loaded from SAM2 image model")

    def set_dino(self, grounding_config_name='GroundingDINO_SwinT_OGC.py', grounding_checkpoint_name='groundingdino_swint_ogc.pth'):
        grounding_config_dir = os.path.join(ROOT_DIR, 'grounding_dino', 'groundingdino', 'config')
        checkpoint_dir = os.path.join(BASE_DIR, 'checkpoints')

        grounding_model_config = os.path.join(grounding_config_dir,grounding_config_name)
        grounding_model_weights = os.path.join(checkpoint_dir,grounding_checkpoint_name)

        self.grounding_model = load_model(grounding_model_config, grounding_model_weights, device=self.device)
        print(f"Grounding model loaded from config {grounding_model_config}, checkpoint {grounding_model_weights}")
        self.grounding_model = self.grounding_model.float()
        print(f"Grounding model converted to float32")
        self.grounding_model = self.grounding_model.to(self.device)
        self.grounding_model.eval()
        print(f"Grounding model sent to {self.device}")


    def load_dataset_meta(self, dataset_meta='sku110_dataset_min.json'):
        with open(os.path.join(self.dataset_dir, dataset_meta)) as f:
            ds_meta = json.load(f)
        return ds_meta


    def detect_boxes(
        self,
        image_pil,
        method="dino",  # Options: "dino" or "retinanet"
        prompt="retail product on shelf, whole item, exclude shelf edges",
        box_threshold=0.05,
        text_threshold=0.3,
    ):
        """
        Detect bounding boxes using Grounding DINO or RetinaNet.
    
        Returns: boxes (list of tensors), confidences, labels
        """
        
        image_np = np.array(image_pil)
        image_tensor = torch.tensor(image_np).permute(2, 0, 1).float() / 255.0
    
        if method == "dino":
            assert self.grounding_model is not None and self.image_predictor is not None, "DINO requires model and image_predictor"

            with torch.cuda.amp.autocast(enabled=False):
                bboxes, confidences, labels = predict(
                    model=self.grounding_model,
                    image=image_tensor,
                    caption=prompt,
                    device=self.device,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold
                )
            print(f"[DINO] {len(bboxes)} boxes detected")

            h, w = image_pil.size 
            new_boxes = []
            for box in bboxes:

                box = box.cpu().numpy()

                x_center, y_center, width, height = box

                # Unnormalize
                x_center *= w
                y_center *= h
                width *= w
                height *= h

                # Convert to x1, y1, x2, y2
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)

                new_boxes.append([x1,y1,x2,y2])
                
            del image_tensor, image_np
            gc.collect()
            torch.cuda.empty_cache()   
            
            return new_boxes, confidences, labels
    
        elif method == "retinanet":
            model = torchvision.models.detection.retinanet_resnet50_fpn(weights='RetinaNet_ResNet50_FPN_Weights.COCO_V1')
            model.to(self.device).eval()
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            input_tensor = transform(image_pil).unsqueeze(0).to(self.device)
    
            with torch.no_grad():
                preds = model(input_tensor)[0]
    
            # Filter by confidence
            threshold = box_threshold
            keep = preds['scores'] >= threshold
            bboxes = preds['boxes'][keep]
            confidences = preds['scores'][keep]
            labels = preds['labels'][keep]

            del image_tensor, image_np
            gc.collect()
            torch.cuda.empty_cache()
            
            print(f"[RetinaNet] {len(bboxes)} boxes detected")
            return bboxes, confidences, labels
    
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'dino' or 'retinanet'.")

    def box2mask(self, sample):
        vis_dir=os.path.join(self.output_dir, 'visualizations')
        masks_output_dir = os.path.join(self.output_dir, 'masks')
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(masks_output_dir, exist_ok=True)

        image_path = sample["image_path"].replace("/SKU110K_fixed/", "images/")
        image_path = os.path.join(self.dataset_dir,image_path)
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        mask_paths, mask_scores = [], []
        
        try:
            # # DINO
            # bboxes, scores, labels = self.detect_boxes(
            #     image_pil,
            #     method="dino",
            #     prompt="retail product, package"
            # )

            # RetinaNet
            raw_bboxes, scores, labels = self.detect_boxes(
                image_pil,
                method="retinanet",
                box_threshold=0.15
            )


            vis_image = image_np.copy()  # Visualization base

            ## split big bboxes recursively:
            # split_bboxes = split_big_boxes(raw_bboxes,
            #                          image_shape=image_pil.size,
            #                          min_area=500, 
            #                          max_depth=1, 
            #                          iou_threshold=0.6, 
            #                          current_depth=0)
            
            # print(f"Created {len(split_bboxes)} bboxes out of {len(bboxes)}")  
            
            # OR filter only "good" bboxes.....
            bboxes = self.filter_bboxes(
                raw_bboxes,
                image_size=image_np.shape[1::-1],  # (W, H)
                max_area_ratio=0.4,
                min_aspect_ratio=0.1,
                max_aspect_ratio=10.0,
                max_iou_with_others=0.8,
                top_k=100
            )

            print(f"\nSelected {len(bboxes)} bboxes out of {len(raw_bboxes)}")  

            self.image_predictor.set_image(image_pil)

            skipped = 0
            for i, gt_box in enumerate(bboxes):
                # Sort box coordinates
                x1, x2 = sorted([gt_box[0], gt_box[2]])
                y1, y2 = sorted([gt_box[1], gt_box[3]])

                box_tensor=torch.stack([x1, y1, x2, y2])

                # print("box_tensor:",box_tensor.shape,box_tensor)
                

                # Predict masks
                sam_masks, scores, _ = self.image_predictor.predict(
                    box=box_tensor,
                    multimask_output=True,
                    return_logits=False,
                )

                # print("sam_masks:",len(sam_masks))

                best_iou = 0
                best_mask = None
                best_score = 0

                for mask, score in zip(sam_masks, scores):
                    iou = self.compute_iou_mask_box(mask, (x1, y1, x2, y2))
                    # print("-- iou:",iou)
                    if iou > best_iou and iou >= IOU_THRESHOLD and np.count_nonzero(mask) >= MIN_PIXELS:
                        best_mask = mask
                        best_iou = iou
                        best_score = score

                if best_mask is not None:
                    mask_path = os.path.join('masks', f"{sample['idx']}_mask_{i}.png")
                    mask_paths.append(mask_path)
                    # print("Creating image png...")
                    Image.fromarray((best_mask * 255).astype(np.uint8)).save(os.path.join(self.output_dir, mask_path))
                    mask_scores.append(float(best_score))

                    # Overlay mask on vis_image
                    mask_color = np.zeros_like(vis_image)
                    mask_color[best_mask > 0] = (0, 255, 0)  # Green
                    # print("Creating visualization...")
                    vis_image = cv2.addWeighted(vis_image, 1.0, mask_color, 0.3, 0)

                # Always draw bbox
                # print("Drawing bbox...")
                x1, y1, x2, y2 = [int(v) for v in box_tensor.cpu().tolist()]
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Red bbox

            # Save visualization if any masks
            if len(mask_paths) > 0:
                os.makedirs(vis_dir, exist_ok=True)
                vis_path = os.path.join(vis_dir, f"{sample['idx']}.jpg")
                # print("Saving visualization...")
                cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

            print(f"{len(mask_paths)} masks created from {len(bboxes)} bboxes")

            # Save masks info
            sample["mask_paths"] = mask_paths if len(mask_paths) > GOOD_MASK_RATIO * len(bboxes) else None
            sample["mask_scores"] = mask_scores if len(mask_paths) > GOOD_MASK_RATIO * len(bboxes) else None

            del image_pil, image_np
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            sample["mask_paths"] = None
            sample["mask_scores"] = None

        return sample




    def draw_gt_boxes(self, image_np, gt_boxes, color=(0, 255, 0), thickness=2):
        image = image_np.copy()
        for box in gt_boxes:
            x1, y1, x2, y2 = map(int, box)
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        return image

    def compute_box_area(self, box):
        """Compute area of a single box."""
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    def compute_iou(self, box1, box2):
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = self.compute_box_area(box1)
        area2 = self.compute_box_area(box2)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def filter_bboxes(self, bboxes, image_size, 
                    max_area_ratio=0.3, 
                    min_aspect_ratio=0.2, 
                    max_aspect_ratio=5.0, 
                    max_iou_with_others=0.8,
                    top_k=None):
        """
        Filters bounding boxes based on size, aspect ratio, and overlap.

        Args:
            bboxes (List of (x1, y1, x2, y2)): Input list of bboxes.
            image_size (tuple): (W, H) of the image.
            max_area_ratio (float): Maximum box area compared to image area.
            min_aspect_ratio (float): Minimum acceptable aspect ratio (w/h).
            max_aspect_ratio (float): Maximum acceptable aspect ratio (w/h).
            max_iou_with_others (float): Maximum allowed IoU between boxes.
            top_k (int): Optional. Keep top-K smallest boxes after filtering.

        Returns:
            List of filtered bboxes.
        """
        w, h = image_size
        image_area = w * h

        filtered = []

        for i, box in enumerate(bboxes):

            x1, x2 = sorted([box[0], box[2]])
            y1, y2 = sorted([box[1], box[3]])

            if x2 <= x1 or y2 <= y1:
                continue  # skip invalid

            box_area = (x2 - x1) * (y2 - y1)
            if box_area / image_area > max_area_ratio:
                continue

                
            if x2 <= x1 or y2 <= y1:
                continue  # skip invalid

            box_area = self.compute_box_area((x1, y1, x2, y2))
            if box_area / image_area > max_area_ratio:
                continue  # too big

            aspect_ratio = (x2 - x1) / (y2 - y1)
            if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                continue  # weird shape

            filtered.append((x1, y1, x2, y2))

        # Step 2: Check overlaps
        final_boxes = []
        for i, box in enumerate(filtered):
            too_much_overlap = False
            for j, other in enumerate(filtered):
                if i != j:
                    if self.compute_iou(box, other) > max_iou_with_others:
                        too_much_overlap = True
                        break
            if not too_much_overlap:
                final_boxes.append(box)

        # Step 3: Optionally select smallest boxes
        if top_k is not None and len(final_boxes) > top_k:
            final_boxes = sorted(final_boxes, key=lambda b: self.compute_box_area(b))[:top_k]

        return final_boxes


    def compute_iou_boxes(self, boxA, boxB):
        """
        Computes IoU (Intersection over Union) between two boxes.
        
        Args:
            boxA, boxB: each box is (x1, y1, x2, y2)
        
        Returns:
            iou: float, IoU score
        """
        xA1, yA1, xA2, yA2 = boxA
        xB1, yB1, xB2, yB2 = boxB

        # Determine coordinates of intersection rectangle
        x_left = max(xA1, xB1)
        y_top = max(yA1, yB1)
        x_right = min(xA2, xB2)
        y_bottom = min(yA2, yB2)

        # Check if there is an intersection
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        # Compute areas
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        areaA = (xA2 - xA1) * (yA2 - yA1)
        areaB = (xB2 - xB1) * (yB2 - yB1)

        # Compute IoU
        iou = intersection_area / float(areaA + areaB - intersection_area)
        return iou


    def intersection_box(self, boxA, boxB):
        """
        Returns the intersection box of two boxes, or None if they don't overlap.

        Args:
            boxA, boxB: each box is (x1, y1, x2, y2)

        Returns:
            (x1, y1, x2, y2) of the intersection, or None
        """
        xA1, yA1, xA2, yA2 = boxA
        xB1, yB1, xB2, yB2 = boxB

        x_left = max(xA1, xB1)
        y_top = max(yA1, yB1)
        x_right = min(xA2, xB2)
        y_bottom = min(yA2, yB2)

        if x_right <= x_left or y_bottom <= y_top:
            return None  # no overlap

        return (x_left, y_top, x_right, y_bottom)


    def compute_iou_mask_box(self, mask, box):
        """
        Computes IoU between binary mask and ground truth box.
        """
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours or len(contours[0]) < 3:
            return 0.0

        try:
            mask_poly = Polygon(contours[0][:, 0, :])
        except Exception as e:
            print(f"Error creating Polygon: {e}")
            return 0.0

        box_poly = shapely_box(*box)

        if not mask_poly.is_valid or mask_poly.is_empty:
            return 0.0

        return mask_poly.intersection(box_poly).area / mask_poly.union(box_poly).area



    def visualize(self, idx,image_np,true_bboxes,full_masks):
        output_dir=os.path.join(self.output_dir,"visualizations")
        # 1. Save GT boxes visualization
        gt_vis = self.draw_gt_boxes(image_np, true_bboxes)
        Image.fromarray(gt_vis).save(os.path.join(output_dir, f"{idx}_gt_boxes.jpg"))
        print(f"Bboxes saved to {os.path.join(output_dir, f'{idx}_gt_boxes.jpg')}")
        
        # print("image_np dtype:", image_np.dtype)
        # print("image_np shape:", image_np.shape)
        
        # 2. Save masks and overlay visualization
        stacked = np.stack(full_masks, axis=0)
        # print("stacked shape: ",stacked.shape)
        global_mask = np.max(stacked, axis=0)
        non_zeros = np.sum(global_mask)/np.ma.count(global_mask)
        # print(f"Global mask: shape {global_mask.shape}, non-zeros {100*non_zeros:.2f}%")
        Image.fromarray((global_mask * 255).astype(np.uint8)).save(os.path.join(output_dir, f"{idx}_global_mask.jpg"))
        print(f"Global mask saved to {os.path.join(output_dir, f'{idx}_global_mask.jpg')}")
        
        mask_3c = np.repeat(global_mask[:, :, np.newaxis], 3, axis=2)
        inv_mask_3c = 1 - mask_3c
        image_float = image_np.astype(np.float32)
        masked_image = image_float * mask_3c + image_float * inv_mask_3c * 0.8
        masked_image = np.clip(masked_image, 0, 255).astype(np.uint8)
        # masked_image = image_np * mask_3c
        Image.fromarray(masked_image).save(os.path.join(output_dir, f'{idx}_overlay.jpg'))

        print(f"Overlay saved to {os.path.join(output_dir, f'{idx}_overlay.jpg')}")

    def preprocess_dataset(self, ds, slice_len=None, split="train", batch_size=10):
        if slice_len is None:
            slice_len = len(ds[split])

        processed_batch = []
        for idx in range(0, slice_len):
            sample = ds[split][idx]
            sample['idx'] = idx
            
            sample = self.box2mask(sample)
            processed_batch.append(sample)

            print(f"{idx}/{slice_len} images processed")

            if (idx + 1) % batch_size == 0:
                batch_idx = (idx + 1) // 10
                with open(os.path.join(self.output_dir,f'processed_batch_{batch_idx}.json'), 'w') as fp:
                    json.dump(processed_batch, fp)    
                print(f"Batch {batch_idx} processed")
                
                del processed_batch
                gc.collect()
                if 'cuda' in self.device:
                    torch.cuda.empty_cache() 
                processed_batch = []

            del sample
            gc.collect()
            if 'cuda' in self.device:
                torch.cuda.empty_cache()

                
    def split_big_boxes(self, bboxes, image_shape, min_area=500, max_area_ratio=0.2, min_width=20, min_height=20,
                        max_depth=2, iou_threshold=0.2, current_depth=0):
        
        h_img, w_img = image_shape
        max_area = max_area_ratio * (h_img * w_img)

        if current_depth > max_depth:
            return bboxes

        split_boxes = []
        for box in bboxes:
            x1, y1, x2, y2 = box
            w_box = x2 - x1
            h_box = y2 - y1
            box_area = w_box * h_box

            if box_area < min_area or w_box < min_width or h_box < min_height:
                split_boxes.append(box)
                continue

            if box_area > max_area:
                # SPLIT manually
                if w_box >= h_box:
                    xm = (x1 + x2) // 2
                    split_boxes.append((x1, y1, xm, y2))
                    split_boxes.append((xm, y1, x2, y2))
                else:
                    ym = (y1 + y2) // 2
                    split_boxes.append((x1, y1, x2, ym))
                    split_boxes.append((x1, ym, x2, y2))
                continue

            # Otherwise try splitting by overlap
            overlapping = []
            for other_box in bboxes:
                if other_box == box:
                    continue
                iou = self.compute_iou_boxes(box, other_box)
                if iou > iou_threshold:
                    overlapping.append(other_box)

            if overlapping:
                for overlap_box in overlapping:
                    split_box = self.intersection_box(box, overlap_box)
                    if split_box:
                        split_boxes.append(split_box)
            else:
                split_boxes.append(box)

        if split_boxes != bboxes:
            return self.split_big_boxes(split_boxes, image_shape, min_area, max_area_ratio,
                                    min_width, min_height, max_depth, iou_threshold, current_depth + 1)
        else:
            return split_boxes







