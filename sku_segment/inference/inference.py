import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import random

from ultralytics import YOLO

model = YOLO("/SAM/Grounded-SAM-2/runs/segment/experiment_yolo_seg/weights/best.pt")
model.eval()
model.conf = 0.10


# Define your validation images folder
val_images_folder = Path("/SAM/Grounded-SAM-2/yolo_dataset_v3/images/val")

# Output folder to save visualizations
output_folder = Path("yolo_dataset_v3/inference_visualizations")
output_folder.mkdir(exist_ok=True, parents=True)

# Helper function to create random colors
def random_color():
    return [random.randint(0, 255) for _ in range(3)]

# Inference and visualization
for img_path in val_images_folder.glob("*.jpg"):  # or "*.png" depending on your dataset
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img_rgb.shape[:2]

    # Predict
    results = model(img_rgb)
    
    # Handle list results
    if isinstance(results, list):
        results = results[0]
    
    # Get masks
    masks = results.masks.data.cpu().numpy() if results.masks is not None else []
    
    # Get bounding boxes
    boxes = results.boxes.xyxy.cpu().numpy()

    
    # Visualization
    for i, mask in enumerate(masks):
        mask = (mask > 0.5).astype(np.uint8)

        # Resize mask to image size
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    
        color = random_color()
        colored_mask = np.zeros_like(img_rgb)
        for c in range(3):
            colored_mask[:, :, c] = mask * color[c]
        img_rgb = cv2.addWeighted(img_rgb, 1.0, colored_mask, 0.5, 0)
    
    # Draw bounding boxes
    for xyxy in boxes:
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img_rgb, 'item', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Save the image
    save_path = str(output_folder / f"{img_path.stem}_viz.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    print(f"Saved visualization to {save_path}")

print("✅ Inference and visualization complete!")
