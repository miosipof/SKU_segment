
import json
import gc
import os
import yaml
from sku_segment.preprocessing.maskgeneration import MaskGenerator
from sku_segment.training.train import TrainYOLO


mask_generator = MaskGenerator(dataset_name='SKU110K_fixed', device='cpu')


"""
Step 1: Load dataset metadata
"""

ds = mask_generator.load_dataset_meta(dataset_meta='sku110_dataset_min.json')
for k,v in ds.items():
    print(f"{k} split: {len(v)} samples")


"""
Step 2: Model initialization
"""

mask_generator.set_sam2(sam2_checkpoint_name='sam2.1_hiera_large.pt', sam2_config_name='sam2.1_hiera_l.yaml')
mask_generator.set_dino(grounding_config_name='GroundingDINO_SwinT_OGC.py', grounding_checkpoint_name='groundingdino_swint_ogc.pth')

"""
Step 3: [Batched, loop over dataset] Prompt Grounding DINO and SAM image predictor to get the box and mask
"""

mask_generator.preprocess_dataset(ds, split="train", slice_len=11, batch_size=10) #len(ds[split])
 

"""
Step 4: Prepare dataset for YOLO format: convert masks to polygons and write labels
"""

processor = TrainYOLO(dataset_dir=mask_generator.dataset_dir, input_dir=mask_generator.output_dir)

# load batched samples from preprocessed dataset with masks
preprocessed_ds = processor.load_batched_samples(max_batches=1)


# # Optional: select best samples with minimum box-over-box overlap
# top_samples = processor.rank_samples_by_overlap(preprocessed_ds, method='overlap')
# with open(os.path.join(mask_generator.output_dir, "ranked_ds.json"), "w") as f:
#     json.dump(top_samples, f)
# with open(os.path.join(mask_generator.output_dir, "ranked_ds.json"), "r") as f:
#     preprocessed_ds = json.load(f)  


# # Optional: check masks overlap
# for i, sample in enumerate(preprocessed_ds):
#     overlap_pixels = processor.check_mask_overlap(sample['mask_paths'])
#     is_void = [processor.is_void_mask(x) for x in sample['mask_paths']]

#     voids = [x for x in is_void if x == True]
#     print(f"Overlap: {overlap_pixels/len(sample['mask_paths']):1f} per mask, void: {len(voids)}/{len(is_void)}")
#     if i == 5:
#         break


# Select subset (optional)
ds_len = 10 #len(preprocessed_ds)
train_len = int(0.8*ds_len)
val_len = ds_len - train_len
preprocessed_ds = preprocessed_ds[:ds_len]
print(f"{len(preprocessed_ds)} samples selected")

yolo_ds_path = os.path.join('datasets','yolo_dataset_v2')
yaml_path = processor.prepare_dataset(preprocessed_ds,train_len,output_dir=yolo_ds_path)


# Optonal: get full dataset stats
# import pandas as pd
# subset = "val"
# dataset_stats = trainer.walk_dataset("/SAM/Grounded-SAM-2/yolo_dataset_cleaned", subset=subset)
# df = pd.DataFrame(dataset_stats)
# df.to_csv(f"yolo_dataset_mask_stats_{subset}.csv", index=False)
# print(df.head())



"""
Step 5: Train YOLO with segmentation task
"""
from ultralytics import YOLO

model = YOLO("checkpoints/yolov8n-seg.pt")
model.train(
    data=yaml_path,
    epochs=30,
    imgsz=512,
    batch=2,
    single_cls=True,
    device='cpu',
    workers=1,
    plots=False,
    verbose=True,
    augment=False,
    name="experiment_yolo_seg"
)