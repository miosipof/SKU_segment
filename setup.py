from setuptools import setup, find_packages
import os

grounding_sam2_path = os.path.abspath("./sku_segment/models/Grounding-SAM-2")
grounding_dino_path = os.path.abspath("./sku_segment/models/Grounding-SAM-2/grounding_dino")

setup(
    name="sku_segment",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "omegaconf",
        "hydra-core",
        "matplotlib",
        "Pillow",
        "supervision",
        f"sam-2 @ file://{grounding_sam2_path}",
        f"grounding_dino @ file://{grounding_dino_path}",
    ],
)
