"""
Dain - Diffusion-based Image Generation Package

A Python implementation of diffusion models for image generation,
with COCO dataset training support.
"""

from .model import DainDiffusionModel
from .train import train_on_coco
from .generate import generate_from_prompt

__version__ = "0.1.0"
__all__ = ['DainDiffusionModel', 'train_on_coco', 'generate_from_prompt']
