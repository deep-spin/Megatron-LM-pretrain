# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image processor class for NVLM-D."""

from typing import Dict, List, Optional, Union, Tuple, Set
import torch
import numpy as np
from PIL import Image

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import ImageInput, valid_images
from transformers.utils import TensorType, logging

logger = logging.get_logger(__name__)

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

class NVLM_D_ImageProcessor(BaseImageProcessor):
    """
    Image processor for NVLM-D. Handles dynamic high-resolution image processing with tile-tagging.
    """

    model_input_names = ["pixel_values"]
    auto_map = {
        "AutoImageProcessor": "image_processing_nvlm_d.NVLM_D_ImageProcessor",
    }

    def __init__(
        self,
        image_size: int = 336,
        max_num: int = 6,
        min_num: int = 1,
        use_thumbnail: bool = True,
        mean: List[float] = CLIP_MEAN,
        std: List[float] = CLIP_STD,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.image_size = image_size
        self.max_num = max_num
        self.min_num = min_num
        self.use_thumbnail = use_thumbnail
        self.mean = mean
        self.std = std   

    def build_transform(self, input_size: int) -> T.Compose:
        """Build the transformation pipeline."""
        # TODO: only CLIP ordering for now
        return T.Compose([
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])

    def find_closest_aspect_ratio(
        self, 
        aspect_ratio: float, 
        target_ratios: Set[Tuple[int, int]], 
        width: int, 
        height: int, 
        image_size: int
    ) -> Tuple[int, int]:
        """Find the closest aspect ratio from target ratios."""
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(
        self, 
        image: Image.Image, 
        min_num: int = None,
        max_num: int = None,
        image_size: int = None,
        use_thumbnail: bool = None
    ) -> List[Image.Image]:
        """Process image into tiles based on aspect ratio."""
        min_num = min_num if min_num is not None else self.min_num
        max_num = max_num if max_num is not None else self.max_num
        image_size = image_size if image_size is not None else self.image_size
        use_thumbnail = use_thumbnail if use_thumbnail is not None else self.use_thumbnail

        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # Calculate target ratios
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) 
            for i in range(1, n + 1) 
            for j in range(1, n + 1) 
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # Find closest aspect ratio
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # Calculate target dimensions
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # Resize and split image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        assert len(processed_images) == blocks
        
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
            
        return processed_images

    def preprocess(
        self,
        images: Union[ImageInput, List[ImageInput]],
        input_size: Optional[int] = None,
        max_num: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        """
        Preprocess an image or batch of images for the NVLM-D model.

        Args:
            images: A single image or list of images
            input_size: Size to resize image patches to
            max_num: Maximum number of patches
            return_tensors: Type of tensors to return ("pt" for PyTorch)

        Returns:
            BatchFeature containing preprocessed pixel values
        """
        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        input_size = input_size if input_size is not None else self.image_size
        max_num = max_num if max_num is not None else self.max_num

        if not isinstance(images, (list, tuple)):
            images = [images]

        transform = self.build_transform(input_size=input_size)
        
        all_pixel_values = []
        for image in images:
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
                
            processed_images = self.dynamic_preprocess(
                image, 
                image_size=input_size,
                max_num=max_num
            )
            pixel_values = [transform(img) for img in processed_images]
            pixel_values = torch.stack(pixel_values)
            all_pixel_values.append(pixel_values)

        if len(all_pixel_values) == 1:
            all_pixel_values = all_pixel_values[0]
        else:
            all_pixel_values = torch.stack(all_pixel_values)

        return BatchFeature(
            data={"pixel_values": all_pixel_values},
            tensor_type=return_tensors
        )