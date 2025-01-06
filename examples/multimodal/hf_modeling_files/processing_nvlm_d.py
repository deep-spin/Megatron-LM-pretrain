# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""
Processor class for NVLM-D.
"""

from typing import List, Union

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging
from transformers.image_utils import ImageInput

logger = logging.get_logger(__name__)


class NVLM_D_ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {},
    }


class NVLM_D_Processor(ProcessorMixin):
    r"""
    Constructs a NVLM-D processor which wraps a NVLM-D image processor and a tokenizer into a single processor.

    Args:
        image_processor ([`NVLM_D_ImageProcessor`]):
            The image processor for handling dynamic high-resolution image processing.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer for processing text.
        image_token (`str`, *optional*, defaults to "<image>"):
            Special token used to denote image location in text.
        tile_token_format (`str`, *optional*, defaults to "<tile_{}>"):
            Format string for tile position tokens.
        global_token (`str`, *optional*, defaults to "<tile_global_thumbnail>"):
            Token used for the global thumbnail image.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"
    auto_map = {
        "AutoProcessor": "processing_nvlm_d.NVLM_D_Processor",
    }

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        image_token="<image>",
        tile_token_format="<tile_{}>",
        global_token="<tile_global_thumbnail>",
        image_context_token="<|vision_pad|>",
        num_image_tokens=144,
        **kwargs,
    ):
        super().__init__(image_processor, tokenizer)
        self.image_token = image_token
        self.tile_token_format = tile_token_format
        self.global_token = global_token
        self.image_context_token = image_context_token
        self.num_image_tokens = num_image_tokens
    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        **kwargs: Unpack[NVLM_D_ProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare inputs for the NVLM-D model. Processes both images and text.

        Args:
            images: The image or batch of images to be processed.
            text: The text or batch of texts to be processed.

        Returns:
            BatchFeature: A BatchFeature with the following fields:
                - input_ids: Token ids for the text
                - attention_mask: Attention mask for text tokens
                - pixel_values: Processed image patches
                - image_sizes: Original sizes of the images
        """
        if images is None and text is None:
            raise ValueError("You have to specify at least images or text.")

        output_kwargs = self._merge_kwargs(
            NVLM_D_ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # Process images if provided
        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
            num_patches = image_inputs.pixel_values.shape[0]  # Get number of patches including thumbnail
        else:
            image_inputs = {}
            num_patches = 0

        # Process text
        if text is not None:
            if isinstance(text, str):
                text = [text]

            # Replace image tokens with appropriate tile tokens
            processed_texts = []
            for txt in text:
                if self.image_token in txt and num_patches > 0:
                    # Generate tile tokens
                    tile_tokens = []
                    for i in range(1, num_patches):  # Start from 1 as per original code
                        tile_tokens.append(self.tile_token_format.format(i))
                    if num_patches > 1:  # Add global thumbnail token if we have multiple patches
                        tile_tokens.append(self.global_token)
                    
                    # Create image token sequence
                    image_token_sequence = ""
                    for tile_token in tile_tokens:
                        image_token_sequence += tile_token + self.image_context_token * self.num_image_tokens
                    
                    # Replace <image> with the full sequence
                    txt = txt.replace(self.image_token, f"<Image>{image_token_sequence}</Image>")
                
                processed_texts.append(txt)

            text_inputs = self.tokenizer(processed_texts, **output_kwargs["text_kwargs"])
        else:
            text_inputs = {}

        return BatchFeature(data={**text_inputs, **image_inputs})

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to the tokenizer's batch_decode.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to the tokenizer's decode.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        """
        Get the model input names from both tokenizer and image processor.
        """
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))