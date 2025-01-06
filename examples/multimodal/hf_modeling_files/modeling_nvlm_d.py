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

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, AutoModelForCausalLM, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput
from transformers.utils import logging

from .configuration_nvlm_d import NVLM_D_Config

@dataclass
class NVLM_D_CausalLMOutputWithPast(ModelOutput):
    """
    Output class for NVLM-D causal language model outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*):
            Contains pre-computed hidden-states for faster sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Attention weights.
        image_hidden_states (`torch.FloatTensor`, *optional*):
            Image features after projection.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


class NVLM_D_MultiModalProjector(nn.Module):
    def __init__(self, config: NVLM_D_Config):
        super().__init__()
        projector_input_size = config.vision_config.hidden_size * int(1 / config.downsample_ratio) ** 2
        self.linear_1 = nn.Linear(projector_input_size, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class NVLM_D_PreTrainedModel(PreTrainedModel):
    """Base class for NVLM-D model."""
    config_class = NVLM_D_Config
    base_model_prefix = "nvlm_d"
    supports_gradient_checkpointing = True
    _no_split_modules = ["CLIPVisionModel", "Qwen2DecoderLayer"]

    def _init_weights(self, module):
        std = self.config.text_config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

# TODO: what to do to support flash attention?
class NVLM_D_Model(NVLM_D_PreTrainedModel):
    """NVLM-D model for dynamic high-resolution image understanding."""
    
    def __init__(self, config: NVLM_D_Config):
        super().__init__(config)
        
        self.config = config
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        self.multi_modal_projector = NVLM_D_MultiModalProjector(config)
        
        # Model attributes
        self.patch_size = config.vision_config.patch_size
        self.select_layer = config.select_layer
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        
        image_size = config.force_image_size or config.vision_config.image_size
        self.num_image_tokens = int((image_size // self.patch_size) ** 2 * (config.downsample_ratio ** 2))
        
        # Generation attributes
        # TODO: hard-coded for now
        self.img_context_token_id = 151654
        self.main_input_name = "input_ids"

    def pixel_shuffle(self, x: torch.Tensor, scale_factor: float = 0.5) -> torch.Tensor:
        """Perform pixel shuffling for downsampling."""
        n, w, h, c = x.size()
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                  int(c / (scale_factor * scale_factor)))
        if self.ps_version != 'v1':
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def get_image_features(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        """Extract and project image features."""
        if self.select_layer == -1:
            vit_embeds = self.vision_tower(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True
            ).last_hidden_state
        else:
            vit_embeds = self.vision_tower(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True
            ).hidden_states[self.select_layer]
        
        vit_embeds = vit_embeds[:, 1:, :]
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return self.multi_modal_projector(vit_embeds)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, NVLM_D_CausalLMOutputWithPast]:
        """Forward pass of the model."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)
            
            # Replace image context tokens with image features
            B, N, C = inputs_embeds.shape
            inputs_embeds = inputs_embeds.reshape(B * N, C)
            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            
            try:
                inputs_embeds[selected] = image_features.reshape(-1, C)
            except Exception as e:
                image_features = image_features.reshape(-1, C)
                n_token = selected.sum()
                inputs_embeds[selected] = image_features[:n_token]
            
            inputs_embeds = inputs_embeds.reshape(B, N, C)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0] if not return_dict else outputs.logits

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return NVLM_D_CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if return_dict else outputs[1],
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        pixel_values=None,
        **kwargs
    ):
        """Prepare inputs for text generation."""
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # Only forward pixel_values on the first call
        if past_key_values is None:
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "inputs_embeds": inputs_embeds,
                "pixel_values": pixel_values,
            }
        else:
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "inputs_embeds": inputs_embeds,
            }
        
        return model_inputs

