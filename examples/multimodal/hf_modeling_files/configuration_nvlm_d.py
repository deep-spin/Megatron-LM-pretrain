# --------------------------------------------------------
# Adapted from https://huggingface.co/nvidia/NVLM-D-72B under MIT License
# --------------------------------------------------------

import copy

from transformers import CONFIG_MAPPING
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class NVLM_D_Config(PretrainedConfig):
    model_type = 'NVLM_D'
    is_composition = True
    auto_map = {
        "AutoModel": "modeling_nvlm_d.NVLM_D_Model",
        "AutoModelForCausalLM": "modeling_nvlm_d.NVLM_D_Model",
        "AutoModelForConditionalGeneration": "modeling_nvlm_d.NVLM_D_Model",
    }

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        use_backbone_lora=0,
        use_llm_lora=0,
        select_layer=-1,
        projector_hidden_act='gelu',
        force_image_size=None,
        downsample_ratio=0.5,
        template="chatml",
        dynamic_image_size=True,
        use_thumbnail=True,
        ps_version='v2',
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        **kwargs
    ):
        super().__init__(**kwargs)

        # if configs are dicts, convert them to config objects
        if isinstance(vision_config, dict):
            vision_config = CONFIG_MAPPING[vision_config['model_type']](**vision_config)
        if isinstance(text_config, dict):
            text_config = CONFIG_MAPPING[text_config['model_type']](**text_config)

        # then use the provided vision and text configs
        self.vision_config = vision_config
        self.text_config = text_config

        # Assign configuration values
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.select_layer = select_layer
        self.projector_hidden_act = projector_hidden_act
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.ps_version = ps_version  # Pixel shuffle version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch

        # Log important parameters
        logger.info(f'vision_select_layer: {self.select_layer}')
        logger.info(f'ps_version: {self.ps_version}')
        logger.info(f'min_dynamic_patch: {self.min_dynamic_patch}')
        logger.info(f'max_dynamic_patch: {self.max_dynamic_patch}')

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Overrides the default `PretrainedConfig.to_dict`.

        Returns:
            Dict[str, Any]: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        output['vision_config'] = self.vision_config.to_dict()
        output['text_config'] = self.text_config.to_dict()
        output['model_type'] = self.model_type
        output['use_backbone_lora'] = self.use_backbone_lora
        output['use_llm_lora'] = self.use_llm_lora
        output['select_layer'] = self.select_layer
        output['force_image_size'] = self.force_image_size
        output['downsample_ratio'] = self.downsample_ratio
        output['template'] = self.template
        output['dynamic_image_size'] = self.dynamic_image_size
        output['use_thumbnail'] = self.use_thumbnail
        output['ps_version'] = self.ps_version
        output['min_dynamic_patch'] = self.min_dynamic_patch
        output['max_dynamic_patch'] = self.max_dynamic_patch

        return output