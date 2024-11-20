import argparse
import os
import torch

from megatron.core import parallel_state
from megatron.core import dist_checkpointing
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from transformers import (
    AutoConfig,
    LlavaForConditionalGeneration,
    AutoTokenizer,
    CLIPVisionConfig,
    AddedToken,
    AutoImageProcessor,
    LlavaProcessor,
)
from transformers import LlavaConfig

from model import model_provider



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcore-load-dir", required=True)
    parser.add_argument("--hf-save-dir", required=True)
    parser.add_argument("--original-text-model-id", required=True)
    parser.add_argument("--original-vision-model-id", required=True)
    parser.add_argument("--target-params-dtype", type=str, default="float16")
    return parser.parse_args()


def main():
    initialize_distributed(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    args = parse_args()
    convert_mcore2hf(args)


def initialize_distributed(tensor_model_parallel_size=1, pipeline_model_parallel_size=1):
    parallel_state.destroy_model_parallel()

    rank = int(os.environ['LOCAL_RANK'])
    world_size = torch.cuda.device_count()
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(world_size=world_size, rank=rank)
    # Megatron core distributed training initialization
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
    )

def convert_mcore2hf(args):
    """Main function to convert MCore checkpoint to HF format"""
    # TODO: add support for casting explicitly to dtype
    dtype = getattr(torch, args.target_params_dtype)

    print(f"> Loading MCore checkpoints")
    assert os.path.exists(f"{args.mcore_load_dir}/latest_checkpointed_iteration.txt")
    assert os.path.isfile(f"{args.mcore_load_dir}/latest_checkpointed_iteration.txt")

    with open(f"{args.mcore_load_dir}/latest_checkpointed_iteration.txt", "r") as f:
        iteration = int(f.read().strip())
    iter_dir = f"{args.mcore_load_dir}/iter_{iteration:07d}"

    # start by loading the args from the checkpoint
    margs = dist_checkpointing.load_common_state_dict(iter_dir)['args']
    print(f"> Loaded args from checkpoint: {margs}")
    args.tensor_model_parallel_size = 1

    # load the model checkpoint itself
    model = model_provider(args=margs)
    sharded_state_dict = model.sharded_state_dict(prefix='')
    checkpoint = dist_checkpointing.load(
        sharded_state_dict=sharded_state_dict, checkpoint_dir=iter_dir
    )

    # import pdb; pdb.set_trace()
    # create the HF config
    hf_config = create_hf_config(args.original_text_model_id, args.original_vision_model_id, margs)

    # create the tokenizer and processor
    processor = create_hf_processor(hf_config, args.original_text_model_id, args.original_vision_model_id)
    processor.save_pretrained(args.hf_save_dir)

    # Convert the state dict
    print(f"> Converting weights from MCore to HF format")
    hf_state_dict = {}

    # Convert vision model weights
    vision_state_dict = convert_mcore2hf_vision_model(checkpoint)

    # Convert language model weights
    language_state_dict = convert_mcore2hf_language_model(checkpoint)

    # Convert projection weights
    projection_state_dict = convert_mcore2hf_vision_projection(checkpoint)

    # Combine all state dicts
    hf_state_dict.update(vision_state_dict)
    hf_state_dict.update(language_state_dict)
    hf_state_dict.update(projection_state_dict)

    # create the HF model
    print(f"> Loading HF model and converted weights")
    hf_model = LlavaForConditionalGeneration(config=hf_config)
    hf_model.load_state_dict(hf_state_dict, strict=True)

    # extend the embeddings
    extend_embeddings(hf_model, hf_config)

    print(f"> Saving HF model to {args.hf_save_dir}")
    hf_model.save_pretrained(args.hf_save_dir)


def create_hf_config(original_text_model_id, original_vision_model_id, margs):
    """Create HF config from Megatron checkpoint"""
    # Extract model args from checkpoint
    assert margs.transformer_impl == "transformer_engine"
    assert margs.position_embedding_type == "rope"
    assert margs.normalization == "RMSNorm"
    assert margs.swiglu
    # assert margs.untie_embeddings_and_output_weights

    # TODO: atm, both vision and language towers
    # assume that there was an initial HF model
    # that we can use to get the config
    # however, in the ideal world, we would directly
    # use the Megatron config to create the HF config
    # but we leave this for later

    # Create CLIP vision config
    # get config for openai/clip-vit-large-patch14-336
    vision_config = CLIPVisionConfig.from_pretrained(original_vision_model_id)

    # Create language model config (using LlamaConfig as base)
    text_config = AutoConfig.from_pretrained(original_text_model_id)

    # Create final LLaVA config combining both
    hf_config = LlavaConfig(
        vision_config=vision_config,
        text_config=text_config,
        # Add any other LLaVA specific configs here
    )
    return hf_config


def create_hf_processor(hf_config, text_model_id, vision_model_id):
    tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    hf_config.image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    hf_config.pad_token_id = tokenizer.pad_token_id

    try:
        from transformers.models.llava.image_processing_llava import LlavaImageProcessor
        image_processor = LlavaImageProcessor(
            do_megatron_pp=True,
        )
    except ImportError:
        print("> WARNING: could not import LlavaImageProcessor, using AutoImageProcessor instead")
        print("> This might lead to performance degradation due to slightly different image pre-processing")
        image_processor = AutoImageProcessor.from_pretrained(vision_model_id)
    
    processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)
    return processor


def convert_mcore2hf_vision_model(mcore_sd):
    """Convert vision model weights from Megatron to HF format"""
    state_dict = {}

    # Vision embedding layers
    state_dict.update(
        {
            "vision_tower.vision_model.embeddings.class_embedding": mcore_sd[
                "vision_model.class_token"
            ].squeeze(),
            "vision_tower.vision_model.embeddings.position_embedding.weight": mcore_sd[
                "vision_model.position_embeddings.weight"
            ],
            "vision_tower.vision_model.embeddings.patch_embedding.weight": mcore_sd[
                "vision_model.conv1.weight"
            ],
            "vision_tower.vision_model.pre_layrnorm.weight": mcore_sd["vision_model.ln_pre.weight"],
            "vision_tower.vision_model.pre_layrnorm.bias": mcore_sd["vision_model.ln_pre.bias"],
        }
    )

    # Vision transformer layers
    # TODO: for some reason, this is not in the args??
    clip_num_layers = 24
    for layer_i in range(clip_num_layers):
        hf_layer_prefix = f"vision_tower.vision_model.encoder.layers.{layer_i}"
        mcore_layer_prefix = f"vision_model.decoder.layers.{layer_i}"

        # Get QKV weights and biases
        qkv_weight = mcore_sd[f"{mcore_layer_prefix}.self_attention.linear_qkv.weight"]
        qkv_bias = mcore_sd[f"{mcore_layer_prefix}.self_attention.linear_qkv.bias"]

        # Split into Q, K, V following CLIP's original ordering
        hidden_size = qkv_weight.shape[1]
        num_heads = 16  # CLIP ViT-L/14 uses 16 heads
        head_dim = hidden_size // num_heads

        # Reshape and split QKV similar to language model approach
        qkv = qkv_weight.reshape(num_heads, 3 * head_dim, -1)

        # Split into Q, K, V components
        q_proj = qkv[:, :head_dim, :]
        k_proj = qkv[:, head_dim : 2 * head_dim, :]
        v_proj = qkv[:, 2 * head_dim :, :]

        # Reshape back to original dimensions
        q_proj = q_proj.reshape(num_heads * head_dim, -1)
        k_proj = k_proj.reshape(num_heads * head_dim, -1)
        v_proj = v_proj.reshape(num_heads * head_dim, -1)

        # Do the same for biases
        qkv_bias = qkv_bias.reshape(num_heads, 3 * head_dim)
        q_bias = qkv_bias[:, :head_dim].reshape(-1)
        k_bias = qkv_bias[:, head_dim : 2 * head_dim].reshape(-1)
        v_bias = qkv_bias[:, 2 * head_dim :].reshape(-1)

        state_dict.update(
            {
                # Attention weights
                f"{hf_layer_prefix}.self_attn.q_proj.weight": q_proj,
                f"{hf_layer_prefix}.self_attn.k_proj.weight": k_proj,
                f"{hf_layer_prefix}.self_attn.v_proj.weight": v_proj,
                f"{hf_layer_prefix}.self_attn.q_proj.bias": q_bias,
                f"{hf_layer_prefix}.self_attn.k_proj.bias": k_bias,
                f"{hf_layer_prefix}.self_attn.v_proj.bias": v_bias,
                # Output projection
                f"{hf_layer_prefix}.self_attn.out_proj.weight": mcore_sd[
                    f"{mcore_layer_prefix}.self_attention.linear_proj.weight"
                ],
                f"{hf_layer_prefix}.self_attn.out_proj.bias": mcore_sd[
                    f"{mcore_layer_prefix}.self_attention.linear_proj.bias"
                ],
                # Layer norms
                f"{hf_layer_prefix}.layer_norm1.weight": mcore_sd[
                    f"{mcore_layer_prefix}.self_attention.linear_qkv.layer_norm_weight"
                ],
                f"{hf_layer_prefix}.layer_norm1.bias": mcore_sd[
                    f"{mcore_layer_prefix}.self_attention.linear_qkv.layer_norm_bias"
                ],
                f"{hf_layer_prefix}.layer_norm2.weight": mcore_sd[
                    f"{mcore_layer_prefix}.mlp.linear_fc1.layer_norm_weight"
                ],
                f"{hf_layer_prefix}.layer_norm2.bias": mcore_sd[
                    f"{mcore_layer_prefix}.mlp.linear_fc1.layer_norm_bias"
                ],
                # MLP weights
                f"{hf_layer_prefix}.mlp.fc1.weight": mcore_sd[
                    f"{mcore_layer_prefix}.mlp.linear_fc1.weight"
                ],
                f"{hf_layer_prefix}.mlp.fc1.bias": mcore_sd[
                    f"{mcore_layer_prefix}.mlp.linear_fc1.bias"
                ],
                f"{hf_layer_prefix}.mlp.fc2.weight": mcore_sd[
                    f"{mcore_layer_prefix}.mlp.linear_fc2.weight"
                ],
                f"{hf_layer_prefix}.mlp.fc2.bias": mcore_sd[
                    f"{mcore_layer_prefix}.mlp.linear_fc2.bias"
                ],
            }
        )

    # NOTE: for some reason, Megatron removes the post_layernorm weights and biases
    # so we need to add them back in for the HF model,
    # ensuring they perform the identity mapping
    state_dict["vision_tower.vision_model.post_layernorm.weight"] = torch.ones(1024)
    state_dict["vision_tower.vision_model.post_layernorm.bias"] = torch.zeros(1024)

    return state_dict


def convert_mcore2hf_vision_model_new(mcore_sd):
    """Convert vision model weights from Megatron to HF format"""
    state_dict = {}

    # Vision embedding layers
    state_dict.update(
        {
            "vision_tower.vision_model.embeddings.class_embedding": mcore_sd[
                "vision_model.class_token"
            ].squeeze(),
            "vision_tower.vision_model.embeddings.position_embedding.weight": mcore_sd[
                "vision_model.position_embeddings.weight"
            ],
            "vision_tower.vision_model.embeddings.patch_embedding.weight": mcore_sd[
                "vision_model.conv1.weight"
            ],
            "vision_tower.vision_model.pre_layrnorm.weight": mcore_sd["vision_model.ln_pre.weight"],
            "vision_tower.vision_model.pre_layrnorm.bias": mcore_sd["vision_model.ln_pre.bias"],
        }
    )

    # Vision transformer layers
    clip_num_layers = 24
    for layer_i in range(clip_num_layers):
        hf_layer_prefix = f"vision_tower.vision_model.encoder.layers.{layer_i}"
        mcore_layer_prefix = f"vision_model.decoder.layers.{layer_i}"

        # Get QKV weights and biases
        qkv_weight = mcore_sd[f"{mcore_layer_prefix}.self_attention.linear_qkv.weight"]
        qkv_bias = mcore_sd[f"{mcore_layer_prefix}.self_attention.linear_qkv.bias"]

        # Calculate dimensions
        hidden_dim = qkv_weight.shape[1]
        num_heads = mcore_sd["args"].num_attention_heads
        head_dim = hidden_dim // num_heads

        # Split QKV weights and biases
        q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)

        # Ensure these are correctly assigned in the state_dict
        state_dict.update(
            {
                f"{hf_layer_prefix}.self_attn.q_proj.weight": q_weight,
                f"{hf_layer_prefix}.self_attn.k_proj.weight": k_weight,
                f"{hf_layer_prefix}.self_attn.v_proj.weight": v_weight,
                f"{hf_layer_prefix}.self_attn.q_proj.bias": q_bias,
                f"{hf_layer_prefix}.self_attn.k_proj.bias": k_bias,
                f"{hf_layer_prefix}.self_attn.v_proj.bias": v_bias,
            }
        )

    # NOTE: for some reason, Megatron removes the post_layernorm weights and biases
    # so we need to add them back in for the HF model,
    # ensuring they perform the identity mapping
    state_dict["vision_tower.vision_model.post_layernorm.weight"] = torch.ones(1024)
    state_dict["vision_tower.vision_model.post_layernorm.bias"] = torch.zeros(1024)
    return state_dict


def convert_mcore2hf_language_model(mcore_sd):
    """Convert language model weights from Megatron to HF format"""
    state_dict = {}

    # Embeddings
    state_dict["language_model.model.embed_tokens.weight"] = mcore_sd[
        "language_model.embedding.word_embeddings.weight"
    ]

    # Final layer norm and output
    state_dict["language_model.model.norm.weight"] = mcore_sd[
        "language_model.decoder.final_layernorm.weight"
    ]
    state_dict["language_model.lm_head.weight"] = mcore_sd["language_model.output_layer.weight"]

    # Transformer layers
    for layer_i in range(mcore_sd["args"].num_layers):
        mcore_prefix = f"language_model.decoder.layers.{layer_i}"
        hf_prefix = f"language_model.model.layers.{layer_i}"

        # Layer norms
        state_dict.update(
            {
                f"{hf_prefix}.input_layernorm.weight": mcore_sd[
                    f"{mcore_prefix}.self_attention.linear_qkv.layer_norm_weight"
                ],
                f"{hf_prefix}.post_attention_layernorm.weight": mcore_sd[
                    f"{mcore_prefix}.mlp.linear_fc1.layer_norm_weight"
                ],
            }
        )

        # Attention weights
        qkv_weight = mcore_sd[f"{mcore_prefix}.self_attention.linear_qkv.weight"]
        # Ensure the shape is divisible by 3

        # load transformer llava and do the same
        # from transformers import LlavaForConditionalGeneration
        # llava_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")

        hidden_size = qkv_weight.shape[1]
        num_kv_heads = mcore_sd["args"].num_query_groups
        num_heads = mcore_sd["args"].num_attention_heads
        num_queries_per_group = num_heads // num_kv_heads
        head_dim = hidden_size // num_heads

        qkv_size, _ = qkv_weight.size()
        expected_qkv_size = num_kv_heads * (num_queries_per_group + 2) * head_dim
        if qkv_size != expected_qkv_size:
            raise ValueError("qkv_size does not match expected size")

        qkv = qkv_weight.reshape(num_kv_heads, (num_queries_per_group + 2) * head_dim, -1)

        # Split qkv into q_proj, k_proj, v_proj
        q_proj = qkv[:, : num_queries_per_group * head_dim, :]
        k_proj = qkv[
            :, num_queries_per_group * head_dim : (num_queries_per_group + 1) * head_dim, :
        ]
        v_proj = qkv[
            :, (num_queries_per_group + 1) * head_dim : (num_queries_per_group + 2) * head_dim, :
        ]

        # Reshape projections to match HuggingFace format
        q_proj = q_proj.reshape(num_kv_heads * num_queries_per_group * head_dim, -1)
        k_proj = k_proj.reshape(num_kv_heads * head_dim, -1)
        v_proj = v_proj.reshape(num_kv_heads * head_dim, -1)

        # import pdb; pdb.set_trace()

        state_dict.update(
            {
                f"{hf_prefix}.self_attn.q_proj.weight": q_proj,
                f"{hf_prefix}.self_attn.k_proj.weight": k_proj,
                f"{hf_prefix}.self_attn.v_proj.weight": v_proj,
                f"{hf_prefix}.self_attn.o_proj.weight": mcore_sd[
                    f"{mcore_prefix}.self_attention.linear_proj.weight"
                ],
            }
        )

        # MLP weights
        # Note: In LLaMA, gate_proj and up_proj together form what was fc1 in the original architecture
        fc1_weight = mcore_sd[f"{mcore_prefix}.mlp.linear_fc1.weight"]
        gate_size = fc1_weight.shape[0] // 2
        state_dict.update(
            {
                f"{hf_prefix}.mlp.gate_proj.weight": fc1_weight[:gate_size],
                f"{hf_prefix}.mlp.up_proj.weight": fc1_weight[gate_size:],
                f"{hf_prefix}.mlp.down_proj.weight": mcore_sd[
                    f"{mcore_prefix}.mlp.linear_fc2.weight"
                ],
            }
        )

    return state_dict


def convert_mcore2hf_vision_projection(mcore_sd):
    """Convert vision projection weights from Megatron to HF format"""
    state_dict = {}

    # Map the weights from Megatron to Hugging Face format
    state_dict["multi_modal_projector.linear_1.weight"] = mcore_sd[
        "vision_projection.encoder.linear_fc1.weight"
    ]
    state_dict["multi_modal_projector.linear_1.bias"] = mcore_sd[
        "vision_projection.encoder.linear_fc1.bias"
    ]
    state_dict["multi_modal_projector.linear_2.weight"] = mcore_sd[
        "vision_projection.encoder.linear_fc2.weight"
    ]
    state_dict["multi_modal_projector.linear_2.bias"] = mcore_sd[
        "vision_projection.encoder.linear_fc2.bias"
    ]

    return state_dict

def extend_embeddings(hf_model, hf_config):
    # Initialize new embeddings for additional tokens
    # We use the average of the pre-expansion embeddings as the mean
    # and a small covariance matrix to ensure the new embeddings are close to the old ones
    # adapted from 
    # https://github.com/huggingface/transformers/blob/bf42c3bd4b088fd9df1086e63d47a8e33048e5e1/src/transformers/models/llava/convert_llava_weights_to_hf.py#L100
    # TODO: it seems this might not be needed anymore in the new versions of HF??
    # double check
    pre_expansion_embeddings = hf_model.language_model.model.embed_tokens.weight.data
    mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    dist = torch.distributions.multivariate_normal.MultivariateNormal(
        mu, covariance_matrix=1e-5 * sigma
    )

    # We add an image token so we resize the model and pad to 64 for performance reasons
    pad_shape = 64
    vocab_size = hf_config.text_config.vocab_size
    hf_model.resize_token_embeddings(vocab_size + 2, pad_shape)
    hf_model.language_model.model.embed_tokens.weight.data[vocab_size:] = torch.stack(
        tuple(
            (
                dist.sample()
                for _ in range(
                    hf_model.language_model.model.embed_tokens.weight.data[vocab_size:].shape[0]
                )
            )
        ),
        dim=0,
    )
    hf_model.language_model.lm_head.weight.data[vocab_size:] = torch.stack(
        tuple(
            (
                dist.sample()
                for _ in range(hf_model.language_model.lm_head.weight.data[vocab_size:].shape[0])
            )
        ),
        dim=0,
    )



if __name__ == "__main__":
    main()
