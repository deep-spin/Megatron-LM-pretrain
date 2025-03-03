# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import argparse
import os
import sys

# Add megatron and the multimodal example to the path.
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir)
    )
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from PIL import Image
import torch
import numpy as np
from transformers import AutoModel, AutoProcessor

from examples.multimodal.model import model_provider
from examples.multimodal.image_processing import get_visual_transform
from examples.multimodal.multimodal_args import add_multimodal_extra_args
from megatron.training import get_model, get_tokenizer
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.core.models.multimodal.llava_model import IGNORE_INDEX

def get_position_ids(input_ids, target, pad_token):
    """Build masks and position id for left to right model."""
    seq_length = input_ids.shape[1]

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

    return position_ids

def prepare_inputs():
    """Prepare identical inputs for both models."""
    # Create image input in numpy
    image = Image.fromarray(np.ones((336, 336, 3), dtype=np.uint8))
    text = "<image>\nDescribe this image."
    return image, text

def run_mcore_llava(model_path, image, text):
    """Run mcore vision model."""
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    # Megatron has some mandatory flags.
    # sys.argv = [
    #     "ignore_me.py",
    #     "--micro-batch-size=1",
    #     "--num-layers=2",
    #     "--vision-model-type=internvit",
    #     "--language-model-type=mistral_7b",
    #     "--tokenizer-prompt-format=mistral",
    #     "--tokenizer-type=MultimodalTokenizer",
    #     "--tokenizer-model=mistralai/Mistral-7B-Instruct-v0.3",
    #     "--vocab-size=1024",
    #     "--hidden-size=64",
    #     "--num-attention-heads=8",
    #     "--seq-length=1024",
    #     "--decoder-seq-length=2048",
    #     "--max-position-embeddings=2048",
    #     "--bf16",
    #     "--img-h=448",
    #     "--img-w=448",
    #     "--patch-dim=14",
    #     "--tensor-model-parallel-size=8",
    #     "--use-te",
    #     f"--pretrained-checkpoint={model_path}",
    # ]
    # Megatron has some mandatory flags.
    sys.argv = [
        "ignore_me.py",
        "--micro-batch-size=1",
        "--vision-model-type=clip",
        "--language-model-type=qwen2.5_7b",
        "--tokenizer-prompt-format=qwen2p0",
        "--tokenizer-type=MultimodalTokenizer",
        "--tokenizer-model=Qwen/Qwen2.5-7B-Instruct",
        "--normalization=RMSNorm",
        "--group-query-attention",
        "--num-query-groups=4",
        "--no-masked-softmax-fusion",
        "--use-flash-attn",
        "--untie-embeddings-and-output-weights",
        "--disable-bias-linear",
        "--add-qkv-bias",
        "--position-embedding-type=rope",
        "--rotary-base=1000000",
        "--swiglu",
        "--no-bias-swiglu-fusion",
        "--no-rope-fusion",
        "--attention-dropout=0.0",
        "--hidden-dropout=0.0",
        "--tensor-model-parallel-size=4",
        "--pipeline-model-parallel-size=1",
        "--num-layers=28",
        "--hidden-size=3584",
        "--num-attention-heads=28",
        "--decoder-seq-length=2048",
        "--max-position-embeddings=131072",
        "--ffn-hidden-size=18944",
        "--norm-epsilon=1e-6",
        "--bf16",
        "--seq-length=256",  # Image embeddings sequence length
        "--image-tag-type=nvlm",
        "--use-tiling",
        "--max-num-tiles=6",
        "--use-thumbnail",
        "--use-tile-tags",
        "--pixel-shuffle",
        "--img-h=336",
        "--img-w=336",
        "--patch-dim=14",
        "--use-te",
        "--disable-vision-class-token",
        f"--pretrained-checkpoint={model_path}",
        "--ckpt-format=torch",
    ]

    print("Initializing megatron...")
    initialize_megatron(extra_args_provider=add_multimodal_extra_args)

    def wrapped_model_provider(pre_process, post_process):
        return model_provider(pre_process, post_process, parallel_output=False)

    # Set up model and load checkpoint.

    print("Loading mcore checkpoint...")
    model = get_model(wrapped_model_provider, wrap_with_ddp=False)
    load_checkpoint(model, None, None)

    model = model[0].module
    model.eval()

    # Get tokenizer
    tokenizer = get_tokenizer()
    
    # Use the multimodal tokenizer's tokenize method
    tiles = get_visual_transform(
        image, 336, 336, 
        use_tiling=True, max_num_tiles=6, use_thumbnail=True, 
        augment=False, 
        vision_model_type="clip"
    )
    tile_count = torch.tensor([len(tiles)], dtype=torch.int, device="cuda")
    tiles = torch.stack(tiles).cuda()
    tokens = tokenizer.tokenize(text)
    input_ids = torch.tensor([tokens], dtype=torch.int, device="cuda")
    position_ids = get_position_ids(input_ids, input_ids, tokenizer.pad)
    attention_mask = None  # Using default causal mask

    # Print input shapes and values
    print("\nMCore inputs:")
    print(f"Image tiles shape: {tiles.shape}")
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Position IDs shape: {position_ids.shape}")
    print(f"First few tokens: {input_ids[0][:10].tolist()}")
    print(f"Image mean/std: {tiles.mean():.3f}/{tiles.std():.3f}")

    tiles = tiles.to(torch.bfloat16)
    
    # Forward pass
    print("Mcore forward pass...")
    with torch.no_grad():
        output, loss_mask = model(
            images=tiles,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            num_image_tiles=tile_count,
        )

    # get just last input_ids.shape[1] tokens of output
    # output = output[-input_ids.shape[1]:]

    return output

def run_hf_llava(model_name, image, text):
    """Run HF LLaVA model."""
    print("Loading HF model...")
    #processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    from examples.multimodal.hf_modeling_files.processing_nvlm_d import NVLM_D_Processor
    from transformers import AutoTokenizer, AutoImageProcessor
    processor = NVLM_D_Processor(
        tokenizer=AutoTokenizer.from_pretrained(model_name, trust_remote_code=True),
        image_processor=AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    )
    model = (
        AutoModel.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        .cuda()
        .eval()
    )

    # Process inputs using the same image tensor
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # DEBUG: detokenize inputs
    tokens = processor.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    print(f"Detokenized tokens: {tokens}")

    # Print input shapes and values
    print("\nHF inputs:")
    print(f"Image tiles shape: {inputs['pixel_values'].shape}")
    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    print(f"Position IDs shape: {inputs['attention_mask'].shape}")
    print(f"First few tokens: {inputs['input_ids'][0][:10].tolist()}")
    print(f"Image mean/std: {inputs['pixel_values'].mean():.3f}/{inputs['pixel_values'].std():.3f}")

    print("HF forward pass...")
    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.logits


def main(mcore_model, hf_model):
    """Compare LLaVA model outputs between mcore and HF given the same fixed input."""

    # Generate common inputs
    images, text = prepare_inputs()
    #print(f"images shape: {images.shape}, text: {text}")
    
    mcore = run_mcore_llava(mcore_model, images, text)

    if torch.distributed.get_rank() == 0:
        hf = run_hf_llava(hf_model, images, text)

        # log shape of mcore and hf
        print(f"mcore shape: {mcore.shape}, hf shape: {hf.shape}")
        
        # Compare logits. Due to different attention implementations and other details,
        # there will be numerical differences.
        diff = (mcore - hf).abs()
        mean_diff = diff.mean().item()
        max_diff = diff.max().item()
        print(f"mean diff {mean_diff}, max diff {max_diff}")
        assert mean_diff < 0.1, f"mean output difference is greater than expected, {mean_diff}"
        assert max_diff < 50, f"max output difference is greater than expected, {max_diff}"

        print("lgtm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check mcore vision model output vs. HF numerically.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mcore-model", type=str, required=True, help="directory for mcore model weights"
    )
    parser.add_argument("--hf-model", type=str, required=True, help="Model name in HF")

    args = parser.parse_args()

    main(args.mcore_model, args.hf_model)
