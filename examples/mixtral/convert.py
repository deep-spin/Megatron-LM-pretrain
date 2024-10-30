import argparse
import os
import torch

from itertools import product
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, MixtralConfig

def main():
    args = parse_args()
    if args.direction == "hf-to-mcore":
        convert_hf2mcore(args)
    elif args.direction == "mcore-to-hf":
        convert_mcore2hf(args)

@torch.inference_mode()
def convert_hf2mcore(args):
    dtype = getattr(torch, args.target_params_dtype)
    pp_size = args.target_pipeline_parallel_size
    tp_size = args.target_tensor_parallel_size
    ep_size = args.target_expert_parallel_size

    if os.path.exists(args.mcore_save_dir):
        print(f"Directory {args.mcore_save_dir} already exists.")
        exit(1)

    hf_config = AutoConfig.from_pretrained(args.hf_load_dir)
    print(f"> Loaded HF config from {args.hf_load_dir}")
    print(f"> HF config: {hf_config}")

    margs = get_megatron_args(
        hf_config=hf_config,
        save_dir=args.mcore_save_dir,
        pp_size=pp_size,
        tp_size=tp_size,
        ep_size=ep_size,
        dtype=dtype,
    )

    hf_sd = AutoModelForCausalLM.from_pretrained(args.hf_load_dir).state_dict()

    mcore_sds = [
        [[{} for _ in range(tp_size)] for _ in range(ep_size)] for _ in range(pp_size)
    ]

    print("> Converting embeddings")
    convert_hf2mcore_embedding(mcore_sds, hf_sd, tp_size=tp_size, ep_size=ep_size, dtype=dtype)

    print("> Converting final_layernorm and lm_head")
    convert_hf2mcore_final_layernorm_and_lm_head(
        mcore_sds, hf_sd, tp_size=tp_size, ep_size=ep_size, dtype=dtype,
    )

    for layer_idx in range(hf_config.num_hidden_layers):
        print("> Converting layer", layer_idx)
        layer_info = get_layer_info(
            layer_idx=layer_idx, pp_size=pp_size, num_layers=hf_config.num_hidden_layers
        )
        print(f"  > HF layer prefix: {layer_info['hf_layer_prefix']}")
        print(f"  > MCore layer prefix: {layer_info['mcore_layer_prefix']}")

        print("  > Converting attention norm")
        convert_hf2mcore_attn_norm(
            mcore_sds, hf_sd, layer_info, tp_size=tp_size, ep_size=ep_size, dtype=dtype
        )

        print("  > Converting attention qkv")
        dim = hf_config.hidden_size // hf_config.num_attention_heads
        convert_hf2mcore_attn_qkv(
            mcore_sds, hf_sd, layer_info,
            dim=dim, tp_size=tp_size, ep_size=ep_size, dtype=dtype,
        )
        
        print("  > Converting attention output")
        convert_hf2mcore_attn_output(
            mcore_sds, hf_sd, layer_info, tp_size=tp_size, ep_size=ep_size, dtype=dtype
        )

        print("  > Converting mlp norm")
        convert_hf2mcore_mlp_norm(
            mcore_sds, hf_sd, layer_info, tp_size=tp_size, ep_size=ep_size, dtype=dtype
        )

        print("  > Converting mlp router")
        convert_hf2mcore_mlp_router(
            mcore_sds, hf_sd, layer_info, tp_size=tp_size, ep_size=ep_size, dtype=dtype,
        )

        print("  > Converting mlp experts W1 and W3")
        convert_hf2mcore_mlp_experts_w1_w3(
            mcore_sds, hf_sd, layer_info,
            num_experts=hf_config.num_local_experts,
            tp_size=tp_size,
            ep_size=ep_size,
            dtype=dtype,
            moe_grouped_gemm=args.moe_grouped_gemm,
        )
        
        print("  > Converting mlp experts W2")
        convert_hf2mcore_mlp_experts_w2(
            mcore_sds, hf_sd, layer_info,
            num_experts=hf_config.num_local_experts,
            tp_size=tp_size,
            ep_size=ep_size,
            dtype=dtype,
            moe_grouped_gemm=args.moe_grouped_gemm,
        )

    for key in hf_sd:
        print(f"Warning: Unconverted key: {key}")

    print(f"> Saving checkpoints")
    os.makedirs(args.mcore_save_dir, exist_ok=True)
    with open(f"{args.mcore_save_dir}/latest_checkpointed_iteration.txt", "w") as f:
        f.write(f"{margs.iteration}\n")
    
    iter_dir = f"{args.mcore_save_dir}/iter_{margs.iteration:07d}"
    for pp_rank, ep_rank, tp_rank in product(range(pp_size), range(ep_size), range(tp_size)):
        print(f"  > Saving pp_rank={pp_rank}, ep_rank={ep_rank}, tp_rank={tp_rank}")
        sub_dir_name = get_checkpoint_sub_dir_name(
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            pp_size=pp_size,
            ep_rank=ep_rank,
            ep_size=ep_size,
        )
        save_dir = f"{iter_dir}/{sub_dir_name}"
        os.makedirs(save_dir, exist_ok=True)
        mcore_sd = mcore_sds[pp_rank][ep_rank][tp_rank]
        ckpt_sd = {
            "args": margs,
            "checkpoint_version": 3.0,
            "iteration": 1,
            "num_floating_point_operations_so_far": 0,
            "model": mcore_sd,
        }
        save_file = f"{save_dir}/model_optim_rng.pt"
        torch.save(ckpt_sd, save_file)


@torch.inference_mode()
def convert_mcore2hf(args):
    dtype = getattr(torch, args.target_params_dtype)
    pp_size = args.source_pipeline_parallel_size
    tp_size = args.source_tensor_parallel_size
    ep_size = args.source_expert_parallel_size

    if os.path.exists(args.hf_save_dir):
        print(f"Directory {args.hf_save_dir} already exists.")
        exit(1)

    print(f"> Loading MCore checkpoints")
    assert os.path.exists(f"{args.mcore_load_dir}/latest_checkpointed_iteration.txt")
    assert os.path.isfile(f"{args.mcore_load_dir}/latest_checkpointed_iteration.txt")
    with open(f"{args.mcore_load_dir}/latest_checkpointed_iteration.txt", "r") as f:
        iteration = int(f.read().strip())

    mcore_sds = [
        [[{} for _ in range(tp_size)] for _ in range(ep_size)] for _ in range(pp_size)
    ]

    iter_dir = f"{args.mcore_load_dir}/iter_{iteration:07d}"
    mcore_args = []
    for pp_rank, ep_rank, tp_rank in product(range(pp_size), range(ep_size), range(tp_size)):
        print(f"  > Loading pp_rank={pp_rank}, ep_rank={ep_rank}, tp_rank={tp_rank}")
        sub_dir_name = get_checkpoint_sub_dir_name(
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            pp_size=pp_size,
            ep_rank=ep_rank,
            ep_size=ep_size,
        )
        save_dir = f"{iter_dir}/{sub_dir_name}"
        ckpt_sd = torch.load(f"{save_dir}/model_optim_rng.pt")
        assert ckpt_sd["checkpoint_version"] == 3.0
        assert ckpt_sd["iteration"] == iteration
        assert ckpt_sd["num_floating_point_operations_so_far"] == 0
        mcore_args.append(ckpt_sd["args"])
        mcore_sds[pp_rank][ep_rank][tp_rank] = ckpt_sd["model"]

    assert all([mcore_args[0] == mcore_arg for mcore_arg in mcore_args])
    margs = mcore_args[0]
    print(f"> Loaded MCore args: {margs}")

    hf_config = get_hf_config(margs)
    print(f"> HF config: {hf_config}")

    hf_sd = {}

    print("> Converting embeddings")
    convert_mcore2hf_embedding(hf_sd, mcore_sds, tp_size=tp_size, ep_size=ep_size, dtype=dtype)

    print("> Converting final_layernorm and lm_head")
    convert_mcore2hf_final_layernorm_and_lm_head(
        hf_sd, mcore_sds, tp_size=tp_size, ep_size=ep_size, dtype=dtype,
    )

    for layer_idx in range(hf_config.num_hidden_layers):
        print("> Converting layer", layer_idx)
        layer_info = get_layer_info(
            layer_idx=layer_idx, pp_size=pp_size, num_layers=hf_config.num_hidden_layers
        )
        print(f"  > HF layer prefix: {layer_info['hf_layer_prefix']}")
        print(f"  > MCore layer prefix: {layer_info['mcore_layer_prefix']}")

        print("  > Converting attention norm")
        convert_mcore2hf_attn_norm(
            hf_sd, mcore_sds, layer_info, tp_size=tp_size, ep_size=ep_size, dtype=dtype
        )

        print("  > Converting attention qkv")
        dim = hf_config.hidden_size // hf_config.num_attention_heads
        convert_mcore2hf_attn_qkv(
            hf_sd, mcore_sds, layer_info,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            dim=dim,
            tp_size=tp_size,
            ep_size=ep_size,
            dtype=dtype,
        )
        
        print("  > Converting attention output")
        convert_mcore2hf_attn_output(
            hf_sd, mcore_sds, layer_info, tp_size=tp_size, ep_size=ep_size, dtype=dtype
        )

        print("  > Converting mlp norm")
        convert_mcore2hf_mlp_norm(
            hf_sd, mcore_sds, layer_info, tp_size=tp_size, ep_size=ep_size, dtype=dtype
        )

        print("  > Converting mlp router")
        convert_mcore2hf_mlp_router(
            hf_sd, mcore_sds, layer_info, tp_size=tp_size, ep_size=ep_size, dtype=dtype,
        )

        print("  > Converting mlp experts W1 and W3")
        convert_mcore2hf_mlp_experts_w1_w3(
            hf_sd, mcore_sds, layer_info,
            num_experts=hf_config.num_local_experts,
            tp_size=tp_size,
            ep_size=ep_size,
            dtype=dtype,
            moe_grouped_gemm=args.moe_grouped_gemm,
        )
        
        print("  > Converting mlp experts W2")
        convert_mcore2hf_mlp_experts_w2(
            hf_sd, mcore_sds, layer_info,
            num_experts=hf_config.num_local_experts,
            tp_size=tp_size,
            ep_size=ep_size,
            dtype=dtype,
            moe_grouped_gemm=args.moe_grouped_gemm,
        )

    for pp_rank, ep_rank, tp_rank in product(range(pp_size), range(ep_size), range(tp_size)):
        for key in mcore_sds[pp_rank][ep_rank][tp_rank]:
            print(f"Warning: Unconverted key: {key} at pp_rank={pp_rank}, ep_rank={ep_rank}, tp_rank={tp_rank}")

    print(f"> Creating HF model")
    os.makedirs(args.hf_save_dir, exist_ok=True)
    with init_empty_weights():
        hf_model = AutoModelForCausalLM.from_config(hf_config)
    hf_model.load_state_dict(hf_sd, assign=True)
    
    print(f"> Saving HF model")
    hf_model.save_pretrained(args.hf_save_dir)

    if args.hf_tokenizer_path is not None:
        print(f"> Saving HF tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer_path)
        tokenizer.save_pretrained(args.hf_save_dir)


def get_megatron_args(*, hf_config, save_dir, pp_size, tp_size, ep_size, dtype):
    """Overwrite command line arguments to parse the arguments for Megatron-LM."""
    import sys
    old_args = list(sys.argv)

    dummy_args = [
        "script.py",
        # Transformer engine
        "--transformer-impl", "transformer_engine",
        # Network size
        "--num-layers", str(hf_config.num_hidden_layers),
        "--hidden-size", str(hf_config.hidden_size),
        "--ffn-hidden-size", str(hf_config.intermediate_size),
        "--num-attention-heads", str(hf_config.num_attention_heads),
        "--group-query-attention",
        "--num-query-groups", str(hf_config.num_key_value_heads),
        "--max-position-embeddings", str(hf_config.max_position_embeddings),
        "--position-embedding-type", "rope",
        "--no-position-embedding",
        "--rotary-base", str(int(hf_config.rope_theta)),
        "--normalization", "RMSNorm",
        "--norm-epsilon", str(hf_config.rms_norm_eps),
        "--swiglu",
        "--untie-embeddings-and-output-weights",
        # Training
        "--disable-bias-linear",
        # Checkpoint
        "--save", save_dir,
        "--ckpt-format", "torch",
        "--save-interval", "1", # Dummy value
        "--no-load-optim",
        "--no-load-rng",
        "--no-save-optim",
        "--no-save-rng",
        # Distributed
        "--tensor-model-parallel-size", str(tp_size),
        "--pipeline-model-parallel-size", str(pp_size),
        # Data
        "--vocab-size", str(hf_config.vocab_size),
        "--micro-batch-size", "1", # Dummy value
        "--seq-length", str(hf_config.max_position_embeddings),
        # Mixture of Experts
        "--num-experts", str(hf_config.num_local_experts),
        "--expert-model-parallel-size", str(ep_size),
        "--moe-router-topk", str(hf_config.num_experts_per_tok),
        "--moe-router-load-balancing-type", "aux_loss",
        "--moe-aux-loss-coeff", str(hf_config.router_aux_loss_coef),
        "--moe-grouped-gemm",
    ]
    # Mixed precision
    if dtype == torch.float16:
        dummy_args.append("--fp16")
    elif dtype == torch.bfloat16:
        dummy_args.append("--bf16")
    
    sys.argv = dummy_args

    from megatron.training.arguments import (parse_args, validate_args)
    
    # Overwrite world size as parse args will check it.
    os.environ["WORLD_SIZE"] = str(pp_size * ep_size * tp_size)
    
    margs = parse_args()
    margs.iteration = 1
    validate_args(margs)
    margs.add_position_embeddings = False
    margs.add_bias_linear = False
    margs.disable_bias_linear = True

    sys.argv = old_args
    return margs


def get_hf_config(margs):
    assert margs.transformer_impl == "transformer_engine"
    assert margs.position_embedding_type == "rope"
    assert margs.normalization == "RMSNorm"
    assert margs.swiglu
    assert margs.untie_embeddings_and_output_weights
    assert margs.disable_bias_linear

    hf_config = MixtralConfig(
        num_hidden_layers=margs.num_layers,
        hidden_size=margs.hidden_size,
        intermediate_size=margs.ffn_hidden_size,
        num_attention_heads=margs.num_attention_heads,
        num_key_value_heads=margs.num_query_groups,
        max_position_embeddings=margs.max_position_embeddings,
        rope_theta=margs.rotary_base,
        vocab_size=margs.vocab_size,
        num_local_experts=margs.num_experts,
        num_experts_per_tok=margs.moe_router_topk,
        router_aux_loss_coef=margs.moe_aux_loss_coeff,
    )
    return hf_config


def convert_hf2mcore_embedding(mcore_sds, hf_sd, *, tp_size, ep_size, dtype):
    hf_embeds_name = "model.embed_tokens.weight"
    mcore_embeds_name = "embedding.word_embeddings.weight"

    hf_embeddings = hf_sd.pop(hf_embeds_name).to(dtype)
    embedding_shards = hf_embeddings.chunk(tp_size, dim=0)
    for ep_rank, tp_rank in product(range(ep_size), range(tp_size)):  
        mcore_sds[0][ep_rank][tp_rank][mcore_embeds_name] = embedding_shards[tp_rank]


def convert_mcore2hf_embedding(hf_sd, mcore_sds, *, tp_size, ep_size, dtype):
    mcore_embeds_name = "embedding.word_embeddings.weight"
    hf_embeds_name = "model.embed_tokens.weight"

    embedding_shards = []
    for tp_rank in range(tp_size):
        ep_shards = []
        for ep_rank in range(ep_size):
            ep_shards.append(mcore_sds[0][ep_rank][tp_rank].pop(mcore_embeds_name))
        assert all([ep_shards[0].equal(shard) for shard in ep_shards])
        embedding_shards.append(ep_shards[0])

    hf_sd[hf_embeds_name] = torch.cat(embedding_shards, dim=0).to(dtype)


def convert_hf2mcore_final_layernorm_and_lm_head(
    mcore_sds, hf_sd, *, tp_size, ep_size, dtype
):
    final_layernorm = hf_sd.pop("model.norm.weight").to(dtype)
    lm_head = hf_sd.pop("lm_head.weight").to(dtype)
    lm_head_shards = lm_head.chunk(tp_size, dim=0)
    
    for ep_rank, tp_rank in product(range(ep_size), range(tp_size)):  
        mcore_sds[-1][ep_rank][tp_rank]["decoder.final_layernorm.weight"] = final_layernorm
        mcore_sds[-1][ep_rank][tp_rank]["output_layer.weight"] = lm_head_shards[tp_rank]
        mcore_sds[-1][ep_rank][tp_rank]["output_layer._extra_state"] = None


def convert_mcore2hf_final_layernorm_and_lm_head(hf_sd, mcore_sds, *, tp_size, ep_size, dtype):
    final_layernorms = []
    lm_head_shards = []

    mcore_layernorm_name = "decoder.final_layernorm.weight"
    mcore_output_name = "output_layer.weight"
    mcore_output_extra = "output_layer._extra_state"
    for tp_rank in range(tp_size):
        ep_lm_head_shards = []
        for ep_rank in range(ep_size):
            final_layernorm = mcore_sds[-1][ep_rank][tp_rank].pop(mcore_layernorm_name)
            final_layernorms.append(final_layernorm)
        
            lm_head_shard = mcore_sds[-1][ep_rank][tp_rank].pop(mcore_output_name)
            _ = mcore_sds[-1][ep_rank][tp_rank].pop(mcore_output_extra)
            ep_lm_head_shards.append(lm_head_shard)

        assert all([ep_lm_head_shards[0].equal(shard) for shard in ep_lm_head_shards])
        lm_head_shards.append(ep_lm_head_shards[0])

    assert all([final_layernorms[0].equal(norm) for norm in final_layernorms])
    hf_sd["model.norm.weight"] = final_layernorms[0].to(dtype)

    lm_head = torch.cat(lm_head_shards, dim=0)
    hf_sd["lm_head.weight"] = lm_head.to(dtype)


def convert_hf2mcore_attn_norm(mcore_sds, hf_sd, layer_info, *, tp_size, ep_size, dtype):
    hf_layer_prefix = layer_info["hf_layer_prefix"]
    mcore_layer_prefix = layer_info["mcore_layer_prefix"]
    pp_rank = layer_info["pp_rank"]

    hf_attn_norm_name = f"{hf_layer_prefix}.input_layernorm.weight"
    mcore_attn_norm_name = f"{mcore_layer_prefix}.self_attention.linear_qkv.layer_norm_weight"
    attn_norm = hf_sd.pop(hf_attn_norm_name).to(dtype)
    for ep_rank, tp_rank in product(range(ep_size), range(tp_size)):
        mcore_sds[pp_rank][ep_rank][tp_rank][mcore_attn_norm_name] = attn_norm


def convert_mcore2hf_attn_norm(hf_sd, mcore_sds, layer_info, *, tp_size, ep_size, dtype):
    hf_layer_prefix = layer_info["hf_layer_prefix"]
    mcore_layer_prefix = layer_info["mcore_layer_prefix"]
    pp_rank = layer_info["pp_rank"]

    hf_attn_norm_name = f"{hf_layer_prefix}.input_layernorm.weight"
    mcore_attn_norm_name = f"{mcore_layer_prefix}.self_attention.linear_qkv.layer_norm_weight"

    attn_norms = []
    for ep_rank, tp_rank in product(range(ep_size), range(tp_size)):
        attn_norm = mcore_sds[pp_rank][ep_rank][tp_rank].pop(mcore_attn_norm_name)
        attn_norms.append(attn_norm)
    assert all([attn_norms[0].equal(norm) for norm in attn_norms])
    hf_sd[hf_attn_norm_name] = attn_norm.to(dtype)


def convert_hf2mcore_attn_qkv(
    mcore_sds, hf_sd, layer_info, *, dim, tp_size, ep_size, dtype,
):
    hf_layer_prefix = layer_info["hf_layer_prefix"]
    mcore_layer_prefix = layer_info["mcore_layer_prefix"]
    pp_rank = layer_info["pp_rank"]

    hf_q_proj_name = f"{hf_layer_prefix}.self_attn.q_proj.weight"
    hf_k_proj_name = f"{hf_layer_prefix}.self_attn.k_proj.weight"
    hf_v_proj_name = f"{hf_layer_prefix}.self_attn.v_proj.weight"

    mcore_qkv_name = f"{mcore_layer_prefix}.self_attention.linear_qkv.weight"
    mcore_qkv_extra = f"{mcore_layer_prefix}.self_attention.linear_qkv._extra_state"

    q_proj = hf_sd.pop(hf_q_proj_name).to(dtype)
    k_proj = hf_sd.pop(hf_k_proj_name).to(dtype)
    v_proj = hf_sd.pop(hf_v_proj_name).to(dtype)
    
    num_q_proj, hidden_size = q_proj.size()
    num_kv_proj, _ = k_proj.size()

    num_heads = num_q_proj // dim
    num_kv_heads = num_kv_proj // dim
    num_queries_per_group = num_heads // num_kv_heads

    qkv = torch.cat([
        q_proj.reshape((num_kv_heads, num_queries_per_group * dim, -1)),
        k_proj.reshape((num_kv_heads, dim, -1)),
        v_proj.reshape((num_kv_heads, dim, -1)),
    ], dim=1).reshape((-1, hidden_size))
    qkv_shards = qkv.chunk(tp_size, dim=0)
    
    for ep_rank, tp_rank in product(range(ep_size), range(tp_size)):
        mcore_sds[pp_rank][ep_rank][tp_rank][mcore_qkv_name] = qkv_shards[tp_rank]
        mcore_sds[pp_rank][ep_rank][tp_rank][mcore_qkv_extra] = None


def convert_mcore2hf_attn_qkv(
    hf_sd, mcore_sds, layer_info, *, num_heads, num_kv_heads, dim, tp_size, ep_size, dtype,
):
    hf_layer_prefix = layer_info["hf_layer_prefix"]
    mcore_layer_prefix = layer_info["mcore_layer_prefix"]
    pp_rank = layer_info["pp_rank"]

    hf_q_proj_name = f"{hf_layer_prefix}.self_attn.q_proj.weight"
    hf_k_proj_name = f"{hf_layer_prefix}.self_attn.k_proj.weight"
    hf_v_proj_name = f"{hf_layer_prefix}.self_attn.v_proj.weight"

    mcore_qkv_name = f"{mcore_layer_prefix}.self_attention.linear_qkv.weight"
    mcore_qkv_extra = f"{mcore_layer_prefix}.self_attention.linear_qkv._extra_state"

    # Collect qkv_shards from mcore_sds
    qkv_shards = []
    for tp_rank in range(tp_size):
        ep_qkv_shards = []
        for ep_rank in range(ep_size):
            ep_qkv_shards.append(mcore_sds[pp_rank][ep_rank][tp_rank].pop(mcore_qkv_name))
            _ = mcore_sds[pp_rank][ep_rank][tp_rank].pop(mcore_qkv_extra)
        first_ep_qkv_shard = ep_qkv_shards[0]
        assert all([shard.equal(first_ep_qkv_shard) for shard in ep_qkv_shards])
        qkv_shards.append(first_ep_qkv_shard.to(dtype))

    # Concatenate qkv_shards along dim=0
    qkv = torch.cat(qkv_shards, dim=0)

    num_queries_per_group = num_heads // num_kv_heads
    qkv_size, _ = qkv.size()
    expected_qkv_size = num_kv_heads * (num_queries_per_group + 2) * dim

    if qkv_size != expected_qkv_size:
        raise ValueError("qkv_size does not match expected size")

    # Reshape qkv to (num_kv_heads, (num_queries_per_group + 2) * dim, -1)
    qkv = qkv.reshape(num_kv_heads, (num_queries_per_group + 2) * dim, -1)

    # Split qkv into q_proj, k_proj, v_proj
    q_proj = qkv[:, :num_queries_per_group * dim, :]
    k_proj = qkv[:, num_queries_per_group * dim : (num_queries_per_group + 1) * dim, :]
    v_proj = qkv[:, (num_queries_per_group + 1) * dim : (num_queries_per_group + 2) * dim, :]

    # Reshape projections to match HuggingFace format
    q_proj = q_proj.reshape(num_kv_heads * num_queries_per_group * dim, -1)
    k_proj = k_proj.reshape(num_kv_heads * dim, -1)
    v_proj = v_proj.reshape(num_kv_heads * dim, -1)

    # Store the projections in hf_sd
    hf_sd[hf_q_proj_name] = q_proj
    hf_sd[hf_k_proj_name] = k_proj
    hf_sd[hf_v_proj_name] = v_proj


def convert_hf2mcore_attn_output(mcore_sds, hf_sd, layer_info, *, tp_size, ep_size, dtype):
    hf_layer_prefix = layer_info["hf_layer_prefix"]
    mcore_layer_prefix = layer_info["mcore_layer_prefix"]
    pp_rank = layer_info["pp_rank"]

    hf_attn_output_name = f"{hf_layer_prefix}.self_attn.o_proj.weight"
    mcore_o_proj_name = f"{mcore_layer_prefix}.self_attention.linear_proj.weight"
    mcore_o_proj_extra = f"{mcore_layer_prefix}.self_attention.linear_proj._extra_state"

    o_proj = hf_sd.pop(hf_attn_output_name).to(dtype)
    o_proj_shards = o_proj.chunk(tp_size, dim=1)

    for ep_rank, tp_rank in product(range(ep_size), range(tp_size)):
        mcore_sds[pp_rank][ep_rank][tp_rank][mcore_o_proj_name] = o_proj_shards[tp_rank]
        mcore_sds[pp_rank][ep_rank][tp_rank][mcore_o_proj_extra] = None


def convert_mcore2hf_attn_output(hf_sd, mcore_sds, layer_info, *, tp_size, ep_size, dtype):
    hf_layer_prefix = layer_info["hf_layer_prefix"]
    mcore_layer_prefix = layer_info["mcore_layer_prefix"]
    pp_rank = layer_info["pp_rank"]

    mcore_o_proj_name = f"{mcore_layer_prefix}.self_attention.linear_proj.weight"
    mcore_o_proj_extra = f"{mcore_layer_prefix}.self_attention.linear_proj._extra_state"
    hf_attn_output_name = f"{hf_layer_prefix}.self_attn.o_proj.weight"

    o_proj_shards = []
    for tp_rank in range(tp_size):
        ep_shards = []
        for ep_rank in range(ep_size):
            ep_shards.append(mcore_sds[pp_rank][ep_rank][tp_rank].pop(mcore_o_proj_name))
            _ = mcore_sds[pp_rank][ep_rank][tp_rank].pop(mcore_o_proj_extra)
        
        assert all([ep_shards[0].equal(shard) for shard in ep_shards])
        o_proj_shards.append(ep_shards[0])
    o_proj = torch.cat(o_proj_shards, dim=1)
    hf_sd[hf_attn_output_name] = o_proj.to(dtype)


def convert_hf2mcore_mlp_norm(mcore_sds, hf_sd, layer_info, *, tp_size, ep_size, dtype):
    hf_layer_prefix = layer_info["hf_layer_prefix"]
    mcore_layer_prefix = layer_info["mcore_layer_prefix"]
    pp_rank = layer_info["pp_rank"]

    hf_mlp_norm_name = f"{hf_layer_prefix}.post_attention_layernorm.weight"
    mcore_mlp_norm_name = f"{mcore_layer_prefix}.pre_mlp_layernorm.weight"
    mlp_norm = hf_sd.pop(hf_mlp_norm_name).to(dtype)
    for ep_rank, tp_rank in product(range(ep_size), range(tp_size)):
        mcore_sds[pp_rank][ep_rank][tp_rank][mcore_mlp_norm_name] = mlp_norm


def convert_mcore2hf_mlp_norm(hf_sd, mcore_sds, layer_info, *, tp_size, ep_size, dtype):
    hf_layer_prefix = layer_info["hf_layer_prefix"]
    mcore_layer_prefix = layer_info["mcore_layer_prefix"]
    pp_rank = layer_info["pp_rank"]

    hf_mlp_norm_name = f"{hf_layer_prefix}.post_attention_layernorm.weight"
    mcore_mlp_norm_name = f"{mcore_layer_prefix}.pre_mlp_layernorm.weight"

    mlp_norms = []
    for ep_rank, tp_rank in product(range(ep_size), range(tp_size)):
        mlp_norm = mcore_sds[pp_rank][ep_rank][tp_rank].pop(mcore_mlp_norm_name)
        mlp_norms.append(mlp_norm)
    assert all([mlp_norms[0].equal(norm) for norm in mlp_norms])
    hf_sd[hf_mlp_norm_name] = mlp_norms[0].to(dtype)


def convert_hf2mcore_mlp_router(mcore_sds, hf_sd, layer_info, *, tp_size, ep_size, dtype):
    hf_layer_prefix = layer_info["hf_layer_prefix"]
    mcore_layer_prefix = layer_info["mcore_layer_prefix"]
    pp_rank = layer_info["pp_rank"]

    hf_router_name = f"{hf_layer_prefix}.block_sparse_moe.gate.weight"
    mcore_router_name = f"{mcore_layer_prefix}.mlp.router.weight"
    router = hf_sd.pop(hf_router_name).to(dtype)
    for ep_rank, tp_rank in product(range(ep_size), range(tp_size)):
        mcore_sds[pp_rank][ep_rank][tp_rank][mcore_router_name] = router


def convert_mcore2hf_mlp_router(hf_sd, mcore_sds, layer_info, *, tp_size, ep_size, dtype):
    hf_layer_prefix = layer_info["hf_layer_prefix"]
    mcore_layer_prefix = layer_info["mcore_layer_prefix"]
    pp_rank = layer_info["pp_rank"]

    hf_router_name = f"{hf_layer_prefix}.block_sparse_moe.gate.weight"
    mcore_router_name = f"{mcore_layer_prefix}.mlp.router.weight"

    routers = []
    for ep_rank, tp_rank in product(range(ep_size), range(tp_size)):
        router = mcore_sds[pp_rank][ep_rank][tp_rank].pop(mcore_router_name)
        routers.append(router)
    assert all([routers[0].equal(router) for router in routers])
    hf_sd[hf_router_name] = routers[0].to(dtype)


def convert_hf2mcore_mlp_experts_w1_w3(
    mcore_sds,
    hf_sd,
    layer_info,
    *,
    num_experts,
    tp_size,
    ep_size,
    dtype,
    moe_grouped_gemm,
):
    hf_layer_prefix = layer_info["hf_layer_prefix"]
    mcore_layer_prefix = layer_info["mcore_layer_prefix"]
    pp_rank = layer_info["pp_rank"]

    hf_experts_w1_names = [
        f"{hf_layer_prefix}.block_sparse_moe.experts.{i}.w1.weight"
        for i in range(num_experts)
    ]
    hf_experts_w3_names = [
        f"{hf_layer_prefix}.block_sparse_moe.experts.{i}.w3.weight"
        for i in range(num_experts)
    ]
    hf_experts_w1 = [hf_sd.pop(name).to(dtype) for name in hf_experts_w1_names]
    hf_experts_w3 = [hf_sd.pop(name).to(dtype) for name in hf_experts_w3_names]

    full_w1 = torch.stack(hf_experts_w1, dim=0)
    full_w3 = torch.stack(hf_experts_w3, dim=0)
    num_experts, out_features, in_features = full_w1.size()
    assert num_experts % ep_size == 0
    
    num_local_experts = num_experts // ep_size
    full_w1 = full_w1.view(
        ep_size, num_local_experts, tp_size, out_features // tp_size, in_features
    )
    full_w1 = full_w1.permute(0, 2, 1, 3, 4)
    full_w3 = full_w3.view(
        ep_size, num_local_experts, tp_size, out_features // tp_size, in_features
    )
    full_w3 = full_w3.permute(0, 2, 1, 3, 4)

    if moe_grouped_gemm:
        linear_f1_prefix = f"{mcore_layer_prefix}.mlp.experts.linear_fc1"
        for ep_rank, tp_rank, local_expert_idx in product(
            range(ep_size), range(tp_size), range(num_local_experts)
        ):
            local_expert_name = f"{linear_f1_prefix}.weight{local_expert_idx}"
            linear_f1_extra = f"{linear_f1_prefix}._extra_state"
            local_w1 = full_w1[ep_rank, tp_rank, local_expert_idx]
            local_w3 = full_w3[ep_rank, tp_rank, local_expert_idx]
            local_expert = torch.cat([local_w1, local_w3], dim=0)
            mcore_sds[pp_rank][ep_rank][tp_rank][local_expert_name] = local_expert
            mcore_sds[pp_rank][ep_rank][tp_rank][linear_f1_extra] = None
    else:
        local_experts_prefix = f"{mcore_layer_prefix}.mlp.experts.local_experts"
        for ep_rank, tp_rank, local_expert_idx in product(
            range(ep_size), range(tp_size), range(num_local_experts)
        ):
            local_expert_name = f"{local_experts_prefix}.{local_expert_idx}.linear_fc1.weight"
            local_expert_extra = f"{local_experts_prefix}.{local_expert_idx}.linear_fc1._extra_state"
            local_w1 = full_w1[ep_rank, tp_rank, local_expert_idx]
            local_w3 = full_w3[ep_rank, tp_rank, local_expert_idx]
            local_expert = torch.cat([local_w1, local_w3], dim=0)
            mcore_sds[pp_rank][ep_rank][tp_rank][local_expert_name] = local_expert
            mcore_sds[pp_rank][ep_rank][tp_rank][local_expert_extra] = None


def convert_mcore2hf_mlp_experts_w1_w3(
    hf_sd,
    mcore_sds,
    layer_info,
    *,
    num_experts,
    tp_size,
    ep_size,
    dtype,
    moe_grouped_gemm,
):
    hf_layer_prefix = layer_info["hf_layer_prefix"]
    mcore_layer_prefix = layer_info["mcore_layer_prefix"]
    pp_rank = layer_info["pp_rank"]

    num_local_experts = num_experts // ep_size

    # Prepare lists to hold the reconstructed w1 and w3 tensors
    hf_experts_w1 = []
    hf_experts_w3 = []

    for ep_rank, local_expert_idx in product(range(ep_size), range(num_local_experts)):
        # Collect shards across tp_ranks
        local_expert_tp_list = []
        for tp_rank in range(tp_size):
            if moe_grouped_gemm:
                linear_f1_prefix = f"{mcore_layer_prefix}.mlp.experts.linear_fc1"
                local_expert_name = f"{linear_f1_prefix}.weight{local_expert_idx}"
                linear_f1_extra = f"{linear_f1_prefix}._extra_state"
            else:
                local_experts_prefix = f"{mcore_layer_prefix}.mlp.experts.local_experts"
                local_expert_name = f"{local_experts_prefix}.{local_expert_idx}.linear_fc1.weight"
                linear_f1_extra = f"{local_experts_prefix}.{local_expert_idx}.linear_fc1._extra_state"

            # Retrieve the local_expert tensor from mcore_sds
            local_expert = mcore_sds[pp_rank][ep_rank][tp_rank].pop(local_expert_name).to(dtype)

            # Pop the extra_state to indicate it was processed
            mcore_sds[pp_rank][ep_rank][tp_rank].pop(linear_f1_extra, None)

            local_expert_tp_list.append(local_expert)

        # Concatenate the shards along the out_features dimension
        local_expert_full = torch.cat(local_expert_tp_list, dim=0)

        # Split local_expert_full into local_w1 and local_w3
        out_features_total = local_expert_full.size(0) // 2
        local_w1 = local_expert_full[:out_features_total, :]
        local_w3 = local_expert_full[out_features_total:, :]

        hf_experts_w1.append(local_w1)
        hf_experts_w3.append(local_w3)

    # Construct the parameter names for HuggingFace format
    hf_experts_w1_names = [
        f"{hf_layer_prefix}.block_sparse_moe.experts.{i}.w1.weight"
        for i in range(num_experts)
    ]
    hf_experts_w3_names = [
        f"{hf_layer_prefix}.block_sparse_moe.experts.{i}.w3.weight"
        for i in range(num_experts)
    ]

    # Store the reconstructed weights in hf_sd
    for i, name in enumerate(hf_experts_w1_names):
        hf_sd[name] = hf_experts_w1[i]

    for i, name in enumerate(hf_experts_w3_names):
        hf_sd[name] = hf_experts_w3[i]


def convert_hf2mcore_mlp_experts_w2(
    mcore_sds,
    hf_sd,
    layer_info,
    *,
    num_experts,
    tp_size,
    ep_size,
    dtype,
    moe_grouped_gemm,
):
    hf_layer_prefix = layer_info["hf_layer_prefix"]
    mcore_layer_prefix = layer_info["mcore_layer_prefix"]
    pp_rank = layer_info["pp_rank"]

    hf_experts_w2_names = [
        f"{hf_layer_prefix}.block_sparse_moe.experts.{i}.w2.weight"
        for i in range(num_experts)
    ]
    hf_experts_w2 = [hf_sd.pop(name).to(dtype) for name in hf_experts_w2_names]
    
    full_w2 = torch.stack(hf_experts_w2, dim=0)
    
    num_experts, out_features, in_features = full_w2.size()
    num_local_experts = num_experts // ep_size

    full_w2 = full_w2.view(
        ep_size, num_local_experts, out_features, tp_size, in_features // tp_size,
    )
    full_w2 = full_w2.permute(0, 3, 1, 2, 4)

    if moe_grouped_gemm:
        linear_f2_prefix = f"{mcore_layer_prefix}.mlp.experts.linear_fc2"
        for ep_rank, tp_rank, local_expert_idx in product(
            range(ep_size), range(tp_size), range(num_local_experts)
        ):
            local_expert_name = f"{linear_f2_prefix}.weight{local_expert_idx}"
            linear_f2_extra = f"{linear_f2_prefix}._extra_state"
            local_w2 = full_w2[ep_rank, tp_rank, local_expert_idx]
            mcore_sds[pp_rank][ep_rank][tp_rank][local_expert_name] = local_w2
            mcore_sds[pp_rank][ep_rank][tp_rank][linear_f2_extra] = None
    else:
        local_experts_prefix = f"{mcore_layer_prefix}.mlp.experts.local_experts"
        for ep_rank, tp_rank, local_expert_idx in product(
            range(ep_size), range(tp_size), range(num_local_experts)
        ):
            local_expert_name = f"{local_experts_prefix}.{local_expert_idx}.linear_fc2.weight"
            local_expert_extra = f"{local_experts_prefix}.{local_expert_idx}.linear_fc2._extra_state"
            local_w2 = full_w2[ep_rank, tp_rank, local_expert_idx]
            mcore_sds[pp_rank][ep_rank][tp_rank][local_expert_name] = local_w2
            mcore_sds[pp_rank][ep_rank][tp_rank][local_expert_extra] = None


def convert_mcore2hf_mlp_experts_w2(
    hf_sd,
    mcore_sds,
    layer_info,
    *,
    num_experts,
    tp_size,
    ep_size,
    dtype,
    moe_grouped_gemm,
):
    hf_layer_prefix = layer_info["hf_layer_prefix"]
    mcore_layer_prefix = layer_info["mcore_layer_prefix"]
    pp_rank = layer_info["pp_rank"]

    num_local_experts = num_experts // ep_size

    # Prepare the list to hold reconstructed weights
    hf_experts_w2 = []

    for ep_rank, local_expert_idx in product(range(ep_size), range(num_local_experts)):
        local_w2_tp_list = []
        for tp_rank in range(tp_size):
            if moe_grouped_gemm:
                linear_f2_prefix = f"{mcore_layer_prefix}.mlp.experts.linear_fc2"
                local_expert_name = f"{linear_f2_prefix}.weight{local_expert_idx}"
                linear_f2_extra = f"{linear_f2_prefix}._extra_state"
            else:
                local_experts_prefix = f"{mcore_layer_prefix}.mlp.experts.local_experts"
                local_expert_name = f"{local_experts_prefix}.{local_expert_idx}.linear_fc2.weight"
                linear_f2_extra = f"{local_experts_prefix}.{local_expert_idx}.linear_fc2._extra_state"

            # Retrieve the local_w2 tensor from mcore_sds
            local_w2 = mcore_sds[pp_rank][ep_rank][tp_rank].pop(local_expert_name).to(dtype)

            # Pop the extra_state to indicate it was processed
            mcore_sds[pp_rank][ep_rank][tp_rank].pop(linear_f2_extra, None)

            local_w2_tp_list.append(local_w2)

        # Concatenate the shards along the in_features dimension
        local_w2_full = torch.cat(local_w2_tp_list, dim=1)
        hf_experts_w2.append(local_w2_full)

    # Construct the parameter names for HuggingFace format
    hf_experts_w2_names = [
        f"{hf_layer_prefix}.block_sparse_moe.experts.{i}.w2.weight"
        for i in range(num_experts)
    ]

    # Store the reconstructed weights in hf_sd
    for name, w2 in zip(hf_experts_w2_names, hf_experts_w2):
        hf_sd[name] = w2


def get_checkpoint_sub_dir_name(*, tp_rank, pp_rank, pp_size, ep_rank, ep_size):
    sub_dir_name = f"mp_rank_{tp_rank:02d}"
    if pp_size > 1: sub_dir_name = f"{sub_dir_name}_{pp_rank:03d}"
    if ep_size > 1: sub_dir_name = f"{sub_dir_name}_{ep_rank:03d}"
    return sub_dir_name


def get_layer_info(*, layer_idx, pp_size, num_layers):
    pp_rank = get_pp_rank(layer_idx=layer_idx, pp_size=pp_size, num_layers=num_layers)
    num_layers_per_pp = num_layers // pp_size
    # Inside each pipeline parallel group, the layer indices start from 0
    pp_layer_idx = layer_idx - pp_rank * num_layers_per_pp
    return {
        "hf_layer_prefix": f"model.layers.{layer_idx}",
        "mcore_layer_prefix": f"decoder.layers.{pp_layer_idx}",
        "pp_rank": pp_rank,
    }

def get_pp_rank(*, layer_idx, pp_size, num_layers):
    assert layer_idx < num_layers
    num_layers_per_pp = num_layers // pp_size
    for pp_rank in range(pp_size):
        curr_min = pp_rank * num_layers_per_pp
        curr_max = (pp_rank + 1) * num_layers_per_pp
        if layer_idx >= curr_min and layer_idx < curr_max:
            return pp_rank
    raise RuntimeError(f"Invalid layer_idx: {layer_idx}")

def parse_args():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="direction", required=True)

    add_hf2mcore_subparser(subparsers)
    add_mcore2hf_subparser(subparsers)
    return parser.parse_args()


def add_hf2mcore_subparser(subparsers):
    hf2mcore_parser = subparsers.add_parser(
        "hf-to-mcore", help="Arguments for 'mcore-to-hf' direction."
    )
    hf2mcore_parser.add_argument("--hf-load-dir", required=True, type=str)
    hf2mcore_parser.add_argument("--mcore-save-dir", required=True, type=str)
    hf2mcore_parser.add_argument("--target-tensor-parallel-size", type=int, default=1)
    hf2mcore_parser.add_argument("--target-pipeline-parallel-size", type=int, default=1)
    hf2mcore_parser.add_argument("--target-expert-parallel-size", type=int, default=1)
    hf2mcore_parser.add_argument("--target-params-dtype", type=str, default='float32')
    hf2mcore_parser.add_argument("--moe-grouped-gemm", action="store_true")

    return hf2mcore_parser

def add_mcore2hf_subparser(subparsers):
    
    # Arguments for 'mg2hf' direction
    mcore2hf_parser = subparsers.add_parser(
        'mcore-to-hf', help="Arguments for 'mcore-to-hf' direction."
    )
    mcore2hf_parser.add_argument("--mcore-load-dir", required=True, type=str)
    mcore2hf_parser.add_argument("--hf-save-dir", required=True, type=str)
    mcore2hf_parser.add_argument("--source-tensor-parallel-size", type=int, default=1)
    mcore2hf_parser.add_argument("--source-pipeline-parallel-size", type=int, default=1)
    mcore2hf_parser.add_argument("--source-expert-parallel-size", type=int, default=1)
    mcore2hf_parser.add_argument("--target-params-dtype", type=str, default='float32')
    mcore2hf_parser.add_argument("--moe-grouped-gemm", action="store_true")
    mcore2hf_parser.add_argument("--hf-tokenizer-path", type=str, default=None)
    
    return mcore2hf_parser

if __name__ == "__main__":
    main()