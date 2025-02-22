import jsonargparse
import re
import os
import torch

from accelerate import init_empty_weights
from torch import Tensor
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, LlamaConfig, MixtralConfig,
)

from typing import Any


def main(
    llama_model_path: str,
    mixtral_model_path: str,
    total_expansion: int,
    active_expansion: int,
    granularity: int,
    router_aux_loss_coef: float = 0.01,
    shuffle: bool = False,
    init_std: float = 0.006,
    moe_reinit_percentage: float = 0.0,
    other_reinit_percentage: float = 0.0,
):
    llama = AutoModelForCausalLM.from_pretrained(llama_model_path)
    assert llama.config.intermediate_size % granularity == 0, "Granularity should divide intermediate size"
    intermediate_size = llama.config.intermediate_size // granularity
    num_experts_per_tok = active_expansion * granularity
    num_local_experts = total_expansion * granularity
    mixtral_cfg = llama_cfg_to_mixtral_cfg(
        llama.config,
        architectures=["MixtralForCausalLM"],
        model_type="mixtral",
        num_experts_per_tok=num_experts_per_tok,
        num_local_experts=num_local_experts,
        intermediate_size=intermediate_size,
        router_aux_loss_coef=router_aux_loss_coef,
    )
    print("> Config Diff:")
    print_dict_diff(llama.config.to_dict(), mixtral_cfg.to_dict())

    mixtral_sd = upcycle_state_dict(
        llama_sd=llama.state_dict(),
        mixtral_cfg=mixtral_cfg,
        shuffle=shuffle,
        init_std=init_std,
        moe_reinit_percentage=moe_reinit_percentage,
        other_reinit_percentage=other_reinit_percentage,
    )

    print(f"> Creating Mixtral model")
    os.makedirs(mixtral_model_path, exist_ok=True)
    with init_empty_weights():
        mixtral = AutoModelForCausalLM.from_config(mixtral_cfg)
    mixtral.load_state_dict(mixtral_sd, assign=True)

    print(f"> Saving Mixtral model")
    mixtral.save_pretrained(mixtral_model_path)

    print(f"> Saving tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
    tokenizer.save_pretrained(mixtral_model_path)



def llama_cfg_to_mixtral_cfg(
    cfg: LlamaConfig,
    **overwrite: Any,
) -> MixtralConfig:
    llama_dict = cfg.to_dict()
    default_mixtral_dict = MixtralConfig().to_dict()
    keys_to_copy = set(llama_dict.keys()) & set(default_mixtral_dict.keys())
    mixtral_dict = {k: llama_dict[k] for k in keys_to_copy}
    mixtral_dict.update(overwrite)
    cfg, unused_args = MixtralConfig.from_dict(
        mixtral_dict, return_unused_kwargs=True,
    )
    assert not unused_args, f"Unused args: {unused_args}"
    return cfg

def upcycle_state_dict(
    *,
    llama_sd: dict[str, Tensor],
    mixtral_cfg: MixtralConfig,
    shuffle: bool,
    init_std: float,
    moe_reinit_percentage: float,
    other_reinit_percentage: float,
):
    mixtral_sd = {}

    print("> Copying embeddings")
    mixtral_sd.update(extract_embeddings(
        llama_sd,
        other_reinit_percentage=other_reinit_percentage,
        init_std=init_std,
    ))

    print("> Copying lm_head")
    mixtral_sd.update(extract_lm_head(
        llama_sd,
        other_reinit_percentage=other_reinit_percentage,
        init_std=init_std,
    ))

    print("> Copying final layernorm")
    mixtral_sd.update(extract_final_layernorm(
        llama_sd,
        other_reinit_percentage=other_reinit_percentage,
        init_std=init_std,
    ))

    for layer_id in range(mixtral_cfg.num_hidden_layers):
        print(f"> Copying layer {layer_id}")
        print("  > Copying attention layernorm")
        mixtral_sd.update(extract_attention_layernorm(
            llama_sd,
            layer_id,
            other_reinit_percentage=other_reinit_percentage,
            init_std=init_std,
        ))

        print("  > Copying attention projections")
        mixtral_sd.update(extract_attention_projection(
            llama_sd,
            layer_id,
            other_reinit_percentage=other_reinit_percentage,
            init_std=init_std,
        ))

        print("  > Copying mlp layernorm")
        mixtral_sd.update(extract_mlp_layernorm(
            llama_sd,
            layer_id,
            other_reinit_percentage=other_reinit_percentage,
            init_std=init_std,
        ))

        print("  > Create router")
        mixtral_sd.update(create_router(
            layer_id=layer_id,
            hidden_size=mixtral_cfg.hidden_size,
            num_local_experts=mixtral_cfg.num_local_experts,
            init_std=init_std,
        ))

        print("  > Upcycle mlp")
        mixtral_sd.update(upcycle_mlp(
            llama_sd=llama_sd,
            layer_id=layer_id,
            num_local_experts=mixtral_cfg.num_local_experts,
            intermediate_size=mixtral_cfg.intermediate_size,
            shuffle=shuffle,
            init_std=init_std,
            moe_reinit_percentage=moe_reinit_percentage,
        ))

    for key in llama_sd:
        print(f"Unused key: {key}")

    return mixtral_sd


def extract_embeddings(
    llama_sd: dict[str, Tensor],
    *,
    other_reinit_percentage: float,
    init_std: float,
) -> dict[str, Tensor]:
    embed_weight = llama_sd.pop("model.embed_tokens.weight")
    reinit_tensor(embed_weight, other_reinit_percentage, init_std)
    return {"model.embed_tokens.weight": embed_weight}


def extract_lm_head(
    llama_sd: dict[str, Tensor],
    *,
    other_reinit_percentage: float,
    init_std: float,
) -> dict[str, Tensor]:
    lm_head_weight = llama_sd.pop("lm_head.weight")
    assert "lm_head.bias" not in llama_sd, "lm_head.bias should not be present"
    reinit_tensor(lm_head_weight, other_reinit_percentage, init_std)
    return {"lm_head.weight": lm_head_weight}


def extract_final_layernorm(
    llama_sd: dict[str, Tensor],
    *,
    other_reinit_percentage: float,
    init_std: float,
) -> dict[str, Tensor]:
    norm_weight = llama_sd.pop("model.norm.weight")
    reinit_tensor(norm_weight, other_reinit_percentage, init_std)
    return {"model.norm.weight": norm_weight}


def extract_attention_layernorm(
    llama_sd: dict[str, Tensor],
    layer_id: int,
    *,
    other_reinit_percentage: float,
    init_std: float,
) -> dict[str, Tensor]:
    norm_weight = llama_sd.pop(f"model.layers.{layer_id}.input_layernorm.weight")
    reinit_tensor(norm_weight, other_reinit_percentage, init_std)
    return {f"model.layers.{layer_id}.input_layernorm.weight": norm_weight}


def extract_attention_projection(
    llama_sd: dict[str, Tensor],
    layer_id: int,
    *,
    other_reinit_percentage: float,
    init_std: float,
) -> dict[str, Tensor]:
    q_proj = llama_sd.pop(f"model.layers.{layer_id}.self_attn.q_proj.weight")
    k_proj = llama_sd.pop(f"model.layers.{layer_id}.self_attn.k_proj.weight")
    v_proj = llama_sd.pop(f"model.layers.{layer_id}.self_attn.v_proj.weight")
    o_proj = llama_sd.pop(f"model.layers.{layer_id}.self_attn.o_proj.weight")
    if layer_id == 0:
        print("Q proj before", q_proj[:10, :10])
    reinit_tensor(q_proj, other_reinit_percentage, init_std)
    if layer_id == 0:
        print("Q proj after", q_proj[:10, :10])
    reinit_tensor(k_proj, other_reinit_percentage, init_std)
    reinit_tensor(v_proj, other_reinit_percentage, init_std)
    reinit_tensor(o_proj, other_reinit_percentage, init_std)
    return {
        f"model.layers.{layer_id}.self_attn.q_proj.weight": q_proj,
        f"model.layers.{layer_id}.self_attn.k_proj.weight": k_proj,
        f"model.layers.{layer_id}.self_attn.v_proj.weight": v_proj,
        f"model.layers.{layer_id}.self_attn.o_proj.weight": o_proj,
    }


def extract_mlp_layernorm(
    llama_sd: dict[str, Tensor],
    layer_id: int,
    *,
    other_reinit_percentage: float,
    init_std: float,
) -> dict[str, Tensor]:
    norm_weight = llama_sd.pop(f"model.layers.{layer_id}.post_attention_layernorm.weight")
    reinit_tensor(norm_weight, other_reinit_percentage, init_std)
    return {f"model.layers.{layer_id}.post_attention_layernorm.weight": norm_weight}


def create_router(
    *, layer_id: int, hidden_size: int, num_local_experts: int, init_std: float,
) -> dict[str, Tensor]:
    router_w = torch.empty(num_local_experts, hidden_size)
    torch.nn.init.normal_(router_w, mean=0.0, std=init_std)
    return {f"model.layers.{layer_id}.block_sparse_moe.gate.weight": router_w}


def upcycle_mlp(
    *,
    llama_sd: dict[str, Tensor],
    layer_id: int,
    num_local_experts: int,
    intermediate_size: int,
    shuffle: bool,
    init_std: float,
    moe_reinit_percentage: float,
) -> dict[str, Tensor]:
    up_proj = llama_sd.pop(f"model.layers.{layer_id}.mlp.up_proj.weight")
    gate_proj = llama_sd.pop(f"model.layers.{layer_id}.mlp.gate_proj.weight")
    down_proj = llama_sd.pop(f"model.layers.{layer_id}.mlp.down_proj.weight")

    orig_intermediate_size, hidden_size = up_proj.shape
    target_intermediate_size = intermediate_size * num_local_experts
    assert target_intermediate_size % orig_intermediate_size == 0, "Intermediate size mismatch"
    num_copies = target_intermediate_size // orig_intermediate_size
    if layer_id == 0:
        print(f"    > Orig Intermediate size: {orig_intermediate_size}")
        print(f"    > Target Intermediate size: {target_intermediate_size}")
        print(f"    > Intermediate size: {intermediate_size}")
        print(f"    > Num copies: {num_copies}")


    up_proj = up_proj.repeat(num_copies, 1)
    gate_proj = gate_proj.repeat(num_copies, 1)
    down_proj = down_proj.repeat(1, num_copies)

    if shuffle:
        perm = torch.randperm(up_proj.shape[0])
        up_proj = up_proj[perm]
        gate_proj = gate_proj[perm]
        down_proj = down_proj[:, perm]

    up_shards = up_proj.chunk(num_local_experts, dim=0)
    gate_shards = gate_proj.chunk(num_local_experts, dim=0)
    down_shards = down_proj.chunk(num_local_experts, dim=1)

    upcycled_sd = {}
    for i in range(num_local_experts):
        assert up_shards[i].shape == (intermediate_size, hidden_size)
        assert gate_shards[i].shape == (intermediate_size, hidden_size)
        assert down_shards[i].shape == (hidden_size, intermediate_size)
        # We need to clone the tensors to avoid some saving errors
        up_shard = up_shards[i].clone()
        gate_shard = gate_shards[i].clone()
        down_shard = down_shards[i].clone()

        if layer_id == 0 and i == 0:
            print("Up shard before", up_shard[:10, :10])

        for tensor in (up_shard, gate_shard, down_shard):
            reinit_tensor(tensor, moe_reinit_percentage, init_std)

        if layer_id == 0 and i == 0:
            print("Up shard after", up_shard[:10, :10])

        upcycled_sd[f"model.layers.{layer_id}.block_sparse_moe.experts.{i}.w3.weight"] = up_shard
        upcycled_sd[f"model.layers.{layer_id}.block_sparse_moe.experts.{i}.w2.weight"] = down_shard
        upcycled_sd[f"model.layers.{layer_id}.block_sparse_moe.experts.{i}.w1.weight"] = gate_shard

    return upcycled_sd


def reinit_tensor(tensor: Tensor, percentage: float, init_std: float):
    mask = torch.rand(tensor.shape, device=tensor.device) <= percentage
    re_init = torch.nn.init.normal_(torch.empty_like(tensor), mean=0.0, std=init_std)
    tensor.copy_(torch.where(mask, re_init, tensor))


def print_dict_diff(dict1, dict2):
    """
    Prints the differences between two dictionaries.

    Args:
        dict1 (dict): The first dictionary.
        dict2 (dict): The second dictionary.
        path (str, optional): The base path for nested keys. Defaults to "".
    """
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    shared_keys = keys1 & keys2
    only_in_dict1 = keys1 - keys2
    only_in_dict2 = keys2 - keys1

    for key in sorted(only_in_dict1):
        print(f"Removed: {key} = {dict1[key]}")

    for key in sorted(only_in_dict2):
        print(f"Added: {key} = {dict2[key]}")

    for key in sorted(shared_keys):
        val1 = dict1[key]
        val2 = dict2[key]
        if  val1 != val2:
            print(f"Modified: {key} = {val1} -> {val2}")

if __name__ == '__main__':
    jsonargparse.CLI(main, as_positional=False)
