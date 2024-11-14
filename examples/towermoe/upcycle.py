import jsonargparse
import re
import os
import torch

from accelerate import init_empty_weights
from torch import Tensor
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, LlamaConfig, MixtralConfig,
)

from typing import Any, Optional


def main(
    llama_model_path: str,
    mixtral_model_path: str,
    num_local_experts: int,
    num_experts_per_tok: int,
    intermediate_size: Optional[int] = None,
    router_aux_loss_coef: float = 0.01,
):
    llama = AutoModelForCausalLM.from_pretrained(llama_model_path)
    if intermediate_size is None:
        intermediate_size = llama.config.intermediate_size
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
):
    mixtral_sd = {}

    print("> Copying embeddings")
    mixtral_sd.update(extract_embeddings(llama_sd))

    print("> Copying lm_head")
    mixtral_sd.update(extract_lm_head(llama_sd))

    print("> Copying final layernorm")
    mixtral_sd.update(extract_final_layernorm(llama_sd))

    for layer_id in range(mixtral_cfg.num_hidden_layers):
        print(f"> Copying layer {layer_id}")
        print("  > Copying attention layernorm")
        mixtral_sd.update(extract_attention_layernorm(llama_sd, layer_id))

        print("  > Copying attention projections")
        mixtral_sd.update(extract_attention_projection(llama_sd, layer_id))

        print("  > Copying mlp layernorm")
        mixtral_sd.update(extract_mlp_layernorm(llama_sd, layer_id))

        print("  > Create router")
        mixtral_sd.update(create_router(
            layer_id=layer_id,
            hidden_size=mixtral_cfg.hidden_size,
            num_local_experts=mixtral_cfg.num_local_experts,
        ))

        print("  > Upcycle mlp")
        mixtral_sd.update(upcycle_mlp(
            llama_sd=llama_sd,
            layer_id=layer_id,
            num_local_experts=mixtral_cfg.num_local_experts,
            intermediate_size=mixtral_cfg.intermediate_size,
        ))

    for key in llama_sd:
        print(f"Unused key: {key}")

    return mixtral_sd


def extract_embeddings(llama_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    embed_weight = llama_sd.pop("model.embed_tokens.weight")
    return {"model.embed_tokens.weight": embed_weight}


def extract_lm_head(llama_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    lm_head_weight = llama_sd.pop("lm_head.weight")
    assert "lm_head.bias" not in llama_sd, "lm_head.bias should not be present"
    return {"lm_head.weight": lm_head_weight}


def extract_final_layernorm(llama_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    norm_weight = llama_sd.pop("model.norm.weight")
    return {"model.norm.weight": norm_weight}


def extract_attention_layernorm(llama_sd: dict[str, Tensor], layer_id: int) -> dict[str, Tensor]:
    norm_weight = llama_sd.pop(f"model.layers.{layer_id}.input_layernorm.weight")
    return {f"model.layers.{layer_id}.input_layernorm.weight": norm_weight}


def extract_attention_projection(llama_sd: dict[str, Tensor], layer_id: int) -> dict[str, Tensor]:
    q_proj = llama_sd.pop(f"model.layers.{layer_id}.self_attn.q_proj.weight")
    k_proj = llama_sd.pop(f"model.layers.{layer_id}.self_attn.k_proj.weight")
    v_proj = llama_sd.pop(f"model.layers.{layer_id}.self_attn.v_proj.weight")
    o_proj = llama_sd.pop(f"model.layers.{layer_id}.self_attn.o_proj.weight")
    return {
        f"model.layers.{layer_id}.self_attn.q_proj.weight": q_proj,
        f"model.layers.{layer_id}.self_attn.k_proj.weight": k_proj,
        f"model.layers.{layer_id}.self_attn.v_proj.weight": v_proj,
        f"model.layers.{layer_id}.self_attn.o_proj.weight": o_proj,
    }


def extract_mlp_layernorm(llama_sd: dict[str, Tensor], layer_id: int) -> dict[str, Tensor]:
    norm_weight = llama_sd.pop(f"model.layers.{layer_id}.post_attention_layernorm.weight")
    return {f"model.layers.{layer_id}.post_attention_layernorm.weight": norm_weight}


def create_router(
    *, layer_id: int, hidden_size: int, num_local_experts: int,
) -> dict[str, Tensor]:
    router_w = torch.empty(num_local_experts, hidden_size)
    torch.nn.init.normal_(router_w, mean=0.0, std=0.02)
    return {f"model.layers.{layer_id}.block_sparse_moe.gate.weight": router_w}


def upcycle_mlp(
    *, llama_sd: dict[str, Tensor], layer_id: int, num_local_experts: int, intermediate_size: int,
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

    up_shards = up_proj.chunk(num_local_experts, dim=0)
    gate_shards = gate_proj.chunk(num_local_experts, dim=0)
    down_shards = down_proj.chunk(num_local_experts, dim=1)

    upcycled_sd = {}
    for i in range(num_local_experts):
        assert up_shards[i].shape == (intermediate_size, hidden_size)
        assert gate_shards[i].shape == (intermediate_size, hidden_size)
        assert down_shards[i].shape == (hidden_size, intermediate_size)
        # We need to clone the tensors to avoid some saving errors
        upcycled_sd[f"model.layers.{layer_id}.block_sparse_moe.experts.{i}.w3.weight"] = up_shards[i].clone()
        upcycled_sd[f"model.layers.{layer_id}.block_sparse_moe.experts.{i}.w2.weight"] = down_shards[i].clone()
        upcycled_sd[f"model.layers.{layer_id}.block_sparse_moe.experts.{i}.w1.weight"] = gate_shards[i].clone()

    return upcycled_sd


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
