import dataclasses
import json
import jsonargparse
import jinja2
import os

from typing import Any, Optional


@dataclasses.dataclass(frozen=True)
class NetworkSizeArgs:
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    max_position_embeddings: int
    num_attention_heads: int
    # Use group query attention if num_query_groups != num_attention_heads
    num_query_groups: int


@dataclasses.dataclass(frozen=True)
class LoggingArgs:
    log_params_norm: bool = False
    log_throughput: bool = False
    log_progress: bool = False


@dataclasses.dataclass(frozen=True)
class RegularizationArgs:
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    weight_decay: float = 0.01
    clip_grad: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8


@dataclasses.dataclass(frozen=True)
class TrainingArgs:
    micro_batch_size: int
    global_batch_size: int


@dataclasses.dataclass(frozen=True)
class LearningRateArgs:
    lr: float
    lr_warmup_iters: int = 0


@dataclasses.dataclass(frozen=True)
class CheckpointingArgs:
    save: Optional[str] = None
    save_interval: Optional[int] = None
    no_save_optim: bool = False
    no_save_rng: bool = False
    load: Optional[str] = None
    no_load_optim: bool = False
    no_load_rng: bool = False
    pretrained_checkpoint: Optional[str] = None
    no_initialization: bool = False


@dataclasses.dataclass(frozen=True)
class DistributedArgs:
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    distributed_optimizer: bool = False


@dataclasses.dataclass(frozen=True)
class ValidationArgs:
    eval_iters: int = 100
    eval_interval: int = 1000


@dataclasses.dataclass(frozen=True)
class DataArgs:
    seq_length: int
    tokenizer_type: str
    train_iters: int
    tokenizer_model: Optional[str] = None
    data_path: list[str] = dataclasses.field(default_factory=list)
    split: Optional[str] = None



@dataclasses.dataclass(frozen=True)
class MoEArgs:
    num_experts: int
    moe_router_topk: int
    expert_model_parallel_size: int = 1
    moe_aux_loss_coeff: float = 0.01
    moe_z_loss_coeff: float = 0.001


@dataclasses.dataclass(frozen=True)
class LaunchArgs:
    job_name: str
    run_dir: str
    gpus_per_node: int
    num_nodes: int = 1
    account: Optional[str] = None
    partition: Optional[str] = None
    qos: Optional[str] = None
    time: Optional[str] = None
    exclusive: bool = False
    port: int = 29400
    activate_env_cmd: str = ""


def main(
    network_size: NetworkSizeArgs,
    logging: LoggingArgs,
    regularization: RegularizationArgs,
    training: TrainingArgs,
    learning_rate: LearningRateArgs,
    checkpointing: CheckpointingArgs,
    distributed: DistributedArgs,
    validation: ValidationArgs,
    data: DataArgs,
    moe: MoEArgs,
    launch: LaunchArgs,
):
    megatron_dir = os.path.abspath(os.path.join(__file__, "..", "..", ".."))

    cfg = {
        "launch": _build_launch_cfg(launch),
        "network_size": dataclasses.asdict(network_size),
        "logging": dataclasses.asdict(logging),
        "regularization": dataclasses.asdict(regularization),
        "training": dataclasses.asdict(training),
        "learning_rate": dataclasses.asdict(learning_rate),
        "checkpointing": dataclasses.asdict(checkpointing),
        "distributed": dataclasses.asdict(distributed),
        "validation": dataclasses.asdict(validation),
        "data": dataclasses.asdict(data),
        "moe": dataclasses.asdict(moe),
        "megatron_dir": megatron_dir,
    }

    towermoe_dir = f"{megatron_dir}/examples/towermoe"

    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(towermoe_dir),
        undefined=jinja2.StrictUndefined,
    )
    template = jinja_env.get_template("towermoe.sbatch.j2")

    run_dir = os.path.abspath(launch.run_dir)

    os.makedirs(run_dir, exist_ok=True)

    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    with open(f"{run_dir}/slurm.sbatch", "w") as f:
        f.write(template.render(cfg))

    if checkpointing.save is not None:
        print(f"Saving checkpoints to {checkpointing.save}")
        os.makedirs(checkpointing.save, exist_ok=True)

    os.system(f"sbatch {run_dir}/slurm.sbatch")

def _build_launch_cfg(launch_args: LaunchArgs) -> dict[str, Any]:
    run_dir = os.path.abspath(launch_args.run_dir)
    cfg = dataclasses.asdict(launch_args)
    cfg["output"] = f"{run_dir}/{launch_args.job_name}-%j.out"
    cfg["error"] = f"{run_dir}/{launch_args.job_name}-%j.err"
    return cfg


if __name__ == "__main__":
    jsonargparse.CLI(main, as_positional=False, parser_mode='omegaconf')
