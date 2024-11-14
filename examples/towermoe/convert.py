import dataclasses
import json
import jsonargparse
import jinja2
import os

from typing import Any, Optional


@dataclasses.dataclass(frozen=True)
class CheckpointingArgs:
    megatron_ckpts_dir: str
    hf_ckpts_dir: str
    hf_tokenizer_path: str


@dataclasses.dataclass(frozen=True)
class DistributedArgs:
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1


@dataclasses.dataclass(frozen=True)
class LaunchArgs:
    job_name: str
    run_dir: str
    account: Optional[str] = None
    partition: Optional[str] = None
    qos: Optional[str] = None
    activate_env_cmd: str = ""
    # Ignored arguments (used only in training)
    cpus_per_node: Optional[int] = None
    exclusive: bool = False


def main(
    checkpointing: CheckpointingArgs,
    distributed: DistributedArgs,
    launch: LaunchArgs,
):
    megatron_dir = os.path.abspath(os.path.join(__file__, "..", "..", ".."))

    cfg = {
        "checkpointing": dataclasses.asdict(checkpointing),
        "distributed": dataclasses.asdict(distributed),
        "launch": dataclasses.asdict(launch),
        "megatron_dir": megatron_dir,
    }

    towermoe_dir = f"{megatron_dir}/examples/towermoe"

    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(towermoe_dir),
        undefined=jinja2.StrictUndefined,
    )
    template = jinja_env.get_template("convert.sh.j2")

    run_dir = os.path.abspath(launch.run_dir)

    os.makedirs(run_dir, exist_ok=True)

    with open(f"{run_dir}/convert.sh", "w") as f:
        f.write(template.render(cfg))

    print(f"Converting checkpoints to {checkpointing.hf_ckpts_dir}")
    os.makedirs(checkpointing.hf_ckpts_dir, exist_ok=True)

    os.system(f"bash {run_dir}/convert.sh")

if __name__ == "__main__":
    jsonargparse.CLI(main, as_positional=False, parser_mode='omegaconf')
