import dataclasses
import json
import jsonargparse
import jinja2
import os

from typing import Any, Optional


@dataclasses.dataclass(frozen=True)
class EvalArgs:
    tasks: str
    hf_ckpt_dir: str
    outputs_dir: str
    num_fewshot: int = 5


@dataclasses.dataclass(frozen=True)
class LaunchArgs:
    job_name: str
    run_dir: str
    cpus_per_task: int
    gpus_per_task: int
    account: Optional[str] = None
    partition: Optional[str] = None
    qos: Optional[str] = None
    activate_env_cmd: str = ""
    # Ignored arguments (used only in training)
    cpus_per_node: Optional[int] = None
    exclusive: bool = False


def main(
    eval: EvalArgs,
    launch: LaunchArgs,
):
    megatron_dir = os.path.abspath(os.path.join(__file__, "..", "..", ".."))

    cfg = {
        "eval": dataclasses.asdict(eval),
        "launch": dataclasses.asdict(launch),
        "megatron_dir": megatron_dir,
    }

    towermoe_dir = f"{megatron_dir}/examples/towermoe"

    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(towermoe_dir),
        undefined=jinja2.StrictUndefined,
    )
    script_name = "eval.sh"
    template = jinja_env.get_template(f"{script_name}.j2")

    run_dir = os.path.abspath(launch.run_dir)

    os.makedirs(run_dir, exist_ok=True)

    with open(f"{run_dir}/{script_name}", "w") as f:
        f.write(template.render(cfg))

    print(f"Evaluating checkpoints {eval.hf_ckpts_dir}")

    os.system(f"bash {run_dir}/{script_name}")

if __name__ == "__main__":
    jsonargparse.CLI(main, as_positional=False, parser_mode='omegaconf')
