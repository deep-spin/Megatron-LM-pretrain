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
    output_dir: str
    num_fewshot: int = 5


@dataclasses.dataclass(frozen=True)
class LaunchArgs:
    cpus_per_task: int = 8
    gpus_per_task: int = 1
    account: Optional[str] = None
    partition: Optional[str] = None
    qos: Optional[str] = None
    activate_env_cmd: str = ""
    # Ignored arguments (used only in training)
    cpus_per_node: Optional[int] = None
    exclusive: bool = False


def main(eval: EvalArgs, launch: LaunchArgs):
    megatron_dir = os.path.abspath(os.path.join(__file__, "..", "..", ".."))

    cfg = {
        "eval": dataclasses.asdict(eval),
        "launch": dataclasses.asdict(launch),
        "megatron_dir": megatron_dir,
    }

    templates_dir = f"{megatron_dir}/examples/towermoe/templates"

    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(templates_dir),
        undefined=jinja2.StrictUndefined,
    )
    script_name = "eval_ckpt.sh"
    template = jinja_env.get_template(f"{script_name}.j2")

    output_dir = os.path.abspath(eval.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/{script_name}", "w") as f:
        f.write(template.render(cfg))

    print(f"Evaluating checkpoints {eval.hf_ckpt_dir}")

    os.system(f"bash {output_dir}/{script_name}")

if __name__ == "__main__":
    jsonargparse.CLI(main, as_positional=False, parser_mode='omegaconf')
