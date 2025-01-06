# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
import argparse
import json
import glob
import os
from dataclasses import dataclass

#from evaluate_mmmu import get_input_output_paths
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

@dataclass
class EvaluationConfig:
    """Evaluation related configuration."""
    task: str

    temperature: float = 1.0
    top_p: float = 0.0
    top_k: int = 0

    out_seq_length: int = 32

    output_path: str = ""

    input_image_path: str = ""
    gt_path: str = ""

    num_partitions: int = 1
    partition_id: int = 0
    num_samples_per_partition: int = 0


def get_output_path(config, dp_rank):
    """Generation output path."""
    return (
        f"{config.output_path}-{config.task}-dprank={dp_rank}-partition={config.partition_id}.jsonl"
    )

def get_input_output_paths(input_path, task):
    """Get all input files and an output path for a merged file."""
    # Single input file.
    if os.path.exists(input_path):
        input_file_paths = [input_path]
        output_file_path = input_path.replace(".jsonl", "-merged.json")
    # Select multiple partitions and dp ranks.
    else:
        cfg = EvaluationConfig(task=task, output_path=input_path, partition_id="*")
        pattern = get_output_path(cfg, dp_rank="*")
        input_file_paths = glob.glob(pattern)

        output_file_path = input_path + f"-{task}-merged.json"

    return input_file_paths, output_file_path


def convert_to_coco_format(input_path):
    """Convert input files to COCO compatible format."""
    input_file_paths, output_file_path = get_input_output_paths(input_path, task="captioning")

    captions = []

    for input_file_path in input_file_paths:
        with open(input_file_path, "r") as input_file:
            for line in input_file:
                res = json.loads(line)

                question_id = res['sample_id']
                caption = res['caption'].rstrip('.').lower()

                captions.append({"image_id": question_id, "caption": caption})

    with open(output_file_path, "w") as output_file:
        json.dump(captions, output_file, indent=4)

    return output_file_path


def coco_captioning_eval(input_path, groundtruth_file):
    """Run COCO captioning evaluation."""
    coco = COCO(groundtruth_file)
    input_file = convert_to_coco_format(input_path)
    coco_result = coco.loadRes(input_file)

    coco_eval = COCOEvalCap(coco, coco_result)

    # Evaluate on the input subset of images.
    coco_eval.params["image_id"] = coco_result.getImgIds()

    coco_eval.evaluate()

    print("========== COCO captioning scores ==========")
    for metric, score in coco_eval.eval.items():
        print(f"{metric} {score * 100:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True, help="Path to input file(s)")
    parser.add_argument(
        "--groundtruth-path", type=str, required=True, help="Path to groundtruth file"
    )
    args = parser.parse_args()

    coco_captioning_eval(args.input_path, args.groundtruth_path)
