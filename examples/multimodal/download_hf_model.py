import argparse

from transformers import AutoModelForCausalLM

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()

def main():
    args = read_args()
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
