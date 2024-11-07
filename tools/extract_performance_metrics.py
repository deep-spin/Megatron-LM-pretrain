import argparse
import re

def main():
    args = parse_args()

    with open(args.train_log, 'r') as f:
        lines = f.readlines()

    time_per_iter_re = r'\|\s*elapsed time per iteration \(ms\):\s*([\d\.]+)\s*\|'
    batch_size_re = r'\|\s*global batch size:\s*(\d+)\s*\|'
    flops_re = r'\|\s*throughput per GPU \(TFLOP/s/GPU\):\s*([\d\.]+)\s*\|'
    sequence_length_re = r'\s+seq_length\s*\.{0,}\s*(\d+)'
    times_per_iter = []
    batch_sizes = set()
    flops_per_gpu = []
    sequence_lengths = []
    for line in lines:
        match_time = re.search(time_per_iter_re, line)
        if match_time:
            times_per_iter.append(float(match_time.group(1)))
        match_batch_size = re.search(batch_size_re, line)
        if match_batch_size:
            batch_sizes.add(int(match_batch_size.group(1)))
        match_flops = re.search(flops_re, line)
        if match_flops:
            flops_per_gpu.append(float(match_flops.group(1)))
        match_seq_length = re.search(sequence_length_re, line)
        if match_seq_length:
            seq_length = int(match_seq_length.group(1))
            sequence_lengths.append(seq_length)

    assert len(batch_sizes) == 1, "Different batch sizes found"
    assert len(sequence_lengths) == 1, "Different sequence lengths found"
    batch_size = batch_sizes.pop()
    sequence_length = sequence_lengths.pop()
    tokens_per_step = batch_size * sequence_length
    avg_time_per_iter = sum(times_per_iter) / len(times_per_iter)
    avg_flops_per_gpu = sum(flops_per_gpu) / len(flops_per_gpu)
    tokens_per_ms = tokens_per_step / avg_time_per_iter
    print(f"Batch size: {batch_size}")
    print(f"Average time per iteration: {avg_time_per_iter}ms")
    print(f"Average throughput per GPU: {avg_flops_per_gpu} TFLOP/s")
    print(f"Tokens per ms: {tokens_per_ms}")
    print(f"Tokens per second: {tokens_per_ms * 1000}")
    print(f"Tokens per day: {tokens_per_ms * 1000 * 60 * 60 * 24}")


def parse_args():
    parser = argparse.ArgumentParser(description='A simple script to analyse performance metrics from the training output')
    parser.add_argument("train_log", help="The training log file")
    return parser.parse_args()

if __name__ == '__main__':
    main()
