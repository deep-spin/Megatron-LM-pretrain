import argparse

from megatron.energon import get_train_dataset, get_loader, WorkerConfig

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = read_args()
    simple_worker_config = WorkerConfig(rank=0, world_size=1, num_workers=1)

    train_ds = get_train_dataset(
        args.dataset_path,
        batch_size=1,
        shuffle_buffer_size=None,
        max_samples_per_sequence=None,
        worker_config=simple_worker_config,
    )

    # print number of samples
    print(f"Number of samples: {len(train_ds)}")