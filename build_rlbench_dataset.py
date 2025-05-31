from rlbench_data_util.rlbench_dataset_builder import RLBenchO1Dataset
import tensorflow_datasets as tfds

builder = tfds.builder("rl_bench_o1_dataset", data_dir="/gpfs/yanghan/openvla-mini-o1/dataset")
builder.download_and_prepare()
ds = builder.as_dataset(split="train")