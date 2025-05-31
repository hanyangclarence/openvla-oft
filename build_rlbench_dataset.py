from rlbench_data_util.rlbench_dataset_builder import RLBenchO1Dataset
import tensorflow_datasets as tfds

builder = tfds.builder("rl_bench_o1_dataset", data_dir="/gpfs/yanghan/openvla-oft/dataset")
builder.download_and_prepare()
ds = builder.as_dataset(split="train")