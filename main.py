# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

"""
Main entry point for model training. Please refer to README.md for usage instructions.
"""

import logging
import os

from typing import List, Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide excessive tensorflow debug messages
import sys

import fbgemm_gpu  # noqa: F401, E402
import gin

import time
import torch
import torch.multiprocessing as mp

from absl import app, flags
from generative_recommenders.trainer.train import train_fn
from generative_recommenders.data.reco_dataset import get_reco_dataset, RecoDataset

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def delete_flags(FLAGS, keys_to_delete: List[str]) -> None:  # pyre-ignore [2]
    keys = [key for key in FLAGS._flags()]
    for key in keys:
        if key in keys_to_delete:
            delattr(FLAGS, key)


delete_flags(flags.FLAGS, ["gin_config_file", "master_port"])
flags.DEFINE_string("gin_config_file", None, "Path to the config file.")
flags.DEFINE_integer("master_port", 12355, "Master port.")
FLAGS = flags.FLAGS  # pyre-ignore [5]


def mp_train_fn(
    rank: int,
    world_size: int,
    master_port: int,
    gin_config_file: Optional[str],
    filename: str,
) -> None:
    if gin_config_file is not None:
        # Hack as absl doesn't support flag parsing inside multiprocessing.
        logging.info(f"Rank {rank}: loading gin config from {gin_config_file}")
        gin.parse_config_file(gin_config_file)

    load_begin = time.time()
    with open(filename, 'rb') as f:
        memory_dict = torch.load(f)
        train_cache = memory_dict['train_cache']
        eval_cache = memory_dict['eval_cache']
    logging.info(f'It takes {time.time() - load_begin}s to load the processed dataset file')
    
    train_fn(rank, world_size, master_port, train_cache, eval_cache)


def _main(argv) -> None:  # pyre-ignore [2]
    world_size = torch.cuda.device_count()
    mp.set_start_method("forkserver")

    gin_config_file = FLAGS.gin_config_file
    if gin_config_file is not None:
        logging.info(f"Main: loading gin config from {gin_config_file}")
        gin.parse_config_file(gin_config_file)

    dataset_name = gin.query_parameter('train_fn.dataset_name')
    max_sequence_length = gin.query_parameter('train_fn.max_sequence_length')
    positional_sampling_ratio = 1.0 

    dataset_id = f'{dataset_name}-l{max_sequence_length}'
    cache_file_path = f'tmp/loaded/{dataset_id}.pt'
    
    if not os.path.exists(cache_file_path) :
        os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
        
        dataset = get_reco_dataset(
            dataset_name=dataset_name,
            max_sequence_length=max_sequence_length,
            chronological=True,
            positional_sampling_ratio=positional_sampling_ratio,
        )
        
        save_begin = time.time()
        # 保存数据到文件
        with open(cache_file_path, 'wb') as f:
            torch.save({
                'train_cache': dataset.train_dataset.get_cache(),
                'eval_cache': dataset.eval_dataset.get_cache(),
            }, f)
        logging.info(f'It takes {time.time() - save_begin}s to save the processed dataset file')
    else :
        logging.info(f'Detected cached file {cache_file_path}')
    
    mp.spawn(
        mp_train_fn,
        args=(world_size, FLAGS.master_port, gin_config_file, cache_file_path),
        nprocs=world_size,
        join=True,
    )


def main() -> None:
    app.run(_main)


if __name__ == "__main__":
    main()
