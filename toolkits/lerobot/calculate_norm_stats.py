# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import numpy as np
import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms
import tqdm
import tyro
from openpi.training.config import DataConfig

from rlinf.data.storage.lerobot import resolve_lerobot_dataset_root
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {
            k: v
            for k, v in x.items()
            if not np.issubdtype(np.asarray(v).dtype, np.str_)
        }


def _calculate_numeric_lerobot_stats(
    dataset_root,
    output_path,
) -> None:
    """Compute state/action statistics directly from LeRobot v2.1 Parquet.

    OpenPI's transformed loader must decode video frames even when only numeric
    normalization statistics are requested. SpaceUR10e stores state and action
    in ordinary Parquet columns, so this path is exact for those features and
    avoids decoding its three MP4 streams.
    """
    import pyarrow.parquet as pq

    stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }
    parquet_files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No LeRobot Parquet files found under {dataset_root}.")

    for parquet_path in tqdm.tqdm(parquet_files, desc="Computing numeric stats"):
        table = pq.read_table(parquet_path, columns=["observation.state", "action"])
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        stats["state"].update(state)
        stats["actions"].update(actions)

    normalize.save(
        output_path,
        {key: value.get_statistics() for key, value in stats.items()},
    )


def _norm_stats_output_path(config, data_config: DataConfig) -> Path:
    """Resolve a stats path without treating an absolute repo id as a child path."""
    assets = config.data.assets
    assets_dir = getattr(assets, "assets_dir", None)
    asset_id = getattr(assets, "asset_id", None)
    if assets_dir:
        return Path(assets_dir) / (asset_id or Path(data_config.repo_id).name)
    return Path(config.assets_dirs) / Path(data_config.repo_id).name


def create_torch_dataloader(
    data_config: DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.TorchDataLoader, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(
        data_config, action_horizon, model_config
    )
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(
        data_config, action_horizon, batch_size, shuffle=False
    )
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    repo_id: str,
    model_path: str | None = None,
    numeric_only: bool = False,
):
    """Calculate OpenPI normalization statistics for a local LeRobot dataset.

    Args:
        config_name: Registered OpenPI data configuration name.
        repo_id: Local LeRobot dataset directory or a cached repo id.
        model_path: Optional Pi0 checkpoint directory that should receive the
            generated statistics. When omitted, the registered config assets
            directory is used.
        numeric_only: Read only LeRobot v2.1 numeric Parquet features instead
            of decoding video frames through the transformed OpenPI loader.
    """
    dataset_root = resolve_lerobot_dataset_root(repo_id)
    if not (dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"LeRobot dataset not found for repo_id={repo_id!r} at {dataset_root}. "
            "Pass a local dataset path, a Hugging Face repo id with data under "
            "HF_LEROBOT_HOME (default: ~/.cache/huggingface/lerobot), or download "
            "the dataset first."
        )
    config = get_openpi_config(
        config_name,
        repo_id=repo_id,
        model_path=model_path,
    )
    data_config = config.data.create(config.assets_dirs, config.model)

    output_path = _norm_stats_output_path(config, data_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if numeric_only:
        _calculate_numeric_lerobot_stats(dataset_root, output_path)
        print(f"Writing stats to: {output_path}")
        return

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
