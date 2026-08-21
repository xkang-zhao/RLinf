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

"""OpenPI data configuration for SpaceUR10e LeRobot v2.1 datasets."""

from __future__ import annotations

import dataclasses
import pathlib

import openpi.models.model as openpi_model
import openpi.transforms as transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import spaceur10e_policy


@dataclasses.dataclass(frozen=True)
class LeRobotSpaceUR10eDataConfig(DataConfigFactory):
    """Repack three SpaceUR10e cameras and direct 7-D delta actions."""

    default_prompt: str | None = None

    @override
    def create(
        self,
        assets_dirs: pathlib.Path,
        model_config: openpi_model.BaseModelConfig,
    ) -> DataConfig:
        repack_transforms = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "images": {
                            "cam1": "observation.images.cam1",
                            "cam2": "observation.images.cam2",
                            "cam3": "observation.images.cam3",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = transforms.Group(
            inputs=[
                spaceur10e_policy.SpaceUR10eInputs(model_type=model_config.model_type)
            ],
            outputs=[spaceur10e_policy.SpaceUR10eOutputs()],
        )
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )
