import dataclasses
import pathlib

import numpy as np
import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import autostack_policy

@dataclasses.dataclass(frozen=True)
class LeRobotAutoStackDataConfig(DataConfigFactory):
    """Configures a LeRobot AutoStack dataset for training and policy inference."""

    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # LeRobot exposes the task instruction as ``prompt`` when
        # ``prompt_from_task`` is enabled. Keep that key so training samples
        # match the dictionary accepted by AutoStackInputs at inference.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation.state": "observation.state",
                        "observation.images.top_camera": "observation.images.top_camera",
                        "observation.images.wrist_camera": "observation.images.wrist_camera",
                        "action": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[autostack_policy.AutoStackInputs(model_type=model_config.model_type.value)],
            outputs=[autostack_policy.AutoStackOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )