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

"""OpenPI transforms for SpaceUR10e's three-camera demonstration data."""

from __future__ import annotations

import dataclasses
from typing import Any

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as openpi_model

STATE_DIM = 13
ACTION_DIM = 7


def _to_hwc_uint8(image: Any) -> np.ndarray:
    """Convert a decoded LeRobot or online RGB image to HWC uint8."""
    array = np.asarray(image)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(
                f"SpaceUR10e expects one image per camera, got shape {array.shape}."
            )
        array = array[0]
    if array.ndim != 3:
        raise ValueError(
            f"SpaceUR10e camera must be a 3-D RGB image, got {array.shape}."
        )
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = array.astype(np.uint8)
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = einops.rearrange(array, "c h w -> h w c")
    if array.shape[-1] != 3:
        raise ValueError(
            f"SpaceUR10e camera must have three channels, got shape {array.shape}."
        )
    return np.ascontiguousarray(array)


def _state_vector(state: Any) -> np.ndarray:
    """Validate the collector's joint-position plus end-effector state."""
    array = np.asarray(state, dtype=np.float32).reshape(-1)
    if array.shape != (STATE_DIM,):
        raise ValueError(
            "SpaceUR10e OpenPI expects observation.state with shape "
            f"({STATE_DIM},), got {array.shape}."
        )
    return array


def _dataset_images(data: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Read camera inputs from either the offline or online representation."""
    if "images" in data:
        images = data["images"]
        return images["cam1"], images["cam3"], images["cam2"]
    return (
        data["observation/image"],
        data["observation/wrist_image"],
        data["observation/extra_view_image"],
    )


@dataclasses.dataclass(frozen=True)
class SpaceUR10eInputs(transforms.DataTransformFn):
    """Map SpaceUR10e data to Pi0's three RGB slots and 13-D state."""

    model_type: openpi_model.ModelType

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        main_image, wrist_image, extra_image = _dataset_images(data)
        inputs: dict[str, Any] = {
            "state": _state_vector(
                data["state"] if "state" in data else data["observation/state"]
            ),
            "image": {
                "base_0_rgb": _to_hwc_uint8(main_image),
                "left_wrist_0_rgb": _to_hwc_uint8(wrist_image),
                "right_wrist_0_rgb": _to_hwc_uint8(extra_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        elif "task" in data:
            inputs["prompt"] = data["task"]
        return inputs


@dataclasses.dataclass(frozen=True)
class SpaceUR10eOutputs(transforms.DataTransformFn):
    """Map Pi0's padded action output back to SpaceUR10e's seven controls."""

    def __call__(self, data: dict[str, Any]) -> dict[str, np.ndarray]:
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}
