import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_auto_stack_example() -> dict:
    """Creates a random input example for the AutoStack policy."""
    return {
        "observation.state": np.random.rand(7),
        "observation.images.top_camera": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        "observation.images.wrist_camera": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        "task": "stack the block",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class AutoStackInputs(transforms.DataTransformFn):
    """Inputs for the AutoStack policy.

    Expected inputs (LeRobot format):
    - observation.images.top_camera: [channel, height, width] RGB image
    - observation.images.wrist_camera: [channel, height, width] RGB image
    - observation.state: [7] - first 7 dims are 6 joints + gripper position
    - action: [7] - delta actions (relative changes)
    - task: string - task description

    Output (model format):
    - image: dict with base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
    - image_mask: dict with True for valid cameras, False for padding
    - state: [7] robot state
    - actions: [action_horizon, 7] action sequence
    - prompt: task instruction
    """

    model_type: str

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("top_camera", "wrist_camera")

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation.images.top_camera"])
        wrist_image = _parse_image(data["observation.images.wrist_camera"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation.state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "action" in data:
            inputs["actions"] = np.asarray(data["action"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AutoStackOutputs(transforms.DataTransformFn):
    """Outputs for the AutoStack policy."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][..., :7])
        return {"actions": actions}
