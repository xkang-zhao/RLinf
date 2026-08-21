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

"""Tests for the SpaceUR10e-specific OpenPI input and output transforms."""

import numpy as np
import pytest
import torch

pytest.importorskip("openpi")

from openpi.models import model as openpi_model

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.openpi.policies.spaceur10e_policy import (
    SpaceUR10eInputs,
    SpaceUR10eOutputs,
)
from rlinf.models.embodiment.openpi_rlinf.eval_action_model import (
    OpenPiPytorchEvalActionModel,
)
from rlinf.models.embodiment.openpi_rlinf.rl_action_model import (
    OpenPiPytorchRLActionModel,
    OpenPiPytorchRLConfig,
)
from rlinf.utils.ckpt_convertor.openpi.pt_to_safetensors import _drop_ppo_only_keys


def test_spaceur10e_inputs_map_all_collector_cameras_and_state():
    transform = SpaceUR10eInputs(model_type=openpi_model.ModelType.PI0)
    output = transform(
        {
            "images": {
                "cam1": np.full((2, 3, 3), 1, dtype=np.uint8),
                "cam2": np.full((2, 3, 3), 2, dtype=np.uint8),
                "cam3": np.full((2, 3, 3), 3, dtype=np.uint8),
            },
            "state": np.arange(13, dtype=np.float32),
            "actions": np.zeros((10, 7), dtype=np.float32),
            "prompt": "Grasp the red cube.",
        }
    )

    np.testing.assert_array_equal(output["state"], np.arange(13, dtype=np.float32))
    assert output["actions"].shape == (10, 7)
    assert output["prompt"] == "Grasp the red cube."
    assert output["image"]["base_0_rgb"][0, 0, 0] == 1
    assert output["image"]["left_wrist_0_rgb"][0, 0, 0] == 3
    assert output["image"]["right_wrist_0_rgb"][0, 0, 0] == 2


def test_spaceur10e_inputs_accept_online_chw_images_and_reject_bad_state():
    transform = SpaceUR10eInputs(model_type=openpi_model.ModelType.PI0)
    online_data = {
        "observation/image": np.ones((3, 2, 4), dtype=np.float32),
        "observation/wrist_image": np.full((3, 2, 4), 0.5, dtype=np.float32),
        "observation/extra_view_image": np.zeros((3, 2, 4), dtype=np.float32),
        "observation/state": np.arange(13, dtype=np.float32),
        "prompt": "Grasp the object.",
    }
    output = transform(online_data)
    assert output["image"]["base_0_rgb"].shape == (2, 4, 3)
    assert output["image"]["base_0_rgb"].dtype == np.uint8

    online_data["observation/state"] = np.zeros(12, dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        transform(online_data)


def test_spaceur10e_outputs_remove_openpi_action_padding():
    output = SpaceUR10eOutputs()({"actions": np.zeros((10, 32), dtype=np.float32)})
    assert output["actions"].shape == (10, 7)


def test_spaceur10e_eval_selects_collector_state_from_rlinf_state():
    eval_model = object.__new__(OpenPiPytorchEvalActionModel)
    eval_model.state_indices = [0, 1, 2, 3, 4, 5, 13, 14, 15, 16, 17, 18, 19]

    states = np.arange(54, dtype=np.float32).reshape(2, 27)
    selected = eval_model._select_configured_state(states)

    assert selected.shape == (2, 13)
    np.testing.assert_array_equal(selected[0], np.r_[0:6, 13:20])


def test_spaceur10e_rl_model_keeps_configured_online_state_indices():
    """RL rollout must apply the same 27-D → 13-D selection as eval."""
    pi0 = torch.nn.Linear(1, 1)
    pi0.action_horizon = 10
    pi0.action_dim = 32
    indices = [0, 1, 2, 3, 4, 5, 13, 14, 15, 16, 17, 18, 19]
    model = OpenPiPytorchRLActionModel(
        pi0,
        num_steps=5,
        action_chunk=10,
        action_env_dim=7,
        rl_cfg=OpenPiPytorchRLConfig(config_name="pi0_spaceur10e"),
        paligemma_width=2,
        state_indices=indices,
    )

    states = np.arange(54, dtype=np.float32).reshape(2, 27)
    np.testing.assert_array_equal(
        model._select_configured_state(states)[0], np.r_[0:6, 13:20]
    )


def test_spaceur10e_rl_model_dispatches_sft_co_training_loss():
    pi0 = torch.nn.Linear(1, 1)
    pi0.action_horizon = 10
    pi0.action_dim = 32
    model = OpenPiPytorchRLActionModel(
        pi0,
        num_steps=5,
        action_chunk=10,
        action_env_dim=7,
        rl_cfg=OpenPiPytorchRLConfig(config_name="pi0_spaceur10e"),
        paligemma_width=2,
    )
    expected = torch.tensor(0.25)
    model.sft_forward = lambda **kwargs: expected

    assert model(forward_type=ForwardType.SFT, data=None) is expected


def test_ppo_checkpoint_export_drops_critic_only_value_head():
    state_dict = {
        "model.img.stem.weight": torch.ones(1),
        "value_head.layers.0.weight": torch.ones(1),
    }

    deploy_state, dropped = _drop_ppo_only_keys(state_dict)

    assert set(deploy_state) == {"model.img.stem.weight"}
    assert dropped == ["value_head.layers.0.weight"]
