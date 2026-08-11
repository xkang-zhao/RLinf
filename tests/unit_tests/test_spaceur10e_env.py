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

import numpy as np
import pytest

from rlinf.envs import SupportedEnvType, get_env_cls
from rlinf.envs.action_utils import prepare_actions_for_spaceur10e
from rlinf.envs.spaceur10e import SpaceUR10eRLinfEnv
from rlinf.envs.spaceur10e.spaceur10e_env import build_state_observation


def test_spaceur10e_is_registered_lazily():
    assert SupportedEnvType.SPACEUR10E.value == "spaceur10e"
    assert get_env_cls("spaceur10e") is SpaceUR10eRLinfEnv


def test_state_observation_has_stable_21_dimensional_order():
    raw_obs = {
        "joint_pos": np.arange(6, dtype=np.float64),
        "base_pose": np.arange(10, 17, dtype=np.float64),
        "ee_pose": np.arange(20, 27, dtype=np.float64),
        "gripper_pos": np.array([0.5], dtype=np.float32),
        "target_pose": np.full(7, 999.0),
    }

    state = build_state_observation(raw_obs)

    assert state.shape == (21,)
    assert state.dtype == np.float32
    np.testing.assert_allclose(
        state,
        np.concatenate(
            [
                raw_obs["joint_pos"],
                raw_obs["base_pose"],
                raw_obs["ee_pose"],
                raw_obs["gripper_pos"],
            ]
        ),
    )
    assert 999.0 not in state


def test_state_observation_rejects_wrong_field_shape():
    raw_obs = {
        "joint_pos": np.zeros(5),
        "base_pose": np.zeros(7),
        "ee_pose": np.zeros(7),
        "gripper_pos": np.zeros(1),
    }

    with pytest.raises(ValueError, match="joint_pos"):
        build_state_observation(raw_obs)


def test_action_adapter_preserves_rotation_and_clips_all_seven_dims():
    raw_actions = np.array(
        [[[2.0, -2.0, 0.25, 0.5, -0.5, 1.5, -1.5]]],
        dtype=np.float32,
    )

    actions = prepare_actions_for_spaceur10e(raw_actions, action_dim=7)

    np.testing.assert_allclose(
        actions,
        [[[1.0, -1.0, 0.25, 0.5, -0.5, 1.0, -1.0]]],
    )


def test_action_adapter_rejects_non_seven_dimensional_actions():
    with pytest.raises(ValueError, match="seven actions"):
        prepare_actions_for_spaceur10e(
            np.zeros((1, 1, 4), dtype=np.float32),
            action_dim=4,
        )
