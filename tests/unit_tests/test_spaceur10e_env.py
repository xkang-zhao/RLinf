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
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from rlinf.envs import SupportedEnvType, get_env_cls
from rlinf.envs.action_utils import prepare_actions_for_spaceur10e
from rlinf.envs.spaceur10e import SpaceUR10eRLinfEnv
from rlinf.envs.spaceur10e import spaceur10e_env as spaceur10e_module
from rlinf.envs.spaceur10e.spaceur10e_env import build_state_observation


def test_spaceur10e_is_registered_lazily():
    assert SupportedEnvType.SPACEUR10E.value == "spaceur10e"
    assert get_env_cls("spaceur10e") is SpaceUR10eRLinfEnv


def test_source_checkout_fallback_requires_fix0613_package_layout(tmp_path):
    checkout = tmp_path / "my_simulation"
    (checkout / "src" / "envs").mkdir(parents=True)
    (checkout / "mjcf").mkdir()

    assert SpaceUR10eRLinfEnv._resolve_repo_root(str(checkout)) == Path(checkout)


def test_state_observation_has_stable_27_dimensional_order():
    raw_obs = {
        "joint_pos": np.arange(6, dtype=np.float64),
        "base_pose": np.arange(10, 17, dtype=np.float64),
        "ee_pose": np.arange(20, 27, dtype=np.float64),
        "target_pose": np.arange(30, 37, dtype=np.float64),
    }

    state = build_state_observation(raw_obs)

    assert state.shape == (27,)
    assert state.dtype == np.float32
    np.testing.assert_allclose(
        state,
        np.concatenate(
            [
                raw_obs["joint_pos"],
                raw_obs["base_pose"],
                raw_obs["ee_pose"],
                raw_obs["target_pose"],
            ]
        ),
    )


def test_state_observation_rejects_wrong_field_shape():
    raw_obs = {
        "joint_pos": np.zeros(5),
        "base_pose": np.zeros(7),
        "ee_pose": np.zeros(7),
        "target_pose": np.zeros(7),
    }

    with pytest.raises(ValueError, match="joint_pos"):
        build_state_observation(raw_obs)


def test_rgb_adapter_cycles_task_definitions_and_exposes_three_cameras(
    monkeypatch, tmp_path
):
    checkout = tmp_path / "my_simulation"
    (checkout / "src" / "envs").mkdir(parents=True)
    (checkout / "mjcf").mkdir()

    tasks_module = ModuleType("planners.auto_grasp.tasks")
    tasks_module.TASK_NAMES = ("cube", "satellite_handle")
    task_specs = {
        "cube": SimpleNamespace(
            name="cube",
            env_id="Fake-Cube-v0",
            instruction="Grab the red cube",
        ),
        "satellite_handle": SimpleNamespace(
            name="satellite_handle",
            env_id="Fake-Satellite-v0",
            instruction="Grab the satellite handle",
        ),
    }
    tasks_module.get_task_spec = task_specs.__getitem__
    planners_module = ModuleType("planners")
    auto_grasp_module = ModuleType("planners.auto_grasp")
    envs_module = ModuleType("envs")
    envs_module.__file__ = str(checkout / "src" / "envs" / "__init__.py")
    monkeypatch.setitem(__import__("sys").modules, "envs", envs_module)
    monkeypatch.setitem(__import__("sys").modules, "planners", planners_module)
    monkeypatch.setitem(
        __import__("sys").modules, "planners.auto_grasp", auto_grasp_module
    )
    monkeypatch.setitem(
        __import__("sys").modules, "planners.auto_grasp.tasks", tasks_module
    )

    class FakeEnv:
        instances = []
        action_space = spaceur10e_module.gym.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

        def __init__(self):
            self.reset_seeds = []
            self.next_success = False
            self.instances.append(self)

        @staticmethod
        def _obs():
            return {
                "joint_pos": np.zeros(6),
                "base_pose": np.zeros(7),
                "ee_pose": np.zeros(7),
                "target_pose": np.zeros(7),
                "third_left_camera": np.zeros((2, 3, 3), dtype=np.uint8),
                "left_wrist_camera": np.ones((2, 3, 3), dtype=np.uint8),
                "third_right_camera": np.full((2, 3, 3), 2, dtype=np.uint8),
            }

        def reset(self, seed=None, options=None):
            self.reset_seeds.append(seed)
            return self._obs(), {}

        def step(self, action):
            return (
                self._obs(),
                42.0,
                self.next_success,
                False,
                {"is_success": self.next_success},
            )

        def close(self):
            pass

    monkeypatch.setattr(
        spaceur10e_module.gym,
        "spec",
        lambda env_id: SimpleNamespace(kwargs={"scene_path": "./mjcf/scene.xml"}),
    )
    monkeypatch.setattr(
        spaceur10e_module.gym,
        "make",
        lambda *args, **kwargs: FakeEnv(),
    )

    env = SpaceUR10eRLinfEnv(
        {
            "repo_root": str(checkout),
            "task_names": "all",
            "obs_mode": "rgb",
        },
        num_envs=3,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    obs, _ = env.reset()

    assert env.instruction == [
        "Grab the red cube",
        "Grab the satellite handle",
        "Grab the red cube",
    ]
    assert obs["states"].shape == (3, 27)
    assert obs["main_images"].shape == (3, 2, 3, 3)
    assert obs["wrist_images"].shape == (3, 2, 3, 3)
    assert obs["extra_view_images"].shape == (3, 1, 2, 3, 3)
    assert obs["task_descriptions"] == env.instruction
    _, rewards, _, _, infos = env.step(np.zeros((3, 7), dtype=np.float32))
    np.testing.assert_array_equal(rewards.cpu(), [0.0, 0.0, 0.0])
    assert infos["task_name"] == ["cube", "satellite_handle", "cube"]

    env.envs[0].next_success = True
    _, rewards, _, _, _ = env.step(np.zeros((3, 7), dtype=np.float32))
    np.testing.assert_array_equal(rewards.cpu(), [1.0, 0.0, 0.0])
    env.reset()
    assert [fake_env.reset_seeds for fake_env in FakeEnv.instances] == [
        [0, 3, 6],
        [1, 4],
        [2, 5],
    ]
    env.close()

    offset_env = SpaceUR10eRLinfEnv(
        {
            "repo_root": str(checkout),
            "task_names": "all",
            "obs_mode": "state",
        },
        num_envs=1,
        seed_offset=1,
        total_num_processes=2,
        worker_info=None,
    )
    assert offset_env.instruction == ["Grab the satellite handle"]
    offset_env.close()


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
