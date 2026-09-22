# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the grouped Warp adapter, including optional CUDA checks."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from rlinf.envs import get_env_cls
from rlinf.envs.spaceur10e.spaceur10e_env import SpaceUR10eRLinfEnv
from rlinf.envs.spaceur10e.spaceur10e_warp_env import SpaceUR10eWarpRLinfEnv


def test_backend_selection_preserves_cpu_default():
    assert get_env_cls("spaceur10e") is SpaceUR10eRLinfEnv
    assert (
        get_env_cls("spaceur10e", {"physics_backend": "warp"}) is SpaceUR10eWarpRLinfEnv
    )


def test_deterministic_physics_config_reaches_warp_group(monkeypatch, tmp_path):
    captured = {}

    class FakeGroup:
        single_action_space = object()

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "rlinf.envs.spaceur10e.spaceur10e_warp_env.importlib.import_module",
        lambda _: SimpleNamespace(SpaceUR10eWarpGroup=FakeGroup),
    )
    env = SpaceUR10eWarpRLinfEnv.__new__(SpaceUR10eWarpRLinfEnv)
    env.cfg = {"deterministic_physics": True, "warp_substeps_per_graph": 10}
    env.video_cfg = {}
    env.ignore_terminations = False
    env._env_tasks = [SimpleNamespace(env_id="SpaceUR10e-Cube-v0")]
    env.repo_root = tmp_path
    env.obs_mode = "state"
    env._initialize_simulators(1)

    assert captured["deterministic_physics"] is True
    assert captured["physics_substeps_per_graph"] == 10


@pytest.mark.parametrize("tasks", ["cube", "all"])
def test_warp_config_composes_with_one_worker(monkeypatch, tasks):
    examples = Path(__file__).resolve().parents[2] / "examples/embodiment"
    monkeypatch.setenv("EMBODIED_PATH", str(examples))
    with initialize_config_dir(config_dir=str(examples / "config"), version_base=None):
        name = (
            "spaceur10e_cube_warp_ppo_openpi_rlinf"
            if tasks == "cube"
            else "spaceur10e_warp_ppo_openpi_rlinf"
        )
        cfg = compose(config_name=name)
    assert cfg.cluster.component_placement.env == 0
    assert cfg.env.train.total_num_envs == 32
    assert cfg.env.train.task_names == (["cube"] if tasks == "cube" else "all")
    assert cfg.actor.model.pi05
    assert cfg.actor.model.num_action_chunks == 30
    assert cfg.env.train.warp_substeps_per_graph == 10
    assert cfg.env.train.deterministic_physics is False
    assert cfg.actor.config_name == "pi05_spaceur10e"
    chunks = (
        cfg.env.train.total_num_envs
        * cfg.env.train.rollout_epoch
        * cfg.env.train.max_steps_per_rollout_epoch
        // 30
    )
    assert chunks % cfg.actor.global_batch_size == 0


def test_warp_formal_compat_config_has_expected_workload(monkeypatch):
    examples = Path(__file__).resolve().parents[2] / "examples/embodiment"
    monkeypatch.setenv("EMBODIED_PATH", str(examples))
    with initialize_config_dir(config_dir=str(examples / "config"), version_base=None):
        cfg = compose(config_name="spaceur10e_warp_ppo_formal_openpi_rlinf")

    assert cfg.runner.max_steps == 100
    assert cfg.runner.save_interval == 100
    assert cfg.env.train.deterministic_physics is False
    assert cfg.env.train.total_num_envs == 64
    assert cfg.env.train.rollout_epoch == 1
    assert cfg.env.train.max_episode_steps == 420
    assert cfg.env.train.max_steps_per_rollout_epoch == 420
    assert cfg.actor.optim.critic_warmup_steps == 20
    chunks = (
        cfg.env.train.total_num_envs
        * cfg.env.train.rollout_epoch
        * cfg.env.train.max_steps_per_rollout_epoch
        // cfg.actor.model.num_action_chunks
    )
    assert chunks == 896
    assert chunks % cfg.actor.global_batch_size == 0


@pytest.fixture
def cuda_env():
    if os.environ.get("RUN_WARP_TESTS") != "1":
        pytest.skip("Set RUN_WARP_TESTS=1 with Warp, CUDA and EGL available")
    cfg = {
        "repo_root": os.environ.get("SPACEUR10E_REPO_ROOT", "/workspace/SpaceRobotEnv"),
        "task_names": ["cube"],
        "obs_mode": "rgb",
        "physics_backend": "warp",
        "ik_backend": "gpu",
        "action_scale": [1.0] * 7,
        "auto_reset": False,
        "max_episode_steps": 5,
        "warp_substeps_per_graph": 10,
    }
    env = SpaceUR10eWarpRLinfEnv(cfg, 2, 0, 1, None)
    yield env
    env.close()


def test_cuda_rgb_partial_reset_and_terminal_chunk(cuda_env):
    env = cuda_env
    first, _ = env.reset(seed=7)
    assert first["states"].shape == (2, 27)
    assert first["main_images"].shape == (2, 480, 640, 3)
    assert first["main_images"].dtype == torch.uint8
    assert first["main_images"].max() > 0
    assert not torch.equal(first["main_images"], first["wrist_images"])
    assert first["extra_view_images"].shape == (2, 1, 480, 640, 3)
    env.step(np.zeros((2, 7)))
    before = env._last_obs
    reset, _ = env.reset(seed=99, options={"env_idx": [1]})
    for key in ("states", "main_images", "wrist_images", "extra_view_images"):
        torch.testing.assert_close(reset[key][0], before[key][0], rtol=0, atol=0)
    # Different episode lengths exercise pausing one world while the other runs.
    obs, reward, term, trunc, infos = env.chunk_step(np.zeros((2, 7, 7)))
    assert len(obs) == 1
    assert reward.shape == term.shape == trunc.shape == (2, 7)
    assert not trunc[:, :-1].any()
    assert trunc[:, -1].all()
    assert not term.any()
    assert env.elapsed_steps.tolist() == [5, 5]
    assert reward.sum() == 0
    np.testing.assert_allclose(
        env.vector_env.groups[0][1].d.time.numpy(), [0.25, 0.25], atol=1e-5
    )
    assert infos[-1]["episode"]["episode_len"].tolist() == [5, 5]
    # RGB buffers returned earlier must not be mutated by subsequent render/reset.
    assert first["main_images"].data_ptr() != obs[0]["main_images"].data_ptr()


def test_cuda_chunk_autoreset_keeps_terminal_observation(cuda_env):
    env = cuda_env
    env.auto_reset = True
    env.reset(seed=7)
    obs, _, _, trunc, infos = env.chunk_step(np.zeros((2, 6, 7)))
    assert trunc[:, -1].all()
    assert env.elapsed_steps.tolist() == [0, 0]
    assert infos[-1]["_final_observation"].all()
    assert infos[-1]["final_info"]["episode"]["episode_len"].tolist() == [5, 5]
    assert not torch.equal(obs[-1]["states"], infos[-1]["final_observation"]["states"])


def test_cuda_deterministic_rgb_reset_after_step():
    if os.environ.get("RUN_WARP_TESTS") != "1":
        pytest.skip("Set RUN_WARP_TESTS=1 with Warp, CUDA and EGL available")
    cfg = {
        "repo_root": os.environ.get("SPACEUR10E_REPO_ROOT", "/workspace/SpaceRobotEnv"),
        "task_names": ["cube"],
        "obs_mode": "rgb",
        "ik_backend": "gpu",
        "action_scale": [1.0] * 7,
        "auto_reset": False,
        "warp_substeps_per_graph": 10,
        "deterministic_rendering": True,
    }
    env = SpaceUR10eWarpRLinfEnv(cfg, 2, 0, 1, None)
    try:
        first, _ = env.reset(seed=7)
        env.step(np.zeros((2, 7)), auto_reset=False)
        second, _ = env.reset(seed=7)
        for key in ("states", "main_images", "wrist_images", "extra_view_images"):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    finally:
        env.close()


def test_cuda_all_tasks_group_order_and_partial_reset():
    if os.environ.get("RUN_WARP_TESTS") != "1":
        pytest.skip("Set RUN_WARP_TESTS=1 with Warp, CUDA and EGL available")
    cfg = {
        "repo_root": os.environ.get("SPACEUR10E_REPO_ROOT", "/workspace/SpaceRobotEnv"),
        "task_names": "all",
        "obs_mode": "rgb",
        "ik_backend": "gpu",
        "action_scale": [1.0] * 7,
        "auto_reset": False,
        "max_episode_steps": 3,
        "warp_substeps_per_graph": 10,
    }
    env = SpaceUR10eWarpRLinfEnv(cfg, 8, 0, 1, None)
    try:
        first, _ = env.reset(seed=7)
        assert len(env.vector_env.groups) == 6
        assert len(set(first["task_descriptions"])) == 8
        assert first["states"].shape == (8, 27)
        assert torch.isfinite(first["states"]).all()
        # Each scene retains the CPU reset distribution and robot geometry.
        import gymnasium as gym

        teacher = env.vector_env.get_planner_state()
        for index, task in enumerate(env._env_tasks):
            cpu = gym.make(
                task.env_id,
                observation_mode="state",
                scene_path=str(
                    env.repo_root / gym.spec(task.env_id).kwargs["scene_path"]
                ),
                arm_path=str(env.repo_root / "mjcf/arm.xml"),
            )
            try:
                cpu_obs, _ = cpu.reset(seed=7 + index)
                np.testing.assert_allclose(
                    teacher["target_pose"][index],
                    cpu.unwrapped._get_target_pose(),
                    atol=3e-7,
                )
                np.testing.assert_allclose(
                    env._raw_batch["ee_pose"][index], cpu_obs["ee_pose"], atol=3e-6
                )
            finally:
                cpu.close()
        env.step(np.zeros((8, 7)))
        before = env._last_obs
        reset, _ = env.reset(seed=99, options={"env_idx": [2]})
        others = [0, 1, 3, 4, 5, 6, 7]
        for key in ("states", "main_images", "wrist_images", "extra_view_images"):
            torch.testing.assert_close(
                reset[key][others], before[key][others], rtol=0, atol=0
            )
        # Reset one world in a shared scene without changing its sibling world.
        obs, reward, term, trunc, info = env.chunk_step(np.zeros((8, 4, 7)))
        assert trunc[:, -1].all()
        assert not term.any()
        assert reward.sum() == 0
        assert env.elapsed_steps.tolist() == [3] * 8
        assert obs[-1]["task_descriptions"] == first["task_descriptions"]
        for _, group in env.vector_env.groups:
            np.testing.assert_allclose(group.d.time.numpy(), 0.15, atol=1e-5)
    finally:
        env.close()
