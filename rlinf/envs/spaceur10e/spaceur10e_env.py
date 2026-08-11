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

from __future__ import annotations

import copy
import importlib
import os
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

__all__ = ["SpaceUR10eRLinfEnv"]


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read a value from either a mapping or attribute-style config."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_state_observation(raw_obs: dict[str, Any]) -> np.ndarray:
    """Build the 21-D policy state in its stable, documented order."""
    expected_shapes = {
        "joint_pos": (6,),
        "base_pose": (7,),
        "ee_pose": (7,),
        "gripper_pos": (1,),
    }
    values = []
    for key, shape in expected_shapes.items():
        value = np.asarray(raw_obs[key], dtype=np.float32)
        if value.shape != shape:
            raise ValueError(
                f"SpaceUR10e observation {key!r} must have shape {shape}, "
                f"got {value.shape}."
            )
        values.append(value)
    return np.concatenate(values, axis=0, dtype=np.float32)


class SpaceUR10eRLinfEnv(gym.Env):
    """Vector-style RLinf adapter around independent SpaceUR10e Gym envs."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: Any,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
        record_metrics: bool = True,
    ):
        """Initialize a CPU batch of independent SpaceUR10e environments."""
        super().__init__()
        self.cfg = cfg
        self.seed = int(_cfg_get(cfg, "seed", 0)) + int(seed_offset)
        self.seed_offset = int(seed_offset)
        self.total_num_processes = int(total_num_processes)
        self.worker_info = worker_info
        self.group_size = int(_cfg_get(cfg, "group_size", 1))
        if num_envs < 1:
            raise ValueError("SpaceUR10eRLinfEnv requires num_envs >= 1.")
        if self.group_size < 1 or int(num_envs) % self.group_size != 0:
            raise ValueError(
                "SpaceUR10e num_envs must be divisible by a positive group_size."
            )
        self.num_group = int(num_envs) // self.group_size
        self.auto_reset = bool(_cfg_get(cfg, "auto_reset", True))
        self.ignore_terminations = bool(
            _cfg_get(cfg, "ignore_terminations", False)
        )
        self.use_rel_reward = bool(_cfg_get(cfg, "use_rel_reward", False))
        self.reward_coef = float(_cfg_get(cfg, "reward_coef", 1.0))
        self.max_episode_steps = int(
            _cfg_get(cfg, "max_episode_steps", 400)
        )
        self.task_prompt = str(
            _cfg_get(cfg, "task_prompt", "Grab the red cube in space.")
        )
        self.record_metrics = bool(record_metrics)
        self.video_cfg = _cfg_get(cfg, "video_cfg", None)
        self._device = torch.device("cpu")
        self._is_start = True

        action_scale = np.asarray(
            _cfg_get(
                cfg,
                "action_scale",
                [0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 1.0],
            ),
            dtype=np.float32,
        )
        if action_scale.shape != (7,) or np.any(action_scale <= 0):
            raise ValueError(
                "SpaceUR10e action_scale must contain seven positive values."
            )
        self.action_scale = action_scale

        self.repo_root = self._resolve_repo_root(
            _cfg_get(cfg, "repo_root", None)
        )
        self.env_id = str(
            _cfg_get(cfg, "gym_id", "SpaceUR10e-Cube-v0")
        )
        self._register_spaceur10e_package()
        self.envs = [self._make_env() for _ in range(int(num_envs))]
        self.single_action_space = self.envs[0].action_space
        self.action_space = self.single_action_space
        self.observation_space = gym.spaces.Dict(
            {
                "states": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_envs, 21),
                    dtype=np.float32,
                )
            }
        )

        self.prev_step_reward = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._elapsed_steps = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self._needs_reset = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._last_obs: dict[str, Any] | None = None
        self.reset_state_ids = None
        self.info_logging_keys = ["success", "fail"]
        self._init_metrics()

    @staticmethod
    def _resolve_repo_root(configured_root: str | None) -> Path:
        candidates = []
        if configured_root:
            candidates.append(Path(configured_root).expanduser())
        if os.environ.get("SPACEUR10E_REPO_ROOT"):
            candidates.append(
                Path(os.environ["SPACEUR10E_REPO_ROOT"]).expanduser()
            )
        candidates.append(Path(__file__).resolve().parents[4])
        for candidate in candidates:
            resolved = candidate.resolve()
            if (resolved / "src" / "envs").is_dir() and (
                resolved / "mjcf"
            ).is_dir():
                return resolved
        raise FileNotFoundError(
            "Cannot locate the SpaceUR10e repository. Set cfg.repo_root or "
            "SPACEUR10E_REPO_ROOT to its absolute path."
        )

    def _register_spaceur10e_package(self) -> None:
        src_root = str(self.repo_root / "src")
        if src_root not in sys.path:
            sys.path.insert(0, src_root)
        importlib.import_module("envs")

    def _absolute_asset_path(self, path: str) -> str:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.repo_root / str(candidate).removeprefix("./")
        return str(candidate.resolve())

    def _make_env(self) -> gym.Env:
        spec = gym.spec(self.env_id)
        spec_kwargs = spec.kwargs or {}
        scene_path = self._absolute_asset_path(spec_kwargs["scene_path"])
        arm_path = self._absolute_asset_path(
            spec_kwargs.get("arm_path", "./mjcf/arm.xml")
        )
        kwargs: dict[str, Any] = {}
        target_init_range = _cfg_get(self.cfg, "target_init_range", None)
        if target_init_range is not None:
            kwargs["target_init_range"] = {
                axis: tuple(target_init_range[axis])
                for axis in ("x", "y", "z")
            }
        return gym.make(
            self.env_id,
            disable_env_checker=True,
            render_mode=None,
            observation_mode="state",
            scene_path=scene_path,
            arm_path=arm_path,
            use_depth=False,
            **kwargs,
        )

    @property
    def total_num_group_envs(self) -> int:
        """Return a compatibility upper bound for reset-state groups."""
        return np.iinfo(np.uint8).max // 2

    @property
    def num_envs(self) -> int:
        """Return the number of independent environments in this process."""
        return len(self.envs)

    @property
    def device(self) -> torch.device:
        """Return the device used for RLinf-facing tensors."""
        return self._device

    @property
    def elapsed_steps(self) -> torch.Tensor:
        """Return per-environment episode lengths."""
        return self._elapsed_steps

    @property
    def is_start(self) -> bool:
        """Return whether the worker requested a fresh rollout."""
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = bool(value)

    @property
    def instruction(self) -> list[str]:
        """Return one language instruction per environment."""
        return [self.task_prompt] * self.num_envs

    def update_reset_state_ids(self) -> None:
        """Keep API compatibility with RLinf reset-state scheduling."""
        self.reset_state_ids = None

    def _wrap_obs(self, raw_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        state = torch.from_numpy(build_state_observation(raw_obs)).to(self.device)
        return {"states": state}

    def _collate_obs(
        self, observations: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        return {
            "states": torch.stack(
                [observation["states"] for observation in observations], dim=0
            )
        }

    def _collate_infos(self, info_list: list[dict[str, Any]]) -> dict[str, Any]:
        keys = set().union(*(info.keys() for info in info_list))
        infos: dict[str, Any] = {}
        for key in sorted(keys):
            values = [info.get(key) for info in info_list]
            if all(
                value is None or isinstance(value, (bool, np.bool_))
                for value in values
            ):
                infos[key] = torch.tensor(
                    [bool(value) for value in values],
                    dtype=torch.bool,
                    device=self.device,
                )
            elif all(
                value is None or isinstance(value, (int, float, np.number))
                for value in values
            ):
                infos[key] = torch.tensor(
                    [0.0 if value is None else float(value) for value in values],
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                infos[key] = values
        return infos

    def _init_metrics(self) -> None:
        self.success_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.fail_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.returns = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

    def _reset_metrics(self, env_idx: torch.Tensor | None = None) -> None:
        if env_idx is None:
            mask = torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        else:
            mask = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            mask[env_idx] = True
        self.prev_step_reward[mask] = 0.0
        self._elapsed_steps[mask] = 0
        self.success_once[mask] = False
        self.fail_once[mask] = False
        self.returns[mask] = 0.0

    def _record_metrics(
        self, reward: torch.Tensor, infos: dict[str, Any]
    ) -> dict[str, Any]:
        if not self.record_metrics:
            return infos
        self.returns += reward
        self.success_once |= infos["success"].bool()
        self.fail_once |= infos["fail"].bool()
        episode_len = self.elapsed_steps.clone()
        infos["episode"] = {
            "success_once": self.success_once.clone(),
            "fail_once": self.fail_once.clone(),
            "return": self.returns.clone(),
            "episode_len": episode_len,
            "reward": self.returns
            / torch.clamp(episode_len.float(), min=1.0),
        }
        return infos

    def _index_cached_obs(self, env_idx: int) -> dict[str, torch.Tensor]:
        if self._last_obs is None:
            raw_obs, _ = self.envs[env_idx].reset(seed=self.seed + env_idx)
            return self._wrap_obs(raw_obs)
        return {"states": self._last_obs["states"][env_idx]}

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Reset all or selected environments and return a full observation batch."""
        options = {} if options is None else dict(options)
        env_idx = options.pop("env_idx", None)
        if env_idx is None:
            indices = list(range(self.num_envs))
            metric_indices = None
            self._needs_reset[:] = False
        else:
            metric_indices = torch.as_tensor(
                env_idx, dtype=torch.int64, device=self.device
            ).reshape(-1)
            indices = metric_indices.cpu().tolist()
            self._needs_reset[metric_indices] = False
        self._reset_metrics(metric_indices)

        observations = []
        info_list = []
        for index, env in enumerate(self.envs):
            if index not in indices:
                observations.append(self._index_cached_obs(index))
                info_list.append({})
                continue
            if isinstance(seed, (list, tuple, np.ndarray)):
                env_seed = int(seed[index])
            elif seed is None:
                env_seed = self.seed + index
            else:
                env_seed = int(seed) + index
            raw_obs, info = env.reset(seed=env_seed, options=options)
            observations.append(self._wrap_obs(raw_obs))
            info_list.append(info if isinstance(info, dict) else {})

        obs = self._collate_obs(observations)
        infos = self._collate_infos(info_list)
        self._last_obs = obs
        self._is_start = False
        return obs, infos

    def _normalize_actions(
        self, actions: np.ndarray | torch.Tensor
    ) -> np.ndarray:
        values = (
            actions.detach().cpu().numpy()
            if isinstance(actions, torch.Tensor)
            else np.asarray(actions)
        )
        if values.ndim == 1:
            values = np.repeat(values[None, :], self.num_envs, axis=0)
        if values.shape != (self.num_envs, 7):
            raise ValueError(
                "SpaceUR10e actions must have shape "
                f"({self.num_envs}, 7), got {values.shape}."
            )
        return np.clip(values, -1.0, 1.0).astype(np.float32, copy=False)

    def step(
        self,
        actions: np.ndarray | torch.Tensor,
        auto_reset: bool = True,
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Apply one normalized action per environment."""
        normalized_actions = self._normalize_actions(actions)
        observations = []
        info_list = []
        rewards = []
        terminations = []
        truncations = []

        for index, env in enumerate(self.envs):
            if self._needs_reset[index]:
                observations.append(self._index_cached_obs(index))
                info_list.append({"success": False, "fail": False})
                rewards.append(0.0)
                terminations.append(True)
                truncations.append(False)
                continue

            env_action = normalized_actions[index] * self.action_scale
            raw_obs, reward, terminated, truncated, info = env.step(env_action)
            self._elapsed_steps[index] += 1
            truncated = bool(truncated) or (
                self._elapsed_steps[index] >= self.max_episode_steps
            )
            info = dict(info) if isinstance(info, dict) else {}
            success = bool(info.get("is_success", False))
            info["success"] = success
            info["fail"] = bool(truncated and not success)
            done = bool(terminated or truncated)
            self._needs_reset[index] = done

            observations.append(self._wrap_obs(raw_obs))
            info_list.append(info)
            rewards.append(float(reward))
            terminations.append(bool(terminated))
            truncations.append(bool(truncated))

        obs = self._collate_obs(observations)
        raw_reward = torch.tensor(
            rewards, dtype=torch.float32, device=self.device
        )
        reward = (
            raw_reward - self.prev_step_reward
            if self.use_rel_reward
            else raw_reward
        )
        self.prev_step_reward = raw_reward
        reward *= self.reward_coef
        termination_tensor = torch.tensor(
            terminations, dtype=torch.bool, device=self.device
        )
        truncation_tensor = torch.tensor(
            truncations, dtype=torch.bool, device=self.device
        )
        infos = self._record_metrics(reward, self._collate_infos(info_list))

        if self.ignore_terminations:
            if "episode" in infos:
                infos["episode"]["success_at_end"] = infos[
                    "success"
                ].clone()
            termination_tensor[:] = False

        dones = termination_tensor | truncation_tensor
        if dones.any() and auto_reset and self.auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)

        self._last_obs = obs
        self._is_start = False
        return obs, reward, termination_tensor, truncation_tensor, infos

    def _handle_auto_reset(
        self,
        dones: torch.Tensor,
        final_obs: dict[str, torch.Tensor],
        infos: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        saved_obs = copy.deepcopy(final_obs)
        saved_info = copy.deepcopy(infos)
        env_idx = torch.arange(self.num_envs, device=self.device)[dones]
        obs, reset_infos = self.reset(options={"env_idx": env_idx})
        reset_infos["final_observation"] = saved_obs
        reset_infos["final_info"] = saved_info
        reset_infos["_final_info"] = dones
        reset_infos["_final_observation"] = dones
        reset_infos["_elapsed_steps"] = dones
        return obs, reset_infos

    def chunk_step(
        self, chunk_actions: np.ndarray | torch.Tensor
    ) -> tuple[
        list[dict[str, torch.Tensor]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[dict[str, Any]],
    ]:
        """Execute actions with shape ``[num_envs, chunk_steps, 7]``."""
        actions = (
            chunk_actions
            if isinstance(chunk_actions, torch.Tensor)
            else torch.from_numpy(np.asarray(chunk_actions))
        )
        if (
            actions.ndim != 3
            or actions.shape[0] != self.num_envs
            or actions.shape[2] != 7
        ):
            raise ValueError(
                "SpaceUR10e chunk_actions must have shape "
                f"({self.num_envs}, chunk_steps, 7), got {tuple(actions.shape)}."
            )

        observations = []
        infos_list = []
        rewards = []
        raw_terminations = []
        raw_truncations = []
        for chunk_index in range(actions.shape[1]):
            obs, reward, terminated, truncated, infos = self.step(
                actions[:, chunk_index], auto_reset=False
            )
            observations.append(obs)
            infos_list.append(infos)
            rewards.append(reward)
            raw_terminations.append(terminated)
            raw_truncations.append(truncated)

        reward_tensor = torch.stack(rewards, dim=1)
        raw_termination_tensor = torch.stack(raw_terminations, dim=1)
        raw_truncation_tensor = torch.stack(raw_truncations, dim=1)
        past_terminations = raw_termination_tensor.any(dim=1)
        past_truncations = raw_truncation_tensor.any(dim=1)
        past_dones = past_terminations | past_truncations
        if past_dones.any() and self.auto_reset:
            observations[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, observations[-1], infos_list[-1]
            )

        chunk_terminations = torch.zeros_like(raw_termination_tensor)
        chunk_truncations = torch.zeros_like(raw_truncation_tensor)
        chunk_terminations[:, -1] = past_terminations
        chunk_truncations[:, -1] = past_truncations
        return (
            observations,
            reward_tensor,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def sample_action_space(self) -> torch.Tensor:
        """Sample one normalized action on the return device."""
        return torch.from_numpy(
            np.asarray(self.action_space.sample(), dtype=np.float32)
        ).to(self.device)

    def close(self) -> None:
        """Close every underlying MuJoCo environment."""
        for env in self.envs:
            env.close()
