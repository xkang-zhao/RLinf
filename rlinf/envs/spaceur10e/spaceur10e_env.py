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

"""RLinf adapter for the eight ``my_simulation`` SpaceUR10e tasks.

The simulator registers six MuJoCo scenes, while its automatic-grasp data
collection registry defines eight task-level goals.  This adapter uses the
latter as its public task interface and exposes both the 27-D proprioceptive
state and the three RGB cameras required by the collected LeRobot datasets.
"""

from __future__ import annotations

import copy
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

__all__ = ["SpaceUR10eRLinfEnv", "build_state_observation"]


STATE_FIELDS = {
    "joint_pos": (6,),
    "base_pose": (7,),
    "ee_pose": (7,),
    "target_pose": (7,),
}
STATE_DIM = sum(np.prod(shape) for shape in STATE_FIELDS.values())
DEFAULT_TASK_NAME = "cube"
DEFAULT_CAMERA_NAMES = {
    "main": "third_left_camera",
    "wrist": "left_wrist_camera",
    "extra": "third_right_camera",
}


@dataclass(frozen=True)
class _TaskDefinition:
    """Task-level metadata used by one or more simulator instances."""

    name: str
    env_id: str
    instruction: str


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read a value from either a mapping or attribute-style config."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_state_observation(raw_obs: dict[str, Any]) -> np.ndarray:
    """Build the stable 27-D policy state from a simulator observation.

    The target pose is intentionally retained: all registered environments
    randomize target placement by default, so dropping it makes a state-only
    policy partially observable.
    """
    values = []
    for key, shape in STATE_FIELDS.items():
        value = np.asarray(raw_obs[key], dtype=np.float32)
        if value.shape != shape:
            raise ValueError(
                f"SpaceUR10e observation {key!r} must have shape {shape}, "
                f"got {value.shape}."
            )
        values.append(value)
    return np.concatenate(values, axis=0, dtype=np.float32)


class SpaceUR10eRLinfEnv(gym.Env):
    """Vector-style RLinf wrapper around SpaceUR10e Gymnasium environments."""

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
        """Initialize independent CPU MuJoCo environments.

        Args:
            cfg: RLinf environment configuration.
            num_envs: Number of environments owned by this worker.
            seed_offset: Offset added to the configured random seed.
            total_num_processes: Number of environment worker processes.
            worker_info: RLinf worker metadata.
            record_metrics: Whether to attach per-episode metrics to infos.
        """
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
        self.ignore_terminations = bool(_cfg_get(cfg, "ignore_terminations", False))
        self.use_rel_reward = bool(_cfg_get(cfg, "use_rel_reward", False))
        self.reward_coef = float(_cfg_get(cfg, "reward_coef", 1.0))
        self.max_episode_steps = int(_cfg_get(cfg, "max_episode_steps", 400))
        self.record_metrics = bool(record_metrics)
        self.video_cfg = _cfg_get(cfg, "video_cfg", None)
        self._device = torch.device("cpu")
        self._is_start = True

        self.obs_mode = str(_cfg_get(cfg, "obs_mode", "state")).lower()
        if self.obs_mode not in {"state", "rgb"}:
            raise ValueError("SpaceUR10e obs_mode must be either 'state' or 'rgb'.")
        self.camera_names = {
            role: str(_cfg_get(cfg, f"{role}_camera", default))
            for role, default in DEFAULT_CAMERA_NAMES.items()
        }

        action_scale = np.asarray(
            _cfg_get(cfg, "action_scale", [0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 1.0]),
            dtype=np.float32,
        )
        if action_scale.shape != (7,) or np.any(action_scale <= 0):
            raise ValueError(
                "SpaceUR10e action_scale must contain seven positive values."
            )
        self.action_scale = action_scale

        self.repo_root = self._resolve_repo_root(_cfg_get(cfg, "repo_root", None))
        self._simulator_module = self._import_simulator()
        if self.repo_root is None:
            self.repo_root = self._repo_root_from_module(self._simulator_module)
        if self.repo_root is None:
            raise ModuleNotFoundError(
                "Could not locate the my_simulation checkout containing mjcf/. "
                "Set SPACEUR10E_REPO_ROOT or env.repo_root to that checkout."
            )

        task_defs = self._resolve_task_definitions()
        if bool(_cfg_get(self.cfg, "shard_tasks_by_worker", False)):
            worker_rank = self.seed_offset % self.total_num_processes
            worker_task_defs = task_defs[worker_rank :: self.total_num_processes]
            if not worker_task_defs:
                worker_task_defs = [task_defs[worker_rank % len(task_defs)]]
            self._env_tasks = [
                worker_task_defs[index % len(worker_task_defs)]
                for index in range(num_envs)
            ]
        else:
            self._env_tasks = [
                task_defs[(self.seed_offset * num_envs + index) % len(task_defs)]
                for index in range(num_envs)
            ]
        self._num_envs = int(num_envs)
        self._initialize_simulators(num_envs)
        self.action_space = self.single_action_space
        self.observation_space = self._build_observation_space()

        self.prev_step_reward = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._elapsed_steps = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self._needs_reset = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Keep evaluation resets reproducible while avoiding the same target
        # initialization on every rollout epoch. The initial reset preserves
        # the historical ``seed + env_index`` behavior; subsequent resets use
        # the next non-overlapping block of per-environment seeds.
        self._reset_counts = np.zeros(self.num_envs, dtype=np.int64)
        self._last_obs: dict[str, Any] | None = None
        self.reset_state_ids = None
        self.info_logging_keys = ["success", "fail"]
        self._init_metrics()

    @staticmethod
    def _resolve_repo_root(
        configured_root: str | os.PathLike[str] | None,
    ) -> Path | None:
        """Return a valid ``fix0613`` source checkout, if configured."""
        candidates: list[Path] = []
        if configured_root:
            candidates.append(Path(configured_root).expanduser())
        if os.environ.get("SPACEUR10E_REPO_ROOT"):
            candidates.append(Path(os.environ["SPACEUR10E_REPO_ROOT"]).expanduser())
        for candidate in candidates:
            resolved = candidate.resolve()
            if (resolved / "src" / "envs").is_dir() and (resolved / "mjcf").is_dir():
                return resolved
        return None

    @staticmethod
    def _repo_root_from_module(module: Any) -> Path | None:
        """Infer a checkout root from the installed ``envs`` package location."""
        module_path = getattr(module, "__file__", None)
        if module_path is None:
            return None
        candidate = Path(module_path).resolve().parents[2]
        if (candidate / "src" / "envs").is_dir() and (candidate / "mjcf").is_dir():
            return candidate
        return None

    def _import_simulator(self) -> Any:
        """Import the simulator registrations, preferring the selected checkout."""
        if self.repo_root is not None:
            src_root = str(self.repo_root / "src")
            if src_root not in sys.path:
                sys.path.insert(0, src_root)
        try:
            return importlib.import_module("envs")
        except ModuleNotFoundError as exc:
            if exc.name != "envs":
                raise
            raise ModuleNotFoundError(
                "my_simulation is not importable. Run "
                "`bash requirements/install.sh embodied --env spaceur10e`, "
                "or set SPACEUR10E_REPO_ROOT to its fix0613 checkout."
            ) from exc

    def _resolve_task_definitions(self) -> list[_TaskDefinition]:
        """Resolve one or more of my_simulation's eight task-level names."""
        requested_names = _cfg_get(self.cfg, "task_names", None)
        requested_name = _cfg_get(self.cfg, "task_name", None)
        configured_env_id = _cfg_get(self.cfg, "gym_id", None)
        configured_prompt = _cfg_get(self.cfg, "task_prompt", None)

        if requested_names is not None and requested_name is not None:
            raise ValueError("Set only one of SpaceUR10e task_name or task_names.")
        if requested_names is None and requested_name is not None:
            requested_names = [requested_name]
        if isinstance(requested_names, str):
            requested_names = (
                self._available_task_names()
                if requested_names == "all"
                else [requested_names]
            )
        if requested_names is None:
            if configured_env_id is not None:
                return [
                    _TaskDefinition(
                        name=str(configured_env_id),
                        env_id=str(configured_env_id),
                        instruction=str(configured_prompt or "Complete the task."),
                    )
                ]
            requested_names = [DEFAULT_TASK_NAME]
        if not requested_names:
            raise ValueError("SpaceUR10e task_names must not be empty.")

        tasks_module = importlib.import_module("planners.auto_grasp.tasks")
        definitions = []
        for task_name in requested_names:
            spec = tasks_module.get_task_spec(str(task_name))
            if configured_env_id is not None and len(requested_names) == 1:
                if str(configured_env_id) != spec.env_id:
                    raise ValueError(
                        f"task_name {spec.name!r} uses {spec.env_id!r}, not "
                        f"configured gym_id {configured_env_id!r}."
                    )
            definitions.append(
                _TaskDefinition(
                    name=spec.name,
                    env_id=spec.env_id,
                    instruction=str(configured_prompt or spec.instruction),
                )
            )
        return definitions

    @staticmethod
    def _available_task_names() -> list[str]:
        """Read the simulator task ordering without importing MuJoCo."""
        tasks_module = importlib.import_module("planners.auto_grasp.tasks")
        return list(tasks_module.TASK_NAMES)

    def _initialize_simulators(self, num_envs: int) -> None:
        """Create the simulation backend and its single-world action space."""
        self.envs = [self._make_env(index) for index in range(num_envs)]
        self.single_action_space = self.envs[0].action_space

    def _make_env(self, index: int) -> gym.Env:
        """Create one Gym environment with checkout-independent asset paths."""
        task = self._env_tasks[index]
        spec = gym.spec(task.env_id)
        kwargs = dict(spec.kwargs or {})
        kwargs["render_mode"] = None
        kwargs["use_depth"] = False
        kwargs["deterministic_rendering"] = bool(
            _cfg_get(self.cfg, "deterministic_rendering", False)
        )
        for key in ("scene_path", "arm_path"):
            configured_path = _cfg_get(self.cfg, key, kwargs.get(key))
            if configured_path is not None:
                path = Path(str(configured_path)).expanduser()
                kwargs[key] = str(path if path.is_absolute() else self.repo_root / path)
        for key in (
            "frame_name",
            "viewer_config",
            "target_body_name",
            "target_joint_name",
        ):
            value = _cfg_get(self.cfg, key, None)
            if value is not None:
                kwargs[key] = value
        target_init_range = _cfg_get(self.cfg, "target_init_range", None)
        if target_init_range is not None:
            kwargs["target_init_range"] = {
                axis: tuple(target_init_range[axis]) for axis in ("x", "y", "z")
            }
        return gym.make(task.env_id, disable_env_checker=True, **kwargs)

    def _build_observation_space(self) -> gym.spaces.Dict:
        """Describe the tensors returned to RLinf actors."""
        spaces: dict[str, gym.Space] = {
            "states": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_envs, STATE_DIM),
                dtype=np.float32,
            )
        }
        if self.obs_mode == "rgb":
            image_shape = (self.num_envs, 480, 640, 3)
            spaces.update(
                {
                    "main_images": gym.spaces.Box(0, 255, image_shape, np.uint8),
                    "wrist_images": gym.spaces.Box(0, 255, image_shape, np.uint8),
                    "extra_view_images": gym.spaces.Box(
                        0, 255, (self.num_envs, 1, 480, 640, 3), np.uint8
                    ),
                }
            )
        return gym.spaces.Dict(spaces)

    @property
    def total_num_group_envs(self) -> int:
        """Return a compatibility upper bound for reset-state groups."""
        return np.iinfo(np.uint8).max // 2

    @property
    def num_envs(self) -> int:
        """Return the number of independent environments in this process."""
        return self._num_envs

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
        """Return each instance's task-level language instruction."""
        return [task.instruction for task in self._env_tasks]

    def update_reset_state_ids(self) -> None:
        """Keep API compatibility with RLinf reset-state scheduling."""
        self.reset_state_ids = None

    @staticmethod
    def _image_from_obs(raw_obs: dict[str, Any], camera_name: str) -> torch.Tensor:
        """Validate and convert one HWC RGB image to a CPU tensor."""
        if camera_name not in raw_obs:
            available = ", ".join(sorted(raw_obs))
            raise KeyError(
                f"SpaceUR10e camera {camera_name!r} is unavailable; "
                f"observation keys are: {available}."
            )
        image = np.asarray(raw_obs[camera_name], dtype=np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"SpaceUR10e camera {camera_name!r} must be HWC RGB, got {image.shape}."
            )
        return torch.from_numpy(np.ascontiguousarray(image))

    def _wrap_obs(self, raw_obs: dict[str, Any], env_index: int) -> dict[str, Any]:
        """Convert one simulator observation to RLinf's common schema."""
        obs: dict[str, Any] = {
            "states": torch.from_numpy(build_state_observation(raw_obs)).to(self.device)
        }
        if self.obs_mode == "rgb":
            obs["main_images"] = self._image_from_obs(
                raw_obs, self.camera_names["main"]
            ).to(self.device)
            obs["wrist_images"] = self._image_from_obs(
                raw_obs, self.camera_names["wrist"]
            ).to(self.device)
            obs["extra_view_images"] = (
                self._image_from_obs(raw_obs, self.camera_names["extra"])
                .unsqueeze(0)
                .to(self.device)
            )
            obs["task_descriptions"] = self._env_tasks[env_index].instruction
        return obs

    def _collate_obs(self, observations: list[dict[str, Any]]) -> dict[str, Any]:
        """Stack tensors and keep per-environment text as a list."""
        keys = set().union(*(observation.keys() for observation in observations))
        result: dict[str, Any] = {}
        for key in sorted(keys):
            values = [observation[key] for observation in observations]
            result[key] = (
                torch.stack(values, dim=0)
                if isinstance(values[0], torch.Tensor)
                else values
            )
        return result

    def _collate_infos(self, info_list: list[dict[str, Any]]) -> dict[str, Any]:
        keys = set().union(*(info.keys() for info in info_list))
        infos: dict[str, Any] = {}
        for key in sorted(keys):
            values = [info.get(key) for info in info_list]
            if all(
                value is None or isinstance(value, (bool, np.bool_)) for value in values
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
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
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
            "reward": self.returns / torch.clamp(episode_len.float(), min=1.0),
        }
        return infos

    def _index_cached_obs(self, env_idx: int) -> dict[str, Any]:
        if self._last_obs is None:
            raw_obs, _ = self.envs[env_idx].reset(seed=self.seed + env_idx)
            return self._wrap_obs(raw_obs, env_idx)
        return {
            key: value[env_idx]
            if isinstance(value, (torch.Tensor, list, tuple))
            else value
            for key, value in self._last_obs.items()
        }

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
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
        selected = set(indices)

        observations = []
        info_list = []
        for index, env in enumerate(self.envs):
            if index not in selected:
                observations.append(self._index_cached_obs(index))
                info_list.append({})
                continue
            if isinstance(seed, (list, tuple, np.ndarray)):
                env_seed = int(seed[index])
            elif seed is not None:
                env_seed = int(seed) + index
            else:
                env_seed = (
                    self.seed + index + int(self._reset_counts[index] * self.num_envs)
                )
                self._reset_counts[index] += 1
            raw_obs, info = env.reset(seed=env_seed, options=options)
            observations.append(self._wrap_obs(raw_obs, index))
            info_list.append(info if isinstance(info, dict) else {})

        obs = self._collate_obs(observations)
        infos = self._collate_infos(info_list)
        self._last_obs = obs
        self._is_start = False
        return obs, infos

    def _normalize_actions(self, actions: np.ndarray | torch.Tensor) -> np.ndarray:
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
        self, actions: np.ndarray | torch.Tensor, auto_reset: bool = True
    ) -> tuple[
        dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]
    ]:
        """Apply one normalized seven-dimensional action per environment."""
        normalized_actions = self._normalize_actions(actions)
        observations = []
        info_list = []
        rewards = []
        terminations = []
        truncations = []

        for index, env in enumerate(self.envs):
            if self._needs_reset[index]:
                observations.append(self._index_cached_obs(index))
                info_list.append(
                    {
                        "success": False,
                        "fail": False,
                        "task_name": self._env_tasks[index].name,
                    }
                )
                rewards.append(0.0)
                terminations.append(True)
                truncations.append(False)
                continue

            env_action = normalized_actions[index] * self.action_scale
            raw_obs, _, terminated, truncated, info = env.step(env_action)
            self._elapsed_steps[index] += 1
            truncated = (
                bool(truncated) or self._elapsed_steps[index] >= self.max_episode_steps
            )
            info = dict(info) if isinstance(info, dict) else {}
            success = bool(info.get("is_success", False))
            info["success"] = success
            info["fail"] = bool(truncated and not success)
            # Keep the task identity next to the terminal metrics so the
            # evaluator can report success rates for a mixed-task vector env.
            info["task_name"] = self._env_tasks[index].name
            self._needs_reset[index] = bool(terminated or truncated)

            observations.append(self._wrap_obs(raw_obs, index))
            info_list.append(info)
            # All eight public tasks use the same terminal binary contract:
            # a confirmed grasp is 1 and every intermediate/failure step is 0.
            rewards.append(1.0 if success else 0.0)
            terminations.append(bool(terminated))
            truncations.append(bool(truncated))

        return self._finish_step(
            self._collate_obs(observations),
            info_list,
            rewards,
            terminations,
            truncations,
            auto_reset=auto_reset,
        )

    def _finish_step(
        self, obs, info_list, rewards, terminations, truncations, *, auto_reset
    ):
        """Apply shared rewards, metrics and auto-reset to a simulator step."""
        raw_reward = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        reward = (
            raw_reward - self.prev_step_reward if self.use_rel_reward else raw_reward
        )
        # Keep reset-time metric clearing from mutating the reward returned by
        # this step when ``reward`` aliases ``raw_reward``.
        self.prev_step_reward = raw_reward.clone()
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
                infos["episode"]["success_at_end"] = infos["success"].clone()
            termination_tensor[:] = False

        dones = termination_tensor | truncation_tensor
        if dones.any() and auto_reset and self.auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)

        self._last_obs = obs
        self._is_start = False
        return obs, reward, termination_tensor, truncation_tensor, infos

    def _handle_auto_reset(
        self, dones: torch.Tensor, final_obs: dict[str, Any], infos: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
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
        list[dict[str, Any]],
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
