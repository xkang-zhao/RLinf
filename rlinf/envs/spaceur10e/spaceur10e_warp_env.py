# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Grouped GPU batch adapter with chunk-boundary RGB rendering for OpenPI PPO."""

from __future__ import annotations

import importlib

import numpy as np
import torch

from .spaceur10e_env import SpaceUR10eRLinfEnv, _cfg_get, build_state_observation


class SpaceUR10eWarpRLinfEnv(SpaceUR10eRLinfEnv):
    """Own one CUDA physics batch per scene with the RLinf observation schema.

    RGB is rendered sequentially using one MuJoCo renderer per scene.
    During chunk_step, only terminal states and the chunk boundary are rendered:
    intermediate images are not consumed by OpenPI. This is not batch rendering.
    """

    def _initialize_simulators(self, num_envs: int) -> None:
        if _cfg_get(self.video_cfg, "save_video", False) or _cfg_get(
            _cfg_get(self.cfg, "data_collection", None), "enabled", False
        ):
            raise ValueError(
                "Warp PPO renders chunk boundaries only; disable video_cfg.save_video "
                "and data_collection.enabled. Use SpaceRobotEnv's GPU collector for datasets."
            )
        if self.ignore_terminations:
            raise ValueError("Warp PPO requires ignore_terminations=false")
        module = importlib.import_module("envs.space_ur10e_warp_group")
        self.vector_env = module.SpaceUR10eWarpGroup(
            env_ids=[task.env_id for task in self._env_tasks],
            device=str(_cfg_get(self.cfg, "warp_device", "cuda:0")),
            ik_backend=str(_cfg_get(self.cfg, "ik_backend", "gpu")),
            nconmax=int(_cfg_get(self.cfg, "warp_nconmax", 128)),
            njmax=int(_cfg_get(self.cfg, "warp_njmax", 512)),
            repo_root=str(self.repo_root),
            physics_substeps_per_graph=_cfg_get(
                self.cfg, "warp_substeps_per_graph", 100
            ),
            deterministic_physics=bool(
                _cfg_get(self.cfg, "deterministic_physics", False)
            ),
        )
        self.single_action_space = self.vector_env.single_action_space
        self._renderers = {}
        self._render_step = True
        self._raw_batch = None
        self._terminal_flags = np.zeros(num_envs, dtype=bool)
        self._truncated_flags = np.zeros(num_envs, dtype=bool)
        try:
            if self.obs_mode == "rgb":
                import mujoco
                from mujoco_robot.robot_sensor import RobotSensor

                for _, env in self.vector_env.groups:
                    data = mujoco.MjData(env.model)
                    self._renderers[id(env)] = (
                        data,
                        RobotSensor(
                            env.model,
                            data,
                            deterministic_rendering=bool(
                                _cfg_get(self.cfg, "deterministic_rendering", False)
                            ),
                        ),
                    )
        except BaseException:
            self.close()
            raise

    def _batch_observation(self, render_mask: np.ndarray) -> dict:
        target = self.vector_env.get_planner_state()["target_pose"]
        states = []
        for i in range(self.num_envs):
            raw = {key: value[i] for key, value in self._raw_batch.items()}
            raw["target_pose"] = target[i]
            states.append(build_state_observation(raw))
        obs = {"states": torch.from_numpy(np.stack(states))}
        if self.obs_mode == "rgb":
            import mujoco

            fields = {
                "main": "main_images",
                "wrist": "wrist_images",
                "extra": "extra_view_images",
            }
            for role, key in fields.items():
                shape = (
                    (self.num_envs, 1, 480, 640, 3)
                    if role == "extra"
                    else (self.num_envs, 480, 640, 3)
                )
                old = None if self._last_obs is None else self._last_obs[key]
                obs[key] = (
                    torch.zeros(shape, dtype=torch.uint8)
                    if old is None
                    else (old.clone() if render_mask.any() else old)
                )
            for i in np.flatnonzero(render_mask):
                env, local = self.vector_env.worlds[i]
                data, sensor = self._renderers[id(env)]
                data.qpos[:] = env._qpos[local]
                mujoco.mj_forward(env.model, data)
                images = sensor.update_camera_view()
                for role, key in fields.items():
                    image = self._image_from_obs(images, self.camera_names[role])
                    obs[key][i] = image.unsqueeze(0) if role == "extra" else image
            obs["task_descriptions"] = self.instruction
        return obs

    def reset(self, *, seed=None, options=None):
        indices = None if options is None else options.get("env_idx")
        mask = (
            np.ones(self.num_envs, dtype=bool)
            if indices is None
            else np.zeros(self.num_envs, dtype=bool)
        )
        if indices is not None:
            mask[torch.as_tensor(indices).cpu().numpy().reshape(-1)] = True
        if not mask.any():
            raise ValueError("reset requires at least one environment")
        seeds = [None] * self.num_envs
        for i in np.flatnonzero(mask):
            if isinstance(seed, (list, tuple, np.ndarray)):
                seeds[i] = int(seed[i])
            elif seed is not None:
                seeds[i] = int(seed) + i
            else:
                seeds[i] = self.seed + i + int(self._reset_counts[i] * self.num_envs)
                self._reset_counts[i] += 1
        self._raw_batch, _ = self.vector_env.reset(
            seed=seeds, options={"reset_mask": mask}
        )
        metric_indices = torch.from_numpy(np.flatnonzero(mask))
        self._reset_metrics(metric_indices)
        self._needs_reset[metric_indices] = False
        self._terminal_flags[mask] = False
        self._truncated_flags[mask] = False
        obs = self._batch_observation(mask)
        self._last_obs = obs
        self._is_start = False
        return obs, {}

    def step(self, actions, auto_reset=True):
        if self._raw_batch is None:
            raise RuntimeError("reset() must precede step()")
        values = self._normalize_actions(actions) * self.action_scale
        if not np.isfinite(values).all():
            raise ValueError("SpaceUR10e actions must be finite")
        active = ~self._needs_reset.numpy()
        infos = [{} for _ in range(self.num_envs)]
        success = np.zeros(self.num_envs, dtype=bool)
        new_done = np.zeros(self.num_envs, dtype=bool)
        if active.any():
            self._raw_batch, _, term, trunc, raw_infos = self.vector_env.step(
                values, active_mask=active
            )
            self._elapsed_steps[torch.from_numpy(active)] += 1
            trunc = trunc | (self._elapsed_steps.numpy() >= self.max_episode_steps)
            self._terminal_flags[active] = term[active]
            self._truncated_flags[active] = trunc[active]
            success = np.asarray(raw_infos["is_success"], dtype=bool) & active
            new_done = (term | trunc) & active
            self._needs_reset[torch.from_numpy(new_done)] = True
            for i in np.flatnonzero(active):
                infos[i] = {key: value[i] for key, value in raw_infos.items()}
        for i in range(self.num_envs):
            infos[i].update(
                success=bool(success[i]),
                fail=bool(new_done[i] and not success[i]),
                task_name=self._env_tasks[i].name,
            )
        obs = self._batch_observation((active if self._render_step else new_done))
        return self._finish_step(
            obs,
            infos,
            success.astype(float).tolist(),
            self._terminal_flags.tolist(),
            self._truncated_flags.tolist(),
            auto_reset=auto_reset,
        )

    def chunk_step(self, chunk_actions):
        actions = torch.as_tensor(chunk_actions)
        if (
            actions.ndim != 3
            or actions.shape[0] != self.num_envs
            or actions.shape[2] != 7
            or actions.shape[1] < 1
        ):
            raise ValueError(
                "chunk_actions must have shape [num_envs, positive chunk_steps, 7]"
            )
        rewards, terms, truncs, infos = [], [], [], []
        try:
            for i in range(actions.shape[1]):
                self._render_step = i == actions.shape[1] - 1
                obs, reward, term, trunc, info = self.step(
                    actions[:, i], auto_reset=False
                )
                rewards.append(reward)
                terms.append(term)
                truncs.append(trunc)
                infos.append(info)
        finally:
            self._render_step = True
        terms, truncs = torch.stack(terms, 1), torch.stack(truncs, 1)
        term_any, trunc_any = terms.any(1), truncs.any(1)
        done = term_any | trunc_any
        if done.any() and self.auto_reset:
            obs, infos[-1] = self._handle_auto_reset(done, obs, infos[-1])
        terms.zero_()
        truncs.zero_()
        terms[:, -1], truncs[:, -1] = term_any, trunc_any
        self._last_obs = obs
        # EnvWorker accepts a list of boundary observations; avoid retaining
        # 30 full RGB batches when only the last one is sent to the policy.
        return [obs], torch.stack(rewards, 1), terms, truncs, infos

    def close(self):
        try:
            for _, sensor in self._renderers.values():
                sensor.close()
        finally:
            self.vector_env.close()
