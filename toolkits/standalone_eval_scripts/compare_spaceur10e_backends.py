# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Paired SFT-policy evaluation on CPU MuJoCo and grouped Warp (no training)."""

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.envs.spaceur10e.spaceur10e_env import SpaceUR10eRLinfEnv
from rlinf.envs.spaceur10e.spaceur10e_warp_env import SpaceUR10eWarpRLinfEnv


def episode_seed(base, index):
    """Stable seed for a global episode slot, independent of backend."""
    return base + index


def action_noise(base, indices, chunk):
    """Use the same initial flow noise for each paired episode and chunk."""
    return torch.stack(
        [
            torch.randn(
                (30, 32),
                generator=torch.Generator().manual_seed(
                    base + 1_000_000 + int(index) * 1000 + chunk
                ),
            )
            for index in indices
        ]
    )


class StepTrace:
    """Record small numeric diagnostics without retaining RGB or changing steps."""

    def __init__(self, env, backend):
        self.env = env
        self.backend = backend
        self.rows = []
        self.original_step = env.step
        env.step = self.step

    def step(self, actions, *args, **kwargs):
        active = ~self.env._needs_reset.cpu().numpy().copy()
        result = self.original_step(actions, *args, **kwargs)
        obs, _, term, trunc, infos = result
        if self.backend == "cpu":
            controls = np.stack([e.unwrapped.data.ctrl.copy() for e in self.env.envs])
        else:
            controls = np.stack(
                [e._control[i].copy() for e, i in self.env.vector_env.worlds]
            )
        row = {
            "active": active,
            "actions": torch.as_tensor(actions).cpu().numpy().copy(),
            "states": obs["states"].cpu().numpy().copy(),
            "controls": controls,
            "terminated": term.cpu().numpy().copy(),
            "truncated": trunc.cpu().numpy().copy(),
        }
        for key in (
            "distance",
            "success_counter",
            "left_contact",
            "right_contact",
            "ik_gpu_accepted",
            "ik_cpu_fallback",
            "ik_residual",
        ):
            value = infos.get(key)
            row[key] = (
                value.cpu().numpy().copy()
                if isinstance(value, torch.Tensor)
                else np.full(self.env.num_envs, np.nan)
            )
        self.rows.append(row)
        return result

    def save(self, path, indices, input_states):
        """Persist one chunk; missing metrics are NaN, inactive rows are masked."""
        arrays = {
            key: np.stack([row[key] for row in self.rows]) for key in self.rows[0]
        }
        with path.open("xb") as file:
            np.savez_compressed(
                file, indices=np.asarray(indices), input_states=input_states, **arrays
            )
        self.rows.clear()


def run_batch(model, backend, indices, args, journal):
    count = len(indices)
    cls = SpaceUR10eRLinfEnv if backend == "cpu" else SpaceUR10eWarpRLinfEnv
    cfg = {
        "repo_root": args.repo_root,
        "task_names": "all",
        "obs_mode": "rgb",
        "seed": args.seed,
        "auto_reset": False,
        "ignore_terminations": False,
        "max_episode_steps": args.max_steps,
        "action_scale": [1.0] * 7,
        "ik_backend": "cpu" if backend == "warp_cpu_ik" else "gpu",
        "warp_substeps_per_graph": 10,
        "deterministic_rendering": args.deterministic_rendering,
        "deterministic_physics": args.deterministic_physics,
        "video_cfg": {"save_video": False},
    }
    started = time.perf_counter()
    env = cls(cfg, count, 0, 1, None)
    try:
        seeds = [episode_seed(args.seed, index) for index in indices]
        obs, _ = env.reset(seed=seeds)
        initial = obs["states"].clone()
        trace = StepTrace(env, backend) if args.trace_steps else None
        setup_seconds = time.perf_counter() - started
        predict_seconds = interact_seconds = 0.0
        for chunk in range((args.max_steps + 29) // 30):
            input_states = obs["states"].cpu().numpy().copy() if trace else None
            noise = action_noise(args.seed, indices, chunk).to("cuda")
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                actions, _ = model.predict_action_batch(obs, mode="eval", noise=noise)
            torch.cuda.synchronize()
            predict_seconds += time.perf_counter() - start
            if not torch.isfinite(actions).all():
                raise RuntimeError("Policy produced non-finite actions")
            actions = actions.detach().cpu()[:, : min(30, args.max_steps - chunk * 30)]
            start = time.perf_counter()
            if backend != "cpu":
                observations, _, _, _, _ = env.chunk_step(actions)
                obs = observations[-1]
            else:
                # Same control steps as CPU chunk_step, but discard intermediate
                # images instead of retaining 30 RGB batches that policy ignores.
                for step in range(actions.shape[1]):
                    obs, _, _, _, _ = env.step(actions[:, step], auto_reset=False)
            interact_seconds += time.perf_counter() - start
            if trace:
                trace.save(
                    args.output / f"trace_{backend}_{indices[0]:03d}_{chunk:02d}.npz",
                    indices,
                    input_states,
                )
            print(
                json.dumps(
                    {
                        "event": "chunk",
                        "backend": backend,
                        "first_episode": indices[0],
                        "chunk": chunk + 1,
                        "done": int(env._needs_reset.sum()),
                        "success": int(env.success_once.sum()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if env._needs_reset.all():
                break
        rows = []
        for local, index in enumerate(indices):
            row = {
                "backend": backend,
                "task": env._env_tasks[local].name,
                "episode_index": index,
                "task_episode": index // 8,
                "seed": seeds[local],
                "success": bool(env.success_once[local]),
                "steps": int(env.elapsed_steps[local]),
                "done": bool(env._needs_reset[local]),
            }
            rows.append(row)
            journal.write(json.dumps(row) + "\n")
            journal.flush()
        timing = {
            "backend": backend,
            "first_episode": indices[0],
            "count": count,
            "setup_seconds": setup_seconds,
            "predict_seconds": predict_seconds,
            "interact_seconds": interact_seconds,
        }
        print(json.dumps(dict(event="batch_complete", **timing)), flush=True)
        return rows, initial, timing
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", default="/workspace/SpaceRobotEnv")
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=420)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument(
        "--deterministic-rendering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable MSAA/dithering on both paths; changes image antialiasing.",
    )
    parser.add_argument(
        "--deterministic-physics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use Warp RUN_TO_RUN deterministic atomics for GPU physics; "
            "ignored by CPU MuJoCo and slower than the default Warp path."
        ),
    )
    parser.add_argument(
        "--include-cpu-ik",
        action="store_true",
        help="Also evaluate Warp physics with CPU IK (third backend).",
    )
    parser.add_argument(
        "--trace-steps",
        action="store_true",
        help="Save numeric per-step NPZ traces; adds diagnostic overhead.",
    )
    args = parser.parse_args()
    if args.batch_size < 8 or args.batch_size % 8 or args.episodes_per_task < 1:
        parser.error(
            "batch-size must be a positive multiple of 8; episodes-per-task must be positive"
        )
    if args.max_steps < 1 or args.max_steps > 420 or args.seed < 0:
        parser.error("max-steps must be in [1,420] and seed nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    backends = (
        ("cpu", "warp", "warp_cpu_ik") if args.include_cpu_ik else ("cpu", "warp")
    )
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(root / "evaluations/spaceur10e/spaceur10e_pi0_rlinf_eval.yaml")
    cfg.rollout.model.model_path = args.model_path
    cfg.rollout.model.pi05 = True
    cfg.rollout.model.num_action_chunks = 30
    cfg.rollout.model.openpi.config_name = "pi05_spaceur10e"
    cfg.rollout.model.openpi.max_token_len = 200
    config = OmegaConf.to_container(cfg.rollout.model, resolve=True)
    metadata = {
        "model": config,
        "episodes_per_task": args.episodes_per_task,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "backends": backends,
        "trace_steps": args.trace_steps,
        "deterministic_rendering": args.deterministic_rendering,
        "deterministic_physics": args.deterministic_physics,
        "torch": torch.__version__,
        # Query distribution metadata without importing MuJoCo Warp before
        # SpaceUR10e can select its process-global deterministic mode.
        "mujoco": version("mujoco"),
        "warp": version("warp-lang"),
        "mujoco_warp": version("mujoco-warp"),
        "note": "Quality comparison; CPU renders every step, Warp at chunk boundaries. Not a parallel throughput benchmark.",
    }
    with (args.output / "metadata.json").open("x") as file:
        json.dump(metadata, file, indent=2)
    from rlinf.models.embodiment.openpi_rlinf import get_model

    model = get_model(OmegaConf.create(config)).to("cuda").eval()
    rows, timings, initial_errors = [], [], []
    with (args.output / "episodes.jsonl").open("x") as journal:
        for first in range(0, args.episodes_per_task * 8, args.batch_size):
            indices = list(
                range(first, min(first + args.batch_size, args.episodes_per_task * 8))
            )
            reference = None
            for backend in backends:
                batch, initial, timing = run_batch(
                    model, backend, indices, args, journal
                )
                rows.extend(batch)
                timings.append(timing)
                if reference is None:
                    reference = initial
                else:
                    error = float((reference - initial).abs().max())
                    initial_errors.append(error)
                    if error > 1e-4:
                        raise RuntimeError(f"Paired initial states differ: {error}")
    summary = {
        "timings": timings,
        "initial_state_max_abs_errors": initial_errors,
        "tasks": {},
    }
    paired = {(row["backend"], row["episode_index"]): row for row in rows}
    summary["paired_success_disagreements"] = [
        {
            "episode_index": index,
            "task": paired["cpu", index]["task"],
            "seed": paired["cpu", index]["seed"],
            "cpu_success": paired["cpu", index]["success"],
            "warp_success": paired["warp", index]["success"],
        }
        for index in range(args.episodes_per_task * 8)
        if paired["cpu", index]["success"] != paired["warp", index]["success"]
    ]
    if args.include_cpu_ik:
        summary["cpu_ik_ablation"] = [
            {
                "episode_index": index,
                "task": paired["cpu", index]["task"],
                **{
                    backend: {
                        key: paired[backend, index][key] for key in ("success", "steps")
                    }
                    for backend in backends
                },
            }
            for index in range(args.episodes_per_task * 8)
        ]
    for task in sorted({row["task"] for row in rows}):
        summary["tasks"][task] = {}
        for backend in backends:
            selected = [
                row for row in rows if row["task"] == task and row["backend"] == backend
            ]
            summary["tasks"][task][backend] = {
                "successes": sum(row["success"] for row in selected),
                "episodes": len(selected),
                "mean_steps": float(np.mean([row["steps"] for row in selected])),
            }
    with (args.output / "summary.json").open("x") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(dict(event="complete", **summary)), flush=True)


if __name__ == "__main__":
    main()
