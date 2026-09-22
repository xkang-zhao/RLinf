# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Measure the CUDA batch adapter without loading a policy or starting Ray."""

import argparse
import json
import time

import numpy as np
import psutil

from rlinf.envs.spaceur10e.spaceur10e_warp_env import SpaceUR10eWarpRLinfEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default="/workspace/SpaceRobotEnv")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--tasks", choices=["cube", "all"], default="cube")
    parser.add_argument("--chunks", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--substeps-per-graph", type=int, default=100)
    args = parser.parse_args()
    if min(args.num_envs, args.chunks, args.chunk_size) < 1:
        parser.error("num-envs, chunks, and chunk-size must be positive")
    cfg = {
        "repo_root": args.repo_root,
        "task_names": "all" if args.tasks == "all" else ["cube"],
        "obs_mode": "rgb",
        "ik_backend": "gpu",
        "action_scale": [1.0] * 7,
        "max_episode_steps": args.chunks * args.chunk_size + 1,
        "auto_reset": False,
        "warp_substeps_per_graph": args.substeps_per_graph,
    }
    start = time.perf_counter()
    env = SpaceUR10eWarpRLinfEnv(cfg, args.num_envs, 0, 1, None)
    init_seconds = time.perf_counter() - start
    try:
        start = time.perf_counter()
        obs, _ = env.reset(seed=7)
        reset_seconds = time.perf_counter() - start
        durations = []
        memory_samples = []
        actions = np.zeros((args.num_envs, args.chunk_size, 7), dtype=np.float32)
        for _ in range(args.chunks):
            start = time.perf_counter()
            observations, rewards, terminated, truncated, _ = env.chunk_step(actions)
            durations.append(time.perf_counter() - start)
            assert (
                rewards.shape
                == terminated.shape
                == truncated.shape
                == actions.shape[:2]
            )
            assert np.isfinite(observations[-1]["states"].numpy()).all()
            memory = psutil.Process().memory_full_info()
            memory_samples.append(
                {
                    "rss_gib": memory.rss / 2**30,
                    "pss_gib": getattr(memory, "pss", 0) / 2**30,
                }
            )
            print(
                json.dumps(
                    {
                        "chunk": len(durations),
                        "seconds": durations[-1],
                        **memory_samples[-1],
                    }
                ),
                flush=True,
            )
        memory = psutil.Process().memory_full_info()
        print(
            json.dumps(
                {
                    "worlds": args.num_envs,
                    "substeps_per_graph": args.substeps_per_graph,
                    "init_seconds": init_seconds,
                    "reset_seconds": reset_seconds,
                    "chunk_seconds": durations,
                    "chunk_memory": memory_samples,
                    "last_chunk_world_control_steps_per_second": args.num_envs
                    * args.chunk_size
                    / durations[-1],
                    "rss_gib": memory.rss / 2**30,
                    "pss_gib": getattr(memory, "pss", 0) / 2**30,
                    "image_shape": list(obs["main_images"].shape),
                    "note": "Zero-action environment smoke test; excludes model inference and PPO.",
                },
                indent=2,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
