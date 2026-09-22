# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from toolkits.standalone_eval_scripts.compare_spaceur10e_backends import (
    StepTrace,
    action_noise,
    episode_seed,
)


def test_episode_seeds_are_unique_and_paired():
    cpu = [episode_seed(123, index) for index in range(80)]
    warp = [episode_seed(123, index) for index in range(80)]
    assert cpu == warp
    assert len(set(cpu)) == 80


def test_noise_is_reproducible_and_independent_of_batch_partition():
    whole = action_noise(123, list(range(32)), 2)
    parts = torch.cat(
        [action_noise(123, list(range(i, i + 8)), 2) for i in range(0, 32, 8)]
    )
    torch.testing.assert_close(whole, parts, rtol=0, atol=0)
    torch.testing.assert_close(
        whole, action_noise(123, list(range(32)), 2), rtol=0, atol=0
    )
    assert not torch.equal(whole, action_noise(123, list(range(32)), 3))
    assert not torch.equal(whole[0], whole[1])


@pytest.mark.parametrize("backend", ["cpu", "warp", "warp_cpu_ik"])
def test_step_trace_preserves_step_and_copies_numeric_data(tmp_path, backend):
    controls = np.ones((2, 7))
    states = torch.zeros(2, 27)
    env = SimpleNamespace(num_envs=2, _needs_reset=torch.tensor([False, True]))
    env.envs = [
        SimpleNamespace(unwrapped=SimpleNamespace(data=SimpleNamespace(ctrl=c)))
        for c in controls
    ]
    vector = SimpleNamespace(_control=controls)
    env.vector_env = SimpleNamespace(worlds=[(vector, 0), (vector, 1)])
    calls = []

    def original(actions, auto_reset=True):
        calls.append(auto_reset)
        return result

    result = (
        {"states": states},
        torch.zeros(2),
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
        {"success_counter": torch.tensor([2, 0])},
    )
    env.step = original
    trace = StepTrace(env, backend)
    assert env.step(torch.zeros(2, 7), auto_reset=False) is result
    states.fill_(3)
    controls.fill(4)
    path = tmp_path / "trace.npz"
    trace.save(path, [0, 1], np.zeros((2, 27)))
    with np.load(path, allow_pickle=False) as saved:
        assert saved["states"].shape == (1, 2, 27)
        assert not saved["states"].any()
        assert (saved["controls"] == 1).all()
        assert saved["active"].tolist() == [[True, False]]
        assert np.isnan(saved["ik_residual"]).all()
        assert saved["success_counter"].tolist() == [[2, 0]]
    assert calls == [False]
    assert not trace.rows
