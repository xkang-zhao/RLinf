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

import logging

import torch
from omegaconf import OmegaConf

from rlinf.algorithms.losses import compute_ppo_actor_loss
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager


class _WarmupModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(2, 2)
        self.value_head = torch.nn.Linear(2, 1)
        self.always_frozen = torch.nn.Linear(2, 2)


def test_critic_warmup_optimizer_registers_actor_with_zero_lr():
    model = _WarmupModel()
    model.always_frozen.requires_grad_(False)

    manager = object.__new__(FSDPModelManager)
    manager._cfg = OmegaConf.create(
        {
            "optim": {
                "lr": 5e-7,
                "value_lr": 5e-5,
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_eps": 1e-8,
                "weight_decay": 0.01,
            },
            "fsdp_config": {"sharding_strategy": "full_shard"},
        }
    )
    manager._logger = logging.getLogger(__name__)
    manager.store_requires_grad_param_name = []

    optimizer = manager.build_optimizer(model, enable_critic_warmup=True)

    optimized_param_ids = {
        id(param) for group in optimizer.param_groups for param in group["params"]
    }
    assert model.actor.weight.requires_grad
    assert model.actor.bias.requires_grad
    assert id(model.actor.weight) in optimized_param_ids
    assert id(model.actor.bias) in optimized_param_ids
    assert id(model.value_head.weight) in optimized_param_ids
    assert id(model.value_head.bias) in optimized_param_ids
    assert not model.always_frozen.weight.requires_grad
    assert optimizer.param_groups[0]["lr"] == 0.0
    assert optimizer.param_groups[1]["lr"] == 5e-5


def test_critic_warmup_actor_loss_keeps_zero_gradient_graph():
    logprobs = torch.tensor([[0.1], [0.2]], requires_grad=True)
    loss, _ = compute_ppo_actor_loss(
        logprobs=logprobs,
        old_logprobs=torch.zeros_like(logprobs),
        clip_ratio_low=0.1,
        clip_ratio_high=0.1,
        advantages=torch.ones_like(logprobs),
        critic_warmup=True,
    )

    assert loss.requires_grad
    assert loss.item() == 0.0
    loss.backward()
    torch.testing.assert_close(logprobs.grad, torch.zeros_like(logprobs))
