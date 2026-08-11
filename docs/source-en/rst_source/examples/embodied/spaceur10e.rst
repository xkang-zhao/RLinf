RL with SpaceUR10e
==================

This example connects the external SpaceUR10e MuJoCo/Gymnasium package to
RLinf and trains a state-only MLP policy with PPO on the cube grasping task.

Environment
-----------

- **Environment**: ``SpaceUR10e-Cube-v0``
- **Observation**: 21 float32 values ordered as ``joint_pos(6)``,
  ``base_pose(7)``, ``ee_pose(7)``, and the measured ``gripper_pos(1)``
  normalized by its joint range.
- **Action**: 7 normalized values for tool-frame XYZ translation, RPY rotation,
  and gripper motion.
- **Reward**: the environment's dense reach, alignment, contact, success, and
  action-penalty reward.

The 21-D observation intentionally excludes the target pose. Therefore this
baseline fixes the cube at ``[1.60, 0.00, 2.65]``. Random target placement
requires adding target information or visual observations.

Dependency Installation
-----------------------

Place the repositories next to each other or mount both into the container.
Then use the RLinf installer to create the environment and install the
simulator package in editable mode:

.. code-block:: bash

   export SPACEUR10E_REPO_ROOT=/workspace/my_simulation
   bash requirements/install.sh embodied --env spaceur10e
   source .venv/bin/activate
   export MUJOCO_GL=egl

Install Pinocchio from conda-forge when the platform's pip packages have an ABI
conflict.

Quick Start
-----------

From the RLinf repository root:

.. code-block:: bash

   export SPACEUR10E_REPO_ROOT=/workspace/my_simulation
   export MUJOCO_GL=egl
   bash examples/embodiment/run_embodiment.sh spaceur10e_ppo_mlp

The environment defaults to one MuJoCo instance. Increase
``env.train.total_num_envs`` only after the single-environment rollout works.
