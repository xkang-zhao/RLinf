RL with SpaceUR10e
==================

This integration connects the ``fix0613`` branch of the external
``my_simulation`` MuJoCo/Gymnasium project to RLinf. It supports the task
registry used by that repository's automatic data collector, together with its
three RGB camera streams.

Tasks and observations
----------------------

``my_simulation`` provides six Gym scenes and eight task-level automatic-grasp
definitions. RLinf exposes these eight stable task names:

``cube``, ``satellite_handle``, ``satellite_left_antenna_panel``,
``satellite2_left_truss_connection``, ``satellite3_left_antenna_panel``,
``satellite3_upper_rod``, ``debris_antenna_panel``, and ``debris_truss``.

The two satellite task pairs share a MuJoCo scene but have different language
goals. The adapter forwards the selected task's instruction and cycles task
names across vector environments and workers when ``task_names: all`` is
configured. Use at least eight total simulator instances to run all task names
concurrently.

State mode returns 27 float32 values in this order:
``joint_pos(6)``, ``base_pose(7)``, ``ee_pose(7)``, and ``target_pose(7)``.
Keeping ``target_pose`` means target-position randomization remains observable
to an MLP policy. RGB mode additionally returns:

- ``main_images`` from ``third_left_camera``;
- ``wrist_images`` from ``left_wrist_camera``;
- ``extra_view_images`` from ``third_right_camera``; and
- ``task_descriptions`` with one instruction per environment.

Images are HWC ``uint8`` tensors, following RLinf's OpenPI/OpenVLA-compatible
observation schema. Actions are seven normalized values for tool-frame XYZ
translation, RPY rotation, and gripper motion.

Dependency installation
-----------------------

Use the ``fix0613`` checkout and install it from the RLinf repository root:

.. code-block:: bash

   export SPACEUR10E_REPO_ROOT=/workspace/my_simulation
   bash requirements/install.sh embodied --env spaceur10e
   source .venv/bin/activate
   export MUJOCO_GL=egl

The adapter resolves registered MJCF paths from ``SPACEUR10E_REPO_ROOT`` (or
``env.repo_root``), so it does not rely on the process working directory.
Install Pinocchio from conda-forge if the PyPI ``pin`` package has a platform
ABI conflict.

Quick start
-----------

The included PPO MLP baseline uses the cube task and the 27-D state:

.. code-block:: bash

   export SPACEUR10E_REPO_ROOT=/workspace/my_simulation
   export MUJOCO_GL=egl
   bash examples/embodiment/run_embodiment.sh spaceur10e_ppo_mlp

Select another task with Hydra overrides:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh spaceur10e_ppo_mlp \
     env.train.task_name=satellite_handle env.eval.task_name=satellite_handle

For a vision-language policy, use the environment fragment
``env/spaceur10e_all_rgb`` in that policy's config. It selects all eight
tasks, returns the three cameras, and supplies their task descriptions.

The task names and profile-specific grasp points originate in
``my_simulation``'s automatic-grasp planner. The Gym environment emits a
terminal binary reward: ``1.0`` for a confirmed grasp and ``0.0`` for every
intermediate or failed step. Tasks that share a scene still share the grasp
success predicate; distinguishing exact parts requires task-specific success
predicates.

OpenPI supervised fine-tuning
-----------------------------

The automatic collector writes LeRobot v2.1 trajectories with a 13-D state
(``joint_pos(6) + ee_pose(7)``), raw 7-D delta actions, and three H.264 camera
streams: ``cam1`` (third-left), ``cam2`` (third-right), and ``cam3``
(left-wrist). ``spaceur10e_sft_openpi_rlinf`` trains Pi0 on all three cameras,
keeps the eight task instructions as language prompts, and uses a 10-step
action horizon.

The following commands merge the eight collector outputs without duplicating
their videos (the merger hard-links files when possible), compute the required
state/action statistics, and start full-parameter SFT. Run them in the preinstalled
RLinf OpenPI environment:

.. code-block:: bash

   cd /workspace/RLinf
   source switch_env openpi
   export PYTHONPATH=$PWD:$PYTHONPATH

   python toolkits/lerobot/merge_lerobot_datasets.py \
     --source-dir /workspace/my_simulation/dataset \
     --output-dir /workspace/my_simulation/dataset/spaceur10e_multitask_lerobot_v21

   export SPACEUR10E_PI0_MODEL_PATH=/workspace/checkpoints/pi0_base_pytorch
   python toolkits/lerobot/calculate_norm_stats.py \
     --config-name pi0_spaceur10e \
     --repo-id /workspace/my_simulation/dataset/spaceur10e_multitask_lerobot_v21 \
     --model-path $SPACEUR10E_PI0_MODEL_PATH --numeric-only

   gsutil -m cp -r gs://openpi-assets/checkpoints/pi0_base /workspace/checkpoints/
   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi_rlinf \
     --input-model /workspace/checkpoints/pi0_base \
     --input-norm-stats $SPACEUR10E_PI0_MODEL_PATH/spaceur10e/multitask/norm_stats.json \
     --output-model $SPACEUR10E_PI0_MODEL_PATH \
     --output-norm-stats $SPACEUR10E_PI0_MODEL_PATH/spaceur10e/multitask/norm_stats.json \
     --no-pi05 --action-dim 32 --action-horizon 10 --max-token-len 48

   bash examples/sft/run_vla_sft.sh spaceur10e_sft_openpi_rlinf

The conversion downloads 11.2 GiB of public JAX base weights and writes the
``model.safetensors`` required at ``SPACEUR10E_PI0_MODEL_PATH``. The supplied
configuration saves every 5,000 steps beneath
``RLinf/logs/<timestamp>-spaceur10e_sft_openpi_rlinf/spaceur10e_pi0_sft/`` and
can be safely scaled by overriding
``actor.global_batch_size``, ``actor.micro_batch_size``, or ``runner.max_steps``.
For online evaluation, use ``env/spaceur10e_all_rgb``: it preserves native
demonstration action units and the model selects the collector's 13 state
coordinates from the environment's 27-D state.

OpenPI simulation evaluation
----------------------------

``evaluations/spaceur10e/spaceur10e_pi0_rlinf_eval.yaml`` is an RLinf
``embodied_eval`` entry point equivalent to the LIBERO evaluators. It loads the
fine-tuned policy, drives the MuJoCo environments, and aggregates the
environment's own ``info["is_success"]`` signal. The default uses a fixed seed
and runs one trajectory for each of the eight tasks (one worker owns eight
parallel environments on a single GPU); the primary metric is
``eval/success_once``. First convert the SFT full weights to the deployment
format:
Mixed-task evaluation also records ``eval/task/<task_name>/success_once``, so
each task's individual success rate is available directly.

.. code-block:: bash

   cd /workspace/RLinf
   source /usr/local/bin/switch_env openpi
   export PYTHONPATH=$PWD:/workspace/my_simulation/src:$PYTHONPATH
   export EMBODIED_PATH=$PWD/examples/embodiment
   export SPACEUR10E_REPO_ROOT=/workspace/my_simulation
   export MUJOCO_GL=egl

In the current multi-policy image, OpenPI is pinned to NumPy 1.x while the
MJCF-capable Pinocchio 4 installation lives in the OpenVLA environment. If
``python -c 'import pinocchio as p; assert hasattr(p.RobotWrapper,
"BuildFromMJCF")'`` fails, reuse that installed binary before launching:

.. code-block:: bash

   export SPACEUR10E_PINOCCHIO_PREFIX=/opt/venv/openvla/lib/python3.11/site-packages/cmeel.prefix
   export PYTHONPATH=$SPACEUR10E_PINOCCHIO_PREFIX/lib/python3.11/site-packages:$PYTHONPATH
   export LD_LIBRARY_PATH=$SPACEUR10E_PINOCCHIO_PREFIX/lib:$LD_LIBRARY_PATH

Do not install the latest PyPI ``pin`` directly into that OpenPI environment:
it requires NumPy 2.x, which conflicts with OpenPI's NumPy 1.x constraint. On
other images, install a Pinocchio build exposing ``RobotWrapper.BuildFromMJCF``
and use the same command.

   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode sft_to_openpi_rlinf \
     --ckpt logs/spaceur10e_sft_3000/pi0_full_sft_3000/checkpoints/global_step_3000 \
     --input-norm-stats /workspace/checkpoints/pi0_base_pytorch/spaceur10e/multitask/norm_stats.json \
     --output-model logs/spaceur10e_sft_3000/pi0_full_sft_3000/deploy/openpi_rlinf_bf16 \
     --output-norm-stats logs/spaceur10e_sft_3000/pi0_full_sft_3000/deploy/openpi_rlinf_bf16/spaceur10e/multitask/norm_stats.json \
     --config-name pi0_spaceur10e --dtype bf16

Then run the evaluator:

.. code-block:: bash

   python evaluations/eval_embodied_agent.py \
     --config-path spaceur10e --config-name spaceur10e_pi0_rlinf_eval

Increase ``env.eval.total_num_envs`` or ``env.eval.rollout_epoch`` to collect
more trials, change ``env.eval.seed`` for another evaluation seed, or override
``rollout.model.model_path=/path/to/converted_checkpoint`` to evaluate another
converted model.
With a fixed ``env.eval.seed``, each repeated rollout uses a new deterministic
reset seed. Ten trials per task therefore have different randomized target
initializations that are reproducible.

OpenPI sparse-reward PPO
------------------------

``spaceur10e_ppo_openpi_rlinf`` restarts from the 3000-step SFT deployment for
100 PPO iterations. The environment emits a terminal reward of ``1`` only after
a confirmed grasp and ``0`` otherwise. The stability settings collect four
trajectories per task per iteration, lower the actor learning rate to ``5e-7``
and warm up the critic for 20 optimizer steps, and co-train on the original
800 demonstrations. Embodied OpenPI does not yet
compute reference-policy log-probabilities, so ``kl_beta`` remains zero; setting
it alone would not provide a real KL constraint.
Co-training runs one SFT batch per global PPO batch and compensates its loss for
gradient accumulation, avoiding a costly three-camera SFT pass per micro-batch.
Reducing SDE noise from ``0.4`` to ``0.15`` increased the measured first-step
approximate KL from ``1.20`` to ``12.88`` by amplifying bf16 recomputation error,
so the configuration retains ``0.4``.
