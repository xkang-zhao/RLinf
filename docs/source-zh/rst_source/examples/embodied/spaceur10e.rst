基于 SpaceUR10e 的强化学习
===========================

该接入面向外部 ``my_simulation`` 仓库的 ``fix0613`` 分支，将其 MuJoCo/
Gymnasium 环境、自动采集任务注册表和三路 RGB 相机接入 RLinf。

任务与观测
----------

``my_simulation`` 有 6 个 Gym 场景和 8 个自动抓取任务定义。RLinf 以如下稳定
任务名暴露全部 8 个任务：

``cube``、``satellite_handle``、``satellite_left_antenna_panel``、
``satellite2_left_truss_connection``、``satellite3_left_antenna_panel``、
``satellite3_upper_rod``、``debris_antenna_panel`` 和 ``debris_truss``。

两组卫星任务分别共享一个 MuJoCo 场景，但语言目标不同。适配器会传出任务指令；
配置 ``task_names: all`` 时，会在向量环境和 worker 之间轮转这 8 个任务。要让
全部任务并发执行，请至少配置 8 个总仿真实例。

状态模式的观测为 27 维 float32，固定顺序是 ``joint_pos(6)``、
``base_pose(7)``、``ee_pose(7)``、``target_pose(7)``。包含 ``target_pose`` 后，
MLP 仍可观测到目标随机初始化。RGB 模式额外返回：

- 来自 ``third_left_camera`` 的 ``main_images``；
- 来自 ``left_wrist_camera`` 的 ``wrist_images``；
- 来自 ``third_right_camera`` 的 ``extra_view_images``；以及
- 每个环境一条任务指令的 ``task_descriptions``。

图像以 HWC ``uint8`` 张量传出，符合 RLinf 的 OpenPI/OpenVLA 观测格式。动作是
7 维归一化值，依次控制工具坐标系 XYZ 平移、RPY 旋转和夹爪运动。

依赖安装
--------

请使用 ``fix0613`` 工作区，在 RLinf 根目录执行：

.. code-block:: bash

   export SPACEUR10E_REPO_ROOT=/workspace/my_simulation
   bash requirements/install.sh embodied --env spaceur10e
   source .venv/bin/activate
   export MUJOCO_GL=egl

适配器会从 ``SPACEUR10E_REPO_ROOT``（或 ``env.repo_root``）解析注册场景的 MJCF
绝对路径，因此不依赖启动命令时的当前目录。若 PyPI 的 ``pin`` 包出现平台 ABI
冲突，请从 conda-forge 安装 Pinocchio。

快速开始
--------

内置 PPO MLP 配置默认在 cube 上使用 27 维状态训练：

.. code-block:: bash

   export SPACEUR10E_REPO_ROOT=/workspace/my_simulation
   export MUJOCO_GL=egl
   bash examples/embodiment/run_embodiment.sh spaceur10e_ppo_mlp

通过 Hydra 覆盖切换为其他任务：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh spaceur10e_ppo_mlp \
     env.train.task_name=satellite_handle env.eval.task_name=satellite_handle

视觉语言策略可以在其配置中引用 ``env/spaceur10e_all_rgb``。该环境片段启用
全部 8 个任务、三路相机和任务描述。

任务名与任务特定抓取点来自 ``my_simulation`` 的自动抓取规划器。Gym 环境使用
终局二值奖励：连续确认抓取成功时返回 ``1.0``，中间步骤和失败返回 ``0.0``。
共享同一场景的任务目前仍共用抓取成功判定；若要严格区分具体抓取部件，还需要
增加任务特定的成功判定。

OpenPI 监督微调
---------------

自动采集器输出 LeRobot v2.1 轨迹：13 维状态（``joint_pos(6) + ee_pose(7)``）、
原始 7 维增量动作，以及三路 H.264 相机流：``cam1``（third-left）、``cam2``
（third-right）、``cam3``（left-wrist）。``spaceur10e_sft_openpi_rlinf`` 会用
Pi0 同时训练三路相机，保留 8 个任务指令作为语言提示，并采用 10 步动作窗口。

下面的命令会将 8 个采集结果合并为一个数据集、计算训练需要的状态/动作统计，
然后启动全参数 SFT。合并工具优先创建视频硬链接，不会复制视频内容。请在预装的
RLinf OpenPI 环境中执行：

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

转换步骤会下载约 11.2 GiB 的公开 JAX Pi0 基座权重，并在
``SPACEUR10E_PI0_MODEL_PATH`` 写入训练所需的 ``model.safetensors``。
当前配置默认训练 10,000 步、每 5,000 步保存一次到
``RLinf/logs/<timestamp>-spaceur10e_sft_openpi_rlinf/spaceur10e_pi0_sft/``；
可通过覆盖 ``actor.global_batch_size``、
``actor.micro_batch_size`` 或 ``runner.max_steps`` 调整。在线评估使用
``env/spaceur10e_all_rgb``：该配置保留采集动作的原始单位，模型会从环境的
27 维状态中自动选出采集器使用的 13 个坐标。

OpenPI 仿真评测
---------------

``evaluations/spaceur10e/spaceur10e_pi0_rlinf_eval.yaml`` 提供与 LIBERO 一致的
RLinf ``embodied_eval`` 入口：它加载微调后的策略、运行 MuJoCo 环境，并直接汇总
环境给出的 ``info["is_success"]``。默认配置以固定种子让全部 8 个任务各执行一条
轨迹（一个 GPU 上由一个 worker 持有 8 个并行环境），最终指标为
``eval/success_once``。先将 SFT 的全精度导出权重转换为部署格式：
混合任务评测还会记录 ``eval/task/<任务名>/success_once``，因此可直接得到每个
任务的独立成功率。

.. code-block:: bash

   cd /workspace/RLinf
   source /usr/local/bin/switch_env openpi
   export PYTHONPATH=$PWD:/workspace/my_simulation/src:$PYTHONPATH
   export EMBODIED_PATH=$PWD/examples/embodiment
   export SPACEUR10E_REPO_ROOT=/workspace/my_simulation
   export MUJOCO_GL=egl

当前多策略镜像中 OpenPI 环境固定使用 NumPy 1.x，而支持 MJCF 的 Pinocchio 4 位于
OpenVLA 环境。若 ``python -c 'import pinocchio as p; assert hasattr(p.RobotWrapper,
"BuildFromMJCF")'`` 失败，可在启动命令前复用该已安装的二进制：

.. code-block:: bash

   export SPACEUR10E_PINOCCHIO_PREFIX=/opt/venv/openvla/lib/python3.11/site-packages/cmeel.prefix
   export PYTHONPATH=$SPACEUR10E_PINOCCHIO_PREFIX/lib/python3.11/site-packages:$PYTHONPATH
   export LD_LIBRARY_PATH=$SPACEUR10E_PINOCCHIO_PREFIX/lib:$LD_LIBRARY_PATH

不要在该 OpenPI 环境中直接安装最新的 PyPI ``pin``；它会要求 NumPy 2.x，和
OpenPI 的 NumPy 1.x 约束不兼容。其他镜像请安装包含
``RobotWrapper.BuildFromMJCF`` 的 Pinocchio，再运行同一命令。

   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode sft_to_openpi_rlinf \
     --ckpt logs/spaceur10e_sft_3000/pi0_full_sft_3000/checkpoints/global_step_3000 \
     --input-norm-stats /workspace/checkpoints/pi0_base_pytorch/spaceur10e/multitask/norm_stats.json \
     --output-model logs/spaceur10e_sft_3000/pi0_full_sft_3000/deploy/openpi_rlinf_bf16 \
     --output-norm-stats logs/spaceur10e_sft_3000/pi0_full_sft_3000/deploy/openpi_rlinf_bf16/spaceur10e/multitask/norm_stats.json \
     --config-name pi0_spaceur10e --dtype bf16

随后运行评测：

.. code-block:: bash

   python evaluations/eval_embodied_agent.py \
     --config-path spaceur10e --config-name spaceur10e_pi0_rlinf_eval

可通过 ``env.eval.total_num_envs``、``env.eval.rollout_epoch`` 和
``env.eval.seed`` 扩大样本量或改变测试种子；用
``rollout.model.model_path=/path/to/converted_checkpoint`` 测试另一份转换后的权重。
在固定 ``env.eval.seed`` 下，每个重复 rollout 会使用新的确定性 reset 种子，因此
每个任务的 10 次评测具有不同且可复现的目标随机初始化。

OpenPI 稀疏奖励 PPO
-------------------

``spaceur10e_ppo_openpi_rlinf`` 从 3000 步 SFT 部署权重重新开始 100 轮 PPO。
环境只在连续确认抓取成功时给终局奖励 ``1``，其余步骤为 ``0``。稳定版配置每轮
为 8 个任务各采样 4 条轨迹，并使用较低的 actor 学习率（``5e-7``）、20 个
optimizer step 的 critic 预热，以及原始 800 条示范数据的
SFT 联合训练约束。Embodied OpenPI 当前没有计算参考策略 log-prob，因此
``kl_beta`` 保持为零，不能把非零配置值当作实际 KL 约束。
联合 SFT 每个全局 PPO batch 执行一次，并按梯度累积倍数补偿 loss 权重，避免
每个 micro-batch 都解码三路视频而显著拖慢训练。
实测将 SDE 噪声从 ``0.4`` 降为 ``0.15`` 会使 bf16 重算误差对应的首轮
approximate KL 从 ``1.20`` 放大至 ``12.88``，因此配置保留 ``0.4``。

.. code-block:: bash

   cd /workspace/RLinf
   source /usr/local/bin/switch_env openpi
   export SPACEUR10E_PINOCCHIO_PREFIX=/opt/venv/openvla/lib/python3.11/site-packages/cmeel.prefix
   export PYTHONPATH=$SPACEUR10E_PINOCCHIO_PREFIX/lib/python3.11/site-packages:$PWD:/workspace/my_simulation/src:$PYTHONPATH
   export LD_LIBRARY_PATH=$SPACEUR10E_PINOCCHIO_PREFIX/lib:$LD_LIBRARY_PATH
   export SPACEUR10E_REPO_ROOT=/workspace/my_simulation
   export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
   bash examples/embodiment/run_embodiment.sh spaceur10e_ppo_openpi_rlinf SPACEUR10E
