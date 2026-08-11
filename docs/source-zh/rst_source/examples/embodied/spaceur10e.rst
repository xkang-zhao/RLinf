基于 SpaceUR10e 的强化学习
===========================

本示例将外部 SpaceUR10e MuJoCo/Gymnasium 仿真包接入 RLinf，并在 Cube
抓取任务上使用 PPO 训练纯状态 MLP 策略。

环境
----

- **环境**：``SpaceUR10e-Cube-v0``；
- **观测**：21 维 float32，顺序固定为 ``joint_pos(6)``、``base_pose(7)``、
  ``ee_pose(7)`` 和按关节范围归一化的实际 ``gripper_pos(1)``；
- **动作**：7 维归一化动作，依次控制工具坐标系 XYZ 平移、RPY 旋转和夹爪；
- **奖励**：底层环境提供的接近、对齐、接触、成功和动作惩罚稠密奖励。

21 维观测按设计不包含目标位姿，因此该基线把 Cube 固定在
``[1.60, 0.00, 2.65]``。若恢复目标位置随机化，必须加入目标信息或视觉观测。

依赖安装
--------

将两个仓库放在相邻目录或同时挂载进容器，然后使用 RLinf 安装脚本创建环境并
以 editable 模式安装仿真包：

.. code-block:: bash

   export SPACEUR10E_REPO_ROOT=/workspace/my_simulation
   bash requirements/install.sh embodied --env spaceur10e
   source .venv/bin/activate
   export MUJOCO_GL=egl

如果平台上的 pip 包存在 ABI 冲突，建议通过 conda-forge 安装 Pinocchio。

快速开始
--------

在 RLinf 仓库根目录运行：

.. code-block:: bash

   export SPACEUR10E_REPO_ROOT=/workspace/my_simulation
   export MUJOCO_GL=egl
   bash examples/embodiment/run_embodiment.sh spaceur10e_ppo_mlp

配置默认只创建一个 MuJoCo 环境。单环境 rollout 跑通后，再逐步增加
``env.train.total_num_envs``。
