# SpaceUR10e：GPU 分组批量 PPO

`spaceur10e_warp_ppo_openpi_rlinf` 使用一个 EnvWorker 管理 32 个
MuJoCo Warp world，覆盖八个任务，每任务四个环境。按六种场景分组，
同组 world 在 GPU 上并行，不同场景组依次执行；不是异构场景全并行。
`spaceur10e_cube_warp_ppo_openpi_rlinf` 保留为 Cube 单场景基线。
正式配置 `spaceur10e_warp_ppo_formal_openpi_rlinf` 使用一个 EnvWorker
在 GPU 0 上管理 64 个环境和八个任务场景。Actor 使用 micro/global batch 8/128，
每轮 896 个动作块对应七个 global batch。该配置参考 LIBERO 的大批量设计，
但没有照搬其面向更大 GPU 配置的 128/2048。
物理和 IK 候选在 CUDA 上批量执行；IK 不收敛时仍回退到 CPU Pinocchio。
现有 CPU 多任务配置 `spaceur10e_ppo_openpi_rlinf` 不受影响。

## 本机启动

本机已建立 `/workspace/SpaceRobotEnv/.venv-warp-rlinf`：它通过
`openpi_base.pth` 复用 `/opt/venv/openpi` 的 PyTorch、OpenPI、Pinocchio 等依赖，
在自身目录安装 MuJoCo 3.12.0、MuJoCo Warp 3.12.0、Warp 1.17.0 和
Gymnasium 1.2.3。原 openpi 环境未升级。此环境依赖原 openpi 环境继续存在。

```bash
cd /workspace/RLinf
source /workspace/SpaceRobotEnv/.venv-warp-rlinf/bin/activate
export GLIBC_TUNABLES="glibc.malloc.arena_max=2:glibc.malloc.trim_threshold=131072"
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=131072
bash examples/embodiment/run_embodiment.sh spaceur10e_warp_ppo_openpi_rlinf
```

默认训练 100 个 PPO step，保存间隔 100，暂时关闭定期评估。
每轮 32 个环境 × 2 轮采样 × 420 控制步，动作块长度 30，共 896 个动作块；
`global_batch_size=64`，每个更新 epoch 为 14 个 global batch。
Actor/rollout 与环境共用 GPU 0，Actor/rollout 启用 offload 为仿真留出显存。
SFT co-training 继续使用原多任务数据集，`actor.sft_num_workers=0`，
避免在仿真之外再启动四个视频读取进程。其他 PPO 配置未指定该字段时，
仍使用 OpenPI TrainConfig 中的 worker 数。

模型和数据的本机绝对路径保留在新 YAML 中；迁移机器时需修改。
`STEPS`、`SAVE_INTER` 环境变量会覆盖启动脚本中的步数和保存间隔。

## RGB 和终止语义

GPU 物理批次每个控制周期仍执行 100 个物理子步，20 Hz 控制频率不变。
新配置设置 `warp_substeps_per_graph: 10`：每张图包含 10 个子步，重复执行
10 次，之后只更新一次成功计数。设为 100 可对照原来的完整展开图。
底层默认仍为 100；参数必须是 100 的正整数因子。此项不改变仿真精度或时长。

训练配置默认保持 `deterministic_physics: false`，使用 CUDA Graph 高吞吐路径。
严格复现实验可设置为 true：它启用 Warp `RUN_TO_RUN` 的稳定归约顺序，替换
三个树形浮点原子归约，并直接启动每个物理子步。严格模式刻意不使用 CUDA
Graph，因为 Warp 1.17 的确定性 scatter 临时缓冲在图回放间不能可靠复位。
该选项是进程级设置，同一 Python 进程不能混用 true/false；切换时需重启进程。
严格模式明显更慢，适用于回归和配对评测，不建议用于 PPO 训练。第一次运行还会
编译带额外确定性记录容量的 CCD 内核，之后从 Warp 缓存加载。

返回原有 27 维机器人/目标状态、三路 640×480 RGB 和对应任务指令；
Pi0.5 的 `state_indices` 继续选择原有机器人状态维度。

目前每种场景使用一个普通 MuJoCo EGL Renderer，逐 world 渲染；不是批量渲染。
一次 30 步动作块内，只在 world 结束或动作块末尾渲染。环境的 `chunk_step`
返回单项末端观测列表、完整 `[num_envs, chunk_steps]` reward/done 和逐步 infos，
兼容 RLinf EnvWorker 取最后观测的逻辑。不会给每个物理控制步保留三路 RGB。
因此本适配器拒绝逐步录像和 `data_collection.enabled`；数据采集应使用
SpaceRobotEnv 自带的 `gpuwarp_collect_autograb_lerobot_v21.py`。

结束后的 world 在动作块内保持最终观测与积分状态，奖励仅在新成功时给出，
其他 world 继续运行。当前 Warp 仍计算固定大小批次，再恢复暂停 world 的状态，
尚未实现按活跃 world 压缩计算。支持局部 reset，自动 reset 在动作块末尾进行，
通过 `final_observation/final_info` 保留结束时数据。

## 验证与环境测时

```bash
cd /workspace/RLinf
source /workspace/SpaceRobotEnv/.venv-warp-rlinf/bin/activate
export PYTHONPATH=/workspace/RLinf:/workspace/SpaceRobotEnv/src
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python toolkits/standalone_eval_scripts/smoke_spaceur10e_warp.py --num-envs 32 --tasks all

cd /workspace/SpaceRobotEnv
RUN_WARP_TESTS=1 python -m pytest \
  /workspace/RLinf/tests/unit_tests/test_spaceur10e_warp_env.py \
  test/test_warp_vector_env.py -q
```

环境测试使用零动作，不代表抓取成功率或 PPO 吞吐。首次运行还包含内核编译和
CUDA Graph 捕获。扩大到 64/128 个环境前，需要分别评估仿真、RGB 渲染、
模型推理和训练显存；当前实现不承诺端到端加速比例。

任务指令与成功判定沿用 CPU 适配器：同场景的不同抓取指令不额外引入
抓取部位奖励。这不是对八种任务抓取成功率或长期 PPO 收敛性的验证。
Cube 的 32 环境、60 控制步、1 轮采样、1 个 PPO step 联调已完成：
step 77.51 秒，其中 rollout 39.93 秒、训练 23.74 秒、权重同步 13.78 秒。
这是短程冒烟测试，不能当作默认 420 控制步、多任务配置的耗时。

八任务 32 环境的同规格短程联调也已完成（2026-09-17，micro batch 2，
critic warmup 0，启用 SFT 混合训练）：step 88.95 秒，rollout 66.88 秒，
训练 14.77 秒，权重同步 7.21 秒。环境交互计时 50.95 秒，
`sft_loss=0.000690`。日志：`/tmp/spaceur10e-all-warp-ppo-smoke.Z42FW9/run.log`。
32 条轨迹均在 60 步截断、成功率为 0；仅验证链路，不代表完整回合效果。
正式配置保留 critic warmup 20，训练 100 step，仅在第 100 step 保存。
不同场景分组开销和逐 world 渲染依然存在，尚不能承诺优于 CPU 多进程。

八任务纯环境零动作测时（32 world、每块 30 控制步、三路 RGB）：
初始化 21.63 秒，reset 2.19 秒，首块 21.11 秒、第二块 11.93 秒，
第二块约 80.50 world-control-step/s；该独立进程 PSS 4.75 GiB。
这是单次两块采样，包含分组物理、IK、CPU/GPU 传输和边界渲染，
不含策略推理，不是长时间稳定吞吐基准。

MuJoCo Warp 会提示部分碰撞形状组合不支持 MULTICCD，多接触生成可能与
CPU MuJoCo 不同。接口和初始状态测试不能证明接触动力学完全等价；
正式切换训练前仍需对照 CPU 评估完整抓取回合和各任务成功率。

## 2026-09-18 完整抓取对照

使用既有自动抓取 planner，按任务顺序使用 seed 7–14；不是 Pi0.5 策略评估。
CPU 和 Warp（每图 10 子步）各八次尝试均达到环境连续成功计数 10。

| 任务 | CPU 控制步 | Warp 控制步 |
| --- | ---: | ---: |
| cube | 96 | 96 |
| satellite_handle | 79 | 79 |
| satellite_left_antenna_panel | 170 | 174 |
| satellite2_left_truss_connection | 102 | 102 |
| satellite3_left_antenna_panel | 139 | 138 |
| satellite3_upper_rod | 90 | 88 |
| debris_antenna_panel | 86 | 86 |
| debris_truss | 107 | 109 |

差异说明不能要求接触轨迹逐位相同；每任务一次也不是成功率统计。
诊断中的 GPU planner 重建 CPU 运动学，不应把它的总耗时当作训练吞吐。
八环境分六组时每组过小，GPU 不一定更快；本轮 CPU 顺序诊断 30.34 秒，
GPU 分组诊断 82.34 秒（均含创建和关闭，不含 RGB）。

复现（不生成数据集、不修改权重）：

```bash
cd /workspace/SpaceRobotEnv
source .venv-warp-rlinf/bin/activate
python scripts/validate_warp_tasks.py --backend cpu
python scripts/validate_warp_tasks.py --backend warp --substeps-per-graph 10
```

原始日志目录：`/tmp/spaceur10e-warp-validation.DAzbZ7`。
本轮 16 项 CUDA/接口/图回放测试、7 项 CPU 适配器回归、3 项诊断成功判定
测试通过；图回放测试包括局部 reset、暂停 world、仿真时间与状态一致性。
4 环境短程对照中每图 100/10 子步的 qpos、qvel、ctrl、time 匹配测试容差；
首次步进约 1.60/0.19 秒，稳定步进约 62.3/59.7 毫秒。这不是 PPO 加速比例。
尝试过 CUDA 条件图，但当前 MuJoCo Warp 的临时分配与条件图不兼容，
失败实现已移除，没有禁用内存池或更改安装的依赖。

128 环境扩展测试（三路 RGB，零动作，每图 10 子步，3 个 30 步动作块）：
初始化 23.78 秒，reset 8.09 秒；动作块耗时 15.53 / 14.62 / 14.66 秒，
末块约 261.91 world-control-step/s。三次 PSS 采样均约 5.166 GiB，
未出现非有限状态或容量溢出。仅覆盖 90 控制步，不是长时间内存稳定性证明；
也不包含 128 环境模型推理和 PPO 显存。因此训练默认仍保留 32 环境。

```bash
cd /workspace/RLinf
export PYTHONPATH=/workspace/RLinf:/workspace/SpaceRobotEnv/src
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
python toolkits/standalone_eval_scripts/smoke_spaceur10e_warp.py \
  --tasks all --num-envs 128 --chunks 3 --substeps-per-graph 10
```

### 完整时域 PPO 单步验证

同日运行 32 环境、单轮采样、回合上限 420 控制步（14 个动作块），
micro batch 2、critic warmup 0、SFT 混合训练开启、不保存权重。
程序退出码 0，测试结束 GPU 已释放。

- PPO step 403.715 秒（不含模型初始化）；采样总阶段 315.9 秒，
  其中环境交互 262.2 秒、策略推理 30.825 秒。
- Actor 更新 78.826 秒，权重同步 8.770 秒。
- 32 条轨迹，21 条成功（65.625%），平均回合长度 207.6875。
  这是一次训练采样，不能作为独立评测成功率或长期收敛结论。
- `actor/total_loss=0.016`，`sft_loss≈0.0012`，`approx_kl≈0.965`。
  仍有 IK 未完全收敛警告，KL 也需要在正式训练中监控。
- `loss_ratio=inf` 来自 `_train_sft_epoch` 的现有诊断分支：某个微批的
  PPO loss 为零时主动记录 inf；该比值不参与优化，总损失不是 inf。
  聚合后的 PPO loss 非零并不意味着所有微批 PPO loss 都非零。

日志：`/tmp/spaceur10e-warp-validation.DAzbZ7/ppo420.log`。
正式配置仍然是 **rollout_epoch=2、critic warmup=20、100 个 PPO step**，
不能把单轮测试的 403.7 秒当作默认两轮采样的每步耗时。
环境交互仍是瓶颈；此轮验证未证明优于 CPU 多进程的同工作量基线。

## CPU / Warp 配对策略评测

### 确定性渲染诊断

对照脚本默认在所有路径统一开启确定性 RGB 模式，
可用 `--no-deterministic-rendering` 复现旧模式；该选择也会写入 metadata。
训练/评估环境配置对应
`env.train.deterministic_rendering` / `env.eval.deterministic_rendering`，默认 false。
此模式关闭 MSAA/dithering，并消除 MuJoCo 无限地面显示位置对上一相机的依赖。
不改变物理参数或成功条件，但会改变图像的边缘抗锯齿效果；它不是旧图像的
逐像素等价替代，也不是 CPU/Warp 物理轨迹完全一致的保证。先在诊断评测中使用。
不要仅靠设定随机种子宣称可重复：需分别检查固定输入推理、reset 状态/RGB、
固定控制回放和完整闭环回合。旧非确定性渲染评测仅保留为历史结果。

对照脚本还默认给 Warp 路径开启 `--deterministic-physics`；可用
`--no-deterministic-physics` 恢复 CUDA Graph 快速路径。该值写入
`metadata.json`。CPU MuJoCo 不读取此选项。严格物理模式保证相同 GPU、软件栈
和输入下的运行间位级复现，不表示 CPU 与 Warp 的 FP64/FP32 接触轨迹会相同。

诊断时可追加 `--include-cpu-ik --trace-steps`：每批依次运行 CPU、Warp + GPU
混合 IK、Warp + CPU IK 三条路径。保留原来的批量 32 和种子顺序，默认共
240 个后端回合；不要用缩小 batch 的结果替代原失败回合的复现。
`trace_<backend>_<首回合索引>_<动作块索引>.npz` 保存逐控制步的状态、动作、
实际控制信号、终止标志、接触、成功计数和 IK 诊断，数组前两维为 step/world。
`input_states` 是该动作块推理前状态，`indices` 对应全局回合索引。
缺失指标为 NaN，已结束 world 必须通过 `active` 掩码排除。轨迹不含 RGB；
记录会增加运行开销，不能当作无仪器开销的吞吐基准。
`summary.json` 中的 `cpu_ik_ablation` 汇总三条路径的逐回合结果。
这仍是闭环策略对照，不是固定动作/控制回放，不能单独证明物理误差的来源。

新增 `python -m toolkits.standalone_eval_scripts.compare_spaceur10e_backends`。
必须从 RLinf 根目录以 `-m` 调用，避免同目录的 `openpi/` 工具包遮蔽
已安装的 OpenPI 依赖。两侧使用同一个已加载的只读 SFT 模型、同一虚拟环境、
eval ODE sampler（5 步）、30 步动作块、相同环境种子和逐回合逐动作块噪声。
默认每任务 10 回合、回合上限 420 步、批量 32，分为 32/32/16 三批。
全局回合 i 的 seed 为 `20260918+i`，任务按八任务注册顺序轮转。

输出目录必须不存在；目录内保存 `metadata.json`、逐批刷新的
`episodes.jsonl` 和全部结束后生成的 `summary.json`。日志输出每块进度。
此脚本不训练、不保存模型。CPU 路径按环境串行交互、每控制步渲染，Warp
路径按场景分组、仅动作块边界渲染；本轮用途是效果一致性，不是公平并行吞吐排名。

```bash
cd /workspace/RLinf
source /workspace/SpaceRobotEnv/.venv-warp-rlinf/bin/activate
export PYTHONPATH=/workspace/RLinf:/workspace/SpaceRobotEnv/src
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
python -m toolkits.standalone_eval_scripts.compare_spaceur10e_backends \
  --model-path /workspace/RLinf/logs/20260911-06:48:22-spaceur10e_sft_openpi_rlinf_pi05/spaceur10e_pi05_sft/deploy/openpi_rlinf_bf16 \
  --output /tmp/spaceur10e-paired-eval-new \
  --episodes-per-task 10 --batch-size 32 --max-steps 420
```
