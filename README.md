# TSR-PPO 任务拆分滚动调度正式实验项目

本项目研究“考虑任务拆分和非等效并行产线的滚动调度及深度强化学习求解”。问题不是“纵剪 -> 横切 -> 复合剪”的顺序型混合流水车间，而是：每个原料批次只属于一种工艺类型，并在该工艺内部的非等效并行机集合中加工；同一任务在不同机器上的加工时间为 `p[j,k]`，任务可按比例拆分到多台合格机器。

## 1. 依赖文件

普通依赖：

```bash
pip install -r requirements.txt
```

CUDA 11.8 训练电脑 PyTorch 依赖参考：

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

注意：普通 `requirements.txt` 不包含 `torch`，避免覆盖已经安装好的 CUDA 版 PyTorch。`requirements-cuda-cu118.txt` 仅作为 CUDA 训练环境说明和兼容记录，推荐优先使用上面的官方 `--index-url` 命令安装。

## 2. 训练电脑部署与运行

当前推荐环境：

- OS: Windows 11 64-bit
- Python: 3.12.4
- GPU: NVIDIA GeForce RTX 3060 12GB
- PyTorch: 2.7.1+cu118
- CUDA runtime shown by PyTorch: 11.8

创建虚拟环境：

```bash
python -m venv .venv
```

激活虚拟环境：

```bash
.venv\Scripts\activate
```

安装 CUDA 11.8 版 PyTorch：

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

验证 CUDA：

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

安装普通依赖：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

quick_debug：

```bash
python -m src.experiments.train --quick_debug --device cuda
```

20 轮小测试：

```bash
python -m src.experiments.train --episodes 20 --scale small --seed 42 --device cuda --output_dir results/test_small_20_cuda
```

1000 轮正式训练：

```bash
python -m src.experiments.train --episodes 1000 --scale medium --seed 42 --device cuda --output_dir results/formal_medium_seed42
```

推荐分阶段训练流程：

```bash
# 1. 代码链路检查
python -m src.experiments.train --preset quick_debug --device cuda

# 2. CUDA、日志、checkpoint、自动画图检查
python -m src.experiments.train --preset smoke_test --device cuda

# 3. 小规模看趋势
python -m src.experiments.train --preset pilot_small --device cuda

# 4. 中规模初步调参
python -m src.experiments.train --preset pilot_medium --device cuda

# 5. 正式 medium
python -m src.experiments.train --preset formal_medium --device cuda
```

## 3. 正式代码结构

```text
src/
  data/
    generator.py              # 生成 train/val/test 实例
    dataset.py                # 加载或自动生成数据集
  env/
    scheduling_env.py         # 滚动调度环境、动作掩码、拆分、reward、raw schedule
  agents/
    ppo.py                    # PyTorch PPO，支持 checkpoint/resume
    tsr_ppo.py                # TSR-PPO 与消融开关
  baselines/
    fifo.py spt.py eat.py ga.py
  experiments/
    train.py                  # 正式训练入口
    evaluate.py               # 同一 test_instances 上评估算法
    run_formal_experiments.py
    run_sensitivity.py
    run_ablation.py
  visualization/
    plot_training.py
    plot_gantt.py
    plot_non_equivalent_machine.py
    plot_heterogeneity.py
  utils/
    io.py logger.py metrics.py seed.py
```

旧版 `src/experiment.py` 等文件保留为兼容入口；正式论文实验请使用 `python -m src.experiments.*`。

## 4. 数据集生成

```bash
python -m src.data.generator --scale medium --seed 42 --output_dir data/generated/formal_medium_seed42
```

固定规模：

- `small`: 50 任务，约 9 台机器，3-4 个滚动周期
- `medium`: 150 任务，约 15 台机器，5-6 个滚动周期
- `large`: 300 任务，约 21 台机器，8-10 个滚动周期

默认工艺比例：横切 70%，纵剪 20%，复合剪 10%。每个实例保存 `instance_id/job_id/process_type/release_time/due_time/quantity/workload/eligible_machines/p[j,k]/rolling_period`。

## 5. 训练入口

quick_debug 默认目录会按设备区分：

```bash
python -m src.experiments.train --quick_debug --device cuda
```

若没有显式传入 `--output_dir`：

- quick_debug 默认输出到 `results/quick_debug_cuda` 或 `results/quick_debug_cpu`
- 正式训练默认输出到 `results/formal_<scale>_seed<seed>`

正式 medium：

```bash
python -m src.experiments.train --episodes 1000 --scale medium --seed 42 --device cuda --output_dir results/formal_medium_seed42
```

正式 small：

```bash
python -m src.experiments.train --episodes 1000 --scale small --seed 42 --device cuda --output_dir results/formal_small_seed42
```

正式 large：

```bash
python -m src.experiments.train --episodes 1000 --scale large --seed 42 --device cuda --output_dir results/formal_large_seed42
```

基础 PPO baseline：

```bash
python -m src.experiments.train --episodes 1000 --scale medium --seed 42 --device cuda --variant PPO --output_dir results/ppo_medium_seed42
```

Resume：

```bash
python -m src.experiments.train --episodes 1000 --scale medium --seed 42 --device cuda --output_dir results/formal_medium_seed42 --resume
```

训练会显示 tqdm episode 进度条，并在 postfix 中显示 `episode/train_reward/train_cmax/eval_cmax_mean/policy_loss/value_loss/entropy/device`。训练结束后输出：

```text
Training finished successfully.
Results saved to: <output_dir>
Figures saved to: <output_dir>/figures
```

`experiment_config.json` 会记录 `python_version/torch_version/torch_cuda_version/cuda_available/gpu_name/device`。

训练结束后会自动生成基础诊断图到 `<output_dir>/figures/`：

- `training_reward_curve.png`
- `training_cmax_curve.png`
- `eval_cmax_curve.png`
- `loss_curve.png`
- `entropy_curve.png`
- `learning_rate_curve.png`（如果 CSV 中存在 learning_rate 字段）

同时会在固定验证集 `val_instances` 上导出：

- `eval_summary.csv`
- `raw_eval_schedule.csv`

如果验证集 raw schedule 可用，还会自动生成：

- `gantt_eval_instance.png`
- `machine_rank_distribution.png`
- `machine_mismatch_distribution.png`
- `efficiency_load_tradeoff.png`

## 6. CSV 编码说明

项目 CSV 写入统一使用无 BOM 的 UTF-8，表头字段保持干净，例如：

```text
episode,train_reward,train_cmax,eval_cmax_mean,...
```

Windows CMD 的 `type` 命令对 UTF-8/BOM 的显示并不可靠，建议使用 VS Code、Excel 或 pandas 读取 CSV。当前 `training_curve.csv`、`raw_schedule_results.csv`、`summary_metrics.csv` 均使用统一编码写入。

`training_curve.csv` 每轮记录训练指标；只有实际 eval 的 episode 写 `eval_cmax_mean/eval_cmax_std/eval_reward_mean/eval_reward_std`，同时新增 `last_eval_cmax_mean/last_eval_cmax_std` 方便进度条和画图平滑显示。

注意：`train_cmax` 每轮可能来自不同训练实例，原始曲线会剧烈波动，不能作为唯一有效性判断依据。建议主要观察：

- moving average 后的 `train_reward` 是否改善；
- 固定验证集 `eval_cmax_mean` 是否下降；
- `eval_cmax_std` 是否稳定；
- 与 FIFO/SPT/EAT/GA/PPO 的同实例测试对比；
- 非等效机器适配损失是否低于基线；
- 机器选择排名分布是否更合理。

不建议每次修改后都跑 1000 轮，应采用 `quick_debug -> smoke_test -> pilot_small -> pilot_medium -> formal` 的分阶段流程。

## 7. 评估所有算法

所有算法必须在同一批 `test_instances` 上评估：

```bash
python -m src.experiments.evaluate ^
  --scale medium ^
  --seed 42 ^
  --dataset_dir data/generated/formal_medium_seed42 ^
  --checkpoint results/formal_medium_seed42/checkpoints/checkpoint_ep1000.pt ^
  --ppo_checkpoint results/ppo_medium_seed42/checkpoints/checkpoint_ep1000.pt ^
  --output_dir results/eval_medium ^
  --include_ga
```

输出：

- `results/eval_medium/raw_schedule_results.csv`
- `results/eval_medium/summary_metrics.csv`
- `results/eval_medium/experiment_config.json`

`raw_schedule_results.csv` 保留字段：

```text
instance_id, scale, algorithm, job_id, subjob_id, process_type,
selected_machine, machine_rank, p_selected, p_best,
machine_mismatch, available_time_before, start_time,
completion_time, split_id, rolling_period, is_carryover, is_new_arrival
```

`summary_metrics.csv` 保留字段：

```text
scale, algorithm, seed, cmax_mean, cmax_std, utilization_mean,
load_balance_std, avg_machine_mismatch, fastest_machine_ratio,
second_fastest_ratio, others_ratio, runtime
```

## 8. 绘图

训练曲线：

```bash
python -m src.visualization.plot_training --input results/formal_medium_seed42/training_curve.csv --output results/figures/formal_medium --ma_window 20
```

该脚本会生成 `training_reward_curve.png`、`training_cmax_curve.png`、`eval_cmax_curve.png`、`loss_curve.png`、`entropy_curve.png`，并在保存每张图时打印 `Saved figure: <path>`。

非等效并行机图：

```bash
python -m src.visualization.plot_non_equivalent_machine ^
  --input results/eval_medium/raw_schedule_results.csv ^
  --instances data/generated/formal_medium_seed42/test_instances.json ^
  --summary results/eval_medium/summary_metrics.csv ^
  --output results/figures/non_equivalent_machine
```

带滚动周期边界的甘特图：

```bash
python -m src.visualization.plot_gantt --input results/eval_medium/raw_schedule_results.csv --algorithm TSR-PPO --rolling_delta 120 --output results/figures/gantt
```

评估汇总图：

```bash
python -m src.visualization.plot_evaluation --summary results/eval_medium/summary_metrics.csv --output results/figures/eval_medium
```

正式论文图生成建议：

- 每次训练后自动生成：`training_reward_curve.png`、`training_cmax_curve.png`、`eval_cmax_curve.png`、`loss_curve.png`、`entropy_curve.png`
- 评估后生成：`algorithm_cmax_comparison.png`、`gantt_tsr_ppo.png`、`machine_rank_distribution.png`、`machine_mismatch_comparison.png`、`efficiency_load_tradeoff.png`
- 消融和敏感性实验完成后再生成：`ablation_cmax.png`、`sensitivity_task_scale.png`、`sensitivity_heterogeneity.png`、`sensitivity_arrival_intensity.png`、`sensitivity_rolling_step.png`、`sensitivity_split_limit.png`、`utilization_balance.png`

## 9. 敏感性与消融实验计划

生成敏感性实验命令清单，不直接启动长训练：

```bash
python -m src.experiments.run_sensitivity --experiment heterogeneity --episodes 1000 --scale medium --output_dir results/sensitivity_heterogeneity
python -m src.experiments.run_sensitivity --experiment arrival_intensity --episodes 1000 --scale medium --output_dir results/sensitivity_arrival
python -m src.experiments.run_sensitivity --experiment rolling_delta --episodes 1000 --scale medium --output_dir results/sensitivity_delta
python -m src.experiments.run_sensitivity --experiment split_limit --episodes 1000 --scale medium --output_dir results/sensitivity_split
python -m src.experiments.run_sensitivity --experiment scale --episodes 1000 --output_dir results/sensitivity_scale
```

生成消融实验命令清单：

```bash
python -m src.experiments.run_ablation --episodes 1000 --scale medium --seed 42 --output_dir results/ablation_plan
```

消融版本包括 PPO、PPO + Mask、PPO + Mask + Split、PPO + Mask + Split + Rolling、TSR-PPO full。

## 10. 仍需正式训练验证

本次代码只准备正式实验训练、日志、checkpoint、评估和绘图接口；不要把 quick_debug 或 20 轮小测试当成论文结果。后续需要在训练电脑上验证：

- 1000 episode 是否稳定收敛；
- CUDA 训练速度和显存占用是否符合预期；
- TSR-PPO 相对 FIFO/SPT/EAT/GA/PPO 的 Cmax 优势；
- 异构程度越高时 TSR-PPO 相对改进率是否扩大；
- 消融模块贡献是否明显；
- large 规模下 GA 运行时间是否需要进一步并行化或调参。
