# TSR-PPO 任务拆分滚动调度正式实验项目

本项目面向“考虑任务拆分和非等效并行产线的滚动调度及深度强化学习求解”。问题设定不是“纵剪 -> 横切 -> 复合剪”的顺序型混合流水车间，而是：每个原料批次只属于一种工艺类型，在该工艺内部的非等效并行机集合中加工，并允许按比例拆分到多台合格机器。

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

正式 PPO 训练使用 PyTorch；数据处理和绘图使用 `numpy/pandas/matplotlib/openpyxl`。

## 2. 新代码结构

```text
src/
  data/
    generator.py              # 生成 train/val/test 实例，small/medium/large 固定规模
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
    evaluate.py               # 同一 test_instances 上评估所有算法
    run_formal_experiments.py # 生成正式实验命令清单
    run_sensitivity.py        # 准备敏感性实验数据和命令
    run_ablation.py           # 准备消融实验命令
  visualization/
    plot_training.py
    plot_gantt.py
    plot_non_equivalent_machine.py
    plot_heterogeneity.py
  utils/
    io.py logger.py metrics.py seed.py
```

旧版 `src/experiment.py` 等文件保留为兼容入口；正式论文实验请使用 `python -m src.experiments.*`。

## 3. 数据集生成

正式生成 medium 数据集：

```bash
python -m src.data.generator --scale medium --seed 42 --output_dir data/generated/formal_medium_seed42
```

quick debug 数据集：

```bash
python -m src.data.generator --quick_debug --seed 42 --output_dir data/generated/quick_debug
```

固定规模：

- `small`: 50 任务，约 9 台机器，3-4 个滚动周期
- `medium`: 150 任务，约 15 台机器，5-6 个滚动周期
- `large`: 300 任务，约 21 台机器，8-10 个滚动周期

默认工艺比例：横切 70%，纵剪 20%，复合剪 10%。每个任务保存 `instance_id/job_id/process_type/release_time/due_time/quantity/workload/eligible_machines/p[j,k]/rolling_period`。

## 4. quick_debug

quick_debug 只用于代码链路检查，episode 不超过 5，不生成正式结论，不覆盖正式结果目录：

```bash
python -m src.experiments.train --quick_debug --episodes 5 --seed 42 --output_dir results/debug
```

## 5. 正式训练 1000 episode

默认正式训练命令：

```bash
python -m src.experiments.train --episodes 1000 --scale medium --seed 42 --output_dir results/formal_medium_seed42
```

训练基础 PPO baseline：

```bash
python -m src.experiments.train --episodes 1000 --scale medium --seed 42 --variant PPO --output_dir results/ppo_medium_seed42
```

使用配置文件：

```bash
python -m src.experiments.train --config configs/formal_medium.json
```

训练只写 CSV 日志和 checkpoint，不在每轮画图。主要输出：

- `results/formal_medium_seed42/training_curve.csv`
- `results/formal_medium_seed42/experiment_config.json`
- `results/formal_medium_seed42/checkpoints/checkpoint_epXXXX.pt`

## 6. Resume

从最近 checkpoint 继续：

```bash
python -m src.experiments.train --episodes 1000 --scale medium --seed 42 --output_dir results/formal_medium_seed42 --resume
```

checkpoint 保存 actor/critic 参数、optimizer 状态、episode、seed 和额外实验信息。

## 7. 评估所有算法

所有算法必须在同一批 `test_instances` 上评估：

```bash
python -m src.experiments.evaluate \
  --scale medium \
  --seed 42 \
  --dataset_dir data/generated/formal_medium_seed42 \
  --checkpoint results/formal_medium_seed42/checkpoints/checkpoint_ep1000.pt \
  --ppo_checkpoint results/ppo_medium_seed42/checkpoints/checkpoint_ep1000.pt \
  --output_dir results/eval_medium \
  --include_ga
```

输出：

- `results/eval_medium/raw_schedule_results.csv`
- `results/eval_medium/summary_metrics.csv`
- `results/eval_medium/experiment_config.json`

`raw_schedule_results.csv` 字段包括 `instance_id, scale, algorithm, job_id, subjob_id, process_type, selected_machine, machine_rank, p_selected, p_best, machine_mismatch, available_time_before, release_time, start_time, completion_time, split_id, rolling_period, is_carryover, is_new_arrival`。

## 8. 生成正式图

训练曲线：

```bash
python -m src.visualization.plot_training --input results/formal_medium_seed42/training_curve.csv --output results/figures/formal_medium
```

非等效并行机图，包括加工时间热力图、机器选择排名、机器适配损失、效率-负载权衡：

```bash
python -m src.visualization.plot_non_equivalent_machine \
  --input results/eval_medium/raw_schedule_results.csv \
  --instances data/generated/formal_medium_seed42/test_instances.json \
  --summary results/eval_medium/summary_metrics.csv \
  --output results/figures/non_equivalent_machine
```

带滚动周期边界的甘特图：

```bash
python -m src.visualization.plot_gantt \
  --input results/eval_medium/raw_schedule_results.csv \
  --algorithm TSR-PPO \
  --rolling_delta 120 \
  --output results/figures/gantt
```

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

消融版本包括：

- PPO
- PPO + Mask
- PPO + Mask + Split
- PPO + Mask + Split + Rolling
- TSR-PPO full

## 10. 从另一台电脑直接运行正式实验

```bash
git pull
pip install -r requirements.txt
python -m src.data.generator --scale medium --seed 42 --output_dir data/generated/formal_medium_seed42
python -m src.experiments.train --episodes 1000 --scale medium --seed 42 --output_dir results/formal_medium_seed42
python -m src.experiments.evaluate --scale medium --seed 42 --dataset_dir data/generated/formal_medium_seed42 --checkpoint results/formal_medium_seed42/checkpoints/checkpoint_ep1000.pt --output_dir results/eval_medium --include_ga
python -m src.visualization.plot_non_equivalent_machine --input results/eval_medium/raw_schedule_results.csv --instances data/generated/formal_medium_seed42/test_instances.json --summary results/eval_medium/summary_metrics.csv --output results/figures/non_equivalent_machine
python -m src.visualization.plot_gantt --input results/eval_medium/raw_schedule_results.csv --algorithm TSR-PPO --rolling_delta 120 --output results/figures/gantt
```

## 11. 仍需正式训练验证的内容

本次重构只准备正式实验代码、配置、日志、checkpoint 和绘图接口；没有运行 1000 episode，也没有生成最终论文图。后续需要在正式机器上验证：

- 1000 episode 是否稳定收敛；
- TSR-PPO 相对 FIFO/SPT/EAT/GA/PPO 的 Cmax 优势；
- 异构程度越高时相对改进率是否扩大；
- 消融模块贡献是否明显；
- quick debug 与正式实验之间的参数规模切换是否符合机器算力预期。
