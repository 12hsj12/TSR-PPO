# TSR-PPO 任务拆分滚动调度

本项目实现“考虑任务拆分和非等效并行产线的混合流水车间滚动调度及深度强化学习求解”。每个原料批次仅对应纵剪、横切、复合剪三类工艺之一；同一工艺内部包含多条非等效并行产线；批次可在合格产线内按比例拆分；系统按滚动周期处理新到达任务、结转任务与产线最早可用时间。

## 项目结构

- `data/raw/`：原始 Excel 数据备份。
- `data/generated/`：可复现实验算例。
- `src/data_generator.py`：读取真实加工记录统计特征，并生成训练/测试算例。
- `src/env.py`：滚动调度环境、动作掩码、产线状态更新。
- `src/splitter.py`：按加工效率与产线最早可用时间解码拆分比例。
- `src/policy.py`、`src/tsr_ppo.py`：轻量 NumPy Actor-Critic 与 PPO/TSR-PPO。
- `src/baselines.py`：FIFO、SPT、EAT、GA 基准算法。
- `src/metrics.py`：Cmax、产线利用率、负载均衡度、运行时间、RPI。
- `src/visualize.py`：收敛曲线、对比图、敏感性分析图、甘特图。
- `src/experiment.py`：统一实验入口。
- `results/figures/`、`results/tables/`、`results/models/`：实验输出。

## 安装依赖

```bash
pip install -r requirements.txt
```

本实现避免强依赖 PyTorch/Stable-Baselines3，采用 NumPy 实现 PPO 的旧策略概率比、clip 更新、价值函数和熵正则，便于在普通论文实验环境中快速复现。

## 一键运行

快速验证完整流程：

```bash
python src/experiment.py --mode quick
```

论文规模实验：

```bash
python src/experiment.py --mode full
```

`--mode all` 当前等价于 quick，用于快速生成所有类型的表格和图片。

## 输出文件

主要表格：

- `results/tables/convergence_history.csv`
- `results/tables/performance_comparison.csv`
- `results/tables/ablation_results.csv`
- `results/tables/sensitivity_results.csv`
- `results/tables/tsr_ppo_schedule.csv`

主要图片：

- `results/figures/convergence_reward.png`
- `results/figures/algorithm_cmax_comparison.png`
- `results/figures/ablation_cmax.png`
- `results/figures/sensitivity_task_scale.png`
- `results/figures/sensitivity_split_limit.png`
- `results/figures/sensitivity_heterogeneity.png`
- `results/figures/sensitivity_rolling_delta.png`
- `results/figures/sensitivity_arrival_intensity.png`
- `results/figures/gantt_tsr_ppo.png`
- `results/figures/utilization_balance.png`

## 算法说明

TSR-PPO 的动作采用“选择原料批次 + 选择产线组合”的分层离散结构。环境对尚未释放批次、已完成批次、工艺不匹配产线、超过拆分上限等动作进行掩码。拆分比例由确定性规则解码：

`权重 = 1 / 加工时间 * 1 / (1 + 产线等待时间)`

再归一化得到各子任务比例，并过滤过小拆分比例。奖励函数由 Cmax 增量、产线空闲惩罚和负载不均衡惩罚组成。

## 可复现性

所有随机生成过程、GA 和 PPO 训练均设置 seed。真实 Excel 仅用于学习产线规模、工艺类别和加工时间分布；训练集与测试集由 `src/data_generator.py` 生成。
