# 1h PPO 与 Baseline 复现实验

本文档记录 `static_comparison_constructive_g240.md` 中使用的 1h PPO、NSGA-III baseline 训练与测试命令。

实验配置为 150 个初始任务、10 辆车、种群规模 32、240 代。动态事件只在最终路线执行阶段处理，PPO 训练和 NSGA-III 初始优化阶段不处理动态事件。

## 1. 环境配置

```bash
cd ~/nsga3drl
conda env create -f environment.yml
conda activate nsga3drl
export PYTHONPATH="$PWD/src"
```

固定版本为 Python 3.11.9、NumPy 1.26.4、SciPy 1.11.4、PyTorch 2.2.2 和 pytest 8.2.2。

## 2. 数据集校验

训练前检查训练集、验证集和测试集：

```bash
python scripts/validate_dataset.py data/train_clustered_extremes_150x10
python scripts/validate_dataset.py data/validation_clustered_extremes_150x10
python scripts/validate_dataset.py data/test_150x10_compatible
```

如果数据集不包含 preference 字段，测试命令中的 `--preference` 参数会提供目标权重；本实验使用命令行偏好覆盖。

## 3. 训练 1h PPO 模型

该模型从随机初始化开始训练，不使用 `--resume`。验证集表现最好的模型会保存为 `*.best.pt`。

```bash
python scripts/train_ppo.py \
  --dataset data/train_clustered_extremes_150x10 \
  --validation-dataset data/validation_clustered_extremes_150x10 \
  --validation-every 18 \
  --validation-instances 5 \
  --updates 18 \
  --episodes-per-update 4 \
  --generations 240 \
  --population 32 \
  --sampling epoch \
  --entropy-start 0.03 \
  --entropy-end 0.01 \
  --entropy-anneal-updates 24 \
  --seed 42 \
  --checkpoint checkpoints/ppo_constructive_ls_g240_1h.pt \
  --metrics results/ppo_constructive_ls_g240_1h.train.json
```

训练输出文件：

```text
checkpoints/ppo_constructive_ls_g240_1h.pt
checkpoints/ppo_constructive_ls_g240_1h.best.pt
results/ppo_constructive_ls_g240_1h.train.json
```

后续测试使用 `ppo_constructive_ls_g240_1h.best.pt`。

## 4. PPO 效率偏好测试

`[1.0, 0.0]` 表示优先最小化总路程。

```bash
python scripts/evaluate_ppo.py \
  --dataset data/test_150x10_compatible \
  --checkpoint checkpoints/ppo_constructive_ls_g240_1h.best.pt \
  --preference 1.0 0.0 \
  --generations 240 \
  --population 32 \
  --seed 42 \
  --metrics results/ppo_constructive_ls_g240_1h.efficiency.eval.json
```

## 5. PPO 公平性偏好测试

`[0.0, 1.0]` 表示优先最小化车辆路径长度方差。

```bash
python scripts/evaluate_ppo.py \
  --dataset data/test_150x10_compatible \
  --checkpoint checkpoints/ppo_constructive_ls_g240_1h.best.pt \
  --preference 0.0 1.0 \
  --generations 240 \
  --population 32 \
  --seed 42 \
  --metrics results/ppo_constructive_ls_g240_1h.fairness.eval.json
```

## 6. Baseline 效率偏好测试

Baseline 使用相同的路线构造、局部搜索、NSGA-III 参数和动态执行器，但不使用 PPO 算子选择。它使用 NSGA-III 的随机变异算子选择。

```bash
python scripts/run_baseline.py \
  --dataset data/test_150x10_compatible \
  --preference 1.0 0.0 \
  --population 32 \
  --generations 240 \
  --seed 42 \
  --metrics results/baseline_constructive_ls_g240.efficiency.eval.json
```

## 7. Baseline 公平性偏好测试

```bash
python scripts/run_baseline.py \
  --dataset data/test_150x10_compatible \
  --preference 0.0 1.0 \
  --population 32 \
  --generations 240 \
  --seed 42 \
  --metrics results/baseline_constructive_ls_g240.fairness.eval.json
```

## 8. 结果文件对应关系

| 实验 | 结果文件 |
|---|---|
| PPO 1h 训练 | `results/ppo_constructive_ls_g240_1h.train.json` |
| PPO 效率偏好 | `results/ppo_constructive_ls_g240_1h.efficiency.eval.json` |
| PPO 公平性偏好 | `results/ppo_constructive_ls_g240_1h.fairness.eval.json` |
| Baseline 效率偏好 | `results/baseline_constructive_ls_g240.efficiency.eval.json` |
| Baseline 公平性偏好 | `results/baseline_constructive_ls_g240.fairness.eval.json` |
| 静态汇总表 | `results/static_comparison_constructive_g240.md` |

结果中的 `initial_total_distance` 和 `initial_variance` 对应动态加点前的 150 个任务；`total_distance` 和 `variance` 对应执行 15 次动态加点后的最终路线。静态方法比较时应使用 `initial_*` 字段。
