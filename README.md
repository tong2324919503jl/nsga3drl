# NSGA-III + DRL 动态加点 Open MTSP MVP

本仓库当前实现范围：

- 开放型双目标 MTSP（总距离、车辆路径长度方差）；
- 分隔符整数编码、合法性修复、交叉/变异/2-opt；
- 自包含的 NSGA-III 风格参考方向环境选择；
- 合成静态/动态数据集生成器；
- 两种可选的动态插点和剩余路径重规划策略；
- 种群数值特征 MLP 与 PPO 算子选择环境。

未包含：GAT、交叉注意力、偏好编码网络、PPO 父代选择、真实路网距离。

数据集字段、动态事件与 `arrival_completed_tasks` 的精确定义见 [DATASET_FORMAT.md](DATASET_FORMAT.md)。

## Conda 环境安装

环境定义位于 `environment.yml`，固定使用 Python 3.11.9、NumPy 1.26.4、SciPy 1.11.4、PyTorch 2.2.2、pytest 8.2.2、pip 24.2 和 setuptools 70.3.0。CPU 环境在项目根目录执行：

```bash
conda env create -f environment.yml
conda activate nsga3drl
```

`environment.yml` 的 PyTorch 包默认由 Conda 根据目标平台解析。若目标机器具备 NVIDIA GPU，并且需要指定 CUDA 运行时，请创建环境后安装与驱动兼容的 PyTorch CUDA 元包。例如 CUDA 11.8：

```bash
conda activate nsga3drl
conda install pytorch-cuda=11.8 -c pytorch -c nvidia
```

不要同时使用 `pip install torch` 或升级 NumPy/SciPy/PyTorch；这会破坏该锁定组合。若要完全按已有环境文件重建，可先执行 `conda env remove -n nsga3drl`，然后再次执行 `conda env create -f environment.yml`。

首次在目标平台成功创建环境后，执行以下命令可导出包含**全部直接与间接 Conda 包、精确构建号和下载地址**的同平台锁定文件：

```bash
conda activate nsga3drl
conda list --explicit > conda-explicit-linux-64.txt
```

后续在相同操作系统/CPU 架构的平台上可不经求解器重建为完全一致的环境：

```bash
conda create -n nsga3drl-repro --file conda-explicit-linux-64.txt
conda activate nsga3drl-repro
```

如果目标平台不是 Linux x86_64，请把导出文件名改为相应平台名，并在该平台上重新导出；Conda 二进制包不能跨平台复用。

## 验证与示例

```bash
pytest
python scripts/run_baseline.py --tasks 30 --vehicles 3 --generations 80
python scripts/generate_dataset.py --output data/demo --instances 30 --tasks 150 --vehicles 10 --dynamic-rate 0.1
python scripts/train_ppo.py --updates 10 --episodes-per-update 4
```

训练 PPO 时，使用 `balanced5` 使偏好 `[0.1,0.9]`、`[0.3,0.7]`、`[0.5,0.5]`、`[0.7,0.3]`、`[0.9,0.1]` 在实例中均衡出现；偏好向量会作为 PPO 状态输入。固定单一偏好可使用 `--preference-mode fixed --preference 0.7 0.3`。

```bash
python scripts/generate_dataset.py --output data/train_150x10 --instances 50 --tasks 150 --vehicles 10 --distribution clustered --dynamic-rate 0.1 --preference-mode balanced5 --seed 20260801
python scripts/generate_dataset.py --output data/test_150x10_preferences --instances 30 --tasks 150 --vehicles 10 --distribution clustered --dynamic-rate 0.1 --preference-mode balanced5 --seed 31415
```

`--distribution mixed` cycles through uniform, clustered and hotspot distributions. With `--paired-preferences --preference-mode balanced5`, `--instances` means the number of base spatial scenarios; each base scenario is emitted five times with identical coordinates/events and different preferences. PPO training defaults to `--sampling epoch`, which visits every instance once before reshuffling.

数据集可以不保存偏好，由运行命令提供：

```bash
python scripts/generate_dataset.py --output data/test_no_preference --instances 20 --tasks 150 --vehicles 10 --distribution mixed --dynamic-rate 0.1 --omit-preference --seed 31415
python scripts/evaluate_ppo.py --dataset data/test_no_preference --checkpoint checkpoints/ppo_ideal_150x10.best.pt --preference 0.5 0.5 --generations 160 --population 32
```

使用已生成数据集而非临时随机实例：

```bash
python scripts/run_baseline.py --instance data/demo/clustered_v10_n150_123456.json --population 64 --generations 50 --seed 42
python scripts/train_ppo.py --dataset data/demo --updates 10 --episodes-per-update 4 --generations 20 --seed 42
python scripts/evaluate_ppo.py --dataset data/test --checkpoint checkpoints/ppo_operator_selector.pt --generations 20
```

NSGA-III/PPO 只优化初始任务；车辆执行初始计划时每完成 10 个任务才触发一次动态插点，动态阶段不再运行 NSGA-III/PPO。

基线为“静态初始 NSGA-III + 随机变异算子选择”，随后使用同一动态执行器；默认采用 `tsp_scored` 逐车 TSP 打分，也可选择 `greedy_score_then_tsp`。动态执行阶段不再调用 NSGA-III 或 PPO。

`run_baseline.py` 和 `evaluate_ppo.py` 输出的 `preferred_solution.total_distance`、`variance` 是动态执行完成后的真实路线目标；`initial_total_distance`、`initial_variance` 是初始 150 点方案的目标。数据集没有偏好字段时，必须通过 `--preference EFFICIENCY FAIRNESS` 提供权重；若数据集有偏好字段，则默认逐实例读取，也可以用命令行覆盖。

训练每次 update 后都会覆盖保存 checkpoint，并写入 `<checkpoint>.train.json`；评估会写入 `<checkpoint>.eval.json`。终端中的 `elapsed` 是已完成部分的准确耗时，`projected_total` 是按当前平均速度预测的整轮总耗时。训练支持从 checkpoint 继续：

```bash
python scripts/train_ppo.py --dataset data/demo --resume checkpoints/ppo_operator_selector.pt --checkpoint checkpoints/ppo_operator_selector.pt --updates 10
```

训练可通过独立验证集选择累计奖励最高的最佳 checkpoint：

```bash
python scripts/train_ppo.py --dataset data/train --validation-dataset data/validation --validation-every 10 --validation-instances 10 --checkpoint checkpoints/ppo.pt
```

最佳模型会写入 `checkpoints/ppo.best.pt`；最终测试应加载该文件，而不是默认的最后一次 update 模型。

评估调用 `greedy_action`，不计算梯度，也不更新参数；因此在相同实例、种群、代数和硬件条件下，测试阶段通常明显快于训练阶段。

所有脚本接受 `--seed`，便于复现。

## 并行、可续跑的批量评估

`scripts/run_parallel_evaluation.py` 可将不同实例、偏好、PPO checkpoint 版本和动态策略拆成独立任务，在多个 CPU 进程中并行评估。每完成一个任务就提交到 SQLite 状态库；中断后重复执行相同命令，会跳过已完成任务。改变数据集、checkpoint、偏好或算法参数时需要使用新的 `--state-db`，否则脚本会拒绝混用不同配置的结果。

```bash
python scripts/run_parallel_evaluation.py \
  --dataset data/test_150x10_compatible \
  --checkpoint ppo1h=checkpoints/ppo_constructive_ls_g240_1h.best.pt \
  --method ppo --method baseline \
  --dynamic-strategy greedy_score_then_tsp \
  --dynamic-strategy tsp_scored \
  --preference 1.0 0.0 --preference 0.8 0.2 --preference 0.6 0.4 \
  --preference 0.5 0.5 --preference 0.4 0.6 --preference 0.2 0.8 \
  --preference 0.0 1.0 \
  --generations 240 --population 32 --seed 42 --workers 8 \
  --output-dir results/preference_comparison \
  --state-db results/preference_comparison/resume.sqlite3
```

`--checkpoint LABEL=PATH` 和 `--method` 可重复提供，以并行比较多个模型版本与 baseline；偏好和动态策略也支持重复提供。每个组合写出独立的兼容评估 JSON，SQLite 数据库是断点续跑的权威记录。按 `Ctrl+C` 会在当前运行任务完成后保存进度；重新执行原命令即可续跑。
python scripts/run_baseline.py \
    --dataset data/test_150x10_compatible \
    --preference 1.0 0.0 \
    --population 32 \
    --generations 160 \
    --seed 42 \
    --metrics results/baseline_150x10.eval.json \
