# NSGA-III + DRL 动态加点 Open MTSP MVP

本仓库当前实现范围：

- 开放型双目标 MTSP（总距离、车辆路径长度方差）；
- 分隔符整数编码、合法性修复、交叉/变异/2-opt；
- 自包含的 NSGA-III 风格参考方向环境选择；
- 合成静态/动态数据集生成器；
- KD-Tree 驱动的动态插点策略；
- 种群数值特征 MLP 与 PPO 算子选择环境。

未包含：GAT、交叉注意力、偏好编码网络、PPO 父代选择、真实路网距离。

数据集字段、动态事件与 `arrival_generation` 的精确定义见 [DATASET_FORMAT.md](DATASET_FORMAT.md)。

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

使用已生成数据集而非临时随机实例：

```bash
python scripts/run_baseline.py --instance data/demo/clustered_v10_n150_123456.json --population 64 --generations 50 --seed 42
python scripts/train_ppo.py --dataset data/demo --updates 10 --episodes-per-update 4 --generations 20 --seed 42
python scripts/evaluate_ppo.py --dataset data/test --checkpoint checkpoints/ppo_operator_selector.pt --generations 20
```

动态事件会在其 `arrival_generation` 对应代开始前插入当前种群，并对受影响路径运行局部 2-opt。HV 的参考点会在每次动态插点后重置，因此 PPO 奖励只比较同一动态问题阶段内的相邻代。

训练每次 update 后都会覆盖保存 checkpoint，并写入 `<checkpoint>.train.json`；评估会写入 `<checkpoint>.eval.json`。终端中的 `elapsed` 是已完成部分的准确耗时，`projected_total` 是按当前平均速度预测的整轮总耗时。训练支持从 checkpoint 继续：

```bash
python scripts/train_ppo.py --dataset data/demo --resume checkpoints/ppo_operator_selector.pt --checkpoint checkpoints/ppo_operator_selector.pt --updates 10
```

评估调用 `greedy_action`，不计算梯度，也不更新参数；因此在相同实例、种群、代数和硬件条件下，测试阶段通常明显快于训练阶段。

所有脚本接受 `--seed`，便于复现。
