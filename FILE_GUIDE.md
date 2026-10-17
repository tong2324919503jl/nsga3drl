# 项目文件说明

本文档按当前仓库结构概述各文件的职责。`data/` 和 `results/` 中的大量实例及实验输出属于生成文件，按目录说明，不逐个重复列出。

## 根目录文件

| 文件 | 作用 |
|---|---|
| `DATASET_FORMAT.md` | MTSP JSON 数据格式、动态事件字段、偏好字段和 `arrival_completed_tasks` 语义。 |
| `IMPLEMENTATION_PLAN.md` | 从建模、NSGA-III、PPO 到动态加点的实现计划与阶段划分。 |
| `REPRODUCE_1H_RESULTS.md` | 复现 1h PPO、baseline 及评估结果所需的命令和文件说明。 |
| `environment.yml` | Conda 环境定义，固定 Python、NumPy、SciPy、PyTorch、pytest 等版本。 |
| `requirements.txt` | pip 依赖版本列表；通常优先使用 `environment.yml`。 |
| `pyproject.toml` | Python 包元数据、运行依赖、开发依赖和 pytest 配置。 |

## 核心源码 `src/nsga3drl`

### 数据模块 `src/nsga3drl/data`

| 文件 | 作用 |
|---|---|
| `__init__.py` | 导出数据生成、加载、数据模型和校验 API。 |
| `schema.py` | 定义 `DynamicEvent` 和 `MTSPInstance`；负责字段校验、事件注入、偏好覆盖、去除事件以及 JSON 序列化/反序列化。 |
| `generator.py` | 生成 uniform、clustered、hotspot 空间分布的数据集，创建动态事件、偏好组合和数据集 manifest。 |
| `io.py` | 从目录读取实例 JSON，并按 manifest/文件顺序返回 `MTSPInstance` 列表。 |
| `validation.py` | 检查实例和数据集结构、任务 ID、动态事件顺序、偏好、manifest 以及任务坐标重叠；输出校验报告和警告。 |

### 问题定义 `src/nsga3drl/problem`

| 文件 | 作用 |
|---|---|
| `__init__.py` | 导出问题和编码相关 API。 |
| `encoding.py` | 实现整数排列加车辆分隔符编码，提供 `encode`、`decode`、合法性检查、修复和随机染色体生成。 |
| `open_mtsp.py` | 实现开放型 MTSP 目标函数：车辆从各自起点出发，不返回起点；计算总路程和车辆路线长度方差。 |

### 包入口

| 文件 | 作用 |
|---|---|
| `src/nsga3drl/__init__.py` | Python 包入口和项目简介，不承载具体算法逻辑。 |

### 实验批处理 `src/nsga3drl/experiments`

| 文件 | 作用 |
|---|---|
| `__init__.py` | 实验工具包入口。 |
| `evaluate.py` | 以单个实例为单位执行 PPO 或 baseline 评估，并统一生成评估记录与汇总结构。 |
| `resume_store.py` | 用 SQLite 持久化完成的任务和运行配置；检查断点续跑参数是否与原任务一致。 |
| `batch.py` | 展开模型版本、偏好、动态策略和实例的任务组合，进行多进程调度、信号处理、结果归档和断点恢复。 |

### NSGA-III 与路线构造 `src/nsga3drl/algorithms`

| 文件 | 作用 |
|---|---|
| `__init__.py` | 导出算法模块 API。 |
| `nsga3.py` | 实现 NSGA-III 风格进化流程：非支配排序、参考方向、超体积、环境选择、交叉、变异、局部改进、动态事件处理和算子信用记录。 |
| `route_construction.py` | 生成较好的初始种群，包括最近起点、sweep、clustered 路线构造，以及 2-opt、跨车 relocate/swap 等局部搜索。 |

### 动态加点 `src/nsga3drl/dynamic`

| 文件 | 作用 |
|---|---|
| `__init__.py` | 导出动态执行器和插点规划器。 |
| `execution.py` | 模拟车辆以相同速度完成任务；每累计完成 10 个任务注入一个动态点，记录事件顺序、路线、动态耗时和决策日志。动态阶段不重新运行 NSGA-III 或 PPO。 |
| `insertion.py` | 实现动态点分配和路线重规划。包含原有 `tsp_scored` 逐车 TSP 打分策略，以及当前新增的 `greedy_score_then_tsp`：先对每辆车做最小增量贪心插入评分，再只对中选车辆执行三级 TSP（Held-Karp、分支定界、启发式）。同时计算顺路分、均衡分、偏好加权总分、负增量和回退信息。 |

### 强化学习 `src/nsga3drl/rl`

| 文件 | 作用 |
|---|---|
| `__init__.py` | 导出 PPO 和环境 API。 |
| `environment.py` | 将 NSGA-III 算子选择包装成环境；提取种群数值特征、偏好和算子信用相关状态，执行动作并以 HV 改变量形成奖励。 |
| `ppo.py` | 定义 MLP Actor-Critic、PPO 配置、采样、优势估计、策略/价值更新、熵正则、动作概率、checkpoint 保存和加载。 |

## 命令行脚本 `scripts`

| 文件 | 作用 |
|---|---|
| `generate_dataset.py` | 命令行生成并立即校验数据集；支持任务/车辆规模、空间分布、动态比例、偏好模式、配对偏好和省略偏好字段。 |
| `validate_dataset.py` | 检查一个数据集目录，输出每个实例的任务数、车辆数、事件数、错误和坐标重叠等警告。 |
| `migrate_dataset_format.py` | 将旧格式事件字段 `arrival_generation` 转换为当前的 `arrival_completed_tasks`，不重新生成坐标，并重新校验数据集。 |
| `train_ppo.py` | 训练 PPO 算子选择器；支持数据集采样、每 epoch 无放回遍历、熵退火、验证集、断点续训、最佳 checkpoint 保存和训练耗时预测。 |
| `evaluate_ppo.py` | 加载 PPO checkpoint，在一个实例或整个数据集上进行确定性评估；输出静态 HV、动态前后路线目标、事件日志、算子统计和耗时。支持 `tsp_scored` 与 `greedy_score_then_tsp`。 |
| `run_baseline.py` | 运行不使用 PPO 的 NSGA-III baseline；支持单实例/数据集、多种群/代数、偏好覆盖和两种动态策略，输出格式与 PPO 评估基本一致。 |
| `run_parallel_evaluation.py` | 并行运行 PPO/baseline、多 checkpoint、多偏好、多动态策略和多实例测试；逐个任务存入 SQLite，重复原命令可跳过已完成部分继续运行。 |
| `纯NSGA3版.py` | 合作者的独立纯 NSGA-III 参考实现，已适配当前实例格式、开放路线目标和动态执行器；用于额外对照，不是 PPO 训练入口。 |

## 测试 `tests`

| 文件 | 作用 |
|---|---|
| `test_dataset.py` | 测试数据生成、manifest、偏好写入/省略、配对偏好、坐标重叠检查和车辆起点警告。 |
| `test_encoding_and_problem.py` | 测试编码解码往返、合法性和开放 MTSP 路程目标。 |
| `test_route_construction.py` | 测试初始路线构造覆盖所有车辆，以及局部搜索保持任务唯一性和非空路线。 |
| `test_nsga3.py` | 测试 NSGA-III 种群合法性和迭代流程。 |
| `test_dynamic.py` | 测试动态点只插入一次、偏好影响车辆分配、负增量、精确 TSP、分支定界、三级求解器选择、两种动态策略及相同速度下的事件顺序。 |
| `test_rl.py` | 测试 PPO 环境状态、动态事件不进入训练、PPO 更新、checkpoint 和一个世代内多算子动作。 |

