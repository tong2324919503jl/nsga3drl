# 数据集格式与动态加点语义

## 1. 数据集目录

```text
data/test_150x10/
  clustered_v10_n150_123456.json
  clustered_v10_n150_789012.json
  ...
  manifest.json
```

每个实例文件是一个独立测试场景；`manifest.json` 记录该目录中的文件清单，供完整性校验使用。

## 2. 单个实例 JSON 的字段说明

示例（为阅读方便只展示 2 辆车、3 个初始点和 1 个动态点）：

```json
{
  "instance_id": "clustered_v2_n3_000042",
  "vehicle_starts": [[12.3, 45.6], [76.1, 20.4]],
  "task_coordinates": [[20.1, 30.2], [41.8, 18.0], [64.5, 75.1]],
  "task_ids": [1, 2, 3],
  "preference": [0.5, 0.5],
  "events": [
    {
      "arrival_completed_tasks": 10,
      "task_id": 4,
      "coordinate": [42.0, 18.5]
    }
  ],
  "seed": 42
}
```

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `instance_id` | string | 实例唯一名称。生成规则为 `{distribution}_v{车辆数}_n{初始任务数}_{随机种子}`。 |
| `vehicle_starts` | `float[vehicle_count][2]` | 各车辆的初始位置，按数组下标对应车辆编号 `0..vehicle_count-1`。二维坐标 `[x, y]`。 |
| `task_coordinates` | `float[task_count][2]` | **初始已知**任务点坐标。默认地图范围为 `[0, 100] x [0, 100]`。 |
| `task_ids` | `int[task_count]` | 与 `task_coordinates` 同下标对应的任务 ID。当前整数分隔符编码要求初始 ID 连续为 `1..task_count`；`0` 保留作车辆路径分隔符。 |
| `preference` | `float[2]` 或缺省 | 用户偏好 `[效率权重, 公平权重]`，两项非负且和为 1。它作为 PPO 状态输入，并用于从最终帕累托前沿选择推荐解；它不改变 NSGA-III 的双目标定义。生成器默认写入 `[0.5, 0.5]`；`--omit-preference` 时从 JSON 移除，运行脚本必须使用 `--preference EFFICIENCY FAIRNESS`。 |
| `events` | object array | 尚未出现的动态任务事件，按已完成任务数处理。动态点**不包含**在 `task_coordinates` 中。 |
| `seed` | integer | 生成此实例使用的随机种子，用于复现相同坐标、事件和空间分布。 |

### 目标函数中坐标的作用

车辆路径不要求回到起点。对于第 `i` 辆车的任务序列 `[p1, p2, ...]`：

```text
L_i = distance(start_i, p1) + distance(p1, p2) + ...
f1 = sum(L_i)                         # 总距离，越小越好
f2 = variance(L_1, ..., L_m)          # 车辆路径长度方差，越小越公平
```

MVP 使用二维欧氏距离；坐标不是经纬度，也不代表真实路网距离。

## 3. 动态事件 `events` 的字段与 `arrival_completed_tasks`

每个事件的格式为：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `arrival_completed_tasks` | non-negative integer | 所有车辆累计完成的原有任务数；达到该数量后插入新任务。 |
| `task_id` | positive integer | 新任务 ID，必须不与初始 `task_ids` 或其他动态事件 ID 重复。生成器从 `初始任务数 + 1` 开始编号。 |
| `coordinate` | `float[2]` | 新任务点坐标 `[x, y]`。 |

### `arrival_completed_tasks` 的精确定义

`arrival_completed_tasks = c` 表示：**车辆总共完成了 c 个原有任务后**，将该任务作为新任务注入当前执行中的路线。它与 NSGA-III 代数无关，也不会触发 NSGA-III 或 PPO。

- `arrival_completed_tasks = 10`：总共完成 10 个原有任务后加入第一个动态点。
- `arrival_completed_tasks = 20`：总共完成 20 个原有任务后加入第二个动态点。
- 对 150 个初始任务和 15 个动态点，默认事件为 `10, 20, ..., 150`。
- 多个事件可以具有相同完成数，表示同一批到达；按 `task_id` 升序逐个插入。
- 当前采用二维欧氏距离且所有车辆速度相同；执行器按各车预计到达时间的全局最小值确定完成顺序，时间相同时按车辆编号稳定打破平局。

生成器当前使用固定间隔：第 `j` 个动态点（从 1 开始计数）的 `arrival_completed_tasks = 10 * j`。例如 150 个初始点、`dynamic_rate=0.1` 时，生成 15 个动态点，分别在完成第 `10, 20, ..., 150` 个原有任务后到达。

动态点到达后，应调用 `DynamicInsertionPlanner.insert(routes, task_id, coordinate)`：它通过 KD-Tree 查询、顺路分与均衡分选择车辆，并在该车的最优位置插入新点。此时不再运行 NSGA-III 或 PPO。

## 4. 动态事件回放行为

`NSGA3.run()` 和 `OperatorSelectionEnv` 只优化初始任务集合。得到推荐路线后，`dynamic.execution.execute_routes()` 以所有车辆等速并行执行固定路线：每次选择预计完成时间最早的下一任务，累计完成 10 个原有任务后注入一个动态点。执行器调用一次 KD-Tree 分配、最小增量插入和受影响路径的局部 2-opt；动态执行阶段不再扩展种群，也不调用 NSGA-III 或 PPO。已完成任务会从待执行路线中移除并记录在 visited 集合中，绝不会再次访问。

动态点不参与初始 PPO 训练状态；PPO 只负责初始 150 点的算子选择。动态执行阶段应比较最终实际执行路线的总距离、方差和响应时间，而不是动态阶段的 HV。

训练和测试的时间记录：训练脚本按 update 记录 rollout 步数、准确耗时和按当前速度预测的总耗时；测试脚本按实例记录准确耗时、事件数和最终 HV，并写出 JSON 报告。两者均使用 `time.perf_counter()`，不依赖操作系统的低精度时间命令。

## 5. `manifest.json` 的字段

```json
[
  {
    "index": 0,
    "path": "clustered_v10_n150_123456.json",
    "seed": 123456,
    "events": 15
  }
]
```

| 字段 | 含义 |
| --- | --- |
| `index` | 数据集内顺序编号。 |
| `path` | 相对于数据集目录的实例文件名。 |
| `seed` | 对应实例的随机种子。 |
| `events` | 对应实例的动态事件总数。 |

执行 `python scripts/validate_dataset.py data/test_150x10` 会检查：坐标是否有限、任务点坐标是否重叠、初始和动态任务 ID 是否唯一且互不冲突、完成任务数是否非负，以及 manifest 是否与实际实例文件一致。默认以欧氏距离 `<= 1e-6` 判定任务点重叠；可通过 `--coordinate-tolerance 0.001` 调整容差。车辆起点与任务点重合不会使数据集失败，但会输出警告。
