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
      "arrival_generation": 10,
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
| `preference` | `float[2]` | 用户偏好 `[效率权重, 公平权重]`，两项非负且和为 1。当前生成器默认 `[0.5, 0.5]`。它不改变 NSGA-III 的双目标定义。 |
| `events` | object array | 尚未出现的动态任务事件，按到达代数处理。动态点**不包含**在 `task_coordinates` 中。 |
| `seed` | integer | 生成此实例使用的随机种子，用于复现相同坐标、事件和空间分布。 |

### 目标函数中坐标的作用

车辆路径不要求回到起点。对于第 `i` 辆车的任务序列 `[p1, p2, ...]`：

```text
L_i = distance(start_i, p1) + distance(p1, p2) + ...
f1 = sum(L_i)                         # 总距离，越小越好
f2 = variance(L_1, ..., L_m)          # 车辆路径长度方差，越小越公平
```

MVP 使用二维欧氏距离；坐标不是经纬度，也不代表真实路网距离。

## 3. 动态事件 `events` 的字段与 `arrival_generation`

每个事件的格式为：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `arrival_generation` | non-negative integer | 新任务进入系统的 NSGA-III 代数。 |
| `task_id` | positive integer | 新任务 ID，必须不与初始 `task_ids` 或其他动态事件 ID 重复。生成器从 `初始任务数 + 1` 开始编号。 |
| `coordinate` | `float[2]` | 新任务点坐标 `[x, y]`。 |

### `arrival_generation` 的精确定义

`arrival_generation = g` 表示：**在第 `g` 代开始、生成该代子代之前**，将该任务作为新任务注入当前调度状态。

- 第 `0` 代是初始种群已经创建、尚未进行任何进化的状态。
- `arrival_generation = 0`：初始化后立即插入，再进行第 1 次进化。
- `arrival_generation = 10`：先完成第 1 至第 9 次进化；在第 10 次进化产生子代前处理该点。
- 多个事件可以具有相同 `arrival_generation`，表示同一时刻批量到达；应按 `task_id` 升序逐个插入以保证可复现。
- 事件代数超过实验最大代数时，该事件不会在该轮实验中生效，校验器不会把它视为格式错误。

生成器当前使用固定间隔：第 `j` 个动态点（从 1 开始计数）的 `arrival_generation = 10 * j`。例如 150 个初始点、`dynamic_rate=0.1` 时，生成 15 个动态点，分别在第 `10, 20, ..., 150` 代到达。

动态点到达后，应调用 `DynamicInsertionPlanner.insert(routes, task_id, coordinate)`：它通过 KD-Tree 查询、顺路分与均衡分选择车辆，并在该车的最优位置插入新点。

## 4. 动态事件回放行为

`NSGA3.run()` 和 `OperatorSelectionEnv` 会在每一代开始检查 `events`。事件到达后，系统会对种群中的每个个体执行 KD-Tree 分配、最小增量插入和受影响路径的局部 2-opt，再扩展该个体的分隔符编码并重新计算目标值。事件日志可通过 `algorithm.event_log` 获取。

由于新任务会扩大目标空间，每次动态插点后都会重置 HV 参考点；因此 HV 和 PPO 奖励只应在同一段“两次动态事件之间”的代数内比较，不应跨事件直接相减。

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

执行 `python scripts/validate_dataset.py data/test_150x10` 会检查：坐标是否有限、任务点坐标是否重叠、初始和动态任务 ID 是否唯一且互不冲突、到达代数是否非负，以及 manifest 是否与实际实例文件一致。默认以欧氏距离 `<= 1e-6` 判定任务点重叠；可通过 `--coordinate-tolerance 0.001` 调整容差。车辆起点与任务点重合不会使数据集失败，但会输出警告。
