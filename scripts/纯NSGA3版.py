import numpy as np
import argparse
import json
import os
import csv
import time
from itertools import combinations

from nsga3drl.data.schema import MTSPInstance
from nsga3drl.dynamic.execution import execute_routes
from nsga3drl.problem.encoding import encode
from nsga3drl.problem.open_mtsp import OpenMTSPProblem

# ==============================================
# 1. 超参数配置（严格对齐组长要求）
# ==============================================
config = {
    "n_cities": 150,
    "n_vehicles": 10,
    "pop_size": 32,
    "n_gen": 160,
    "pc": 0.8,
    "pm": 0.2,
    "n_divisions": 30,  # 双目标30个划分足够，和种群规模匹配，避免死循环
    "seed": 42
}


# ==============================================
# 2. 数据集读取
# ==============================================
def load_dataset(json_path):
    """Load the shared MTSP schema, including dynamic-event metadata."""
    return MTSPInstance.load(json_path)


# ==============================================
# 3. 距离预计算（核心性能优化）
# ==============================================
def precompute_distances(cities, vehicle_starts):
    n = len(cities)
    # 城市间距离矩阵
    city_dist = np.zeros((n, n))
    for i in range(n):
        diff = cities[i] - cities
        city_dist[i] = np.linalg.norm(diff, axis=1)
    # 每辆车起点到每个城市的距离
    depot_dist = np.zeros((len(vehicle_starts), n))
    for v in range(len(vehicle_starts)):
        diff = vehicle_starts[v] - cities
        depot_dist[v] = np.linalg.norm(diff, axis=1)
    return city_dist, depot_dist


# ==============================================
# 4. 解码与适应度计算
# ==============================================
def decode_chromosome(chromosome, n_vehicles):
    n_cities = len(chromosome) - (n_vehicles - 1)
    perm = chromosome[:n_cities].astype(int)
    splits = np.sort(chromosome[n_cities:].astype(int))
    routes = []
    prev = 0
    for split in splits:
        routes.append(perm[prev:split])
        prev = split
    routes.append(perm[prev:])
    return routes


def calc_objectives(chromosome, city_dist, depot_dist, n_vehicles):
    routes = decode_chromosome(chromosome, n_vehicles)
    route_lengths = np.zeros(n_vehicles)
    for v in range(n_vehicles):
        route = routes[v]
        if len(route) == 0:
            continue
        # Open MTSP: start -> first task -> ... -> final task.  There is no
        # return leg to the vehicle start, matching the PPO/baseline definition.
        length = depot_dist[v, route[0]]
        length += np.sum(city_dist[route[:-1], route[1:]])
        route_lengths[v] = length
    f1 = np.sum(route_lengths)
    f2 = np.var(route_lengths)
    return np.array([f1, f2])


# ==============================================
# 5. NSGA-3 核心算子（修复死循环+优化）
# ==============================================
def generate_reference_points(n_obj, n_divisions):
    ref_points = []
    for combo in combinations(range(n_divisions + n_obj - 1), n_obj - 1):
        point = np.zeros(n_obj)
        prev = -1
        for i, c in enumerate(combo):
            point[i] = c - prev - 1
            prev = c
        point[-1] = n_divisions + n_obj - 2 - prev
        ref_points.append(point / n_divisions)
    return np.array(ref_points)


def fast_non_dominated_sort(fitness):
    n = len(fitness)
    domination_count = np.zeros(n, dtype=int)
    dominated_set = [[] for _ in range(n)]
    ranks = np.zeros(n, dtype=int)
    front = [[]]

    for p in range(n):
        for q in range(p + 1, n):
            p_dom_q = np.all(fitness[p] <= fitness[q]) and np.any(fitness[p] < fitness[q])
            q_dom_p = np.all(fitness[q] <= fitness[p]) and np.any(fitness[q] < fitness[p])
            if p_dom_q:
                dominated_set[p].append(q)
                domination_count[q] += 1
            elif q_dom_p:
                dominated_set[q].append(p)
                domination_count[p] += 1
        if domination_count[p] == 0:
            ranks[p] = 0
            front[0].append(p)

    i = 0
    while len(front[i]) > 0:
        next_front = []
        for p in front[i]:
            for q in dominated_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    ranks[q] = i + 1
                    next_front.append(q)
        i += 1
        front.append(next_front)
    return ranks, front


def normalize_population(fitness, ideal_point):
    return fitness - ideal_point


def associate(pop_norm, ref_points):
    n = len(pop_norm)
    assoc_ref = np.zeros(n, dtype=int)
    min_dist = np.full(n, np.inf)
    ref_norm = np.linalg.norm(ref_points, axis=1)

    for i in range(n):
        # 向量投影计算，向量化加速
        proj_scalar = np.dot(pop_norm[i], ref_points.T) / (ref_norm ** 2 + 1e-12)
        proj = proj_scalar[:, None] * ref_points
        dist = np.linalg.norm(pop_norm[i] - proj, axis=1)
        best_idx = np.argmin(dist)
        assoc_ref[i] = best_idx
        min_dist[i] = dist[best_idx]
    return assoc_ref, min_dist


def niching_selection(combined_fit, front, assoc_ref, niche_count, pop_size):
    next_pop_idx = []
    for f in front[:-1]:
        next_pop_idx.extend(f)

    last_front = front[-1]
    if len(next_pop_idx) + len(last_front) <= pop_size:
        next_pop_idx.extend(last_front)
        return np.array(next_pop_idx)

    need = pop_size - len(next_pop_idx)
    selected = set(next_pop_idx)
    remaining = [p for p in last_front if p not in selected]

    # 修复死循环：最多循环need次，避免无限空转
    for _ in range(need):
        min_count = np.min(niche_count)
        min_refs = np.where(niche_count == min_count)[0]

        found = False
        for ref_idx in min_refs:
            candidates = [p for p in remaining if assoc_ref[p] == ref_idx]
            if len(candidates) == 0:
                niche_count[ref_idx] = np.inf
                continue
            # 选距离最近的
            distances = [np.linalg.norm(combined_fit[p]) for p in candidates]
            best = candidates[np.argmin(distances)]
            next_pop_idx.append(best)
            remaining.remove(best)
            niche_count[ref_idx] += 1
            found = True
            break

        # 极端情况：所有参考点都无候选，直接从剩余里随机选
        if not found and remaining:
            pick = remaining.pop(0)
            next_pop_idx.append(pick)
        elif not remaining:
            break

    return np.array(next_pop_idx[:pop_size])


# ==============================================
# 6. 遗传操作
# ==============================================
def init_population(pop_size, n_cities, n_vehicles):
    pop = []
    for _ in range(pop_size):
        perm = np.random.permutation(n_cities)
        splits = np.sort(np.random.choice(range(1, n_cities), size=n_vehicles - 1, replace=False))
        pop.append(np.concatenate([perm, splits]))
    return np.array(pop)


def ox_crossover(parent1, parent2, n_cities):
    n_splits = len(parent1) - n_cities
    perm1, perm2 = parent1[:n_cities], parent2[:n_cities]
    splits1, splits2 = parent1[n_cities:], parent2[n_cities:]

    a, b = np.sort(np.random.randint(0, n_cities, 2))
    child_perm = np.zeros(n_cities, dtype=int)
    child_perm[a:b] = perm1[a:b]

    pos = b
    for city in perm2:
        if city not in child_perm[a:b]:
            if pos >= n_cities:
                pos = 0
            child_perm[pos] = city
            pos += 1

    if np.random.rand() < 0.5 and n_splits > 0:
        cross_pt = np.random.randint(0, n_splits)
        child_splits = np.concatenate([splits1[:cross_pt], splits2[cross_pt:]])
    else:
        child_splits = splits1.copy()

    child_splits = np.unique(child_splits)
    while len(child_splits) < n_splits:
        child_splits = np.unique(np.append(child_splits, np.random.randint(1, n_cities)))
    child_splits = np.sort(child_splits[:n_splits])

    return np.concatenate([child_perm, child_splits])


def swap_mutation(chromosome, n_cities):
    n_splits = len(chromosome) - n_cities
    perm = chromosome[:n_cities].copy()
    splits = chromosome[n_cities:].copy()

    if np.random.rand() < 0.5:
        i, j = np.random.randint(0, n_cities, 2)
        perm[i], perm[j] = perm[j], perm[i]

    if n_splits > 0 and np.random.rand() < 0.3:
        idx = np.random.randint(0, n_splits)
        new_val = splits[idx] + np.random.choice([-1, 1])
        if 1 <= new_val <= n_cities - 1 and new_val not in splits:
            splits[idx] = new_val
            splits.sort()

    return np.concatenate([perm, splits])


def tournament_selection(pop, fitness, ranks, tour_size=2):
    candidates = np.random.randint(0, len(pop), tour_size)
    best = candidates[0]
    for c in candidates[1:]:
        if ranks[c] < ranks[best]:
            best = c
        elif ranks[c] == ranks[best] and np.all(fitness[c] <= fitness[best]):
            best = c
    return pop[best].copy()


# ==============================================
# 7. NSGA-3 主流程
# ==============================================
def nsga3_run(cities, vehicle_starts, config):
    n_cities = config["n_cities"]
    n_vehicles = config["n_vehicles"]
    pop_size = config["pop_size"]
    n_gen = config["n_gen"]

    city_dist, depot_dist = precompute_distances(cities, vehicle_starts)
    ref_points = generate_reference_points(n_obj=2, n_divisions=config["n_divisions"])

    pop = init_population(pop_size, n_cities, n_vehicles)
    fitness = np.array([calc_objectives(ind, city_dist, depot_dist, n_vehicles) for ind in pop])

    for gen in range(1, n_gen + 1):
        ranks, fronts = fast_non_dominated_sort(fitness)

        # 生成子代
        offspring = []
        for _ in range(pop_size):
            p1 = tournament_selection(pop, fitness, ranks)
            p2 = tournament_selection(pop, fitness, ranks)
            if np.random.rand() < config["pc"]:
                child = ox_crossover(p1, p2, n_cities)
            else:
                child = p1.copy()
            if np.random.rand() < config["pm"]:
                child = swap_mutation(child, n_cities)
            offspring.append(child)
        offspring = np.array(offspring)
        off_fitness = np.array([calc_objectives(ind, city_dist, depot_dist, n_vehicles) for ind in offspring])

        # 合并父子代
        combined_pop = np.vstack([pop, offspring])
        combined_fit = np.vstack([fitness, off_fitness])

        # 环境选择
        ranks, fronts = fast_non_dominated_sort(combined_fit)
        ideal_point = np.min(combined_fit, axis=0)
        pop_norm = normalize_population(combined_fit, ideal_point)
        assoc_ref, _ = associate(pop_norm, ref_points)

        niche_count = np.zeros(len(ref_points))
        for i in range(len(combined_pop)):
            if ranks[i] == 0:
                niche_count[assoc_ref[i]] += 1

        next_idx = niching_selection(combined_fit, fronts, assoc_ref, niche_count, pop_size)
        pop = combined_pop[next_idx]
        fitness = combined_fit[next_idx]

        # 每20代打印进度
        if gen % 20 == 0 or gen == 1:
            best_f1 = np.min(fitness[:, 0])
            best_f2 = np.min(fitness[:, 1])
            print(f"第{gen:3d}代 | 最小总距离: {best_f1:.2f} | 最小负载差: {best_f2:.2f}")

    # 提取帕累托解集
    final_ranks, _ = fast_non_dominated_sort(fitness)
    pareto_idx = np.where(final_ranks == 0)[0]
    return pop[pareto_idx], fitness[pareto_idx]


# ==============================================
# 8. 批量运行入口
# ==============================================
def preferred_index(fitness, preference):
    """Choose one Pareto solution using the same normalised preference rule."""
    low = fitness.min(axis=0)
    high = fitness.max(axis=0)
    normalised = (fitness - low) / np.maximum(high - low, 1e-12)
    return int(np.argmin(normalised @ np.asarray(preference, dtype=float)))


def chromosome_routes(chromosome, task_ids, vehicle_count):
    """Map this script's zero-based city indices back to dataset task IDs."""
    return [[int(task_ids[city]) for city in route] for route in decode_chromosome(chromosome, vehicle_count)]


def main():
    parser = argparse.ArgumentParser(description="Pure NSGA-III reference with the shared open-MTSP and dynamic-execution definition.")
    parser.add_argument("--dataset", required=True, help="Directory containing compatible MTSP JSON instances")
    parser.add_argument("--preference", type=float, nargs=2, required=True, metavar=("EFFICIENCY", "FAIRNESS"))
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--generations", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/pure_nsga3_results.csv")
    args = parser.parse_args()
    preference = tuple(args.preference)
    if min(preference) < 0 or not np.isclose(sum(preference), 1.0):
        parser.error("--preference must contain two non-negative values summing to 1")

    dataset = os.path.abspath(args.dataset)
    output_file = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    json_files = sorted(f for f in os.listdir(dataset) if f.endswith('.json') and f != 'manifest.json')
    if not json_files:
        parser.error(f"no instance JSON files found in {dataset}")

    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            "instance_id", "preference", "initial_total_distance", "initial_variance",
            "executed_total_distance", "executed_variance", "pareto_solutions",
            "events", "active_tasks", "dynamic_time_seconds", "elapsed_seconds",
        ])

    print(f"Found {len(json_files)} instances. Running pure NSGA-III...\n")
    for index, filename in enumerate(json_files, 1):
        json_path = os.path.join(dataset, filename)
        started = time.perf_counter()
        try:
            instance = load_dataset(json_path)
            cities = instance.task_coordinates
            vehicle_starts = instance.vehicle_starts
            run_config = dict(config, n_cities=len(cities), n_vehicles=len(vehicle_starts), pop_size=args.population, n_gen=args.generations)
            np.random.seed(args.seed + index - 1)
            pareto_pop, pareto_fit = nsga3_run(cities, vehicle_starts, run_config)
            chosen = preferred_index(pareto_fit, preference)
            initial_routes = chromosome_routes(pareto_pop[chosen], instance.task_ids, instance.vehicle_count)
            execution = execute_routes(instance, initial_routes, preference)
            final_problem = OpenMTSPProblem(execution.instance)
            final_objectives = final_problem.evaluate(encode(execution.routes, instance.vehicle_count))
            elapsed = time.perf_counter() - started
            with open(output_file, 'a', newline='', encoding='utf-8-sig') as f:
                csv.writer(f).writerow([
                    instance.instance_id, list(preference), round(float(pareto_fit[chosen, 0]), 6), round(float(pareto_fit[chosen, 1]), 6),
                    round(float(final_objectives[0]), 6), round(float(final_objectives[1]), 6), len(pareto_fit),
                    len(execution.event_log), final_problem.task_count, round(execution.elapsed_seconds, 6), round(elapsed, 6),
                ])
            print(
                f"[{index}/{len(json_files)}] {instance.instance_id} "
                f"initial_distance={pareto_fit[chosen, 0]:.3f} initial_variance={pareto_fit[chosen, 1]:.3f} "
                f"executed_distance={final_objectives[0]:.3f} executed_variance={final_objectives[1]:.3f} "
                f"events={len(execution.event_log)} active_tasks={final_problem.task_count} elapsed={elapsed:.3f}s"
            )
        except Exception as error:
            print(f"[{index}/{len(json_files)}] {filename} failed: {error}")
    print(f"Finished. Results saved to {output_file}")


if __name__ == "__main__":
    main()
