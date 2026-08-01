(nsga3drl) yihan_tong@dell-PowerEdge-T640:~/nsga3drl$ python scripts/run_baseline.py \
>     --instance "$(find data/test_150x10 -type f -name '*.json' ! -name 'manifest.json' | head -n 1)" \
>     --population 64 \
>     --generations 160 \
>     --seed 42
dynamic events processed=15/15; final active tasks=165; dynamic_time=2.576s
final HV=5852651.4659; pareto solutions=15; elapsed=697.725s
total_distance=2658.558, variance=136.151
total_distance=2536.468, variance=482.860
total_distance=2527.845, variance=543.640
total_distance=2675.052, variance=78.208
total_distance=2643.057, variance=199.624