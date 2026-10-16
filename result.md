for SPEC in \
>   "1.0 0.0 e100_f000" \
>   "0.8 0.2 e080_f020" \
>   "0.6 0.4 e060_f040" \
>   "0.5 0.5 e050_f050" \
>   "0.4 0.6 e040_f060" \
>   "0.2 0.8 e020_f080" \
>   "0.0 1.0 e000_f100"
> do
>   read -r EFF FAIR TAG <<< "$SPEC"
>   python scripts/evaluate_ppo.py \
>     --dataset "$TEST_DATASET" \
>     --checkpoint "$PPO_CHECKPOINT" \
>     --preference "$EFF" "$FAIR" \
>     --generations 240 \
>     --population 32 \
>     --seed 42 \
>     --dynamic-strategy greedy_score_then_tsp \
>     --metrics "results/preference_comparison/ppo_greedy_${TAG}.eval.json"
> done
Traceback (most recent call last):
  File "/home/yihan_tong/nsga3drl/scripts/evaluate_ppo.py", line 160, in <module>
    main()
  File "/home/yihan_tong/nsga3drl/scripts/evaluate_ppo.py", line 37, in main
    trainer = PPOTrainer.load(args.checkpoint)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yihan_tong/nsga3drl/src/nsga3drl/rl/ppo.py", line 80, in load
    checkpoint = torch.load(path, map_location=device)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yihan_tong/miniconda3/envs/nsga3drl/lib/python3.11/site-packages/torch/serialization.py", line 998, in load
    with _open_file_like(f, 'rb') as opened_file:
         ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yihan_tong/miniconda3/envs/nsga3drl/lib/python3.11/site-packages/torch/serialization.py", line 445, in _open_file_like
    return _open_file(name_or_buffer, mode)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yihan_tong/miniconda3/envs/nsga3drl/lib/python3.11/site-packages/torch/serialization.py", line 426, in __init__
    super().__init__(open(name, mode))
                     ^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: ''
Traceback (most recent call last):
  File "/home/yihan_tong/nsga3drl/scripts/evaluate_ppo.py", line 160, in <module>
    main()
  File "/home/yihan_tong/nsga3drl/scripts/evaluate_ppo.py", line 37, in main
    trainer = PPOTrainer.load(args.checkpoint)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yihan_tong/nsga3drl/src/nsga3drl/rl/ppo.py", line 80, in load
    checkpoint = torch.load(path, map_location=device)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yihan_tong/miniconda3/envs/nsga3drl/lib/python3.11/site-packages/torch/serialization.py", line 998, in load
    with _open_file_like(f, 'rb') as opened_file:
         ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yihan_tong/miniconda3/envs/nsga3drl/lib/python3.11/site-packages/torch/serialization.py", line 445, in _open_file_like
    return _open_file(name_or_buffer, mode)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yihan_tong/miniconda3/envs/nsga3drl/lib/python3.11/site-packages/torch/serialization.py", line 426, in __init__
    super().__init__(open(name, mode))
                     ^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: ''
Traceback (most recent call last):
  File "/home/yihan_tong/nsga3drl/scripts/evaluate_ppo.py", line 160, in <module>
    main()
  File "/home/yihan_tong/nsga3drl/scripts/evaluate_ppo.py", line 37, in main
    trainer = PPOTrainer.load(args.checkpoint)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yihan_tong/nsga3drl/src/nsga3drl/rl/ppo.py", line 80, in load
    checkpoint = torch.load(path, map_location=device)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yihan_tong/miniconda3/envs/nsga3drl/lib/python3.11/site-packages/torch/serialization.py", line 998, in load
    with _open_file_like(f, 'rb') as opened_file:
         ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yihan_tong/miniconda3/envs/nsga3drl/lib/python3.11/site-packages/torch/serialization.py", line 445, in _open_file_like
    return _open_file(name_or_buffer, mode)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yihan_tong/miniconda3/envs/nsga3drl/lib/python3.11/site-packages/torch/serialization.py", line 426, in __init__
    super().__init__(open(name, mode))
                     ^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: ''
Traceback (most recent call last):
  File "/home/yihan_tong/nsga3drl/scripts/evaluate_ppo.py", line 160, in <module>
    main()
  File "/home/yihan_tong/nsga3drl/scripts/evaluate_ppo.py", line 37, in main
    trainer = PPOTrainer.load(args.checkpoint)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yihan_tong/nsga3drl/src/nsga3drl/rl/ppo.py", line 80, in load
    checkpoint = torch.load(path, map_location=device)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yihan_tong/miniconda3/envs/nsga3drl/lib/python3.11/site-packages/torch/serialization.py", line 998, in load
    with _open_file_like(f, 'rb') as opened_file:
         ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yihan_tong/miniconda3/envs/nsga3drl/lib/python3.11/site-packages/torch/serialization.py", line 445, in _open_file_like
    return _open_file(name_or_buffer, mode)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yihan_tong/miniconda3/envs/nsga3drl/lib/python3.11/site-packages/torch/serialization.py", line 426, in __init__
    super().__init__(open(name, mode))
                     ^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: ''
Traceback (most recent call last):
  File "/home/yihan_tong/nsga3drl/scripts/evaluate_ppo.py", line 160, in <module>
    main()
  File "/home/yihan_tong/nsga3drl/scripts/evaluate_ppo.py", line 37, in main
    trainer = PPOTrainer.load(args.checkpoint)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yihan_tong/nsga3drl/src/nsga3drl/rl/ppo.py", line 80, in load
    checkpoint = torch.load(path, map_location=device)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yihan_tong/miniconda3/envs/nsga3drl/lib/python3.11/site-packages/torch/serialization.py", line 998, in load
    with _open_file_like(f, 'rb') as opened_file:
         ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yihan_tong/miniconda3/envs/nsga3drl/lib/python3.11/site-packages/torch/serialization.py", line 445, in _open_file_like
    return _open_file(name_or_buffer, mode)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yihan_tong/miniconda3/envs/nsga3drl/lib/python3.11/site-packages/torch/serialization.py", line 426, in __init__
    super().__init__(open(name, mode))
                     ^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: ''
Traceback (most recent call last):
  File "/home/yihan_tong/nsga3drl/scripts/evaluate_ppo.py", line 160, in <module>
    main()
  File "/home/yihan_tong/nsga3drl/scripts/evaluate_ppo.py", line 37, in main
    trainer = PPOTrainer.load(args.checkpoint)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yihan_tong/nsga3drl/src/nsga3drl/rl/ppo.py", line 80, in load
    checkpoint = torch.load(path, map_location=device)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yihan_tong/miniconda3/envs/nsga3drl/lib/python3.11/site-packages/torch/serialization.py", line 998, in load
    with _open_file_like(f, 'rb') as opened_file:
         ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yihan_tong/miniconda3/envs/nsga3drl/lib/python3.11/site-packages/torch/serialization.py", line 445, in _open_file_like
    return _open_file(name_or_buffer, mode)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yihan_tong/miniconda3/envs/nsga3drl/lib/python3.11/site-packages/torch/serialization.py", line 426, in __init__
    super().__init__(open(name, mode))
                     ^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: ''
Traceback (most recent call last):
  File "/home/yihan_tong/nsga3drl/scripts/evaluate_ppo.py", line 160, in <module>
    main()
  File "/home/yihan_tong/nsga3drl/scripts/evaluate_ppo.py", line 37, in main
    trainer = PPOTrainer.load(args.checkpoint)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yihan_tong/nsga3drl/src/nsga3drl/rl/ppo.py", line 80, in load
    checkpoint = torch.load(path, map_location=device)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yihan_tong/miniconda3/envs/nsga3drl/lib/python3.11/site-packages/torch/serialization.py", line 998, in load
    with _open_file_like(f, 'rb') as opened_file:
         ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yihan_tong/miniconda3/envs/nsga3drl/lib/python3.11/site-packages/torch/serialization.py", line 445, in _open_file_like
    return _open_file(name_or_buffer, mode)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yihan_tong/miniconda3/envs/nsga3drl/lib/python3.11/site-packages/torch/serialization.py", line 426, in __init__
    super().__init__(open(name, mode))
                     ^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: ''
(nsga3drl) yihan_tong@dell-PowerEdge-T640:~/nsga3drl$ 