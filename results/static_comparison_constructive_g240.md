# Static Route Comparison

All four methods are compared on the same 20 clustered instances with 150 original tasks and 10 vehicles. The preference is `[1.0, 0.0]`, so the primary selection criterion is total distance. Only the initial 150-task plan is compared here; dynamic events are excluded from the distance comparison. The HSNR column uses the requested `f1_min` field.

## Aggregate Results

| Method | Mean distance | Mean variance | Mean reported time | Wins against PPO |
|---|---:|---:|---:|---:|
| PPO + constructive initialization | **656.24** | 465.09 | 37.65 s | 17/20 vs baseline |
| NSGA-III baseline + same construction | 661.27 | **492.23** | 28.18 s | 3/20 vs PPO |
| Collaborator static method | 692.07 | not reported | 5.07 s* | 1/20 vs PPO |
| HSNR min-sum collaborator (`f1_min`) | 763.25 | not reported | 30.62 s** | 1/20 vs PPO |

The PPO initial distance is 5.03 units shorter than the new baseline on average, about 0.76% better. It is 35.83 units shorter than the first collaborator result, about 5.18% better, and 107.01 units shorter than HSNR `f1_min`, about 14.02% better. PPO beats the first collaborator on 19 of 20 instances, and beats HSNR on 19 of 20 instances.

The reported times are not a strict hardware benchmark: PPO includes neural-network action selection and the current evaluation wrapper, while collaborator outputs use their own implementation and timing convention. `*` is the first collaborator's reported `time_total` mean. `**` is HSNR's mean time for one run; its displayed `f1_min` is the best result selected from five independent 30-second runs, so it is not a single-run result.

## Per-Instance Distance

| Instance | PPO initial | Baseline initial | Collaborator | HSNR `f1_min` | Best |
|---|---:|---:|---:|---:|---|
| clustered_v10_n150_1081530686 | 528.33 | 530.11 | 578.67 | 696.27 | PPO |
| clustered_v10_n150_1097657231 | 582.91 | 583.88 | 643.38 | 722.59 | PPO |
| clustered_v10_n150_1167425778 | 641.74 | 641.74 | 649.43 | 700.22 | PPO/Baseline |
| clustered_v10_n150_1202413557 | 485.55 | 495.38 | 508.98 | 636.69 | PPO |
| clustered_v10_n150_1302740407 | 713.26 | 718.63 | 723.98 | 914.35 | PPO |
| clustered_v10_n150_1357791198 | 759.10 | 766.18 | 818.68 | 760.03 | PPO |
| clustered_v10_n150_1367864806 | 606.31 | 606.75 | 694.22 | 757.55 | PPO |
| clustered_v10_n150_1394609702 | 661.75 | 685.67 | 765.44 | 761.06 | PPO |
| clustered_v10_n150_1566581934 | 644.12 | 644.12 | **629.85** | 789.30 | Collaborator |
| clustered_v10_n150_161576974 | 776.61 | 781.91 | 862.46 | **744.97** | HSNR |
| clustered_v10_n150_1746484539 | 579.28 | 579.40 | 628.53 | 762.04 | PPO |
| clustered_v10_n150_1826701614 | 680.79 | 684.75 | 690.16 | 825.73 | PPO |
| clustered_v10_n150_1960127675 | 562.87 | 564.94 | 568.97 | 809.75 | PPO |
| clustered_v10_n150_2008052738 | 663.50 | 666.18 | 678.03 | 794.55 | PPO |
| clustered_v10_n150_2084654369 | 691.87 | 695.77 | 766.14 | 795.87 | PPO |
| clustered_v10_n150_35492826 | 672.95 | 672.95 | 684.55 | 809.65 | PPO/Baseline |
| clustered_v10_n150_376383645 | 704.63 | 709.58 | 726.37 | 748.12 | PPO |
| clustered_v10_n150_579362555 | 713.54 | 719.94 | 734.96 | 757.59 | PPO |
| clustered_v10_n150_661058651 | 763.43 | 784.06 | 790.43 | 773.66 | PPO |
| clustered_v10_n150_87989972 | 692.28 | 693.55 | 698.09 | 705.07 | PPO |

## Fairness Preference: Static Reference

The following results use preference `[0.0, 1.0]` and only the initial 150-task plan. For PPO and baseline, the selected solution is the Pareto solution with the lowest route-length variance. `Max route` is then calculated as the maximum of that selected solution's ten vehicle route lengths.

For HSNR, `f2_min` is its minimum variance and `max_route_min` is its minimum longest-vehicle-route value. As requested, both values use the minimum over its five runs. These two HSNR minima can originate from different runs and do not necessarily describe one common solution. PPO and baseline each use one run, so the comparison is metric-aligned but remains favourable to HSNR's multi-run selection.

| Method | Variance value | Max-route value | Mean initial distance | Mean reported time |
|---|---:|---:|---:|---:|
| PPO 1h | **43.77** | **96.05** | **847.24** | 48.61 s |
| NSGA-III baseline | 49.84 | 98.25 | 850.67 | 28.49 s |
| HSNR min-max collaborator | 253.87 (`f2_min`) | 112.37 (`max_route_min`) | not used for this comparison | 31.96 s |

PPO has the lowest variance on 11 of 20 instances, baseline on 8, and HSNR on 1. For maximum route length, PPO is lowest on 11 instances, baseline on 7, and HSNR on 2.

| Instance | PPO variance | PPO max route | Baseline variance | Baseline max route | HSNR `f2_min` variance | HSNR `max_route_min` | Variance best | Max-route best |
|---|---:|---:|---:|---:|---:|---:|---|---|
| clustered_v10_n150_1081530686 | 42.18 | 93.79 | 38.56 | 78.48 | 70.58 | 94.33 | Baseline | Baseline |
| clustered_v10_n150_1097657231 | 5.80 | 81.92 | 22.08 | 76.58 | 280.77 | 103.76 | PPO | Baseline |
| clustered_v10_n150_1167425778 | 11.08 | 107.81 | 59.73 | 111.23 | 210.27 | 104.38 | PPO | HSNR |
| clustered_v10_n150_1202413557 | 33.01 | 70.98 | 13.49 | 106.91 | 520.81 | 97.08 | Baseline | PPO |
| clustered_v10_n150_1302740407 | 42.12 | 99.74 | 66.78 | 101.97 | 472.40 | 128.69 | PPO | PPO |
| clustered_v10_n150_1357791198 | 165.60 | 117.59 | 105.26 | 108.54 | 105.27 | 114.57 | Baseline | Baseline |
| clustered_v10_n150_1367864806 | 59.29 | 105.59 | 37.42 | 113.19 | 127.68 | 108.05 | Baseline | PPO |
| clustered_v10_n150_1394609702 | 37.93 | 95.51 | 24.57 | 94.11 | 995.96 | 149.07 | Baseline | Baseline |
| clustered_v10_n150_1566581934 | 78.90 | 101.46 | 78.54 | 103.28 | 132.07 | 116.52 | Baseline | PPO |
| clustered_v10_n150_161576974 | 8.41 | 112.30 | 15.76 | 100.87 | 65.39 | 114.64 | PPO | Baseline |
| clustered_v10_n150_1746484539 | 20.57 | 86.41 | 24.71 | 89.22 | 302.11 | 112.08 | PPO | PPO |
| clustered_v10_n150_1826701614 | 83.28 | 99.09 | 71.87 | 111.52 | 65.20 | 102.11 | HSNR | PPO |
| clustered_v10_n150_1960127675 | 31.12 | 73.10 | 44.28 | 73.81 | 344.20 | 105.91 | PPO | PPO |
| clustered_v10_n150_2008052738 | 15.30 | 87.84 | 29.66 | 84.62 | 86.19 | 108.50 | PPO | Baseline |
| clustered_v10_n150_2084654369 | 73.92 | 116.68 | 180.79 | 115.18 | 208.70 | 106.68 | PPO | HSNR |
| clustered_v10_n150_35492826 | 5.42 | 77.92 | 5.61 | 78.65 | 474.44 | 124.72 | PPO | PPO |
| clustered_v10_n150_376383645 | 11.71 | 86.30 | 46.76 | 92.66 | 245.36 | 114.28 | PPO | PPO |
| clustered_v10_n150_579362555 | 76.14 | 98.71 | 58.23 | 104.42 | 60.85 | 105.38 | Baseline | PPO |
| clustered_v10_n150_661058651 | 46.25 | 113.62 | 32.16 | 105.77 | 157.20 | 121.16 | Baseline | Baseline |
| clustered_v10_n150_87989972 | 27.33 | 94.60 | 40.57 | 113.94 | 151.95 | 115.41 | PPO | PPO |

HSNR time is the mean time of one run. Its displayed `f2_min` and `max_route_min` are each selected from five runs.

## Interpretation

The constructive route generation and local search are responsible for the largest improvement: both PPO and baseline are now close to, or better than, the collaborator static results. PPO adds a smaller but measurable improvement over the same NSGA-III baseline, winning 17 of 20 instances. HSNR is the best method on `clustered_v10_n150_161576974`; PPO is best or tied best on the other 19 instances.

The next fair experiment is to report this static table together with the dynamic metrics separately. Do not compare the collaborator's static distance to PPO's final dynamic distance, because the latter contains 15 additional tasks.
