(nsga3drl) yihan_tong@dell-PowerEdge-T640:~/nsga3drl$ python scripts/evaluate_ppo.py \
>     --dataset data/test_150x10 \
>     --checkpoint checkpoints/ppo_dynamic_150x10.pt \
>     --generations 160 \
>     --population 32 \
>     --seed 42
instance=clustered_v10_n150_1081530686 final_hv=11018825.6024 reward_sum=0.86821 events=15 active_tasks=165 elapsed=94.732s projected_total=1894.810s
instance=clustered_v10_n150_1097657231 final_hv=23986306.2520 reward_sum=0.59750 events=15 active_tasks=165 elapsed=93.174s projected_total=1879.148s
instance=clustered_v10_n150_1167425778 final_hv=16389095.5510 reward_sum=0.55304 events=15 active_tasks=165 elapsed=92.952s projected_total=1872.444s
instance=clustered_v10_n150_1202413557 final_hv=17223673.0643 reward_sum=0.95816 events=15 active_tasks=165 elapsed=93.757s projected_total=1873.119s
instance=clustered_v10_n150_1302740407 final_hv=31365698.9485 reward_sum=0.76142 events=15 active_tasks=165 elapsed=93.987s projected_total=1874.442s
instance=clustered_v10_n150_1357791198 final_hv=44717253.1320 reward_sum=0.67030 events=15 active_tasks=165 elapsed=93.231s projected_total=1872.804s
instance=clustered_v10_n150_1367864806 final_hv=44873530.8960 reward_sum=0.82655 events=15 active_tasks=165 elapsed=93.525s projected_total=1872.475s
instance=clustered_v10_n150_1394609702 final_hv=39061820.2277 reward_sum=0.74496 events=15 active_tasks=165 elapsed=93.019s projected_total=1870.964s
instance=clustered_v10_n150_1566581934 final_hv=26473845.2057 reward_sum=0.70972 events=15 active_tasks=165 elapsed=93.049s projected_total=1869.854s
instance=clustered_v10_n150_161576974 final_hv=23821362.7805 reward_sum=0.85834 events=15 active_tasks=165 elapsed=93.516s projected_total=1869.901s
instance=clustered_v10_n150_1746484539 final_hv=125110277.1670 reward_sum=0.50334 events=15 active_tasks=165 elapsed=92.839s projected_total=1868.708s
instance=clustered_v10_n150_1826701614 final_hv=49290528.2894 reward_sum=0.82415 events=15 active_tasks=165 elapsed=93.557s projected_total=1868.911s
instance=clustered_v10_n150_1960127675 final_hv=10195274.3299 reward_sum=0.65202 events=15 active_tasks=165 elapsed=92.714s projected_total=1867.786s
instance=clustered_v10_n150_2008052738 final_hv=26910107.1899 reward_sum=0.72446 events=15 active_tasks=165 elapsed=92.030s projected_total=1865.844s
instance=clustered_v10_n150_2084654369 final_hv=81167071.4412 reward_sum=0.93894 events=15 active_tasks=165 elapsed=93.320s projected_total=1865.880s
instance=clustered_v10_n150_35492826 final_hv=59874869.2209 reward_sum=0.90225 events=15 active_tasks=165 elapsed=93.435s projected_total=1866.057s
instance=clustered_v10_n150_376383645 final_hv=24310259.5956 reward_sum=0.72270 events=15 active_tasks=165 elapsed=93.263s projected_total=1866.010s
instance=clustered_v10_n150_579362555 final_hv=54888838.3395 reward_sum=1.00797 events=15 active_tasks=165 elapsed=93.624s projected_total=1866.370s
instance=clustered_v10_n150_661058651 final_hv=57479273.2446 reward_sum=0.99258 events=15 active_tasks=165 elapsed=93.170s projected_total=1866.214s
instance=clustered_v10_n150_87989972 final_hv=7664525.9252 reward_sum=0.48239 events=15 active_tasks=165 elapsed=92.822s projected_total=1865.726s
summary instances=20 mean_final_hv=38791121.8202 mean_reward_sum=0.76495 mean_events=15.00 elapsed=1865.726s metrics=checkpoints/ppo_dynamic_150x10.eval.json
(nsga3drl) yihan_tong@dell-PowerEdge-T640:~/nsga3drl$ python scripts/evaluate_ppo.py \
>     --dataset data/test_150x10_preferences \
>     --checkpoint checkpoints/ppo_dynamic_150x10.pt \
>     --generations 160 \
>     --population 32 \
>     --seed 42
instance=clustered_v10_n150_1039606416 final_hv=45750130.8904 reward_sum=0.66325 events=15 active_tasks=165 elapsed=105.238s projected_total=3157.644s
instance=clustered_v10_n150_1073074663 final_hv=56948104.4623 reward_sum=0.75105 events=15 active_tasks=165 elapsed=99.743s projected_total=3074.964s
instance=clustered_v10_n150_1097276529 final_hv=144032098.2971 reward_sum=0.80136 events=15 active_tasks=165 elapsed=99.316s projected_total=3043.139s
instance=clustered_v10_n150_1136516574 final_hv=38381494.3850 reward_sum=0.86325 events=15 active_tasks=165 elapsed=99.964s projected_total=3032.083s
instance=clustered_v10_n150_1450494010 final_hv=13420455.8464 reward_sum=0.70531 events=15 active_tasks=165 elapsed=100.040s projected_total=3025.904s
instance=clustered_v10_n150_1510975705 final_hv=36573047.3286 reward_sum=1.06203 events=15 active_tasks=165 elapsed=100.943s projected_total=3026.304s
instance=clustered_v10_n150_162666930 final_hv=30694584.8633 reward_sum=0.72946 events=15 active_tasks=165 elapsed=100.390s projected_total=3024.217s
instance=clustered_v10_n150_1637942552 final_hv=39014361.7270 reward_sum=1.11989 events=15 active_tasks=165 elapsed=100.336s projected_total=3022.450s
instance=clustered_v10_n150_1764200989 final_hv=11493348.0319 reward_sum=1.00961 events=15 active_tasks=165 elapsed=99.553s projected_total=3018.467s
instance=clustered_v10_n150_1798304279 final_hv=217402799.3953 reward_sum=0.98483 events=15 active_tasks=165 elapsed=100.136s projected_total=3017.030s
instance=clustered_v10_n150_1846487998 final_hv=82915233.6675 reward_sum=0.87403 events=15 active_tasks=165 elapsed=100.142s projected_total=3015.868s
instance=clustered_v10_n150_1851511491 final_hv=18155711.1148 reward_sum=0.79572 events=15 active_tasks=165 elapsed=99.723s projected_total=3013.853s
instance=clustered_v10_n150_2002243403 final_hv=23919082.5687 reward_sum=0.94510 events=15 active_tasks=165 elapsed=100.128s projected_total=3013.082s
instance=clustered_v10_n150_2008252322 final_hv=64427291.6461 reward_sum=0.96476 events=15 active_tasks=165 elapsed=99.845s projected_total=3011.815s
instance=clustered_v10_n150_2039239133 final_hv=26925562.8380 reward_sum=1.02705 events=15 active_tasks=165 elapsed=99.769s projected_total=3010.566s
instance=clustered_v10_n150_2114844020 final_hv=15218143.7552 reward_sum=0.67664 events=15 active_tasks=165 elapsed=100.181s projected_total=3010.244s
instance=clustered_v10_n150_2145690041 final_hv=62786739.1887 reward_sum=0.82515 events=15 active_tasks=165 elapsed=100.366s projected_total=3010.288s
instance=clustered_v10_n150_247395568 final_hv=55063180.8631 reward_sum=0.77115 events=15 active_tasks=165 elapsed=100.409s projected_total=3010.398s
instance=clustered_v10_n150_351456463 final_hv=41857597.9261 reward_sum=0.88435 events=15 active_tasks=165 elapsed=99.877s projected_total=3009.656s
instance=clustered_v10_n150_358644765 final_hv=54996422.5454 reward_sum=0.85760 events=15 active_tasks=165 elapsed=99.437s projected_total=3008.329s
instance=clustered_v10_n150_370035341 final_hv=142553355.6341 reward_sum=0.84385 events=15 active_tasks=165 elapsed=100.127s projected_total=3008.114s
instance=clustered_v10_n150_447787736 final_hv=135674832.2667 reward_sum=0.76343 events=15 active_tasks=165 elapsed=99.028s projected_total=3006.420s
instance=clustered_v10_n150_463929282 final_hv=52859010.4339 reward_sum=1.11465 events=15 active_tasks=165 elapsed=102.121s projected_total=3008.908s
instance=clustered_v10_n150_527333212 final_hv=182604445.5478 reward_sum=0.77660 events=15 active_tasks=165 elapsed=99.733s projected_total=3008.202s
instance=clustered_v10_n150_600255899 final_hv=37647174.8487 reward_sum=0.71720 events=15 active_tasks=165 elapsed=100.647s projected_total=3008.650s
instance=clustered_v10_n150_6545106 final_hv=13079536.3595 reward_sum=0.74923 events=15 active_tasks=165 elapsed=103.732s projected_total=3012.624s
instance=clustered_v10_n150_7975317 final_hv=111706301.1432 reward_sum=0.62854 events=15 active_tasks=165 elapsed=99.409s projected_total=3011.500s
instance=clustered_v10_n150_920720100 final_hv=12707207.4698 reward_sum=0.52984 events=15 active_tasks=165 elapsed=97.723s projected_total=3008.650s
instance=clustered_v10_n150_99066772 final_hv=98654905.0495 reward_sum=0.89314 events=15 active_tasks=165 elapsed=100.079s projected_total=3008.433s
instance=clustered_v10_n150_99941638 final_hv=16642384.9429 reward_sum=0.58501 events=15 active_tasks=165 elapsed=99.286s projected_total=3007.438s
summary instances=30 mean_final_hv=62803484.8346 mean_reward_sum=0.83044 mean_events=15.00 elapsed=3007.438s metrics=checkpoints/ppo_dynamic_150x10.eval.json
python scripts/evaluate_ppo.py \
    --dataset data/test_150x10_preferences \
    --checkpoint checkpoints/ppo_dynamic_150x10.pt \
    --generations 160 \
    --population 32 \
    --seed 42 \
    --metrics results/ppo_test_150x10_preferences.json