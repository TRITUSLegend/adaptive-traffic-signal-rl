## Generated results tables

### Mean waiting time per vehicle, by scenario

| Controller | balanced | ns_heavy | peak | ew_heavy | surge | light |
|---|---|---|---|---|---|---|
| Fixed-time (untuned) | 29.6 s | 52.6 s | 72.1 s | 46.5 s | 75.5 s | 28.7 s |
| Fixed-time (Webster-tuned) | 30.5 s | 31.8 s | 46.5 s | 32.7 s | 45.3 s | 23.6 s |
| Greedy queue heuristic | 18.5 s | 22.3 s | 52.7 s | 22.2 s | 50.8 s | 13.4 s |
| **DQN (learned)** | 19.7 s | 20.6 s | 39.4 s | 24.9 s | 37.6 s | 14.0 s |
| PPO (learned) | 19.2 s | 19.8 s | 51.7 s | 25.1 s | 46.6 s | 15.0 s |

### Change in waiting time vs the untuned fixed-time baseline (negative is better)

| Controller | balanced | ns_heavy | peak | ew_heavy | surge | light |
|---|---|---|---|---|---|---|
| Fixed-time (Webster-tuned) | +3.0% | -39.4% | -35.5% | -29.8% | -40.0% | -17.7% |
| Greedy queue heuristic | -37.6% | -57.6% | -26.9% | -52.3% | -32.8% | -53.3% |
| **DQN (learned)** | -33.3% | -60.7% | -45.3% | -46.4% | -50.3% | -51.2% |
| PPO (learned) | -35.0% | -62.4% | -28.3% | -46.2% | -38.3% | -47.6% |

### Scenario: `balanced`

| Controller | Mean wait (s) | Mean queue (veh) | Time lost (s) | Throughput (veh/h) | Wait vs fixed-time |
|---|---|---|---|---|---|
| Fixed-time (untuned) | 29.6 ± 0.8 | 13.0 ± 0.7 | 38.0 | 1546 | &mdash; |
| Fixed-time (Webster-tuned) | 30.5 ± 0.8 | 13.5 ± 0.7 | 38.7 | 1544 | +3.0% |
| Greedy queue heuristic | 18.5 ± 0.7 | 8.1 ± 0.5 | 27.1 | 1549 | -37.6% |
| **DQN (learned)** | 19.7 ± 1.3 | 8.7 ± 0.8 | 27.7 | 1552 | **-33.3%** |
| PPO (learned) | 19.2 ± 0.9 | 8.4 ± 0.6 | 27.3 | 1550 | -35.0% |

### Scenario: `ns_heavy`

| Controller | Mean wait (s) | Mean queue (veh) | Time lost (s) | Throughput (veh/h) | Wait vs fixed-time |
|---|---|---|---|---|---|
| Fixed-time (untuned) | 52.6 ± 9.8 | 26.1 ± 4.9 | 70.4 | 1726 | &mdash; |
| Fixed-time (Webster-tuned) | 31.8 ± 0.5 | 15.9 ± 0.3 | 40.7 | 1750 | -39.4% |
| Greedy queue heuristic | 22.3 ± 1.2 | 11.1 ± 0.6 | 33.7 | 1753 | -57.6% |
| **DQN (learned)** | 20.6 ± 1.1 | 10.2 ± 0.6 | 29.6 | 1755 | **-60.7%** |
| PPO (learned) | 19.8 ± 0.3 | 9.8 ± 0.2 | 28.5 | 1755 | -62.4% |

### Scenario: `peak`

| Controller | Mean wait (s) | Mean queue (veh) | Time lost (s) | Throughput (veh/h) | Wait vs fixed-time |
|---|---|---|---|---|---|
| Fixed-time (untuned) | 72.1 ± 2.7 | 43.9 ± 1.8 | 102.9 | 2093 | &mdash; |
| Fixed-time (Webster-tuned) | 46.5 ± 5.1 | 29.6 ± 3.8 | 60.3 | 2222 | -35.5% |
| Greedy queue heuristic | 52.7 ± 8.6 | 33.5 ± 6.2 | 79.9 | 2197 | -26.9% |
| **DQN (learned)** | 39.4 ± 5.3 | 25.1 ± 4.0 | 53.4 | 2228 | **-45.3%** |
| PPO (learned) | 51.7 ± 10.9 | 32.9 ± 7.7 | 71.5 | 2211 | -28.3% |

### Scenario: `ew_heavy`

| Controller | Mean wait (s) | Mean queue (veh) | Time lost (s) | Throughput (veh/h) | Wait vs fixed-time |
|---|---|---|---|---|---|
| Fixed-time (untuned) | 46.5 ± 5.9 | 23.1 ± 3.0 | 61.1 | 1735 | &mdash; |
| Fixed-time (Webster-tuned) | 32.7 ± 1.1 | 16.3 ± 0.7 | 41.6 | 1754 | -29.8% |
| Greedy queue heuristic | 22.2 ± 1.8 | 11.1 ± 0.9 | 33.5 | 1760 | -52.3% |
| **DQN (learned)** | 24.9 ± 0.6 | 12.4 ± 0.4 | 33.4 | 1760 | **-46.4%** |
| PPO (learned) | 25.1 ± 1.5 | 12.5 ± 0.8 | 35.5 | 1759 | -46.2% |

### Scenario: `surge`

| Controller | Mean wait (s) | Mean queue (veh) | Time lost (s) | Throughput (veh/h) | Wait vs fixed-time |
|---|---|---|---|---|---|
| Fixed-time (untuned) | 75.5 ± 3.8 | 43.8 ± 2.3 | 109.0 | 1987 | &mdash; |
| Fixed-time (Webster-tuned) | 45.3 ± 4.8 | 29.2 ± 3.6 | 58.1 | 2245 | -40.0% |
| Greedy queue heuristic | 50.8 ± 5.6 | 32.5 ± 3.9 | 80.1 | 2211 | -32.8% |
| **DQN (learned)** | 37.6 ± 3.0 | 24.2 ± 2.3 | 51.8 | 2252 | **-50.3%** |
| PPO (learned) | 46.6 ± 6.0 | 30.0 ± 4.2 | 64.4 | 2236 | -38.3% |

### Scenario: `light`

| Controller | Mean wait (s) | Mean queue (veh) | Time lost (s) | Throughput (veh/h) | Wait vs fixed-time |
|---|---|---|---|---|---|
| Fixed-time (untuned) | 28.7 ± 0.8 | 8.5 ± 0.3 | 36.3 | 1044 | &mdash; |
| Fixed-time (Webster-tuned) | 23.6 ± 0.9 | 7.0 ± 0.2 | 30.8 | 1043 | -17.7% |
| Greedy queue heuristic | 13.4 ± 0.6 | 3.9 ± 0.2 | 20.6 | 1046 | -53.3% |
| **DQN (learned)** | 14.0 ± 0.6 | 4.1 ± 0.2 | 20.5 | 1047 | **-51.2%** |
| PPO (learned) | 15.0 ± 0.3 | 4.4 ± 0.1 | 21.4 | 1047 | -47.6% |
