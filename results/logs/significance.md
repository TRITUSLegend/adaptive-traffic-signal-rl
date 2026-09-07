## Addendum A: paired significance tests

Every controller drives the *same* five hours of traffic on a given
scenario, so the comparison is paired and the differences below are
per-seed. `Mean diff` is A - B in seconds of mean waiting time, so a
negative value means A waits less.

Two tests are reported. The **paired t-test** is the primary result:
it uses the magnitude of each difference and so has usable power at
n = 5. The **exact Wilcoxon signed-rank test** is a distribution-free
cross-check that assumes nothing about normality.

> **Read the Wilcoxon column with its ceiling in mind.** With 5 paired
> observations there are only 2^5 = 32 equally likely sign assignments
> under the null, so the smallest two-sided p-value it can return is
> 0.0625. It can never report significance at the 0.05 level here, however
> large the effect. A `0.0625` in that column is the test's floor - it means
> *every* seed favoured the same controller, which is the strongest
> signal the test is capable of producing at this sample size, not a
> null result.

Significance markers: `***` p<0.001, `**` p<0.01, `*` p<0.05,
`n.s.` not significant at 0.05. Metric: `mean_waiting_time_s`.

### DQN vs Greedy queue

| Scenario | Mean diff (s) | \|t\| | p (paired t) | Verdict | p (Wilcoxon) |
|---|---|---|---|---|---|
| `balanced` | +1.27 | 3.93 | 0.0170 | * | 0.0625 |
| `ns_heavy` | -1.65 | 3.55 | 0.0239 | * | 0.0625 |
| `peak` | -13.26 | 7.47 | 0.0017 | ** | 0.0625 |
| `ew_heavy` | +2.72 | 3.00 | 0.0401 | * | 0.1250 |
| `surge` | -13.20 | 7.48 | 0.0017 | ** | 0.0625 |
| `light` | +0.61 | 1.44 | 0.2238 | n.s. | 0.3125 |

### DQN vs Fixed-time (untuned)

| Scenario | Mean diff (s) | \|t\| | p (paired t) | Verdict | p (Wilcoxon) |
|---|---|---|---|---|---|
| `balanced` | -9.87 | 28.88 | 0.0000 | *** | 0.0625 |
| `ns_heavy` | -31.93 | 7.77 | 0.0015 | ** | 0.0625 |
| `peak` | -32.63 | 23.44 | 0.0000 | *** | 0.0625 |
| `ew_heavy` | -21.61 | 8.45 | 0.0011 | ** | 0.0625 |
| `surge` | -37.96 | 19.91 | 0.0000 | *** | 0.0625 |
| `light` | -14.68 | 32.98 | 0.0000 | *** | 0.0625 |

### DQN vs Fixed-time (Webster-tuned)

| Scenario | Mean diff (s) | \|t\| | p (paired t) | Verdict | p (Wilcoxon) |
|---|---|---|---|---|---|
| `balanced` | -10.75 | 34.75 | 0.0000 | *** | 0.0625 |
| `ns_heavy` | -11.21 | 19.53 | 0.0000 | *** | 0.0625 |
| `peak` | -7.04 | 3.89 | 0.0176 | * | 0.0625 |
| `ew_heavy` | -7.76 | 11.28 | 0.0004 | *** | 0.0625 |
| `surge` | -7.78 | 5.46 | 0.0055 | ** | 0.0625 |
| `light` | -9.59 | 32.83 | 0.0000 | *** | 0.0625 |

### PPO vs DQN

| Scenario | Mean diff (s) | \|t\| | p (paired t) | Verdict | p (Wilcoxon) |
|---|---|---|---|---|---|
| `balanced` | -0.49 | 1.38 | 0.2398 | n.s. | 0.3125 |
| `ns_heavy` | -0.85 | 2.13 | 0.1003 | n.s. | 0.1250 |
| `peak` | +12.23 | 4.02 | 0.0158 | * | 0.0625 |
| `ew_heavy` | +0.12 | 0.16 | 0.8829 | n.s. | 1.0000 |
| `surge` | +9.05 | 5.93 | 0.0041 | ** | 0.0625 |
| `light` | +1.01 | 3.17 | 0.0339 | * | 0.0625 |

### Greedy queue vs Fixed-time (untuned)

| Scenario | Mean diff (s) | \|t\| | p (paired t) | Verdict | p (Wilcoxon) |
|---|---|---|---|---|---|
| `balanced` | -11.13 | 33.44 | 0.0000 | *** | 0.0625 |
| `ns_heavy` | -30.28 | 7.37 | 0.0018 | ** | 0.0625 |
| `peak` | -19.36 | 6.70 | 0.0026 | ** | 0.0625 |
| `ew_heavy` | -24.33 | 10.10 | 0.0005 | *** | 0.0625 |
| `surge` | -24.76 | 11.88 | 0.0003 | *** | 0.0625 |
| `light` | -15.30 | 25.80 | 0.0000 | *** | 0.0625 |

### What the tests change

Against the greedy heuristic the DQN is significantly better on `ns_heavy`, `peak`, `surge`,
significantly worse on `balanced`, `ew_heavy`, and indistinguishable on `light`.

**§5.1 survives, and gets sharper.** The report's reading - that most
of the gain comes from being adaptive at all, and that the learned
policy earns its keep specifically under saturation - is what the
tests show. Every scenario where the DQN wins significantly is at or
beyond capacity, and the margins there (13 s) are an order of
magnitude larger than anywhere else.

**§6 needs revising in one respect.** It advises treating "any gap
of a second or two as noise". That is too conservative for a
*paired* comparison: because every controller drives the identical
five hours, the seed-to-seed variation cancels, and a gap of
1.3 s on `balanced` is resolved at p<0.05. The honest threshold is
lower than the report assumed - the only DQN-versus-greedy gap that
does *not* separate is the 0.6 s on `light`.

That cuts both ways, and the losing direction should be stated as
plainly as the winning one: the DQN being beaten by the heuristic
on `balanced` and `ew_heavy` is a real effect, not noise.

The §6 caveat that "no significance test was run" no longer applies.
The separate caveat that five runs is a small sample still does: these
tests establish that differences exist, not how large they would be on
a different day, and the Wilcoxon column shows how quickly a
distribution-free test runs out of resolution at this n.
