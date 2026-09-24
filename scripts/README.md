# scripts/

Offline analysis, the stiffness benchmarks and the paper's figures. Training and
serving of the SmolVLA policies are in `compliance_vla.policy`
(`../src/compliance_vla/policy/`), and robot deployment in `../smolvla_policy/`.
Dataset I/O, forward kinematics and label extraction are in `../data_extraction/`,
which these scripts put on `sys.path` themselves.

| File | Role |
| --- | --- |
| `offline_stiffness_benchmark.py` | M8 learnability check: learned vs constant vs nearest-neighbour stiffness prediction on held-out sessions |
| `fit_b1_oracle_stiffness.py` | B1 baseline: per-axis constant stiffness grid-searched on the validation split |
| `analyze_cross_operator_adverbs.py` | Cross-operator manner-word -> contact-force analysis (Figure 6) |
| `analyze_data_two_color.py` | Sanity checks and adverb analysis for the `data_two_color` collection |
| `analyze_data_factorial.py` | Factor-decorrelation checks for the `data_factorial_2c` collection |
| `figures/` | Figure generation |

Run from the release root, e.g.:

```bash
python scripts/fit_b1_oracle_stiffness.py        # CPU-only, under a minute
python scripts/offline_stiffness_benchmark.py
```

Both read the frozen split from `../data_extraction/build_dataset_splits.py`
(train=demo4, val=demo1, test=demo3) and write to `../reports/`.
`analyze_data_two_color.py` also imports a manifest helper from
`../src/compliance_vla/policy/diagnose_language_grounding.py`.
