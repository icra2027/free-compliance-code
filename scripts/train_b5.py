#!/usr/bin/env python3
"""Day 11 B5-only entrypoint -- thin wrapper, kept for the exact command line
used to produce reports/b5_seed0_smoke_train_log.json (Day 11's smoke test).

Day 12 generalized this into scripts/train_policy.py (adds B0/B2, shares one
dataset/training loop across all three baselines per proposal §6.1's
"matched data and backbone" requirement). New usage should prefer:

    python scripts/train_policy.py --policy b5 ...

This wrapper just forces --policy b5 and forwards every other argument.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_policy  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--policy", "b5", *sys.argv[1:]]
    train_policy.main()
