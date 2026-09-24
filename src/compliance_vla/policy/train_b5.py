#!/usr/bin/env python3
"""B5-only entrypoint -- thin wrapper, kept for the exact command line used to
produce reports/b5_seed0_smoke_train_log.json (the B5 smoke test).

src/compliance_vla/policy/train_policy.py generalizes it (adds the other baselines,
shares one dataset/training loop across them for the "matched data and
backbone" requirement). New usage should prefer:

    python -m compliance_vla.policy.train_policy --policy b5 ...

This wrapper just forces --policy b5 and forwards every other argument.
"""

import sys

from compliance_vla.policy import train_policy

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--policy", "b5", *sys.argv[1:]]
    train_policy.main()
