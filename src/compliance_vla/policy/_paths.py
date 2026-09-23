"""Locates the release's sibling directories from inside the installed package.

The policy code loads a few things that are not Python modules of this package:
the research/analysis helpers under ``scripts/``, the label extraction under
``hardware/fr3_bilateral_teleop/dataset_tools/labeling/``, and calibration artifacts such
as ``scripts/tool_offset.npy``. In the original working repository those sat at
fixed relative offsets from the training code, and each module recomputed them
with its own ``os.path.dirname`` chain.

Resolving them in one place instead means the release layout is stated once. It
also means an editable install (where the package still lives inside the release
tree) and a copied tree both work, while a non-editable install into
site-packages -- where these siblings genuinely are not present -- fails with a
message saying so rather than with a confusing ImportError from three frames
down.

Set ``COMPLIANCE_VLA_ROOT`` to override, which is what to do if the package is
installed somewhere other than the release tree.
"""

import os
import sys
from pathlib import Path

__all__ = [
    "RELEASE_ROOT",
    "SCRIPTS_DIR",
    "EXTERNAL_SCRIPTS",
    "REPORTS_DIR",
    "ensure_on_sys_path",
    "require_release_tree",
]


def _find_release_root() -> Path:
    override = os.environ.get("COMPLIANCE_VLA_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    # .../<release>/src/compliance_vla/policy/_paths.py -> parents[3] == <release>
    return Path(__file__).resolve().parents[3]


RELEASE_ROOT = _find_release_root()
SCRIPTS_DIR = RELEASE_ROOT / "scripts"
# Where extract_impedance_labels.py lives. The name predates the move into dataset_tools/
# and is kept because compliance_vla.policy.labels re-exports it.
EXTERNAL_SCRIPTS = RELEASE_ROOT / "hardware" / "fr3_bilateral_teleop" / "dataset_tools" / "labeling"
REPORTS_DIR = RELEASE_ROOT / "reports"


def require_release_tree() -> None:
    """Raises with an actionable message if the sibling directories are absent."""
    missing = [str(p) for p in (SCRIPTS_DIR, EXTERNAL_SCRIPTS) if not p.is_dir()]
    if missing:
        raise RuntimeError(
            "compliance_vla.policy needs the release tree's sibling directories, but "
            f"these are missing: {missing}. This happens when the package is installed "
            "outside the release tree. Install it editable from the release root "
            "(pip install -e .), or set COMPLIANCE_VLA_ROOT to the release directory."
        )


def ensure_on_sys_path() -> None:
    """Puts scripts/ and the label-extraction directory on sys.path, nearest-first.

    The modules imported from there (dataset_io, panda_fk, extract_impedance_labels,
    run_extraction_on_dataset) are plain scripts rather than an installed package,
    which is how they were run for the paper and how they remain runnable
    standalone on the rig.
    """
    require_release_tree()
    for path in (EXTERNAL_SCRIPTS, SCRIPTS_DIR):
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)
