"""The packaged core must stay identical to the code that produced the results.

`src/compliance_vla/` was built by lifting functions verbatim out of the
rig-side scripts that were actually run for the paper, and splitting them into
importable modules. Nothing about the algorithms was rewritten -- deliberately,
because a reimplementation that quietly diverged would mean the tested code and
the code behind the numbers were two different things.

That property is only worth claiming if something enforces it. These tests
compare each packaged function against its original, as an abstract syntax tree
with argument annotations stripped, so formatting, comments and the
argparse.Namespace-to-dataclass change are tolerated while any change to the
actual logic fails.

If a test here fails, the right response is usually to change BOTH copies, or to
delete the entry and say in the README that the packaged version has diverged --
not to loosen the comparison.
"""

import ast
from pathlib import Path

import pytest

PACKAGE = Path("src/compliance_vla")
TELEOP = Path("hardware/fr3_bilateral_teleop")
LABELING = TELEOP / "dataset_tools" / "labeling"
RIG_SCRIPTS = TELEOP / "scripts"

# packaged module -> (original file, [function/class names lifted verbatim])
PROVENANCE = {
    PACKAGE / "geometry.py": (
        LABELING / "extract_impedance_labels.py",
        [
            "quat_conjugate", "quat_multiply", "quat_to_rotvec", "quat_to_rotmat",
            "build_inplane_basis", "fit_contact_frame", "compute_pose_error",
            "numerically_differentiate", "rotate_force_to_contact_frame",
        ],
    ),
    PACKAGE / "extraction.py": (
        LABELING / "extract_impedance_labels.py",
        [
            "load_demo_csv", "_fit_one_window", "extract_axis",
            "nearest_sample_indices", "compute_mask", "extract_demo",
        ],
    ),
    PACKAGE / "synthetic.py": (
        LABELING / "extract_impedance_labels.py",
        ["synthetic_demo"],
    ),
    PACKAGE / "wrench.py": (
        RIG_SCRIPTS / "fit_residual_bias.py",
        [
            "load_sweep_csvs", "RFFRidgeBiasModel", "rms_per_axis",
            "session_split", "fit_and_report", "synthetic_dataset",
        ],
    ),
    PACKAGE / "frames.py": (
        PACKAGE / "policy" / "controller_frame_utils.py",
        ["rotate_diag_stiffness_to_base", "chunk_step_index"],
    ),
}


class _StripAnnotations(ast.NodeTransformer):
    """Removes argument and return annotations.

    The packaged copies swap an argparse.Namespace annotation for the
    equivalent frozen dataclass. That is a type annotation change and nothing
    else -- the attribute access in the body is identical -- so it must not be
    reported as a logic change.
    """

    def visit_arg(self, node):
        node.annotation = None
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        node.returns = None
        return self.generic_visit(node)


def _definitions(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stripped = _StripAnnotations().visit(node)
            out[node.name] = ast.dump(ast.fix_missing_locations(stripped))
    return out


@pytest.mark.parametrize(
    "packaged, original, name",
    [
        (packaged, original, name)
        for packaged, (original, names) in PROVENANCE.items()
        for name in names
    ],
    ids=lambda v: v if isinstance(v, str) else Path(v).stem,
)
def test_packaged_definition_matches_its_original(release_root, packaged, original, name):
    packaged_defs = _definitions(release_root / packaged)
    original_defs = _definitions(release_root / original)

    assert name in packaged_defs, f"{name} is missing from {packaged}"
    assert name in original_defs, f"{name} is missing from {original}"
    assert packaged_defs[name] == original_defs[name], (
        f"{name} in {packaged} has diverged from {original}. Update both copies, "
        f"or remove it from PROVENANCE and document the divergence.")


def test_every_provenance_source_exists(release_root):
    for packaged, (original, _) in PROVENANCE.items():
        assert (release_root / packaged).is_file(), f"missing packaged module {packaged}"
        assert (release_root / original).is_file(), f"missing original {original}"


def test_provenance_covers_the_public_surface_of_each_packaged_module(release_root):
    """Every non-private top-level definition in a provenance-tracked module
    must either be tracked or be new code introduced by the release.

    Listing the exceptions explicitly is what stops an untracked function from
    being added to a "verbatim" module without anyone noticing.
    """
    introduced_by_the_release = {
        "extraction.py": {"ExtractionConfig"},
        "wrench.py": {"BiasModelConfig", "per_axis_noise_floor"},
        "geometry.py": set(),
        "synthetic.py": set(),
        "frames.py": set(),
    }
    for packaged, (_, tracked) in PROVENANCE.items():
        defined = {n for n in _definitions(release_root / packaged) if not n.startswith("__")}
        allowed = set(tracked) | introduced_by_the_release[packaged.name]
        untracked = defined - allowed
        assert not untracked, (
            f"{packaged.name} defines {sorted(untracked)}, which is neither tracked as "
            f"verbatim nor listed as introduced by the release")
