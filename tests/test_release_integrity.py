"""Structural checks on the release as a shipped artifact.

Everything else in this suite tests behaviour. This module tests that what a
reviewer downloads is coherent: that every Python file still parses after the
anonymization rewrite, that the numpy-only core really is numpy-only, that the
two forked hardware packages are fully renamed and carry their upstream licenses,
and that no build droppings were shipped.
"""

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib

# Working directories that are never part of the release (VCS metadata, caches,
# virtualenvs, colcon output).
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
             "build", "install", "log", ".venv", "venv", "node_modules", ".eggs"}

CORE_MODULES = [
    "geometry.py", "extraction.py", "synthetic.py", "wrench.py",
    "benchmark.py", "ensembling.py", "safety.py", "frames.py",
]


def _python_files(release_root: Path):
    for path in sorted(release_root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.relative_to(release_root).parts):
            continue
        yield path


def test_every_shipped_python_file_parses(release_root):
    """The anonymization rewrites file contents, so a bad substitution would
    show up here as a syntax error rather than at a reviewer's first run."""
    broken = {}
    for path in _python_files(release_root):
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            broken[str(path.relative_to(release_root))] = f"line {exc.lineno}: {exc.msg}"
    assert not broken, f"files that no longer parse: {broken}"


def test_required_top_level_files_are_present(release_root):
    for name in ("README.md", "LICENSE", "pyproject.toml", "Makefile"):
        assert (release_root / name).is_file(), f"missing {name}"


def test_pyproject_is_valid_and_declares_a_light_core(release_root):
    with (release_root / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)

    project = config["project"]
    assert project["name"] == "compliance-vla"
    # The core must stay installable without a deep-learning stack: that is what
    # lets a reviewer verify the method's claims on an ordinary machine.
    required = " ".join(project["dependencies"]).lower()
    assert "torch" not in required
    assert "torch" in " ".join(project["optional-dependencies"]["policy"]).lower()


def test_the_core_package_does_not_import_torch(release_root):
    """Checked statically, so the result does not depend on whether torch
    happens to be installed in the environment running the tests."""
    offenders = []
    for name in CORE_MODULES:
        tree = ast.parse((release_root / "src/compliance_vla" / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(n.split(".")[0] in {"torch", "lerobot", "pandas", "sklearn", "matplotlib"}
                   for n in names):
                offenders.append(f"{name}: {names}")
    assert not offenders, f"core modules pulled in a heavy dependency: {offenders}"


# Run in a subprocess with torch actively blocked. Asserting against
# sys.modules in-process would be order-dependent -- test_policy.py imports the
# policy subpackage, so a full-suite run would see it already loaded -- and
# would not prove the core works when torch is genuinely absent.
_CORE_IMPORTS_WITHOUT_TORCH = """
import sys


class TorchBlocker:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is unavailable in this environment")
        return None


sys.meta_path.insert(0, TorchBlocker())

import compliance_vla

for module in compliance_vla.__all__:
    if module != "policy":
        __import__("compliance_vla." + module)

assert "compliance_vla.policy" not in sys.modules, "importing the package pulled in policy"
assert "torch" not in sys.modules, "something imported torch"
print("OK")
"""


def test_the_core_imports_with_torch_unavailable():
    """A reviewer without a deep-learning stack must still be able to run the
    part of this release that carries the method's claims."""
    result = subprocess.run(
        [sys.executable, "-c", _CORE_IMPORTS_WITHOUT_TORCH],
        capture_output=True, text=True)
    assert result.returncode == 0, (
        f"the core failed to import without torch:\n{result.stderr}")
    assert result.stdout.strip().endswith("OK")


def test_all_core_modules_are_importable_and_listed(release_root):
    import compliance_vla

    listed = set(compliance_vla.__all__)
    on_disk = {Path(name).stem for name in CORE_MODULES}
    assert on_disk <= listed, f"undocumented core modules: {on_disk - listed}"
    for module in on_disk:
        __import__(f"compliance_vla.{module}")


def test_no_stray_artifacts_outside_the_ignored_caches(release_root):
    """Caches regenerate on every test run, so they are gitignored and excluded
    by `make dist` rather than asserted absent here. What this catches is junk
    that no cache directory explains -- a stray .pyc beside its source, or an
    editor/OS dropping."""
    junk = [
        str(p.relative_to(release_root))
        for pattern in ("*.pyc", "*.pyo", ".DS_Store", "*.swp", "*~")
        for p in release_root.rglob(pattern)
        if not any(part in SKIP_DIRS for part in p.relative_to(release_root).parts)
    ]
    assert not junk, f"stray artifacts present in the release: {junk}"


def test_gitignore_covers_everything_the_distribution_must_exclude(release_root):
    """`make dist` and .gitignore must agree on what never ships; if they drift,
    a reviewer tarball can pick up caches or colcon build output."""
    gitignore = (release_root / ".gitignore").read_text()
    for pattern in ("__pycache__/", "*.egg-info/", ".pytest_cache/",
                    "hardware/build/", "hardware/install/", "hardware/log/", ".DS_Store"):
        assert pattern in gitignore, f".gitignore does not cover {pattern}"

    makefile = (release_root / "Makefile").read_text()
    for excluded in ("__pycache__", "*.egg-info", ".pytest_cache",
                     "hardware/build", "hardware/install", "hardware/log"):
        assert excluded in makefile, f"make dist does not exclude {excluded}"


def test_no_colcon_build_output_is_shipped(release_root):
    """Building the hardware packages in place leaves build/, install/ and log/
    next to them; those are working directories, not part of the release."""
    present = [d for d in ("build", "install", "log") if (release_root / "hardware" / d).exists()]
    assert not present, f"colcon output present under hardware/: {present} -- remove before distributing"


# --------------------------------------------------------------------------
# Forked hardware packages
# --------------------------------------------------------------------------

# release package -> (upstream package it was forked from, its license file)
FORKS = {
    "variable_impedance_controllers": ("crisp_controllers", "LICENSE.md"),
    "fr3_bilateral_teleop": ("franka_ros2_teleop", "LICENSE"),
}

# Files that are meant to name the upstream package: its license, the record of
# what was changed, and its own history.
_PROVENANCE_FILES = {"LICENSE", "LICENSE.md", "NOTICE", "CHANGELOG.md"}


@pytest.mark.parametrize("package", sorted(FORKS))
def test_each_fork_is_a_ros2_package_under_its_own_name(release_root, package):
    root = release_root / "hardware" / package
    manifest = (root / "package.xml").read_text()
    assert re.search(rf"<name>{package}</name>", manifest), f"{package}: package.xml <name> mismatch"
    assert f"project({package})" in (root / "CMakeLists.txt").read_text()
    assert (root / "include" / package).is_dir(), f"{package}: include/{package}/ missing"


@pytest.mark.parametrize("package", sorted(FORKS))
def test_each_fork_exports_its_plugins_under_its_own_namespace(release_root, package):
    """A plugin still registered under the upstream name would clash with the
    original package if both were installed in one workspace."""
    plugins = (release_root / "hardware" / package / f"{package}.xml").read_text()
    assert f'<library path="{package}">' in plugins
    names = re.findall(r'<class name="([^"]+)"\s+type="([^"]+)"', plugins)
    assert names, f"{package}: no plugin classes found"
    for name, cpp_type in names:
        assert name.startswith(f"{package}/"), name
        assert cpp_type.startswith(f"{package}::"), cpp_type


@pytest.mark.parametrize("package", sorted(FORKS))
def test_each_fork_keeps_its_upstream_license_and_records_its_base_commit(release_root, package):
    """Redistributing a modified upstream package means shipping its license and
    saying where it came from and what changed."""
    upstream, license_file = FORKS[package]
    root = release_root / "hardware" / package
    assert (root / license_file).is_file(), f"{package}: upstream {license_file} missing"
    notice = (root / "NOTICE").read_text()
    assert upstream in notice, f"{package}: NOTICE does not name {upstream}"
    assert re.search(r"\bcommit [0-9a-f]{40}\b", notice), f"{package}: NOTICE lacks a full base commit"
    assert "Changes from the original" in notice


def test_no_upstream_package_name_survives_the_rename(release_root):
    """In code and config, the upstream package names may appear only inside
    URLs. A leftover include path, controller type string or FindPackageShare()
    would silently resolve to the original package, or fail, on a machine that
    has it installed. Prose (Markdown) and the provenance files are exempt:
    they are where the upstream is supposed to be named."""
    url = re.compile(r"https?://\S+")
    upstream = re.compile(r"\b(?:crisp_controllers|franka_ros2_teleop)\b")
    leftovers = []
    for path in sorted(release_root.rglob("*")):
        rel = path.relative_to(release_root)
        if not path.is_file() or any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.name in _PROVENANCE_FILES or path.suffix == ".md" or rel.parts[0] == "tests":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if upstream.search(url.sub("", line)):
                leftovers.append(f"{rel}:{lineno}")
    assert not leftovers, f"upstream package names left after the rename: {leftovers}"


@pytest.mark.parametrize("package", sorted(FORKS))
def test_every_installed_program_exists(release_root, package):
    """colcon is not run by this suite, so a script moved without updating its
    install(PROGRAMS ...) entry would only surface as a build failure on the rig."""
    root = release_root / "hardware" / package
    cmake = (root / "CMakeLists.txt").read_text()
    listed = [
        entry
        for block in re.findall(r"install\(PROGRAMS(.*?)DESTINATION", cmake, flags=re.S)
        for entry in block.split()
    ]
    missing = [entry for entry in listed if not (root / entry).is_file()]
    assert not missing, f"{package}: install(PROGRAMS) lists missing files: {missing}"


def test_dataset_tools_hold_only_offline_code(release_root):
    """dataset_tools/ is the no-ROS half of fr3_bilateral_teleop: everything in it
    must run on a machine without a ROS 2 install."""
    tools = release_root / "hardware" / "fr3_bilateral_teleop" / "dataset_tools"
    scripts = sorted(tools.rglob("*.py"))
    assert scripts, "dataset_tools/ is empty"
    ros = [str(p.relative_to(tools)) for p in scripts
           if re.search(r"^\s*(?:import|from)\s+(?:rclpy|rosbag2_py)\b", p.read_text(), flags=re.M)]
    assert not ros, f"dataset_tools/ scripts importing ROS: {ros}"
