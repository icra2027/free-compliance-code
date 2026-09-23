.PHONY: help install install-all test test-fast lint clean dist

help:
	@echo "make install         - install the numpy-only core plus test tools"
	@echo "make install-all     - also install the policy and analysis extras"
	@echo "make test            - run the full test suite"
	@echo "make test-fast       - skip the slow end-to-end extraction tests"
	@echo "make lint            - ruff check on the core package and tests"
	@echo "make dist            - build a clean reviewer tarball"

install:
	python3 -m pip install -e '.[dev]'

install-all:
	python3 -m pip install -e '.[dev,policy,analysis]'

test:
	python3 -m pytest

# The extraction tests run the full pipeline over a synthetic demonstration and
# dominate the runtime; this target is the one to use while iterating.
test-fast:
	python3 -m pytest -m 'not slow'

lint:
	python3 -m ruff check src tests

clean:
	rm -rf build dist src/*.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

# Excludes caches and build output, so the artifact a reviewer receives
# contains only the release itself.
dist: clean
	# --exclude='.git' rather than --exclude-vcs: the latter also strips
	# .gitignore, which IS part of the release (the hardware packages ship their own too).
	tar --exclude='.git' \
	    --exclude='__pycache__' --exclude='*.egg-info' \
	    --exclude='.pytest_cache' --exclude='.ruff_cache' \
	    --exclude='hardware/build' --exclude='hardware/install' --exclude='hardware/log' \
	    -czf ../compliance-vla-anonymous.tar.gz .
	@echo "wrote ../compliance-vla-anonymous.tar.gz"
