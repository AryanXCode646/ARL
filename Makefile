.PHONY: help install install-dev install-all test test-cov lint format typecheck clean

PYTHON ?= python3

help:
	@echo "AdaptiveRL Development Tasks:"
	@echo "  make install       Install package in editable mode"
	@echo "  make install-dev   Install development dependencies"
	@echo "  make install-all   Install all dependencies (including RL engines)"
	@echo "  make test          Run test suite"
	@echo "  make test-cov      Run test suite with coverage report"
	@echo "  make lint          Run code linter checks"
	@echo "  make format        Format codebase"
	@echo "  make typecheck     Run static type checker"
	@echo "  make clean         Remove build and cache artifacts"

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

install-all:
	$(PYTHON) -m pip install -e ".[all]"

test:
	$(PYTHON) -m pytest -v tests/

test-cov:
	$(PYTHON) -m pytest --cov=adaptive_rl --cov-report=term-missing tests/

lint:
	$(PYTHON) -m ruff check src/ tests/

format:
	$(PYTHON) -m ruff format src/ tests/

typecheck:
	$(PYTHON) -m mypy src/

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/ .coverage htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} +
