# qphys -- development tasks.
PYTHON ?= python
PIP    ?= $(PYTHON) -m pip

.PHONY: help install dev test test-all lint format figures experiments palette lock info clean

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

install:  ## runtime install
	$(PIP) install -e .

dev:  ## development install + git hooks
	$(PIP) install -e '.[dev]'
	pre-commit install

test:  ## the fast suite: no training, no seed sweeps
	$(PYTHON) -m pytest -m "not slow" -q

test-all:  ## everything, including network training and seed sweeps
	$(PYTHON) -m pytest -q

lint:  ## ruff check + format check + mypy
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests
	$(PYTHON) -m mypy

format:  ## apply ruff fixes and formatting
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m ruff format src tests

figures:  ## regenerate every figure from results/
	$(PYTHON) -m qphys.common.figures

palette:  ## re-validate the chart palette (colour is computed, not eyeballed)
	$(PYTHON) -m qphys.common.palette "#2a78d6,#eb6834,#1baf7a,#4a3aa7" light all

experiments:  ## re-run every measured result (hours -- writes results/)
	$(PYTHON) -m qphys.cli run all

lock:  ## pin the environment that produced results/
	$(PIP) freeze --exclude-editable > requirements.lock

info:  ## resolved paths and switches
	$(PYTHON) -m qphys.cli info

clean:
	rm -rf build dist src/*.egg-info .pytest_cache .ruff_cache
