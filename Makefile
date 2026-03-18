VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
OMEGA := $(VENV)/bin/Omega-spatial
OMEGA_MODULE := $(PY) -m omega_spatial.cli
EXAMPLES_DIR := examples
OUTPUT_DIR := results/examples_run

.PHONY: help setup install run-examples

help:
	@echo "Targets:"
	@echo "  make setup         - create virtualenv and install package"
	@echo "  make install       - install package into existing virtualenv"
	@echo "  make run-examples  - run Omega-spatial on first file in examples/"

setup:
	python3 -m venv "$(VENV)"
	"$(PIP)" install -U pip
	"$(PIP)" install -e .

install:
	@if [ ! -x "$(PY)" ]; then echo "Virtualenv missing. Run: make setup"; exit 1; fi
	"$(PIP)" install -e .

run-examples:
	@if [ ! -x "$(PY)" ]; then echo "Virtualenv missing. Run: make setup"; exit 1; fi
	@if [ ! -d "$(EXAMPLES_DIR)" ]; then echo "examples/ not found. Create it and add data."; exit 1; fi
	@INPUT=$$(ls -1 "$(EXAMPLES_DIR)"/*.h5ad "$(EXAMPLES_DIR)"/*.csv "$(EXAMPLES_DIR)"/*.tsv "$(EXAMPLES_DIR)"/*.tar "$(EXAMPLES_DIR)"/*.tar.gz 2>/dev/null | head -n 1); \
	if [ -z "$$INPUT" ]; then \
		echo "No .h5ad/.csv/.tsv/.tar/.tar.gz found in examples/. Add your subsampled data first."; \
		exit 1; \
	fi; \
	echo "Running Omega-spatial with $$INPUT"; \
	$(OMEGA_MODULE) run --input "$$INPUT" --output "$(OUTPUT_DIR)"
