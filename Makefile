# One-command reproducibility. Run `make help` to see targets.
# Uses a plain Python venv + pip (no conda).
.PHONY: help setup reproduce test lint grade clean data

VENV := .venv
PY := $(VENV)/bin/python
CONFIG ?= config.yaml
OUTPUT_DIR ?=

help:
	@echo "make data       - download and verify UNSW-NB15 training CSV"
	@echo "make setup      - create .venv and install requirements"
	@echo "make reproduce  - cluster + evaluate, writing results/ (THE grading command)"
	@echo "make test       - run smoke tests"
	@echo "make lint       - byte-compile check (stdlib only)"
	@echo "make grade      - run the objective grading harness"
	@echo "make clean      - remove generated results"

data:
	$(PY) -m src.download_data

setup:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

# The single command a grader runs. Must recreate your reported results.
reproduce:
	$(PY) -m src.reproduce --config $(CONFIG) $(if $(OUTPUT_DIR),--output-dir "$(OUTPUT_DIR)",)

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m py_compile src/*.py tests/*.py

grade:
	$(PY) grading/grade.py

clean:
	rm -f results/*.png results/*.json results/euclidean/*.png results/euclidean/*.json results/mahalanobis/*.png results/mahalanobis/*.json
