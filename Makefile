PYTHON ?= python3
.PHONY: help env pins hydrate test smoke check env-full accept-s0 native-build paper-baseline spark-baseline

help:
	@echo "make env             Create the local isac virtual environment"
	@echo "make check           Run the portable tests and four-opportunity replay"
	@echo "make hydrate         Fetch all three locked external Git worktrees"
	@echo "make env-full        Install the pinned Sionna and RT sources in isac"
	@echo "make accept-s0       Run every Sprint 0 gate and archive the evidence (BACKENDS=\"cpu cuda\")"
	@echo "make native-build    Build the native Spark Release and correctness targets"
	@echo "make paper-baseline  Execute the DeepSig one-layer comparison on Spark"

env:
	bash scripts/bootstrap_env.sh core

env-full:
	bash scripts/bootstrap_env.sh full

pins:
	$(PYTHON) scripts/check_pins.py

hydrate: pins
	git submodule update --init --checkout
	$(PYTHON) scripts/check_pins.py --worktrees

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

smoke:
	PYTHON="$(PYTHON)" bash scripts/run_smoke.sh

check: pins test smoke

accept-s0:
	bash scripts/accept_s0.sh $(BACKENDS)

native-build:
	bash scripts/build_ocudu.sh

paper-baseline:
	bash scripts/run_spark_baseline.sh --profile paper --skip-build

spark-baseline:
	bash scripts/run_spark_baseline.sh
