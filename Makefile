PYTHON ?= python3
.PHONY: help pins hydrate test smoke check spark-baseline

help:
	@echo "make check    Verify parent submodule pins, run CPU tests and run the smoke fixture"
	@echo "make hydrate  Fetch pinned upstream source trees (network access required)"
	@echo "Spark demo: docs/demo/SPARK.md and docs/demo/ADVISOR_DEMO.md"

pins:
	$(PYTHON) scripts/check_pins.py

hydrate: pins
	git submodule update --init --checkout -- deps/ocudu deps/sionna deps/sionna-rt
	$(PYTHON) scripts/check_pins.py --worktrees

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

smoke:
	bash scripts/run_smoke.sh

spark-baseline:
	bash scripts/run_spark_baseline.sh

check: pins test smoke
