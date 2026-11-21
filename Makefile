PYTHON ?= python3
.PHONY: help env pins hydrate test smoke check env-full accept-s0 accept-s1 accept-s1.1 accept-s1.2 accept-s2.1 accept-s2.2 native-deps native-build adapter-build episode-pair episode-set paper-baseline spark-baseline

help:
	@echo "make env             Create the local isac virtual environment"
	@echo "make check           Run the portable tests and four-opportunity replay"
	@echo "make hydrate         Fetch all three locked external Git worktrees"
	@echo "make env-full        Install the pinned Sionna and RT sources in isac"
	@echo "make accept-s0       Run every Sprint 0 gate and archive the evidence (BACKENDS=\"cpu cuda\")"
	@echo "make native-deps     Install the missing OCUDU development libraries with apt-get (safe to repeat)"
	@echo "make native-build    Build the native Spark Release and correctness targets"
	@echo "make accept-s1.1     Build, then run every Sprint 1.1 gate and archive the evidence (BACKENDS=\"cpu cuda\", JOBS=8)"
	@echo "make accept-s1.2     Measure the DeepSig Table II row and the four-RX smoke on the S1.1 build, grade and archive"
	@echo "make accept-s1       Run the Sprint 1.1 gate, then the Sprint 1.2 gate (BACKENDS=\"cpu cuda\", JOBS=8)"
	@echo "make adapter-build   Build the project's native adapter out of tree against the accepted OCUDU build"
	@echo "make accept-s2.1     Build the adapter, compare both transmit references of each grant, archive the records"
	@echo "make accept-s2.2     Build the grant fixture of an episode, trace the matched pair and a small set, grade and archive"
	@echo "make episode-pair    Trace the reference pair alone into a new directory under data/episodes (CONFIG=, OUT=)"
	@echo "make episode-set     Trace an episode set into a new directory under data/episodes (CONFIG=, OUT=, VARIANT=)"
	@echo "make paper-baseline  Execute the DeepSig one-layer comparison alone on Spark"
	@echo "make spark-baseline  Execute the short four-RX smoke alone on Spark"

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

native-deps:
	bash scripts/install_native_deps.sh

native-build:
	bash scripts/build_ocudu.sh

accept-s1.1:
	bash scripts/accept_s1_1.sh $(if $(JOBS),--jobs $(JOBS)) $(BACKENDS)

accept-s1.2:
	bash scripts/accept_s1_2.sh

# Command-line variables such as BACKENDS and JOBS reach both gates. The second runs only if the first passed.
accept-s1:
	$(MAKE) accept-s1.1
	$(MAKE) accept-s1.2

adapter-build:
	$(PYTHON) scripts/native_adapter.py build

accept-s2.1:
	bash scripts/accept_s2_1.sh

accept-s2.2:
	bash scripts/accept_s2_2.sh

# Offline channel generation. The output directory must be new. VARIANT selects the Mitsuba variant.
VARIANT ?= llvm_ad_mono_polarized
STAMP := $(shell date -u +%Y%m%dT%H%M%SZ)

episode-pair:
	$(PYTHON) scripts/export_rt_episode.py --pair $(or $(CONFIG),configs/episodes/reference_pair.json) \
	    --output-dir $(or $(OUT),data/episodes/reference-pair-$(STAMP)) --variant $(VARIANT)

episode-set:
	$(PYTHON) scripts/export_rt_episode.py --set $(or $(CONFIG),configs/episodes/demo_set.json) \
	    --output-dir $(or $(OUT),data/episodes/demo-set-$(STAMP)) --variant $(VARIANT)

paper-baseline:
	bash scripts/run_spark_baseline.sh --profile paper --skip-build

spark-baseline:
	bash scripts/run_spark_baseline.sh
