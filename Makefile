PYTHON ?= python3
.PHONY: help env pins hydrate test smoke check env-full accept-s0 accept-s1 accept-s1.1 accept-s1.2 accept-s2 accept-s2.1 accept-s2.2 accept-s2.3 accept-s2.4 accept-s3.1 accept-s3.2 accept-s3.3 accept-s3.4 accept-s5.2 native-deps native-build native-build-o3 adapter-build adapter-build-o3 episode-pair episode-set paper-baseline spark-baseline

help:
	@echo "make env             Create the local isac virtual environment"
	@echo "make check           Run the portable tests and four-opportunity replay"
	@echo "make hydrate         Fetch the locked external Git worktrees: three dependencies, two reference sources"
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
	@echo "make accept-s2.3     Build the native channel application, apply the archived pair with its grid gain, grade and archive"
	@echo "make accept-s2.4     Build the receiver dispatch, decode the replayed grids on the host and the CUDA path, grade and archive (REHEARSAL=1 on a host without CUDA)"
	@echo "make accept-s2       Audit the four accepted Sprint 2 runs as one chain and close Sprint 2 (RERUN=1 runs the four gates again first)"
	@echo "make accept-s3.1     Build the tapped receiver replay, export the pilot LS (O2) from the grids of the accepted Sprint 2.4 run, compare the decodes, grade and archive (REHEARSAL=1 on a host without CUDA)"
	@echo "make accept-s3.2     Build the observation ring and the timed path, publish the pair of the accepted Sprint 3.1 run to a consumer in the modes audit and timed, grade and archive (REHEARSAL=1 on a host without CUDA)"
	@echo "make accept-s3.3     Form the radar cube, the delay-Doppler view and the causal features of the pair from the ring of the accepted Sprint 3.2 run, from the files of its observer and from a live ring, verify the known return, grade and archive (REHEARSAL=1 on a host without CUDA)"
	@echo "make accept-s3.4     Train one linear presence head on receiver-derived records of an episode set, freeze its artifact, score held-out records in a second CUDA process and offline, grade and archive (S33_RUN=, EPISODE_SET=, REHEARSAL=1 on a host without CUDA)"
	@echo "make accept-s5.2     Build the patched receiver from its own tree and the adapter with the O3 route, export the receiver's CUDA channel estimate on the plans of the accepted Sprint 3.1 run, compare it with the matched estimator and the decodes with the unpatched build, grade and archive (S31_RUN=, JOBS=8)"
	@echo "make native-build-o3 Build the pinned receiver with the patch series of patches/ocudu, from a separate tree, into build/ocudu-spark-o3"
	@echo "make adapter-build-o3 Build the adapter against the patched receiver, with the O3 route, into build/ocudu-adapter-o3"
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

# The Aerial reference source tracks about 400 MB of test vectors and engines with Git LFS. Nothing here
# reads them, so they stay pointer files. No dependency uses Git LFS.
hydrate: pins
	GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --checkout
	$(PYTHON) scripts/check_pins.py --worktrees --references

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

accept-s2.3:
	bash scripts/accept_s2_3.sh

# REHEARSAL=1 grades the host path on a build without CUDA. It ends REHEARSED and never ACCEPTED.
accept-s2.4:
	bash scripts/accept_s2_4.sh $(if $(REHEARSAL),--host-rehearsal)

# The four accepted runs as one chain, read from their archives. RERUN=1 first runs the four gates
# again, in order, each on the run of the one before.
accept-s2:
	bash scripts/accept_s2.sh $(if $(RERUN),--rerun)

# The control is the accepted Sprint 2.4 run: the newest one, or S24_RUN. REHEARSAL=1 grades the
# host path against a rehearsed Sprint 2.4 run. It ends REHEARSED and never ACCEPTED.
accept-s3.1:
	bash scripts/accept_s3_1.sh $(if $(S24_RUN),--s24-run $(S24_RUN)) $(if $(REHEARSAL),--host-rehearsal)

# The control is the accepted Sprint 3.1 run: the newest one, or S31_RUN. It names the plan of the
# pair and the Sprint 2.4 run with the untapped decodes. REHEARSAL=1 grades the host path against
# a rehearsed Sprint 3.1 run. It ends REHEARSED and never ACCEPTED.
accept-s3.2:
	bash scripts/accept_s3_2.sh $(if $(S31_RUN),--s31-run $(S31_RUN)) $(if $(REHEARSAL),--host-rehearsal)

# The control is the accepted Sprint 3.2 run: the newest one, or S32_RUN. It holds the records its
# consumer kept from the ring and the files of its audited observer. This gate changes no native
# source and runs the reference of the sensing core, NumPy on the CPU. REHEARSAL=1 grades the host
# path against a rehearsed Sprint 3.2 run. It ends REHEARSED and never ACCEPTED.
accept-s3.3:
	bash scripts/accept_s3_3.sh $(if $(S32_RUN),--s32-run $(S32_RUN)) $(if $(REHEARSAL),--host-rehearsal)

# The control is the accepted Sprint 3.3 run: the newest one, or S33_RUN. The stimulus is an episode
# set under data/episodes: the newest demo-set-*, or EPISODE_SET. "make episode-set" traces one.
# This gate changes no native source. It trains and scores the reference of the head, PyTorch in
# the interpreter of isac, which needs the full profile. REHEARSAL=1 grades the host path against
# a rehearsed Sprint 3.3 run and scores on the CPU. It ends REHEARSED and never ACCEPTED.
accept-s3.4:
	bash scripts/accept_s3_4.sh $(if $(S33_RUN),--s33-run $(S33_RUN)) $(if $(EPISODE_SET),--episode-set $(EPISODE_SET)) $(if $(REHEARSAL),--host-rehearsal)

# Sprint 5.2 sits inside phase S3. The control is the accepted Sprint 3.1 run: the newest one, or
# S31_RUN. The gate builds the patched receiver from a separate tree into build/ocudu-spark-o3 and
# the adapter with the O3 route into build/ocudu-adapter-o3. The submodule and the accepted build
# are not touched. There is no rehearsal: the estimate exists only on a CUDA device.
accept-s5.2:
	bash scripts/accept_s5_2.sh $(if $(S31_RUN),--s31-run $(S31_RUN)) $(if $(JOBS),--jobs $(JOBS))

native-build-o3:
	$(PYTHON) scripts/native_ocudu.py build --patched $(if $(JOBS),--jobs $(JOBS))

adapter-build-o3:
	$(PYTHON) scripts/native_adapter.py build --o3

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
