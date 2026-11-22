# Native sensing core

The sensing core that every timed sensing stage runs in: CUDA kernels and a TensorRT engine behind a C API, the sensor process `sense_host` and the Python binding `ocudu_cuda_isac.sensing.runtime`. Sprint 3.6 built it for the cuSense-like branch of Sprint 3.5. [Contract of the core](../../docs/interfaces/SENSE_CORE.md) | [code levels](../../docs/architecture/RUNTIME.md) | [sprint cards](../../docs/dev/DEVELOPMENT.md#phase-s3)

```bash
python scripts/native_core.py probe              # checkpoint D0: TensorRT builds and runs an engine on this GPU
python scripts/native_core.py build              # build/sense-core, CTest, and the receipt isac-core.json
python scripts/native_core.py build --host-only  # the host twin: no CUDA compiler, for a development host
python tests/unit/test_sense_core.py -v          # the unit gate, beside the Python reference
```

| Ticket | Adds |
|---|---|
| S3.6 `sense_core` | Built and accepted on the Spark (run `s36-spark-ecf8-20261122T194723Z`, TensorRT 11.5.0.63): the C API, the observation view, the ingest units, the cuSense front-end kernels with their host twin, the engine host, the plan, `sense_host`, `sense_feed`, `sense_engine`, `sense_probe`, the binding, `scripts/native_core.py` and the receipt `isac-core.json` |
| S6.12 `native_spectral` | Radar cube, delay-Doppler, clutter and CFAR kernels |
| S6.13 `native_angle` | Covariance, Bartlett and MUSIC kernels with a batched Hermitian eigensolver |
| S10.1, S10.2, S11.2 | Nothing here. The Aerial adapters link this core unchanged. |

| Directory | Holds |
|---|---|
| `include/sense_core/` | `sense_core.h`, the C API: view, plan, answer, decision, counters |
| `src/core/` | SHA-256, JSON, the reader of a frozen artifact, the registry, the tolerances |
| `src/kernels/` | `units.h`, the units written once; the CUDA device and its host twin |
| `src/engine/` | The backends `tensorrt`, `libtorch` and `host_reference`, and the engine manifest |
| `src/plan/` | The window, the submission, the decision, the JSON lines, the optional stage |
| `host/` | `sense_host`, its loop and its record sources |
| `tools/` | `sense_feed`, `sense_engine`, `sense_probe` |
| `python/` | The pybind11 extension |
| `tests/` | The unit cases of CTest |

Rules that hold for every file here:

1. **No host.** Nothing in this directory includes or links OCUDU, Aerial or a simulator. Its `CMakeLists.txt` is its own project with `LANGUAGES C CXX CUDA`, and the OCUDU adapter's build never compiles it. `tests/unit/test_core_link.py` checks the link lines.
2. **One input.** The core consumes the observation view of the [observation contract](../../docs/interfaces/OBSERVATION.md#the-observation-view) and returns decision records.
3. **One registry.** A unit carries the name of its Python reference under `src/ocudu_cuda_isac/sensing/`, and a parity test ties the two.
4. **No interpreter.** `sense_host` starts on a host with no Python. The binding is for tests and offline scoring.
5. **C boundary, common subset.** The public API is C with opaque handles. Sources stay within C++17 and CUDA C++17, so that the core builds beside the pinned OCUDU and inside the pinned Aerial tree. The optional LibTorch backend is the one file compiled as C++20, because the headers of its library need it.
6. **Written once.** A unit of the ingest or of the front end is one function in `src/kernels/units.h`, compiled by `nvcc` into a kernel and by the host compiler into the twin, without fused multiply-add on either side.
7. **Bounded.** A plan allocates when it is created and never after, makes no device call while a window is open, and submits a window once.
