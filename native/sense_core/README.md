# Native sensing core

Specified, not yet built. This directory will hold the sensing core that every timed sensing stage runs in: CUDA kernels and TensorRT engines behind a C API, the sensor process `sense_host` and the Python binding `ocudu_cuda_isac.sensing.runtime`. [Code levels and the native core](../../docs/architecture/RUNTIME.md) | [sprint cards](../../docs/dev/DEVELOPMENT.md#phase-s3)

| Ticket | Adds |
|---|---|
| S3.6 `sense_core` | The C API, the observation view, ingest units, the cuSense front-end kernels, the engine host, plans, `sense_host`, the binding, `scripts/native_core.py` and the receipt `isac-core.json` |
| S6.12 `native_spectral` | Radar cube, delay-Doppler, clutter and CFAR kernels |
| S6.13 `native_angle` | Covariance, Bartlett and MUSIC kernels with a batched Hermitian eigensolver |
| S10.1, S10.2, S11.2 | Nothing here. The Aerial adapters link this core unchanged. |

Rules that hold from the first file placed here:

1. **No host.** Nothing in this directory includes or links OCUDU, Aerial or a simulator. Its `CMakeLists.txt` is its own project with `LANGUAGES CXX CUDA`, and the OCUDU adapter's build never compiles it.
2. **One input.** The core consumes the observation view of the [observation contract](../../docs/interfaces/OBSERVATION.md#the-observation-view) and returns decision records.
3. **One registry.** A unit carries the name of its Python reference under `src/ocudu_cuda_isac/sensing/`, and a parity test ties the two.
4. **No interpreter.** `sense_host` starts on a host with no Python. The binding is for tests and offline scoring.
5. **C boundary, common subset.** The public API is C with opaque handles. Sources stay within C++17 and CUDA C++17, so that the core builds beside the pinned OCUDU and inside the pinned Aerial tree.
