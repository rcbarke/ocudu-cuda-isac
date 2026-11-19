# Environments

Keep the CPU fixture usable with Python's standard library. For trace generation create a separate environment from the two pinned Sionna source paths. Record installed Mitsuba, Dr.Jit, PyTorch, CUDA/runtime and exact package versions from an actually working run. Native OCUDU builds use the branch's CMake dependency instructions and explicit GPU architecture. Keep the real-time launch independent of the offline ray-tracing Python environment. See [UPSTREAM.md](../docs/integration/UPSTREAM.md) and [SPARK.md](../docs/demo/SPARK.md).
